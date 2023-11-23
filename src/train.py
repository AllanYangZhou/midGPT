import typing as tp
from functools import partial
from dataclasses import dataclass
import os
import equinox as eqx
import jax
from jax.experimental import mesh_utils
import jmp
import optax
import orbax.checkpoint as ocp
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import trange
from .model import GPT, GPTConfig

jax.config.update("jax_threefry_partitionable", True)

jnp, jrandom, vmap, scan = jax.numpy, jax.random, jax.vmap, jax.lax.scan
P, KeyArray = jax.sharding.PartitionSpec, tp.Any
jtu, NamedSharding = jax.tree_util, jax.sharding.NamedSharding
with_sharding_constraint = jax.lax.with_sharding_constraint
Array, Mesh = jax.Array, jax.sharding.Mesh


@dataclass
class ExperimentConfig:
    rundir: str  # Directory containing ckpts and logs.
    data_dir: str  # Dataset directory
    learning_rate: float
    batch_size: int  # GLOBAL across all devices (not per device)
    warmup_steps: int
    min_lr: float  # Final LR after decay
    lr_decay_steps: int
    max_steps: int  # No. of grad steps: iters / g_accum_iters
    beta2: float
    weight_decay: float
    eval_interval: int
    policy: str
    g_accum_iters: int  # Accumulate this many grads before step
    shard_model: bool
    model_config: GPTConfig
    debug: bool = False


def get_batch(data, block_size: int, batch_size: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.take(data, np.arange(block_size) + ix[:, None], axis=0).astype(np.int32)
    y = np.take(data, np.arange(1, block_size + 1) + ix[:, None], axis=0).astype(np.int32)
    return x, y


def make_training_fns(
        config: ExperimentConfig, optimizer: optax.GradientTransformationExtraArgs,
        mesh: Mesh, shard_model: bool) -> tp.Tuple[tp.Callable, tp.Callable]:
    policy = jmp.get_policy(config.policy)
    def loss_fn(model: GPT, x: Array, y: Array, key: tp.Optional[KeyArray]) -> Array:
        if key is not None:
            key = jrandom.split(key, x.shape[0])
        logits = vmap(model)(x, key=key)
        orig_dtype = logits.dtype
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), y)  # compute loss in float32
        return loss.astype(orig_dtype).mean()

    @partial(eqx.filter_jit, donate='all')
    def step(model: GPT, opt_state, x: Array, y: Array, key: KeyArray):
        model_half = policy.cast_to_compute(model)
        loss, grad = eqx.filter_value_and_grad(loss_fn)(model_half, x, y, key)
        grad = shard_gpt(grad, mesh, shard_model)
        grad = policy.cast_to_param(grad)
        updates, opt_state = optimizer.update(grad, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    data_sharding = NamedSharding(mesh, P('data', None))
    fast_loss_fn = eqx.filter_jit(loss_fn)
    def evaluate(model: GPT, data: np.ndarray) -> Array:
        model = policy.cast_to_compute(model)
        model = eqx.Partial(model, inference=True)
        tot_loss = jnp.zeros(())
        for i in range(200):
            x, y = get_batch(data, config.model_config.block_size, config.batch_size)
            x, y = jax.device_put((x, y), data_sharding)
            loss = fast_loss_fn(model, x, y, None)
            tot_loss = tot_loss + loss
        return tot_loss / 200

    return step, evaluate


def get_layers(model: GPT, layer_cls: tp.Type[eqx.Module]) -> tp.Iterable[eqx.Module]:
    """Get all layers of model matching layer_cls."""
    matches_cls = lambda x: isinstance(x, layer_cls)
    return filter(lambda x: matches_cls(x), jtu.tree_leaves(model, is_leaf=matches_cls))


def count_params(model: GPT) -> int:
    dupe = jnp.size(model.lm_head.weight)  # embedding and final layer are shared.
    tot = sum([jnp.size(x) for x in jtu.tree_leaves(model) if isinstance(x, jax.Array)])
    return tot - dupe - jnp.size(model.wpe.weight)  # non-embedding only.


def shard_gpt(
        model: GPT, mesh: Mesh, shard_model: bool, sharding_fn=with_sharding_constraint
) -> eqx.Module:
    """Shard model parameters (or replicate if shard_model is False)."""
    if shard_model:
        lin_sharding = NamedSharding(mesh, P(None, 'data'))
        ln_sharding = NamedSharding(mesh, P('data',))
    else:
        # currently, no strategy for biases
        lin_sharding = NamedSharding(mesh, P(None, None))
        ln_sharding = NamedSharding(mesh, P(None,))
    get_lin_wts = lambda m: [l.weight for l in get_layers(m, (eqx.nn.Linear, eqx.nn.Embedding))]
    sharded_lin_wts = [sharding_fn(w, lin_sharding) for w in get_lin_wts(model)]
    model = eqx.tree_at(get_lin_wts, model, sharded_lin_wts)

    get_ln_wts = lambda m: [l.weight for l in get_layers(m, eqx.nn.LayerNorm)]
    sharded_ln_wts = [sharding_fn(w, ln_sharding) for w in get_ln_wts(model)]
    model = eqx.tree_at(get_ln_wts, model, sharded_ln_wts)

    n_wts = len([x for x in jtu.tree_leaves(model) if isinstance(x, jax.Array)])
    assert n_wts == len(sharded_lin_wts) + len(sharded_ln_wts), 'Some parameters are not being sharded!'
    return model


def train(config: ExperimentConfig):
    writer = SummaryWriter(os.path.join(config.rundir, 'logs'), flush_secs=30)
    devices = jax.devices()
    print(devices)
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=('data',))

    train_data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    options = ocp.CheckpointManagerOptions(
        max_to_keep=1, save_interval_steps=config.eval_interval)
    mngr = ocp.CheckpointManager(
        os.path.abspath(os.path.join(config.rundir, 'ckpt_mngr')),
        ocp.PyTreeCheckpointer(),
        options=options)

    # optax operates on iters, not grad steps
    warmup_iters = config.warmup_steps * config.g_accum_iters
    lr_decay_iters = config.lr_decay_steps * config.g_accum_iters
    scheduler = optax.warmup_cosine_decay_schedule(
        0, config.learning_rate, warmup_iters, lr_decay_iters, end_value=config.min_lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=config.beta2),
        optax.add_decayed_weights(config.weight_decay),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
        optax.apply_every(config.g_accum_iters),
    )
    step, evaluate = make_training_fns(config, optimizer, mesh, config.shard_model)

    key = jrandom.PRNGKey(0)
    def init_sharded_model(model_key):
        model = GPT(config.model_config, model_key)
        return shard_gpt(model, mesh, config.shard_model)
    key, key1 = jrandom.split(key)
    # Use jit with sharding constraints to init sharded model.
    model = eqx.filter_jit(init_sharded_model)(key1)
    print(f'Model has {count_params(model)} parameters.')
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    first_step = 0
    if mngr.latest_step() is not None:  # Restore existing checkpoint.
        model_leaves, opt_state_leaves = mngr.restore(mngr.latest_step())
        model = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)
        opt_state = jtu.tree_unflatten(jtu.tree_structure(opt_state), opt_state_leaves)
        first_step = mngr.latest_step() + 1
    data_sharding = NamedSharding(mesh, P('data', None))
    postfix_values = {}  # values to display in the progress bar
    max_iters = config.max_steps * config.g_accum_iters
    pbar = trange(first_step, max_iters, initial=first_step, total=max_iters)
    eval_interval = config.eval_interval * config.g_accum_iters
    for itr in pbar:
        if not config.debug and (itr % eval_interval == 0):
            train_loss = evaluate(model, train_data).item()
            val_loss = evaluate(model, val_data).item()
            postfix_values['train_loss'] = train_loss
            postfix_values['val_loss'] = val_loss
            writer.add_scalar('loss/train', train_loss, itr)
            writer.add_scalar('loss/val', val_loss, itr)
        key, key1 = jrandom.split(key)
        x, y = get_batch(train_data, config.model_config.block_size, config.batch_size)
        if itr == 1: jax.profiler.start_trace(os.path.join(config.rundir, 'logs'))
        x, y = jax.device_put((x, y), data_sharding)
        model, opt_state, loss = step(model, opt_state, x, y, key1)
        if itr == 1: loss.block_until_ready(); jax.profiler.stop_trace()
        if not config.debug: mngr.save(itr, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)))
        postfix_values['loss'] = loss.item()
        postfix_values['lr'] = scheduler(opt_state[3].count).item()
        if pbar.format_dict['rate'] is not None:
            postfix_values['thruput'] = pbar.format_dict['rate'] * config.batch_size
        pbar.set_postfix(**postfix_values)
    pbar.close()
    writer.close()
