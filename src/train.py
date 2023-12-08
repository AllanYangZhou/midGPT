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
from .model import GPT, GPTConfig, shard_gpt, count_params

jax.config.update("jax_threefry_partitionable", True)

jnp, jrandom, vmap, scan, jtu = jax.numpy, jax.random, jax.vmap, jax.lax.scan, jax.tree_util
Array, KeyArray = jax.Array, tp.Any
Mesh, NamedSharding = jax.sharding.Mesh, jax.sharding.NamedSharding
P, with_sharding_constraint = jax.sharding.PartitionSpec, jax.lax.with_sharding_constraint


@dataclass
class ExperimentConfig:
    rundir: str  # Directory containing ckpts and logs.
    data_dir: str  # Dataset directory
    learning_rate: float
    batch_size: int  # GLOBAL across all devices (not per device)
    warmup_steps: int
    min_lr: float  # Final LR after decay
    lr_decay_steps: int
    max_steps: int  # No. of grad steps
    beta2: float
    weight_decay: float
    eval_interval: int
    policy: str  # JMP mixed precision policy string
    g_accum_iters: int  # Accumulate this many grads before step
    shard_model: bool
    model_config: GPTConfig
    debug: bool = False


def get_batch(
        data, block_size: int, batch_size: int, g_accum_iters: tp.Optional[int]=None
) -> tp.Tuple[np.ndarray, np.ndarray]:
    bs = batch_size * (g_accum_iters or 1)
    ix = np.random.randint(0, len(data) - block_size, size=(bs,))
    x = np.take(data, np.arange(block_size) + ix[:, None], axis=0).astype(np.int32)
    y = np.take(data, np.arange(1, block_size + 1) + ix[:, None], axis=0).astype(np.int32)
    if g_accum_iters is not None:  # reshape to (g_accum_steps, batch_size, block_size)
        x = x.reshape(g_accum_iters, batch_size, block_size)
        y = y.reshape(g_accum_iters, batch_size, block_size)
    return x, y


def make_training_fns(
        config: ExperimentConfig, optimizer: optax.GradientTransformationExtraArgs,
        mesh: Mesh) -> tp.Tuple[tp.Callable, tp.Callable]:
    policy = jmp.get_policy(config.policy)
    def loss_fn(model_params: GPT, model_static: GPT, x: Array, y: Array, key: tp.Optional[KeyArray]) -> Array:
        model = eqx.combine(model_params, model_static)
        if key is not None:
            key = jrandom.split(key, x.shape[0])
        logits = vmap(model)(x, key=key)
        orig_dtype = logits.dtype
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), y)  # compute loss in float32
        return loss.mean().astype(orig_dtype)

    @partial(eqx.filter_jit, donate='all')
    def step(model: GPT, opt_state, x_GxBxT: Array, y_GxBxT: Array, key: KeyArray):
        G = config.g_accum_iters
        # put params in compute dtype (probably bfloat16), and split params out
        model_params, model_static = eqx.partition(policy.cast_to_compute(model), eqx.is_array)
        # compute loss and grad on microbatch, then scan over microbatches
        def microstep(grad_so_far, xykey_g: tp.Tuple[Array, Array, KeyArray]):
            loss, grad = jax.value_and_grad(loss_fn)(model_params, model_static, *xykey_g)
            if config.shard_model: grad = shard_gpt(grad, mesh)
            grad_so_far = jtu.tree_map(lambda x, y: x + y, grad, grad_so_far)
            return grad_so_far, loss
        all_keys = jrandom.split(key, config.g_accum_iters)
        init_grad = jtu.tree_map(jnp.zeros_like, model_params)
        grad, loss_G = scan(microstep, init_grad, (x_GxBxT, y_GxBxT, all_keys))
        # Grad accumulated (summed) over G, so divide.
        loss, grad = jnp.mean(loss_G, axis=0), jtu.tree_map(lambda x: x / G, grad)
        # put grad back in params dtype
        grad = policy.cast_to_param(grad)
        updates, opt_state = optimizer.update(grad, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def simple_loss(model: GPT, x: Array, y: Array, key: tp.Optional[KeyArray]) -> Array:
        """Same as loss_fn, but doesn't split params into compute/static."""
        model_params, model_static = eqx.partition(model, eqx.is_array)
        return loss_fn(model_params, model_static, x, y, key)

    data_sharding = NamedSharding(mesh, P('data', None))  # (B, D)
    def evaluate(model: GPT, data: np.ndarray) -> Array:
        model = policy.cast_to_compute(model)
        model = eqx.Partial(model, inference=True)
        tot_loss = jnp.zeros(())
        num_eval_steps = 1 if config.debug else 200
        for i in range(num_eval_steps):
            x_BxD, y_BxD = get_batch(data, config.model_config.block_size, config.batch_size)
            x_BxD, y_BxD = jax.device_put((x_BxD, y_BxD), data_sharding)
            loss = simple_loss(model, x_BxD, y_BxD, None)
            tot_loss = tot_loss + loss
        return tot_loss / num_eval_steps

    return step, evaluate


def train(config: ExperimentConfig):
    writer = SummaryWriter(os.path.join(config.rundir, 'logs'), flush_secs=30)
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=('data',))

    train_data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    options = ocp.CheckpointManagerOptions(
        max_to_keep=1, save_interval_steps=config.eval_interval)
    mngr = ocp.CheckpointManager(
        os.path.abspath(os.path.join(config.rundir, 'ckpt_mngr')),
        ocp.PyTreeCheckpointer(),
        options=options)

    scheduler = optax.warmup_cosine_decay_schedule(
        0, config.learning_rate, config.warmup_steps, config.lr_decay_steps,
        end_value=config.min_lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=config.beta2),
        optax.scale_by_schedule(scheduler),
        optax.add_decayed_weights(config.weight_decay),
        optax.scale(-1),
    )
    step, evaluate = make_training_fns(config, optimizer, mesh)

    key = jrandom.PRNGKey(0)
    def init_sharded_model(model_key):
        model = GPT(config.model_config, model_key)
        if config.shard_model: model = shard_gpt(model, mesh)
        return model
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
    data_sharding = NamedSharding(mesh, P(None, 'data', None))  # (G, B, D)
    postfix_values = {}  # values to display in the progress bar
    pbar = trange(first_step, config.max_steps, initial=first_step, total=config.max_steps)
    for itr in pbar:
        if itr % config.eval_interval == 0:
            train_loss = evaluate(model, train_data).item()
            val_loss = evaluate(model, val_data).item()
            postfix_values['train_loss'] = train_loss
            postfix_values['val_loss'] = val_loss
            writer.add_scalar('loss/train', train_loss, itr)
            writer.add_scalar('loss/val', val_loss, itr)
        key, key1 = jrandom.split(key)
        x_GxBxD, y_GxBxD = get_batch(
            train_data, config.model_config.block_size, config.batch_size, config.g_accum_iters
        )
        if config.debug and itr == 0:
            jax.profiler.start_trace(os.path.join(config.rundir, 'logs'))
        x_GxBxD, y_GxBxD = jax.device_put((x_GxBxD, y_GxBxD), data_sharding)
        model, opt_state, loss = step(model, opt_state, x_GxBxD, y_GxBxD, key1)
        if config.debug and itr == 0:
            loss.block_until_ready(); jax.profiler.stop_trace()
        if not config.debug: mngr.save(itr, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)))
        postfix_values['loss'] = loss.item()
        postfix_values['lr'] = scheduler(opt_state[3].count).item()
        if pbar.format_dict['rate'] is not None:
            postfix_values['thpt'] = pbar.format_dict['rate'] * config.batch_size * config.g_accum_iters
        pbar.set_postfix(**postfix_values)
    pbar.close()
    writer.close()
