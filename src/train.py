import csv
import time
import typing as tp
from dataclasses import dataclass
import os
import equinox as eqx
import jax
from jax.experimental import mesh_utils
import jmp
import optax
import orbax.checkpoint as ocp
import numpy as np
from tqdm import trange
from .model import GPT, GPTConfig

jnp, jrandom, vmap, scan = jax.numpy, jax.random, jax.vmap, jax.lax.scan
P, PRNGKey = jax.sharding.PartitionSpec, jax.random.PRNGKey
jtu, NamedSharding = jax.tree_util, jax.sharding.NamedSharding
with_sharding_constraint = jax.lax.with_sharding_constraint


def get_batch(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.take(data, np.arange(block_size) + ix[:, None], axis=0).astype(np.int32)
    y = np.take(data, np.arange(1, block_size + 1) + ix[:, None], axis=0).astype(np.int32)
    return x, y


def make_training_fns(config, optimizer, mesh, shard_model: bool):
    policy = jmp.get_policy(config.policy)
    def loss_fn(model, x, y, key: tp.Optional[PRNGKey]):
        model = policy.cast_to_compute(model)
        if key is not None:
            key = jrandom.split(key, x.shape[0])
        logits = vmap(model)(x, key=key)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss.mean()

    @eqx.filter_jit
    def step(model, opt_state, x, y, key: PRNGKey):
        loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
        grad = shard_gpt(grad, mesh, shard_model, sharding_fn=with_sharding_constraint)
        updates, opt_state = optimizer.update(grad, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    data_sharding = NamedSharding(mesh, P('data', None))
    fast_loss_fn = eqx.filter_jit(loss_fn)
    def evaluate(model, data):
        model = eqx.Partial(model, inference=True)
        tot_loss = jnp.zeros(())
        for i in range(200):
            x, y = get_batch(data, config.model_config.block_size, config.batch_size)
            x, y = jax.device_put((jnp.array(x), jnp.array(y)), data_sharding)
            loss = fast_loss_fn(model, x, y, None)
            tot_loss = tot_loss + loss
        return tot_loss / 200

    return step, evaluate


@dataclass
class ExperimentConfig:
    rundir: str
    data_dir: str
    learning_rate: float
    batch_size: int
    warmup_steps: int
    min_lr: float
    lr_decay_steps: int
    max_steps: int
    beta2: float
    weight_decay: float
    eval_interval: int
    policy: str
    g_accum_steps: int
    shard_model: bool
    model_config: GPTConfig


def get_layers(model, layer_cls):
    """Get all layers of model matching layer_cls."""
    matches_cls = lambda x: isinstance(x, layer_cls)
    return filter(lambda x: matches_cls(x), jtu.tree_leaves(model, is_leaf=matches_cls))


def count_params(model):
    return sum([jnp.size(x) for x in jtu.tree_leaves(model) if isinstance(x, jax.Array)])


def shard_gpt(model, mesh, shard_model: bool, sharding_fn=jax.device_put):
    """FSDP model parameter sharding. Assumes bias=False."""
    if shard_model:
        lin_sharding = NamedSharding(mesh, P(None, 'data'))
        ln_sharding = NamedSharding(mesh, P('data',))
    else:
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


def log_metric(filename, step, metric_type, value):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = int(time.time())
        writer.writerow([timestamp, step, metric_type, value])


def train(config):
    eqx.tree_pprint(config)
    n_devices = len(jax.devices())
    print(f"Using {n_devices} devices.")
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((n_devices,)), axis_names=('data',))

    train_data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    metrics_path = os.path.join(config.rundir, 'metrics.csv')

    options = ocp.CheckpointManagerOptions(
        max_to_keep=1, save_interval_steps=config.eval_interval)
    mngr = ocp.CheckpointManager(
        os.path.abspath(os.path.join(config.rundir, 'ckpt_mngr')),
        ocp.PyTreeCheckpointer(),
        options=options)

    # Under grad accum, scheduler is only updated every g_accum_steps
    warmup_steps = config.warmup_steps // config.g_accum_steps
    lr_decay_steps = config.lr_decay_steps // config.g_accum_steps
    scheduler = optax.warmup_cosine_decay_schedule(
        0, config.learning_rate, warmup_steps, lr_decay_steps, end_value=config.min_lr)
    optimizer = optax.MultiSteps(optax.chain(
        optax.scale_by_adam(b2=config.beta2),
        optax.add_decayed_weights(config.weight_decay),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    ), every_k_schedule=config.g_accum_steps)
    step, evaluate = make_training_fns(config, optimizer, mesh, config.shard_model)

    key = jrandom.PRNGKey(0)
    key, key1 = jrandom.split(key)
    model = GPT(config.model_config, key1)
    print(f'Model has {count_params(model)} parameters.')
    model = shard_gpt(model, mesh, config.shard_model)
    jax.debug.visualize_array_sharding(model.lm_head.weight)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    jax.debug.visualize_array_sharding(opt_state.inner_opt_state[0].mu.lm_head.weight)
    first_step = 0
    if mngr.latest_step() is not None:
        model_leaves, opt_state_leaves = mngr.restore(mngr.latest_step())
        model = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)
        opt_state = jtu.tree_unflatten(jtu.tree_structure(opt_state), opt_state_leaves)
        first_step = mngr.latest_step() + 1
    else:
        # Initialize the metrics CSV
        with open(metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Unix Timestamp', 'Step', 'Metric', 'Value'])
    data_sharding = NamedSharding(mesh, P('data', None))
    postfix_values = {}
    pbar = trange(first_step, config.max_steps, initial=first_step, total=config.max_steps)
    for i in pbar:
        if i % config.eval_interval == 0:
            train_loss = evaluate(model, train_data).item()
            val_loss = evaluate(model, val_data).item()
            postfix_values['train_loss'] = train_loss
            postfix_values['val_loss'] = val_loss
            log_metric(metrics_path, i, 'train_loss', train_loss)
            log_metric(metrics_path, i, 'val_loss', val_loss)
        key, key1 = jrandom.split(key)
        x, y = get_batch(train_data, config.model_config.block_size, config.batch_size)
        x, y = jax.device_put((jnp.array(x), jnp.array(y)), data_sharding)
        loss, model, opt_state = step(model, opt_state, x, y, key1)
        mngr.save(i, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)))
        postfix_values['loss'] = loss.item()
        postfix_values['lr'] = scheduler(opt_state.inner_opt_state[2].count).item()
        if pbar.format_dict['rate'] is not None:
            postfix_values['thruput'] = pbar.format_dict['rate'] * config.batch_size
        pbar.set_postfix(**postfix_values)
    pbar.close()
