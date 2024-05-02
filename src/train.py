import typing as tp
from functools import partial
from dataclasses import dataclass
import os
import equinox as eqx
import jax
from jax.experimental import mesh_utils
import optax
import orbax.checkpoint as ocp
import numpy as np
import wandb
from tqdm import trange
from .model import GPT, GPTConfig, shard_gpt, count_params
from .sharding import reshard, get_shard_fn

jax.config.update("jax_threefry_partitionable", True)

jnp, jrandom, vmap, scan, jtu = jax.numpy, jax.random, jax.vmap, jax.lax.scan, jax.tree_util
Array = jax.Array
KeyArray = tp.Any
Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
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
    param_dtype: str  # bfloat16 or float32
    compute_dtype: str
    g_accum_iters: int  # Accumulate this many grads before step
    shard_model: bool
    model_config: GPTConfig
    debug: bool = False


def cast_pytree(pytree: tp.Any, dtype: jnp.dtype) -> tp.Any:
    """Cast a pytree of arrays to a given dtype, ignore non-arrays."""
    def cast(x):
        if eqx.is_array(x):
            return x.astype(dtype)
        return x
    return jtu.tree_map(cast, pytree)


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
    def loss_fn(model_params: GPT, model_static: GPT, x: Array, y: Array, key: tp.Optional[KeyArray]) -> Array:
        model = eqx.combine(model_params, model_static)
        if key is not None:
            key = jrandom.split(key, x.shape[0])
        logits = vmap(model)(x, key=key).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @partial(eqx.filter_jit, donate='all')
    def step(model: GPT, opt_state, x_GxBxT: Array, y_GxBxT: Array, key: KeyArray):
        G = config.g_accum_iters
        params, static = eqx.partition((model), eqx.is_array)
        params_cpt = cast_pytree(params, jnp.dtype(config.compute_dtype))
        # compute loss and grad on microbatch, then scan over microbatches
        def microstep(grad_so_far, xykey_g: tp.Tuple[Array, Array, KeyArray]):
            loss, grad = jax.value_and_grad(loss_fn)(params_cpt, static, *xykey_g)
            grad = shard_gpt(grad, mesh, config.shard_model)
            grad_so_far = jtu.tree_map(lambda x, y: x + y, grad, grad_so_far)
            return grad_so_far, loss
        all_keys = jrandom.split(key, config.g_accum_iters)
        init_grad = jtu.tree_map(jnp.zeros_like, params)
        grad, loss_G = scan(microstep, init_grad, (x_GxBxT, y_GxBxT, all_keys))
        # Grad accumulated (summed) over G, so divide.
        loss, grad = jnp.mean(loss_G, axis=0), jtu.tree_map(lambda x: x / G, grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        model = eqx.combine(optax.apply_updates(params, updates), static)
        return model, opt_state, loss

    @eqx.filter_jit
    def simple_loss(model: tp.Union[GPT, eqx.Partial], x: Array, y: Array, key: tp.Optional[KeyArray]) -> Array:
        """Same as loss_fn, but doesn't split params into compute/static."""
        model_params, model_static = eqx.partition(model, eqx.is_array)
        return loss_fn(model_params, model_static, x, y, key)

    data_sharding = NamedSharding(mesh, P(None, ('replica', 'data'), None))  # (G, B, D)
    shard_fn = get_shard_fn(mesh, data_sharding)
    def evaluate(model: GPT, data: np.ndarray) -> float:
        eval_model = eqx.Partial(cast_pytree(model, jnp.dtype(config.compute_dtype)), inference=True)
        tot_loss = 0
        num_eval_steps = 1 if config.debug else 200
        for i in range(num_eval_steps):
            x_1xBxD_np, y_1xBxD_np = get_batch(data, config.model_config.block_size, config.batch_size, 1)
            x_1xBxD, y_1xBxD = jtu.tree_map(shard_fn, (x_1xBxD_np, y_1xBxD_np))
            x_BxD, y_BxD = x_1xBxD.squeeze(0), y_1xBxD.squeeze(0)
            loss = simple_loss(eval_model, x_BxD, y_BxD, None).item()
            tot_loss = tot_loss + loss
        return tot_loss / num_eval_steps

    return step, evaluate


def split_array_by_idx(arr_N, proc_idx, n_proc):
    n = int(arr_N.shape[0] / n_proc) + 1  # n per proc
    return arr_N[proc_idx * n:(proc_idx+1)*n]


def train(config: ExperimentConfig):
    n_proc, proc_idx = jax.process_count(), jax.process_index()
    n_devices = jax.device_count()  # Assumes num_devices is multiple of 8.
    mesh = Mesh(mesh_utils.create_device_mesh((n_devices // 8, 8)), axis_names=('replica', 'data'))

    train_data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r').copy()
    val_data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r').copy()
    print("Raw shapes", train_data.shape, val_data.shape)
    train_data = split_array_by_idx(train_data, proc_idx, n_proc)
    val_data = split_array_by_idx(val_data, proc_idx, n_proc)
    print(f"Process {proc_idx}/{n_proc}. train_data.shape={train_data.shape}, val_data.shape={val_data.shape}.")

    if not config.debug:
        options = ocp.CheckpointManagerOptions(
            max_to_keep=1, save_interval_steps=config.eval_interval)
        mngr = ocp.CheckpointManager(
            config.rundir,
            ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
            options=options)

    scheduler = optax.warmup_cosine_decay_schedule(
        0, config.learning_rate, config.warmup_steps, config.lr_decay_steps,
        end_value=config.min_lr)
    @jax.jit
    def get_lr(_opt_state):
        return scheduler(_opt_state[3].count)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=config.beta2),
        optax.add_decayed_weights(config.weight_decay / config.learning_rate),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    step, evaluate = make_training_fns(config, optimizer, mesh)

    key = jrandom.PRNGKey(0)
    def init_model(model_key):
        model = GPT(config.model_config, model_key)
        model = cast_pytree(model, config.param_dtype)
        model = shard_gpt(model, mesh, config.shard_model)
        return model
    key, key1 = jrandom.split(key)
    # Use jit with sharding constraints to init sharded model+opt.
    model= eqx.filter_jit(init_model)(key1)
    print(f'Model has {count_params(model)} parameters.')
    def repl_opt_scalars(x: Array):
        if x.ndim == 0:
            x = reshard(x, NamedSharding(mesh, P()))
        return x
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = jtu.tree_map(repl_opt_scalars, opt_state)
    first_step = 0
    if not config.debug and mngr.latest_step() is not None:  # Restore existing checkpoint.
        ex_state = (jtu.tree_leaves(model), jtu.tree_leaves(opt_state))
        ex_shardings = jtu.tree_map(lambda x: x.sharding if eqx.is_array(x) else None, ex_state)
        restore_args = ocp.checkpoint_utils.construct_restore_args(ex_state, ex_shardings)
        model_leaves, opt_state_leaves = mngr.restore(
            mngr.latest_step(), restore_kwargs={'restore_args': restore_args})
        model = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)
        opt_state = jtu.tree_unflatten(jtu.tree_structure(opt_state), opt_state_leaves)
        first_step = mngr.latest_step() + 1
    data_sharding = NamedSharding(mesh, P(None, ('replica', 'data'), None))  # (G, B, D)
    shard_fn = get_shard_fn(mesh, data_sharding)
    postfix_values = {}  # values to display in the progress bar
    pbar = trange(
        first_step, config.max_steps, initial=first_step, total=config.max_steps,
        disable=jax.process_index() != 0)
    for itr in pbar:
        if itr % config.eval_interval == 0:
            train_loss = evaluate(model, train_data)
            val_loss = evaluate(model, val_data)
            postfix_values['train_loss'] = train_loss
            postfix_values['val_loss'] = val_loss
            if jax.process_index() == 0:
                wandb.log({'loss/train': train_loss, 'loss/val': val_loss}, step=itr)
        key, key1 = jrandom.split(key)
        x_GxBxD, y_GxBxD = get_batch(
            train_data, config.model_config.block_size, config.batch_size, config.g_accum_iters)
        if config.debug and itr == 0:
            jax.profiler.start_trace(config.rundir)
        x_GxBxD, y_GxBxD = jtu.tree_map(shard_fn, (x_GxBxD, y_GxBxD))
        model, opt_state, loss = step(model, opt_state, x_GxBxD, y_GxBxD, key1)
        if config.debug and itr == 0:
            loss.block_until_ready()
            jax.profiler.stop_trace()
        if jax.process_index() == 0 and itr % 20 == 0:
            wandb.log({'loss/optimized': loss.item()}, step=itr)
        if not config.debug:
            mngr.save(itr, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)))
        postfix_values['loss'] = loss.item()
        postfix_values['lr'] = get_lr(opt_state).item()
        if pbar.format_dict['rate'] is not None:
            postfix_values['thpt'] = pbar.format_dict['rate'] * config.batch_size * config.g_accum_iters
        pbar.set_postfix(**postfix_values)
    pbar.close()
    if jax.process_index() == 0:
        wandb.finish()
    if not config.debug:
        mngr.wait_until_finished()
