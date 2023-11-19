import typing as tp
from dataclasses import dataclass
import os
import equinox as eqx
import jax
import jmp
import optax
import numpy as np
from tqdm import tqdm
from .model import GPT, GPTConfig

jnp, jrandom, vmap, scan = jax.numpy, jax.random, jax.vmap, jax.lax.scan
PRNGKey = jrandom.PRNGKey


def get_batch(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.take(data, np.arange(block_size) + ix[:, None], axis=0).astype(np.int32)
    y = np.take(data, np.arange(1, block_size + 1) + ix[:, None], axis=0).astype(np.int32)
    return x, y


def make_training_fns(config, optimizer):
    def loss_fn(model, x, y, key: tp.Optional[PRNGKey]):
        model = config.policy.cast_to_compute(model)
        if key is not None:
            key = jrandom.split(key, x.shape[0])
        logits = vmap(model)(x, key=key)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss.mean()

    @eqx.filter_jit
    def step(model, opt_state, x, y, key: PRNGKey):
        loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
        updates, opt_state = optimizer.update(grad, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    fast_loss_fn = eqx.filter_jit(loss_fn)
    def evaluate(model, data):
        model = eqx.Partial(model, inference=True)
        tot_loss = jnp.zeros(())
        for i in range(200):
            x, y = get_batch(data, config.model_config.block_size, config.batch_size)
            x, y = jnp.array(x), jnp.array(y)
            loss = fast_loss_fn(model, x, y, None)
            tot_loss = tot_loss + loss
        return tot_loss / 200

    return step, evaluate


@dataclass
class ExperimentConfig:
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
    policy: jmp.Policy
    model_config: GPTConfig


def train(config):
    print(config)
    train_data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    key = jrandom.PRNGKey(0)
    key, key1 = jrandom.split(key)
    model = GPT(config.model_config, key1)

    scheduler = optax.warmup_cosine_decay_schedule(
        0, config.learning_rate, config.warmup_steps,
        config.lr_decay_steps, end_value=config.min_lr)
    optimizer = optax.chain(
        optax.scale_by_adam(b2=config.beta2),
        optax.add_decayed_weights(config.weight_decay),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    step, evaluate = make_training_fns(config, optimizer)

    pbar = tqdm(range(config.max_steps))
    postfix_values = {}
    for i in pbar:
        if i % config.eval_interval == 0:
            train_loss = evaluate(model, train_data)
            val_loss = evaluate(model, val_data)
            postfix_values['train_loss'] = train_loss.item()
            postfix_values['val_loss'] = val_loss.item()
        key, key1 = jrandom.split(key)
        x, y = get_batch(train_data, config.model_config.block_size, config.batch_size)
        x, y = jnp.array(x), jnp.array(y)
        loss, model, opt_state = step(model, opt_state, x, y, key1)
        postfix_values['loss'] = loss.item()
        postfix_values['lr'] = scheduler(i).item()
        pbar.set_postfix(**postfix_values)
    pbar.close()
