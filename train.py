import typing as tp
from dataclasses import dataclass, field
import os
import pickle
import equinox as eqx
import jax
import jax.random as jrandom
import jax.numpy as jnp
import jmp
import optax
import numpy as np
from tqdm import tqdm
from model import GPT, GPTConfig

PRNGKey = jrandom.PRNGKey
vmap = jax.vmap
scan = jax.lax.scan


def get_vocab_size(data_dir):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
    if vocab_size is None:
        vocab_size = 50304
    return vocab_size


@dataclass
class ExperimentConfig:
    dataset: str = 'shakespeare_char'
    learning_rate: float = 1e-3
    batch_size: int = 64
    warmup_steps: int = 100
    min_lr: float = 1e-4
    lr_decay_steps: int = 5000
    max_steps: int = 5000
    beta2: float = 0.99
    weight_decay: float = 0.1
    eval_interval: int = 2000
    # Need an A40/A100 to natively support bfloat16.
    # policy: jmp.Policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    policy: jmp.Policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    model_config: GPTConfig = field(init=False)

    def __post_init__(self):
        vocab_size = get_vocab_size(os.path.join('data', self.dataset))
        self.model_config = GPTConfig(
            block_size=256, vocab_size=vocab_size, n_layer=6, n_head=6,
            n_embd=384, dropout=0.2, bias=False,
        )


def get_batch(data, block_size, batch_size, key: PRNGKey):
    ix = jrandom.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.take(data, np.arange(block_size) + ix[:, None], axis=0).astype(np.int32)
    y = jnp.take(data, np.arange(1, block_size + 1) + ix[:, None], axis=0).astype(np.int32)
    return jnp.array(x), jnp.array(y)



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

    def evaluate(model, data, key: PRNGKey):
        model = eqx.Partial(model, inference=True)
        def _helper(loss_so_far, key):
            x, y = get_batch(data, config.model_config.block_size, config.batch_size, key)
            loss = loss_fn(model, x, y, None)
            return loss_so_far + loss, None
        tot_loss = jnp.zeros(())
        keys = jrandom.split(key, 200)
        losses, _ = scan(_helper, tot_loss, keys)
        return losses / keys.shape[0]

    return step, evaluate


def main():
    config = ExperimentConfig()
    print(config)
    data_dir = os.path.join('data', config.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
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
            key, key1, key2 = jrandom.split(key, 3)
            train_loss = evaluate(model, train_data, key1)
            val_loss = evaluate(model, val_data, key2)
            postfix_values['train_loss'] = train_loss.item()
            postfix_values['val_loss'] = val_loss.item()
        key, key1, key2 = jrandom.split(key, 3)
        x, y = get_batch(train_data, config.model_config.block_size, config.batch_size, key1)
        loss, model, opt_state = step(model, opt_state, x, y, key2)
        postfix_values['loss'] = loss.item()
        postfix_values['lr'] = scheduler(i).item()
        pbar.set_postfix(**postfix_values)
    pbar.close()


if __name__ == '__main__':
    main()