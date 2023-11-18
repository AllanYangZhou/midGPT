from dataclasses import dataclass
import typing as tp
import equinox as eqx
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom


class MLP(eqx.Module):
    c_fc: eqx.Module
    c_proj: eqx.Module
    dropout: eqx.Module

    def __init__(self, n_embd, bias, dropout, key):
        key1, key2 = jrandom.split(key)
        self.c_fc = eqx.nn.Linear(n_embd, 4 * n_embd, use_bias=bias, key=key1)
        self.c_proj = eqx.nn.Linear(4 * n_embd, n_embd, use_bias=bias, key=key2)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key):
        x = jax.nn.gelu(self.c_fc(x))
        return self.dropout(self.c_proj(x), key=key)


class CausalSelfAttention(eqx.Module):
    n_head: int
    n_embd: int
    c_attn: eqx.Module
    c_proj: eqx.Module
    attn_dropout: eqx.Module
    resid_dropout: eqx.Module

    def __init__(self, n_embd, n_head, bias, dropout, key):
        key1, key2 = jrandom.split(key)
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = eqx.nn.Linear(n_embd, 3 * n_embd, use_bias=bias, key=key1)
        self.c_proj = eqx.nn.Linear(n_embd, n_embd, use_bias=bias, key=key2)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key):
        key1, key2 = jrandom.split(key)
        T, C = x.shape
        qkv = vmap(self.c_attn)(x)  # (T, 3 * C)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # (T, C)
        n_per_head = self.n_embd // self.n_head
        # (T, C) -> (n_head, T, n_per_head)
        q = jnp.transpose(jnp.reshape(q, (T, self.n_head, n_per_head)), (1, 0, 2))
        k = jnp.transpose(jnp.reshape(k, (T, self.n_head, n_per_head)), (1, 0, 2))
        v = jnp.transpose(jnp.reshape(v, (T, self.n_head, n_per_head)), (1, 0, 2))
        att = q @ jnp.transpose(k, (0, 2, 1))  # (n_head, T, T)
        # Causal mask
        idx_x, idx_y = jnp.triu_indices(T, 0)
        att = att.at[..., idx_x, idx_y].set(float('-inf'))
        att = jax.nn.softmax(att / jnp.sqrt(n_per_head), axis=-1)
        att = self.attn_dropout(att, key=key1)
        out = jnp.reshape(jnp.transpose(att @ v, (1, 0, 2)), (T, C))
        return self.resid_dropout(vmap(self.c_proj)(out), key=key2)


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: MLP
    ln1: eqx.Module
    ln2: eqx.Module
    def __init__(self, n_embd, n_head, bias, dropout, key):
        key1, key2 = jrandom.split(key)
        self.attn = CausalSelfAttention(
            n_embd=n_embd, n_head=n_head, bias=bias, dropout=dropout, key=key1)
        self.mlp = MLP(n_embd=n_embd, bias=bias, dropout=dropout, key=key2)
        self.ln1 = eqx.nn.LayerNorm(n_embd, eps=1e-5, use_bias=bias)
        self.ln2 = eqx.nn.LayerNorm(n_embd, eps=1e-5, use_bias=bias)

    def __call__(self, x, key):
        ln1, ln2 = vmap(self.ln1), vmap(self.ln2)
        key1, key2 = jrandom.split(key)
        x = x + self.attn(ln1(x), key1)
        key2 = jrandom.split(key2, x.shape[0])
        return x + vmap(self.mlp)(ln2(x), key2)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(eqx.Module):
    wte: eqx.Module
    wpe: eqx.Module
    drop: eqx.Module
    blocks: tp.List[eqx.Module]
    ln_f: eqx.Module
    lm_head: eqx.Module

    def __init__(self, config, key):
        key1, key2, key3, key4 = jrandom.split(key, 4)
        self.wte = eqx.nn.Embedding(config.vocab_size, config.n_embd, key=key1)
        self.wpe = eqx.nn.Embedding(config.block_size, config.n_embd, key=key2)
        self.drop = eqx.nn.Dropout(config.dropout)
        block_keys = jrandom.split(key3, config.n_layer)
        self.blocks = [Block(
            config.n_embd, config.n_head, config.bias, config.dropout, bkey
        ) for bkey in block_keys]
        self.ln_f = eqx.nn.LayerNorm(config.n_embd, eps=1e-5, use_bias=config.bias)
        self.lm_head = eqx.nn.Linear(config.n_embd, config.vocab_size, use_bias=config.bias, key=key4)

    def __call__(self, x, key):  # (T, vocab_size)
        key, key1 = jrandom.split(key)
        x = vmap(self.wte)(x) + vmap(self.wpe)(jnp.arange(x.shape[0]))
        x = self.drop(x, key=key1)
        keys = jrandom.split(key, len(self.blocks))
        for subkey, block in zip(keys, self.blocks):
            x = block(x, subkey)
        x = vmap(self.ln_f)(x)
        logits = vmap(self.lm_head)(x)
        return logits  # (T, vocab_size)


if __name__ == "__main__":
    key = jrandom.PRNGKey(0)
    key, key1, key2, key3 = jrandom.split(key, 4)
    config = GPTConfig()
    gpt = GPT(config, key1)
    inp = jrandom.randint(key2, (10, 100), 0, config.vocab_size)
    out = vmap(gpt)(inp, jrandom.split(key3, inp.shape[0]))
    print(inp.shape, out.shape)
