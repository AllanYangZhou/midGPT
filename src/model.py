from dataclasses import dataclass
import math
import typing as tp
import equinox as eqx
import jax

jnp, jrandom, vmap, Array = jax.numpy, jax.random, jax.vmap, jax.Array


class Embedding(eqx.Module):
    """eqx Embedding requires vmapping over T dimension, which ends up being very slow."""
    num_embeddings: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    weight: Array

    def __init__(self, num_embeddings, embedding_size, weight=None, *, key=None, **kwargs):
        super().__init__(**kwargs)
        if weight is None:
            self.weight = jrandom.normal(key, (num_embeddings, embedding_size))
        else:
            self.weight = weight
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size

    @jax.named_scope("Embedding")
    def __call__(self, x, *, key=None):  # x: (T,)
        return jnp.take(self.weight, x, axis=0)


def reinit_linear(layer: eqx.nn.Linear, key, w_std=0.02):
    weight = jrandom.normal(key, layer.weight.shape) * w_std
    layer = eqx.tree_at(lambda layer: layer.weight, layer, weight)
    if layer.bias is not None:
        bias = jnp.zeros_like(layer.bias)
        layer = eqx.tree_at(lambda layer: layer.bias, layer, bias)
    return layer


class MLP(eqx.Module):
    c_fc: eqx.Module
    c_proj: eqx.Module
    dropout: eqx.Module

    def __init__(self, n_embd, bias, dropout, c_proj_std, key):
        key1, key2 = jrandom.split(key)
        self.c_fc = reinit_linear(eqx.nn.Linear(n_embd, 4 * n_embd, use_bias=bias, key=key1), key1)
        self.c_proj = reinit_linear(eqx.nn.Linear(
            4 * n_embd, n_embd, use_bias=bias, key=key2), key2, w_std=c_proj_std)
        self.dropout = eqx.nn.Dropout(dropout)

    @jax.named_scope('mlp')
    def __call__(self, x, inference=False, key=None):
        x = jax.nn.gelu(self.c_fc(x))
        return self.dropout(self.c_proj(x), inference=inference, key=key)


class CausalSelfAttention(eqx.Module):
    n_head: int
    n_embd: int
    c_attn: eqx.Module
    c_proj: eqx.Module
    attn_dropout: eqx.Module
    resid_dropout: eqx.Module

    def __init__(self, n_embd, n_head, bias, dropout, c_proj_std, key):
        key1, key2 = jrandom.split(key)
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = reinit_linear(eqx.nn.Linear(n_embd, 3 * n_embd, use_bias=bias, key=key1), key1)
        self.c_proj = reinit_linear(eqx.nn.Linear(
            n_embd, n_embd, use_bias=bias, key=key2), key2, w_std=c_proj_std)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)

    @jax.named_scope('causal_sa')
    def __call__(self, x, inference=False, key=None):
        attndrop_key, projdrop_key = jrandom.split(key) if key is not None else (None, None)
        T, C = x.shape
        qkv = vmap(self.c_attn)(x)  # (T, 3 * C)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # (T, C)
        n_per_head = self.n_embd // self.n_head
        # (T, C) -> (n_head, T, n_per_head)
        q = jnp.transpose(jnp.reshape(q, (T, self.n_head, n_per_head)), (1, 0, 2))
        k = jnp.transpose(jnp.reshape(k, (T, self.n_head, n_per_head)), (1, 0, 2))
        v = jnp.transpose(jnp.reshape(v, (T, self.n_head, n_per_head)), (1, 0, 2))
        att = q @ jnp.transpose(k, (0, 2, 1))  # (n_head, T, T)
        causal_mask = jnp.tril(jnp.ones((1, T, T))) == 0
        att = jnp.where(causal_mask, float('-inf'), att)
        # Softmax should be in full precision.
        orig_dtype = att.dtype
        att = jax.nn.softmax(att.astype(jnp.float32) / jnp.sqrt(n_per_head), axis=-1)
        att = att.astype(orig_dtype)
        att = self.attn_dropout(att, inference=inference, key=attndrop_key)
        out = jnp.reshape(jnp.transpose(att @ v, (1, 0, 2)), (T, C))
        out = self.resid_dropout(vmap(self.c_proj)(out), inference=inference, key=projdrop_key)
        return out


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: MLP
    ln1: eqx.Module
    ln2: eqx.Module
    def __init__(self, n_embd, n_head, bias, dropout, c_proj_std, key):
        key1, key2 = jrandom.split(key)
        self.attn = CausalSelfAttention(
            n_embd=n_embd, n_head=n_head, bias=bias, dropout=dropout, c_proj_std=c_proj_std,
            key=key1)
        self.mlp = MLP(
            n_embd=n_embd, bias=bias, dropout=dropout, c_proj_std=c_proj_std, key=key2)
        self.ln1 = eqx.nn.LayerNorm(n_embd, eps=1e-5, use_bias=bias)
        self.ln2 = eqx.nn.LayerNorm(n_embd, eps=1e-5, use_bias=bias)

    @jax.named_scope('block')
    def __call__(self, x, inference=False, key=None):
        attn_key, mlp_key = (None, None)
        if key is not None:
            attn_key, mlp_key = jrandom.split(key)
            mlp_key = jrandom.split(mlp_key, x.shape[0])
        ln1, ln2 = vmap(self.ln1), vmap(self.ln2)
        x = x + self.attn(ln1(x), inference=inference, key=attn_key)
        return x + vmap(eqx.Partial(self.mlp, inference=inference))(ln2(x), key=mlp_key)


@dataclass
class GPTConfig:
    block_size: int  # Max sequence length
    vocab_size: int  # No. of tokens
    n_layer: int  # No. of transformer blocks
    n_head: int  # No. attention heads
    n_embd: int  # Hidden dimension
    dropout: float
    bias: bool  # Whether or not to use biases in linear layers


class GPT(eqx.Module):
    wte: eqx.Module
    wpe: eqx.Module
    drop: eqx.Module
    blocks: tp.List[eqx.Module]
    ln_f: eqx.Module
    lm_head: eqx.Module

    def __init__(self, config, key):
        block_key, head_key, wpe_key = jrandom.split(key, 3)
        self.drop = eqx.nn.Dropout(config.dropout)
        c_proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        self.blocks = [Block(
            config.n_embd, config.n_head, config.bias, config.dropout, c_proj_std, bkey
        ) for bkey in jrandom.split(block_key, config.n_layer)]
        self.ln_f = eqx.nn.LayerNorm(config.n_embd, eps=1e-5, use_bias=config.bias)
        self.lm_head = reinit_linear(eqx.nn.Linear(
            config.n_embd, config.vocab_size, use_bias=config.bias, key=head_key), head_key)
        self.wte = Embedding(config.vocab_size, config.n_embd, weight=self.lm_head.weight)
        wpe_wt = 0.02 * jrandom.normal(wpe_key, (config.block_size, config.n_embd))
        self.wpe = Embedding(config.block_size, config.n_embd, weight=wpe_wt)

    @jax.named_scope('gpt')
    def __call__(self, x, inference=False, key=None):  # (T,)
        drop_key, block_keys = None, (None,) * len(self.blocks)
        if key is not None:
            drop_key, block_keys = jrandom.split(key)
            block_keys = jrandom.split(block_keys, len(self.blocks))
        x = self.wte(x) + self.wpe(jnp.arange(x.shape[0]))
        x = self.drop(x, inference=inference, key=drop_key)
        for block_key, block in zip(block_keys, self.blocks):
            x = block(x, inference=inference, key=block_key)
        x = vmap(self.ln_f)(x)
        logits = vmap(self.lm_head)(x)
        return logits  # (T, vocab_size)
