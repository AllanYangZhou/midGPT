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
    def __call__(self, x_T, *, key=None):
        return jnp.take(self.weight, x_T, axis=0)


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
    def __call__(self, x_D, inference=False, key=None):
        x_D = jax.nn.gelu(self.c_fc(x_D))
        return self.dropout(self.c_proj(x_D), inference=inference, key=key)


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
    def __call__(self, x_TxD, inference=False, key=None):
        adrop_key, pdrop_key = jrandom.split(key) if key is not None else (None, None)
        T, D = x_TxD.shape
        Q_TxD, K_TxD, V_TxD = jnp.split(vmap(self.c_attn)(x_TxD), 3, axis=-1)
        C = self.n_embd // self.n_head
        Q_HxTxC = jnp.transpose(jnp.reshape(Q_TxD, (T, self.n_head, C)), (1, 0, 2))
        K_HxTxC = jnp.transpose(jnp.reshape(K_TxD, (T, self.n_head, C)), (1, 0, 2))
        V_HxTxC = jnp.transpose(jnp.reshape(V_TxD, (T, self.n_head, C)), (1, 0, 2))
        A_HxTxT = Q_HxTxC @ jnp.transpose(K_HxTxC, (0, 2, 1))
        causal_mask = jnp.tril(jnp.ones((1, T, T))) == 0
        A_HxTxT = jnp.where(causal_mask, float('-inf'), A_HxTxT)
        # Softmax should be in full precision.
        orig_dtype = A_HxTxT.dtype
        A_HxTxT = jax.nn.softmax(A_HxTxT.astype(jnp.float32) / jnp.sqrt(C), axis=-1)
        A_HxTxT = A_HxTxT.astype(orig_dtype)
        A_HxTxT = self.attn_dropout(A_HxTxT, inference=inference, key=adrop_key)
        out_TxD = jnp.reshape(jnp.transpose(A_HxTxT @ V_HxTxC, (1, 0, 2)), (T, D))
        out_TxD = self.resid_dropout(vmap(self.c_proj)(out_TxD), inference=inference, key=pdrop_key)
        return out_TxD


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
    def __call__(self, x_TxD, inference=False, key=None):
        attn_key, mlp_key = (None, None)
        if key is not None:
            attn_key, mlp_key = jrandom.split(key)
            mlp_key = jrandom.split(mlp_key, x_TxD.shape[0])
        x_TxD = x_TxD + self.attn(vmap(self.ln1)(x_TxD), inference=inference, key=attn_key)
        mlp = vmap(eqx.Partial(self.mlp, inference=inference))
        return x_TxD + mlp(vmap(self.ln2)(x_TxD), key=mlp_key)


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
    def __call__(self, x_T, inference=False, key=None):
        # Either (inference=False and key) or (inference=True and key=None)
        drop_key, block_keys = None, (None,) * len(self.blocks)
        if key is not None:
            drop_key, block_keys = jrandom.split(key)
            block_keys = jrandom.split(block_keys, len(self.blocks))
        x_TxD = self.wte(x_T) + self.wpe(jnp.arange(x_T.shape[0]))
        x_TxD = self.drop(x_TxD, inference=inference, key=drop_key)
        for block_key, block in zip(block_keys, self.blocks):
            x_TxD = block(x_TxD, inference=inference, key=block_key)
        x_TxD = vmap(self.ln_f)(x_TxD)
        logits_TxV = vmap(self.lm_head)(x_TxD)
        return logits_TxV