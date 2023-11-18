import typing as tp
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom


class MLP(eqx.Module):
    c_fc: eqx.Module
    c_proj: eqx.Module
    dropout: eqx.Module

    def __init__(self, n_embd, bias, dropout, key):
        key1, key2 = jrandom.split(key)
        self.c_fc = jax.vmap(eqx.nn.Linear(n_embd, 4 * n_embd, use_bias=bias, key=key1))
        self.c_proj = jax.vmap(eqx.nn.Linear(4 * n_embd, n_embd, use_bias=bias, key=key2))
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
        self.c_attn = jax.vmap(jax.vmap(
            eqx.nn.Linear(n_embd, 3 * n_embd, use_bias=bias, key=key1)))
        self.c_proj = jax.vmap(jax.vmap(
            eqx.nn.Linear(n_embd, n_embd, use_bias=bias, key=key2)))
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key):
        key1, key2 = jrandom.split(key)
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # (B, T, C)
        n_per_head = self.n_embd // self.n_head
        # (B, T, C) -> (B, n_head, T, n_per_head)
        q = jnp.transpose(jnp.reshape(q, (B, T, self.n_head, n_per_head)), (0, 2, 1, 3))
        k = jnp.transpose(jnp.reshape(k, (B, T, self.n_head, n_per_head)), (0, 2, 1, 3))
        v = jnp.transpose(jnp.reshape(v, (B, T, self.n_head, n_per_head)), (0, 2, 1, 3))
        att = q @ jnp.transpose(k, (0, 1, 3, 2))  # (B, n_head, T, T)
        # Causal mask
        idx_x, idx_y = jnp.triu_indices(T, 0)
        att = att.at[..., idx_x, idx_y].set(float('-inf'))
        att = jax.nn.softmax(att / jnp.sqrt(n_per_head), axis=-1)
        att = self.attn_dropout(att, key=key1)
        out = jnp.reshape(jnp.transpose(att @ v, (0, 2, 1, 3)), (B, T, C))
        return self.resid_dropout(self.c_proj(out), key=key2)


if __name__ == "__main__":
    key = jrandom.PRNGKey(0)
    key, key1, key2 = jrandom.split(key, 3)
    causal_sa = CausalSelfAttention(512, 8, False, 0.1, key=key1)
    inp = jnp.ones((10, 100, 512))
    out = causal_sa(inp, key1)
    print(inp.shape, out.shape)
    key, key1, key2 = jrandom.split(key, 3)
    mlp = MLP(100, False, 0.1, key1)
    inp = jnp.ones((10, 100))
    out = mlp(inp, key2)
    print(inp.shape, out.shape)