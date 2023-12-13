import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import jax
import jax.numpy as jnp
from src.layers import fixed_pos_embedding, apply_rotary_pos_emb


def test_rotary():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    T, C = 32, 64
    Q_HxTxC = jax.random.normal(key1, (8, T, C))
    K_HxTxC = jax.random.normal(key2, (8, T, C))
    # Shift K, Q along T dimension
    shift = 5
    Qshift_HxTxC = jnp.roll(Q_HxTxC, shift, axis=1)
    Kshift_HxTxC = jnp.roll(K_HxTxC, shift, axis=1)
    sin_TxCp, cos_TxCp = fixed_pos_embedding(C, T)
    Q_HxTxC = apply_rotary_pos_emb(Q_HxTxC, sin_TxCp, cos_TxCp)
    K_HxTxC = apply_rotary_pos_emb(K_HxTxC, sin_TxCp, cos_TxCp)
    A_HxTxT = Q_HxTxC @ jnp.transpose(K_HxTxC, (0, 2, 1))

    Qshift_HxTxC = apply_rotary_pos_emb(Qshift_HxTxC, sin_TxCp, cos_TxCp)
    Kshift_HxTxC = apply_rotary_pos_emb(Kshift_HxTxC, sin_TxCp, cos_TxCp)
    Ashift_HxTxT = Qshift_HxTxC @ jnp.transpose(Kshift_HxTxC, (0, 2, 1))
    
    A_HxTxT_shifted = jnp.roll(A_HxTxT, shift, axis=(-2, -1))
    print(jnp.abs(Ashift_HxTxT[:, shift:, shift:] - A_HxTxT_shifted[:, shift:, shift:]).max())
    return


if __name__ == "__main__":
    test_rotary()
