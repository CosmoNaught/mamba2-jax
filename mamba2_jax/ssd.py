# mamba2_jax/ssd.py

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange, repeat


def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Pad zeros at the end of the sequence dimension (axis=1)."""
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def segsum(x: jnp.ndarray) -> jnp.ndarray:
    """
    More stable segment sum calculation, JAX version of the PyTorch implementation.

    Input:
        x: (..., T)
    Output:
        x_segsum: (..., T, T)
    """
    T = x.shape[-1]

    # Repeat over a new trailing axis
    x_rep = repeat(x, "... d -> ... d e", e=T)  # (..., T, T)

    # Mask lower-triangular (strict) for the partial sums
    mask_lower = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_rep = jnp.where(mask_lower, x_rep, 0.0)

    # Cumulative sum over the 'd' axis (second-to-last)
    x_segsum = jnp.cumsum(x_rep, axis=-2)

    # Keep only lower triangle (including diagonal), -inf elsewhere
    mask_diag = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    neg_inf = jnp.array(-jnp.inf, dtype=x_segsum.dtype)
    x_segsum = jnp.where(mask_diag, x_segsum, neg_inf)

    return x_segsum


def ssd_naive(
    x: jnp.ndarray,         # (B, L, H, P)
    dt: jnp.ndarray,        # (B, L, H)
    A: jnp.ndarray,         # (H,)
    B_mat: jnp.ndarray,     # (B, L, H, N)
    C_mat: jnp.ndarray,     # (B, L, H, N)
    chunk_size: int,
    D: jnp.ndarray,         # (H,)
    dt_bias: jnp.ndarray,   # (H,)
    dt_min: float,
    dt_max: float,
    initial_states: Optional[jnp.ndarray] = None,  # (B, 1, H, P, N)
    return_final_states: bool = False,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    JAX port of the PyTorch `_ssd_naive` function.

    Args:
        x:   (batch_size, seq_len, num_heads, head_dim)
        dt:  (batch_size, seq_len, num_heads)
        A:   (num_heads)
        B_mat, C_mat: (batch_size, seq_len, num_heads, ssm_state_size)

    Returns:
        y:          (batch_size, seq_len, num_heads, head_dim)
        final_state (optional): (batch_size, num_heads, head_dim, ssm_state_size)
    """
    B_size, seq_len, num_heads, head_dim = x.shape

    # Padding size for equal-sized chunks
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # dt softplus and clamping (with bias)
    dt = jax.nn.softplus(dt + dt_bias)  # broadcast over batch/seq
    dt = jnp.clip(dt, dt_min, dt_max)

    # Pad tensors along sequence dimension
    x_padded = _pad_seq_dim(x, pad_size)          # (B, L_pad, H, P)
    dt_padded = _pad_seq_dim(dt, pad_size)        # (B, L_pad, H)
    B_padded = _pad_seq_dim(B_mat, pad_size)      # (B, L_pad, H, N)
    C_padded = _pad_seq_dim(C_mat, pad_size)      # (B, L_pad, H, N)

    L_pad = x_padded.shape[1]

    # D residual
    D_residual = D.reshape(1, 1, num_heads, 1) * x_padded  # (B, L_pad, H, P)

    # Discretize x and A
    x_disc = x_padded * dt_padded[..., None]               # (B, L_pad, H, P)
    A_disc = A.astype(x_disc.dtype) * dt_padded            # (B, L_pad, H)

    # Chunk everything
    def pad_and_chunk(t):
        return rearrange(t, "b (c l) ... -> b c l ...", l=chunk_size)

    x_blk = pad_and_chunk(x_disc)       # (B, Ck, Lc, H, P)
    A_blk = pad_and_chunk(A_disc)       # (B, Ck, Lc, H)
    B_blk = pad_and_chunk(B_padded)     # (B, Ck, Lc, H, N)
    C_blk = pad_and_chunk(C_padded)     # (B, Ck, Lc, H, N)

    # A_cumsum over intra-chunk time dimension
    A_blk2 = rearrange(A_blk, "b c l h -> b h c l")     # (B, H, Ck, Lc)
    A_cumsum = jnp.cumsum(A_blk2, axis=-1)              # (B, H, Ck, Lc)

    # 1. Intra-chunk (diagonal blocks)
    L_mat = jnp.exp(segsum(A_cumsum))                   # (B, H, Ck, Lc, Lc)

    # C_blk: (B, Ck, Lc, H, N) -> b c l h n
    # B_blk: (B, Ck, Lc, H, N) -> b c s h n
    # L_mat: (B, H, Ck, Lc, Lc)-> b h c l s
    # x_blk: (B, Ck, Lc, H, P) -> b c s h p
    Y_diag = jnp.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        C_blk,
        B_blk,
        L_mat,
        x_blk,
    )  # (B, Ck, Lc, H, P)

    # 2. States within each chunk
    decay_states = jnp.exp(A_cumsum[..., -1:, :] - A_cumsum)  # (B, H, Ck, Lc)

    states = jnp.einsum(
        "bclhn,bhcl,bclhp->bchpn",
        B_blk,
        decay_states,
        x_blk,
    )  # (B, Ck, H, P, N)

    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1, ...])  # (B, 1, H, P, N)

    states = jnp.concatenate([initial_states, states], axis=1)  # (B, Ck+1, H, P, N)

    # A at chunk boundaries
    A_end = A_cumsum[..., -1]                          # (B, H, Ck)
    A_end_padded = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))  # (B, H, Ck+1)

    decay_chunk = jnp.exp(segsum(A_end_padded))        # (B, H, Ck+1, Ck+1)

    new_states = jnp.einsum(
        "bhzc,bchpn->bzhpn",
        decay_chunk,
        states,
    )  # (B, Ck+1, H, P, N)

    states, final_state = new_states[:, :-1, ...], new_states[:, -1, ...]  # (B, Ck, H, P, N), (B, H, P, N)

    # 4. Convert states -> outputs
    state_decay_out = jnp.exp(A_cumsum)             # (B, H, Ck, Lc)

    Y_off = jnp.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        C_blk,
        states,
        state_decay_out,
    )  # (B, Ck, Lc, H, P)

    y = Y_diag + Y_off                               # (B, Ck, Lc, H, P)
    y = rearrange(y, "b c l h p -> b (c l) h p")     # (B, L_pad, H, P)
    y = y + D_residual                               # add residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    if return_final_states:
        return y, final_state
    else:
        return y, None
