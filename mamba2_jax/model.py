# mamba2_jax/model.py

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

from .config import Mamba2Config
from .ssd import ssd_naive


# ---- Activations ----

ACT2FN_JAX: Dict[str, callable] = {
    "silu": nn.silu,
    "gelu": nn.gelu,
    "relu": nn.relu,
    "tanh": jnp.tanh,
}


# ---- RMSNorm with optional residual gating ----

class Mamba2RMSNorm(nn.Module):
    """
    JAX/Flax version of Mamba2RMSNorm (RMSNorm + optional residual gating).
    """
    hidden_size: int
    eps: float = 1e-6
    normalize: bool = False  # gate on residual if True

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, residual: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.hidden_size,),
        )

        x = hidden_states.astype(jnp.float32)

        if residual is not None and self.normalize:
            res = residual.astype(jnp.float32)
            x = x * nn.silu(res)

        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        x = x * weight  # broadcast over last dim

        return x.astype(hidden_states.dtype)


# ---- Depthwise Conv1d (causal-ish) ----

class DepthwiseConv1d(nn.Module):
    """
    Depthwise 1D conv over sequence, causal (left-padding only).
    Expects input shape (batch, seq_len, channels).
    """
    features: int          # == in_channels == out_channels
    kernel_size: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding=((self.kernel_size - 1, 0),),   # causal: pad on the left only
            feature_group_count=self.features,      # depthwise
            use_bias=self.use_bias,
        )
        y = conv(x)  # (batch, seq_len, channels), same length as input
        return y


# ---- Mamba2 Mixer ----

class Mamba2Mixer(nn.Module):
    """
    JAX/Flax implementation of the Mamba2 mixer, using the naive SSD path.
    No Triton, no caching in this first port.
    """
    config: Mamba2Config
    layer_idx: int

    def setup(self):
        cfg = self.config
        self.hidden_size = cfg.hidden_size
        self.ssm_state_size = cfg.state_size
        self.conv_kernel_size = cfg.conv_kernel
        self.intermediate_size = cfg.intermediate_size
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.chunk_size = cfg.chunk_size
        self.dt_min, self.dt_max = cfg.time_step_limit
        self.use_bias = cfg.use_bias
        self.use_conv_bias = cfg.use_conv_bias

        # Parallel projection of input hidden states
        self.in_proj = nn.Dense(
            2 * (self.intermediate_size + self.ssm_state_size) + self.num_heads,
            use_bias=self.use_bias,
        )

        conv1d_dim = self.intermediate_size + 2 * self.ssm_state_size
        self.conv1d = DepthwiseConv1d(
            features=conv1d_dim,
            kernel_size=cfg.conv_kernel,
            use_bias=self.use_conv_bias,
        )

        self.activation_name = cfg.hidden_act
        if self.activation_name not in ACT2FN_JAX:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        self.act = ACT2FN_JAX[self.activation_name]

        # dt bias parameter (num_heads,)
        def init_dt_bias(key, shape):
            low, high = cfg.time_step_min, cfg.time_step_max
            floor = cfg.time_step_floor
            u = jax.random.uniform(key, shape)
            log_min = jnp.log(low)
            log_max = jnp.log(high)
            dt = jnp.exp(u * (log_max - log_min) + log_min)
            dt = jnp.maximum(dt, floor)
            # inverse softplus: x = y + log(-expm1(-y))
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt.astype(jnp.float32)

        self.dt_bias = self.param("dt_bias", init_dt_bias, (cfg.num_heads,))

        # A_log, D
        def init_A_log(key, shape):
            low, high = cfg.A_initializer_range
            A = jax.random.uniform(key, shape, minval=low, maxval=high)
            return jnp.log(A).astype(jnp.float32)

        self.A_log = self.param("A_log", init_A_log, (cfg.num_heads,))

        self.D = self.param(
            "D",
            lambda key, shape: jnp.ones(shape, dtype=jnp.float32),
            (cfg.num_heads,),
        )

        # Residual normalization (RMSNorm) inside the mixer
        self.norm = Mamba2RMSNorm(self.intermediate_size, eps=1e-5, normalize=True)

        self.out_proj = nn.Dense(self.hidden_size, use_bias=self.use_bias)

    def _conv1d(self, xBC: jnp.ndarray) -> jnp.ndarray:
        # xBC: (B, L, conv1d_dim)
        x = self.conv1d(xBC)   # (B, L, conv1d_dim)
        x = self.act(x)
        return x

    def __call__(
        self,
        hidden_states: jnp.ndarray,         # (B, L, hidden_size)
        initial_state: Optional[jnp.ndarray] = None,  # (B, H, P, N)
        return_final_state: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        JAX equivalent of `Mamba2Mixer._forward` using the naive SSD path.
        """
        cfg = self.config
        B_size, L, _ = hidden_states.shape

        # 1) Parallel projection
        zxbcdt = self.in_proj(hidden_states)  # (B, L, proj_dim)

        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.ssm_state_size
            - self.num_heads
        ) // 2

        z0, x0, z, xBC, dt = jnp.split(
            zxbcdt,
            [
                d_mlp,
                2 * d_mlp,
                2 * d_mlp + self.intermediate_size,
                2 * d_mlp + self.intermediate_size + self.intermediate_size + 2 * self.ssm_state_size,
            ],
            axis=-1,
        )
        # Shapes:
        # z0:  (B, L, d_mlp)
        # x0:  (B, L, d_mlp)
        # z:   (B, L, intermediate_size)
        # xBC: (B, L, intermediate_size + 2*state_size)
        # dt:  (B, L, num_heads)

        # 2) Depthwise causal convolution over x,B,C
        xBC = self._conv1d(xBC)
        x, B_t, C_t = jnp.split(
            xBC,
            [
                self.intermediate_size,
                self.intermediate_size + self.ssm_state_size,
            ],
            axis=-1,
        )
        # x:   (B, L, intermediate_size)
        # B_t: (B, L, state_size)
        # C_t: (B, L, state_size)

        # 3) Naive SSD
        if initial_state is not None:
            # Convert to (B, 1, H, P, N) to match ssd_naive expectations
            init_state = rearrange(initial_state, "b h p n -> b 1 h p n")
        else:
            init_state = None

        A = -jnp.exp(self.A_log.astype(jnp.float32))  # 1-SS(a) scalar (negative)

        # Broadcast B and C across heads: (B, L, 1, N) -> (B, L, H, N)
        B_exp = jnp.expand_dims(B_t, axis=2)  # (B, L, 1, N)
        C_exp = jnp.expand_dims(C_t, axis=2)  # (B, L, 1, N)
        B_mat = jnp.broadcast_to(B_exp, (B_size, L, self.num_heads, self.ssm_state_size))
        C_mat = jnp.broadcast_to(C_exp, (B_size, L, self.num_heads, self.ssm_state_size))

        y, final_state = ssd_naive(
            x=rearrange(x, "b l (h p) -> b l h p", p=self.head_dim),
            dt=dt,
            A=A,
            B_mat=B_mat,
            C_mat=C_mat,
            chunk_size=self.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            initial_states=init_state,
            return_final_states=return_final_state,
        )

        y = rearrange(y, "b l h p -> b l (h p)")  # (B, L, intermediate_size)

        # 4) Residual gate normalization
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = jnp.concatenate([self.act(z0) * x0, y], axis=-1)

        # 5) Output projection
        y = self.out_proj(y)  # (B, L, hidden_size)

        return y, final_state


# ---- Block ----

class Mamba2Block(nn.Module):
    config: Mamba2Config
    layer_idx: int

    def setup(self):
        cfg = self.config
        self.residual_in_fp32 = cfg.residual_in_fp32
        self.norm = Mamba2RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(cfg, layer_idx=self.layer_idx)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,  # (B, H, P, N)
        return_final_state: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        residual = hidden_states
        hs = hidden_states.astype(jnp.float32)

        hs = self.norm(hs)

        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)

        hs_out, last_state = self.mixer(
            hs,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )

        out = residual + hs_out
        return out, last_state


# ---- Top-level Model ----

class Mamba2Model(nn.Module):
    """
    JAX/Flax Mamba2 backbone (no LM head yet).
    """
    config: Mamba2Config

    def setup(self):
        cfg = self.config
        self.embeddings = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.hidden_size,
        )
        self.layers = [
            Mamba2Block(cfg, layer_idx=i)
            for i in range(cfg.num_hidden_layers)
        ]
        self.norm_f = Mamba2RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,      # (B, L)
        inputs_embeds: Optional[jnp.ndarray] = None,  # (B, L, H)
        initial_states: Optional[List[jnp.ndarray]] = None,  # list of (B, H, P, N)
        output_hidden_states: bool = False,
        output_last_ssm_states: bool = False,
    ) -> Dict[str, Optional[List[jnp.ndarray]]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            hidden_states = self.embeddings(input_ids)  # (B, L, hidden_size)
        else:
            hidden_states = inputs_embeds

        cfg = self.config

        if initial_states is None:
            initial_states = [None] * cfg.num_hidden_layers
        elif len(initial_states) != cfg.num_hidden_layers:
            raise ValueError("initial_states length must equal num_hidden_layers")

        all_hidden_states = [] if output_hidden_states else None
        all_last_states = [] if output_last_ssm_states else None

        for layer, init_state in zip(self.layers, initial_states):
            hidden_states, last_state = layer(
                hidden_states,
                initial_state=init_state,
                return_final_state=output_last_ssm_states,
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_last_ssm_states:
                all_last_states.append(last_state)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "last_ssm_states": all_last_states,
        }
