# mamba2_jax/head.py

from typing import Optional

import jax.numpy as jnp
from flax import linen as nn

from .config import Mamba2Config
from .model import Mamba2Model


class Mamba2Forecaster(nn.Module):
    """
    JAX/Flax analogue of the PyTorch Mamba2Head for timeseries.

    Mirrors:
        - input_proj: Linear(input_dim -> d_model)
        - mamba2:     Mamba2Model(d_model, n_layers, ...)
        - output_proj: Linear(d_model -> forecast_horizon * output_dim)
        - forward: use last hidden state, project, reshape to (B, T_h, D_out)
    """
    input_dim: int
    d_model: int = 768
    n_layers: int = 4
    output_dim: int = 1
    forecast_horizon: int = 24

    # You can override these if you want to experiment
    d_state: int = 128
    headdim: int = 64
    d_conv: int = 4
    chunk_size: int = 256

    def setup(self):
        self.input_proj = nn.Dense(self.d_model)
        cfg = Mamba2Config(
            vocab_size=1,          # irrelevant for inputs_embeds-only usage
            hidden_size=self.d_model,
            state_size=self.d_state,
            head_dim=self.headdim,
            conv_kernel=self.d_conv,
            chunk_size=self.chunk_size,
            num_hidden_layers=self.n_layers,
            expand=2,
            use_bias=False,
            use_conv_bias=True,
            hidden_act="silu",
        )

        self.mamba2 = Mamba2Model(cfg)

        # 3. Output projection
        self.output_proj = nn.Dense(self.output_dim * self.forecast_horizon)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, forecast_horizon, output_dim)
        """
        # (B, L, input_dim) -> (B, L, d_model)
        x_proj = self.input_proj(x)

        # Run Mamba2 backbone using inputs_embeds
        outputs = self.mamba2(
            input_ids=None,
            inputs_embeds=x_proj,
            output_hidden_states=False,
            output_last_ssm_states=False,
        )
        hidden_states = outputs["last_hidden_state"]  # (B, L, d_model)

        # Take last timestep
        last_hidden = hidden_states[:, -1, :]         # (B, d_model)

        # Project to horizon * output_dim
        out = self.output_proj(last_hidden)           # (B, H*T)
        out = out.reshape(
            x.shape[0],
            self.forecast_horizon,
            self.output_dim,
        )                                             # (B, H, D_out)
        return out
