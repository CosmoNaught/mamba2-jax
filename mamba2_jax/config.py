# mamba2_jax/config.py

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Mamba2Config:
    """
    Minimal JAX-side configuration.
    """

    vocab_size: int = 50280
    pad_token_id: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0

    hidden_size: int = 768
    state_size: int = 128
    head_dim: int = 64
    chunk_size: int = 256
    expand: int = 2
    conv_kernel: int = 4
    num_hidden_layers: int = 24
    layer_norm_epsilon: float = 1e-5

    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"

    emb_initializer_range: float = 0.02
    conv_initializer_range: float | None = None
    A_initializer_range: Tuple[float, float] = (1.0, 16.0)

    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: Tuple[float, float] = (0.0, float("inf"))

    residual_in_fp32: bool = True
    rescale_prenorm_residual: bool = False
    tie_embedding_weights: bool = True
    output_last_ssm_states: bool = False
    use_cache: bool = True          # currently unused in JAX version

    @property
    def intermediate_size(self) -> int:
        return int(self.expand * self.hidden_size)

    @property
    def num_heads(self) -> int:
        return self.intermediate_size // self.head_dim
