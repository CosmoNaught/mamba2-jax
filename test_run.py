import jax
import jax.numpy as jnp

from mamba2_jax.config import Mamba2Config
from mamba2_jax.model import Mamba2Model

cfg = Mamba2Config(
    vocab_size=1000,
    hidden_size=64,
    state_size=16,
    head_dim=8,
    num_hidden_layers=2,
)

model = Mamba2Model(cfg)

key = jax.random.PRNGKey(0)
dummy_ids = jnp.ones((2, 16), dtype=jnp.int32)

params = model.init(key, input_ids=dummy_ids)
outputs = model.apply(params, input_ids=dummy_ids)

print(outputs["last_hidden_state"].shape)  # (2, 16, 64)
