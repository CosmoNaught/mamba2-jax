"""
examples/04_streaming_states_demo.py

Demonstrate carrying SSM state across chunks using Mamba2Model.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp

from mamba2_jax import Mamba2Config, Mamba2Model


def main():
    cfg = Mamba2Config(
        vocab_size=512,
        hidden_size=128,
        state_size=32,
        head_dim=16,
        chunk_size=16,
        conv_kernel=4,
        expand=2,
        num_hidden_layers=3,
    )

    model = Mamba2Model(cfg)

    key = jax.random.PRNGKey(0)
    key_init, key_data = jax.random.split(key)

    batch_size, seq_len = 1, 32
    ids_1 = jax.random.randint(
        key_data, (batch_size, seq_len), 0, cfg.vocab_size
    )
    key_data, _ = jax.random.split(key_data)
    ids_2 = jax.random.randint(
        key_data, (batch_size, seq_len), 0, cfg.vocab_size
    )

    variables = model.init(key_init, input_ids=ids_1)
    params = variables["params"]

    # zero initial states, one per layer
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_heads
    init_states = [
        jnp.zeros((batch_size, num_heads, cfg.head_dim, cfg.state_size))
        for _ in range(num_layers)
    ]

    # First chunk: start from zeros
    out1 = model.apply(
        {"params": params},
        input_ids=ids_1,
        initial_states=init_states,
        output_last_ssm_states=True,
    )
    last_states_1 = out1["last_ssm_states"]
    hidden_1 = out1["last_hidden_state"]
    print("Chunk 1 last_hidden_state shape:", hidden_1.shape)

    # Second chunk with carried-over state
    out2 = model.apply(
        {"params": params},
        input_ids=ids_2,
        initial_states=last_states_1,
        output_last_ssm_states=True,
    )
    hidden_2 = out2["last_hidden_state"]

    # Second chunk starting from fresh zeros
    out_zero = model.apply(
        {"params": params},
        input_ids=ids_2,
        initial_states=init_states,
        output_last_ssm_states=False,
    )
    hidden_zero = out_zero["last_hidden_state"]

    diff = jnp.mean(jnp.abs(hidden_2 - hidden_zero))
    print("Mean |difference| with vs. without history:", float(diff))


if __name__ == "__main__":
    main()
