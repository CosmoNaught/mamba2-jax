"""
examples/05_inspect_hidden_states.py

Show how to get per-layer hidden states from Mamba2Model.
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
        vocab_size=256,
        hidden_size=128,
        state_size=32,
        head_dim=16,
        chunk_size=16,
        conv_kernel=4,
        expand=2,
        num_hidden_layers=4,
    )

    model = Mamba2Model(cfg)

    key = jax.random.PRNGKey(42)
    key_init, key_data = jax.random.split(key)

    batch_size, seq_len = 2, 24
    input_ids = jax.random.randint(
        key_data, (batch_size, seq_len), 0, cfg.vocab_size
    )

    variables = model.init(key_init, input_ids=input_ids)
    params = variables["params"]

    outputs = model.apply(
        {"params": params},
        input_ids=input_ids,
        output_hidden_states=True,
        output_last_ssm_states=False,
    )

    all_hidden_states = outputs["hidden_states"]
    last_hidden_state = outputs["last_hidden_state"]

    print(f"Number of hidden state tensors: {len(all_hidden_states)}")
    for i, h in enumerate(all_hidden_states):
        print(f"  Layer {i}: {h.shape}")

    # final element == last_hidden_state
    assert jnp.allclose(last_hidden_state, all_hidden_states[-1])

    # simple per-layer metric
    for i, h in enumerate(all_hidden_states):
        mag = jnp.mean(jnp.abs(h))
        print(f"  Layer {i} mean |activation|: {float(mag):.6f}")


if __name__ == "__main__":
    main()
