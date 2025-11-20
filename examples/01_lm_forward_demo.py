"""
examples/01_lm_forward_demo.py

Minimal forward pass through Mamba2ForCausalLM.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp

from mamba2_jax import Mamba2Config, Mamba2ForCausalLM

def main():
    # Small config for speed / sanity checks
    cfg = Mamba2Config(
        vocab_size=128,
        hidden_size=64,
        state_size=16,
        head_dim=8,
        conv_kernel=4,
        chunk_size=16,
        num_hidden_layers=2,
        expand=2,
    )

    model = Mamba2ForCausalLM(cfg)

    # PRNG setup
    key = jax.random.PRNGKey(0)
    key_init, key_data = jax.random.split(key)

    batch_size = 2
    seq_len = 16

    # Dummy token ids
    input_ids = jax.random.randint(
        key_data,
        (batch_size, seq_len),
        minval=0,
        maxval=cfg.vocab_size,
    )

    # Initialise parameters
    variables = model.init(key_init, input_ids=input_ids)
    params = variables["params"]

    # Forward pass (with labels just to get a loss)
    outputs = model.apply(
        {"params": params},
        input_ids=input_ids,
        labels=input_ids,
    )
    logits = outputs["logits"]
    loss = outputs["loss"]

    print("LM logits shape:", logits.shape)   # (batch_size, seq_len, vocab_size)
    print("LM loss:", float(loss))
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert jnp.isfinite(loss)


if __name__ == "__main__":
    main()
