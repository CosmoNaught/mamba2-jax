"""
examples/02_lm_tiny_train.py

Tiny language-model training loop on synthetic token data.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp
import optax

from mamba2_jax import Mamba2Config, Mamba2ForCausalLM


def main():
    cfg = Mamba2Config(
        vocab_size=128,
        hidden_size=64,
        state_size=16,
        head_dim=8,
        chunk_size=16,
        expand=2,
        conv_kernel=4,
        num_hidden_layers=2,
    )

    model = Mamba2ForCausalLM(cfg)

    key = jax.random.PRNGKey(0)
    key_init, key_data = jax.random.split(key)

    batch_size, seq_len = 2, 16
    dummy_ids = jax.random.randint(
        key_data,
        (batch_size, seq_len),
        minval=0,
        maxval=cfg.vocab_size,
    )

    variables = model.init(key_init, input_ids=dummy_ids)
    params = variables["params"]

    # Optimiser
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(p, batch_ids):
        out = model.apply({"params": p}, input_ids=batch_ids, labels=batch_ids)
        return out["loss"]

    @jax.jit
    def train_step(p, opt_state, batch_ids):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, batch_ids)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss_val

    key_train = key_data
    for step in range(5):
        key_train, key_batch = jax.random.split(key_train)
        batch_ids = jax.random.randint(
            key_batch,
            (batch_size, seq_len),
            minval=0,
            maxval=cfg.vocab_size,
        )
        params, opt_state, loss_val = train_step(params, opt_state, batch_ids)
        print(f"[LM] step {step} loss:", float(loss_val))

    print("Training loop finished without NaNs.")


if __name__ == "__main__":
    main()
