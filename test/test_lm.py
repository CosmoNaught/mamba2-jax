# test/test_lm.py

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp
import optax

from mamba2_jax.config import Mamba2Config
from mamba2_jax.lm import Mamba2ForCausalLM


def test_lm_forward_and_train_step():
    # Small config for speed / stability
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

    key = jax.random.PRNGKey(0)
    key_init, key_batch = jax.random.split(key)

    batch_size = 2
    seq_len = 16

    # Dummy token ids
    input_ids = jax.random.randint(
        key_batch,
        (batch_size, seq_len),
        minval=0,
        maxval=cfg.vocab_size,
    )

    # Initialise parameters
    variables = model.init(key_init, input_ids=input_ids)
    params = variables["params"]

    # NOTE: pass {"params": params}, not params directly
    outputs = model.apply({"params": params}, input_ids=input_ids, labels=input_ids)
    logits = outputs["logits"]
    loss = outputs["loss"]

    print("LM logits shape:", logits.shape)
    print("LM loss:", float(loss))

    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert jnp.isfinite(loss)

    # ---- Tiny training loop (full JAX shebang) ----

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

    key_train = key_batch
    for step in range(3):
        key_train, key_data = jax.random.split(key_train)
        batch_ids = jax.random.randint(
            key_data, (batch_size, seq_len), 0, cfg.vocab_size
        )
        params, opt_state, loss_val = train_step(params, opt_state, batch_ids)
        print(f"[LM] step {step} loss:", float(loss_val))
        assert jnp.isfinite(loss_val)


if __name__ == "__main__":
    test_lm_forward_and_train_step()
