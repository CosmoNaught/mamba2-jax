# test/test_timeseries.py

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp
import optax

from mamba2_jax.timeseries import Mamba2Forecaster

def test_timeseries_forecaster_forward_and_train_step():
    forecaster = Mamba2Forecaster(
        input_dim=10,
        d_model=64,
        n_layers=2,
        output_dim=1,
        forecast_horizon=8,
        d_state=16,
        headdim=8,
        d_conv=4,
        chunk_size=16,
    )

    key = jax.random.PRNGKey(0)
    key_init, key_data = jax.random.split(key)

    batch_size = 4
    context_len = 32
    input_dim = 10

    x = jax.random.normal(
        key_data,
        (batch_size, context_len, input_dim),
    )

    key_data, key_target = jax.random.split(key_data)
    y_true = jax.random.normal(
        key_target,
        (batch_size, 8, 1),
    )

    variables = forecaster.init(key_init, x)
    params = variables["params"]

    y_pred = forecaster.apply({"params": params}, x)
    print("Timeseries y_pred shape:", y_pred.shape)

    assert y_pred.shape == y_true.shape

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(p, batch_x, batch_y):
        preds = forecaster.apply({"params": p}, batch_x)
        return jnp.mean((preds - batch_y) ** 2)

    @jax.jit
    def train_step(p, opt_state, batch_x, batch_y):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, batch_x, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss_val

    for step in range(3):
        params, opt_state, loss_val = train_step(params, opt_state, x, y_true)
        print(f"[TS] step {step} loss:", float(loss_val))
        assert jnp.isfinite(loss_val)


if __name__ == "__main__":
    test_timeseries_forecaster_forward_and_train_step()
