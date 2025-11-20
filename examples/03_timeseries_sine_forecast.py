"""
examples/03_timeseries_sine_forecast.py

Toy time-series forecasting with Mamba2Forecaster on a noisy sine wave.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp
import optax

from mamba2_jax import Mamba2Forecaster


def make_sine_batch(key, batch_size, context_len, horizon):
    """
    Generate noisy sine-wave data.

    Returns:
        x: (B, context_len, 1)
        y: (B, horizon, 1)
    """
    total_len = context_len + horizon
    t = jnp.linspace(0.0, 4.0 * jnp.pi, total_len)
    base = jnp.sin(t)  # (T,)

    base = base[None, :, None]                 # (1, T, 1)
    base = jnp.repeat(base, batch_size, 0)     # (B, T, 1)
    noise = 0.1 * jax.random.normal(key, base.shape)
    series = base + noise

    x = series[:, :context_len, :]
    y = series[:, context_len:, :]
    return x, y


def main():
    batch_size = 8
    context_len = 32
    horizon = 8

    model = Mamba2Forecaster(
        input_dim=1,
        d_model=64,
        n_layers=2,
        output_dim=1,
        forecast_horizon=horizon,
        d_state=32,
        headdim=16,
        d_conv=4,
        chunk_size=16,
    )

    key = jax.random.PRNGKey(0)
    key_init, key_data = jax.random.split(key)

    x_init, y_init = make_sine_batch(key_data, batch_size, context_len, horizon)

    variables = model.init(key_init, x_init)
    params = variables["params"]

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(p, batch_x, batch_y):
        preds = model.apply({"params": p}, batch_x)
        return jnp.mean((preds - batch_y) ** 2)

    @jax.jit
    def train_step(p, opt_state, batch_x, batch_y):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, batch_x, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss_val

    for step in range(10):
        key_data, key_batch = jax.random.split(key_data)
        x_batch, y_batch = make_sine_batch(
            key_batch, batch_size, context_len, horizon
        )
        params, opt_state, loss_val = train_step(params, opt_state, x_batch, y_batch)
        if step % 2 == 0:
            print(f"[TS] step {step} MSE:", float(loss_val))

    y_pred = model.apply({"params": params}, x_init)
    print("Final forecast shape:", y_pred.shape)  # (B, horizon, 1)


if __name__ == "__main__":
    main()
