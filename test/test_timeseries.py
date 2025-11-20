import jax
import jax.numpy as jnp

from mamba2_jax import Mamba2Forecaster


def test_forecaster_output_shape():
    model = Mamba2Forecaster(
        input_dim=1,
        d_model=32,
        n_layers=2,
        output_dim=1,
        forecast_horizon=5,
    )

    key = jax.random.PRNGKey(0)
    batch_size, input_length, input_dim = 4, 10, 1

    x = jax.random.normal(key, (batch_size, input_length, input_dim))

    variables = model.init(key, x)
    params = variables["params"]

    y_pred = model.apply({"params": params}, x)

    assert y_pred.shape == (batch_size, 5, 1)
    assert jnp.issubdtype(y_pred.dtype, jnp.floating)
    assert jnp.all(jnp.isfinite(y_pred))


def test_forecaster_gradients_are_finite():
    model = Mamba2Forecaster(
        input_dim=1,
        d_model=32,
        n_layers=2,
        output_dim=1,
        forecast_horizon=5,
    )

    key_x, key_y = jax.random.split(jax.random.PRNGKey(1))
    batch_size, input_length, input_dim = 4, 10, 1

    x = jax.random.normal(key_x, (batch_size, input_length, input_dim))
    y_true = jax.random.normal(key_y, (batch_size, 5, 1))

    variables = model.init(key_x, x)
    params = variables["params"]

    def loss_fn(p):
        y_pred = model.apply({"params": p}, x)
        return jnp.mean((y_pred - y_true) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    assert jnp.isfinite(loss)

    def all_finite(x):
        return jnp.all(jnp.isfinite(x))

    assert jax.tree_util.tree_all(jax.tree_util.tree_map(all_finite, grads))
