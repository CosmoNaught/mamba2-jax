import jax
import jax.numpy as jnp

from mamba2_jax.ssd import ssd_naive


def test_ssd_residual_path_matches_D_times_x():
    B, L, H, P, N = 2, 7, 3, 4, 5
    key = jax.random.PRNGKey(0)

    x = jax.random.normal(key, (B, L, H, P))
    dt = jnp.ones((B, L, H)) * 0.5
    A = jnp.zeros((H,))
    B_mat = jnp.zeros((B, L, H, N))
    C_mat = jnp.zeros((B, L, H, N))
    D = jnp.linspace(0.5, 1.5, H)
    dt_bias = jnp.zeros((H,))

    y, final_state = ssd_naive(
        x=x,
        dt=dt,
        A=A,
        B_mat=B_mat,
        C_mat=C_mat,
        chunk_size=4,
        D=D,
        dt_bias=dt_bias,
        dt_min=1e-4,
        dt_max=1e4,
        initial_states=None,
        return_final_states=True,
    )

    expected = D.reshape(1, 1, H, 1) * x

    assert y.shape == x.shape
    assert final_state is not None
    assert final_state.shape == (B, H, P, N)

    # use jnp.allclose instead of jnp.testing.assert_allclose
    assert jnp.allclose(y, expected, rtol=1e-5, atol=1e-6)


def test_ssd_zero_input_gives_zero_output():
    B, L, H, P, N = 1, 5, 2, 3, 4

    x = jnp.zeros((B, L, H, P))
    dt = jnp.ones((B, L, H)) * 0.5
    A = jnp.ones((H,))
    key = jax.random.PRNGKey(1)
    B_mat = jax.random.normal(key, (B, L, H, N))
    C_mat = jax.random.normal(key, (B, L, H, N))
    D = jnp.ones((H,))
    dt_bias = jnp.zeros((H,))

    initial_states = jnp.zeros((B, 1, H, P, N))

    y, final_state = ssd_naive(
        x=x,
        dt=dt,
        A=A,
        B_mat=B_mat,
        C_mat=C_mat,
        chunk_size=3,
        D=D,
        dt_bias=dt_bias,
        dt_min=1e-4,
        dt_max=1e4,
        initial_states=initial_states,
        return_final_states=True,
    )

    assert y.shape == x.shape
    assert final_state.shape == (B, H, P, N)

    # same here – pure jnp
    assert jnp.allclose(y, jnp.zeros_like(x), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(final_state, jnp.zeros_like(final_state), rtol=1e-6, atol=1e-6)
