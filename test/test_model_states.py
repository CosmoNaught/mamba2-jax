import jax
import jax.numpy as jnp

from mamba2_jax import Mamba2Config, Mamba2Model


def make_tiny_config() -> Mamba2Config:
    return Mamba2Config(
        vocab_size=128,
        hidden_size=64,
        state_size=16,
        head_dim=16,
        conv_kernel=4,
        chunk_size=8,
        num_hidden_layers=2,
        expand=2,
        hidden_act="silu",
    )


def test_model_hidden_and_ssm_states_shapes():
    cfg = make_tiny_config()
    model = Mamba2Model(cfg)

    key = jax.random.PRNGKey(0)
    batch_size, seq_len = 2, 8

    input_ids = jax.random.randint(
        key,
        (batch_size, seq_len),
        minval=0,
        maxval=cfg.vocab_size,
    )

    variables = model.init(key, input_ids=input_ids)
    params = variables["params"]

    outputs = model.apply(
        {"params": params},
        input_ids=input_ids,
        output_hidden_states=True,
        output_last_ssm_states=True,
    )

    last_hidden = outputs["last_hidden_state"]
    hidden_states_list = outputs["hidden_states"]
    last_ssm_states_list = outputs["last_ssm_states"]

    # Final hidden
    assert last_hidden.shape == (batch_size, seq_len, cfg.hidden_size)
    assert jnp.all(jnp.isfinite(last_hidden))

    # One per block + final layer norm
    assert len(hidden_states_list) == cfg.num_hidden_layers + 1
    for h in hidden_states_list:
        assert h.shape == (batch_size, seq_len, cfg.hidden_size)
        assert jnp.all(jnp.isfinite(h))

    # One SSM state per layer, with (B, H, P, N)
    assert len(last_ssm_states_list) == cfg.num_hidden_layers
    for s in last_ssm_states_list:
        assert s.shape[0] == batch_size
        assert s.shape[1] == cfg.num_heads
        assert s.shape[2] == cfg.head_dim
        assert s.shape[3] == cfg.state_size
        assert jnp.all(jnp.isfinite(s))


def test_model_accepts_initial_states():
    """
    Smoke test that we can feed SSM final states back in as initial_states
    without shape errors.
    """
    cfg = make_tiny_config()
    model = Mamba2Model(cfg)

    key = jax.random.PRNGKey(1)
    batch_size, seq_len = 2, 8

    input_ids = jax.random.randint(
        key,
        (batch_size, seq_len),
        minval=0,
        maxval=cfg.vocab_size,
    )

    variables = model.init(key, input_ids=input_ids)
    params = variables["params"]

    first = model.apply(
        {"params": params},
        input_ids=input_ids,
        output_last_ssm_states=True,
    )
    initial_states = first["last_ssm_states"]

    second = model.apply(
        {"params": params},
        input_ids=input_ids,
        initial_states=initial_states,
        output_last_ssm_states=True,
    )

    assert len(second["last_ssm_states"]) == cfg.num_hidden_layers
