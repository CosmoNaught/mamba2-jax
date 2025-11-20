import jax
import jax.numpy as jnp

from mamba2_jax import Mamba2Config, Mamba2ForCausalLM


def make_tiny_config() -> Mamba2Config:
    # Small config to keep tests fast and CPU-friendly
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


def test_lm_forward_shapes_and_loss():
    cfg = make_tiny_config()
    model = Mamba2ForCausalLM(cfg)

    key = jax.random.PRNGKey(0)
    batch_size, seq_len = 2, 8

    input_ids = jax.random.randint(
        key,
        (batch_size, seq_len),
        minval=0,
        maxval=cfg.vocab_size,
    )

    # Init & forward with labels so the loss path is exercised
    variables = model.init(key, input_ids=input_ids, labels=input_ids)
    outputs = model.apply(variables, input_ids=input_ids, labels=input_ids)

    logits = outputs["logits"]
    loss = outputs["loss"]

    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert jnp.issubdtype(logits.dtype, jnp.floating)
    assert jnp.isfinite(loss)


def test_lm_gradients_are_finite():
    cfg = make_tiny_config()
    model = Mamba2ForCausalLM(cfg)

    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8

    input_ids = jax.random.randint(
        key,
        (batch_size, seq_len),
        minval=0,
        maxval=cfg.vocab_size,
    )

    variables = model.init(key, input_ids=input_ids, labels=input_ids)
    params = variables["params"]

    def loss_fn(p):
        outputs = model.apply({"params": p}, input_ids=input_ids, labels=input_ids)
        return outputs["loss"]

    loss, grads = jax.value_and_grad(loss_fn)(params)

    assert jnp.isfinite(loss)

    def all_finite(x):
        return jnp.all(jnp.isfinite(x))

    assert jax.tree_util.tree_all(jax.tree_util.tree_map(all_finite, grads))
