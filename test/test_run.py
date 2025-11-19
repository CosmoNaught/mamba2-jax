import jax
import jax.numpy as jnp
import optax

from mamba2_jax.config import Mamba2Config
from mamba2_jax.lm import Mamba2ForCausalLM


def main():
    cfg = Mamba2Config(
        vocab_size=1000,
        hidden_size=64,
        state_size=16,
        head_dim=8,
        num_hidden_layers=2,
    )

    model = Mamba2ForCausalLM(cfg)

    key = jax.random.PRNGKey(0)
    B, L = 2, 16
    dummy_ids = jax.random.randint(key, (B, L), minval=0, maxval=cfg.vocab_size)

    # init params
    init_key, data_key = jax.random.split(key)
    params = model.init(init_key, input_ids=dummy_ids)

    # simple optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, batch_ids):
        outputs = model.apply(params, input_ids=batch_ids, labels=batch_ids)
        return outputs["loss"]

    @jax.jit
    def train_step(params, opt_state, batch_ids):
        (loss_value, grads) = jax.value_and_grad(loss_fn)(params, batch_ids)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # run a couple of dummy steps
    for step in range(3):
        step_key, data_key = jax.random.split(data_key)
        batch_ids = jax.random.randint(step_key, (B, L), 0, cfg.vocab_size)
        params, opt_state, loss_value = train_step(params, opt_state, batch_ids)
        print(f"step {step} loss:", float(loss_value))

    # final forward just to check shape
    outputs = model.apply(params, input_ids=dummy_ids)
    print("logits shape:", outputs["logits"].shape)


if __name__ == "__main__":
    main()
