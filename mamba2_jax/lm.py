# mamba2_jax/lm.py

from typing import Dict, Optional

import jax.numpy as jnp
from flax import linen as nn
import optax

from .config import Mamba2Config
from .model import Mamba2Model


class Mamba2ForCausalLM(nn.Module):
    """
    JAX/Flax Causal LM head on top of the Mamba2 backbone.
    """
    config: Mamba2Config

    def setup(self):
        cfg = self.config
        self.backbone = Mamba2Model(cfg)
        self.lm_head = nn.Dense(
            cfg.vocab_size,
            use_bias=False,
        )
        # Note: we are not literally weight-tying embeddings and lm_head yet.

    def __call__(
        self,
        input_ids: jnp.ndarray,                 # (B, L)
        labels: Optional[jnp.ndarray] = None,   # (B, L)
        train: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        # 1. Backbone forward
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            output_hidden_states=False,
            output_last_ssm_states=False,
        )
        hidden_states = backbone_outputs["last_hidden_state"]  # (B, L, H)

        # 2. LM head
        logits = self.lm_head(hidden_states)  # (B, L, vocab_size)

        # 3. Optional loss (shifted)
        loss = None
        if labels is not None:
            # shift: tokens < n predict token n
            shift_logits = logits[:, :-1, :]   # (B, L-1, V)
            shift_labels = labels[:, 1:]       # (B, L-1)

            # flatten batch & time
            shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])  # (B*(L-1), V)
            shift_labels = shift_labels.reshape(-1)                          # (B*(L-1),)

            loss_vec = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits,
                shift_labels,
            )
            loss = loss_vec.mean()

        out = {"logits": logits}
        if loss is not None:
            out["loss"] = loss

        return out
