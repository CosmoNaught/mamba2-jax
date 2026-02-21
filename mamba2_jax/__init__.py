# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mamba2-JAX: Pure JAX/Flax NNX implementation of Mamba2."""

__version__ = "1.0.0"

from .modeling import (
    ACT2FN,
    Mamba2Block,
    Mamba2Cache,
    Mamba2Config,
    Mamba2ForCausalLM,
    Mamba2Forecaster,
    Mamba2Mixer,
    Mamba2Model,
    RMSNorm,
    create_empty_cache,
    forward,
    segsum,
    ssd_forward,
)
from .params import (
    count_parameters,
    create_model_from_huggingface,
    create_model_from_torch_checkpoint,
    create_random_forecaster,
    create_random_model,
    load_pytorch_weights,
)
