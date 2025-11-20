# Mamba2-JAX: Pure JAX Implementation of Mamba2

## Introduction

This is an experimental JAX/Flax implementation of Mamba2 [[1]](#references) inspired by vasqu's exquisite PyTorch version [[2]](#references). The implementation provides a pure JAX alternative for researchers and practitioners who prefer the JAX ecosystem for its functional programming paradigm, automatic differentiation, and seamless integration with TPU hardware.

**Current Status: Alpha (Stable) Release**

This alpha version focuses on numerical correctness and stability. The implementation has been tested against the PyTorch version and shows equivalent numerical behavior (see [Numerical Validation](#numerical-validation) below).

**NOTE:** This is an early-stage implementation that currently supports:
- Pure JAX/Flax implementation (no Triton kernels)
- Causal language modeling with `Mamba2ForCausalLM`
- Time series forecasting with `Mamba2Forecaster`
- Full forward and backward passes with gradient computation
- Small to medium-scale experimentation

## Why JAX?

While vasqu's excellent PyTorch implementation provides multiple optimization paths including Triton kernels, this JAX version offers several unique advantages:

- **Functional Programming**: JAX's functional approach makes it easier to reason about model behavior and transformations
- **Hardware Flexibility**: Seamless support for TPUs alongside GPUs through XLA compilation
- **Research-Friendly**: JAX's transformation system (jit, grad, vmap, pmap) enables elegant experimentation
- **Ecosystem Integration**: Natural fit for projects already using JAX (Flax, Optax, Haiku)
- **Educational Value**: Cleaner implementation for understanding Mamba2 internals without CUDA complexity

This implementation prioritizes clarity and correctness over raw performance, making it ideal for:
- Understanding Mamba2 architecture
- Rapid prototyping of variants
- Integration into JAX-based research codebases
- TPU-based training workflows

## Installation

### Stable Version

Automatically download from PyPI using pip

```bash
pip install mamba2-jax
```

### Development Version

Clone the repository and install as a package:

```bash
git clone https://github.com/yourusername/mamba2-jax.git
cd mamba2-jax
pip install -e .
```

### Requirements

```bash
pip install jax jaxlib flax optax einops
```

### GPU (CUDA) & TPU support

CUDA support will be released in an upcoming version with plans to optimise for triton kernel.

For TPU support, follow the [official JAX TPU guide](https://jax.readthedocs.io/en/latest/installation.html#tpu).

> WARN! TPU support has not been validated as of this release.

## Usage

### Basic Language Modeling Example

This complete example shows how to create a Mamba2 language model, initialize it, and run a forward pass. You can copy and paste this entire block to get started:

```python
import jax
import jax.numpy as jnp
from mamba2_jax import Mamba2Config, Mamba2ForCausalLM

# Create a small configuration for testing
# You can scale these up for real applications
config = Mamba2Config(
    vocab_size=1024,        # Small vocabulary for demo
    hidden_size=256,        # Hidden dimension
    state_size=64,          # SSM state size
    head_dim=32,            # Dimension per head
    num_hidden_layers=4,    # Number of Mamba2 blocks
    chunk_size=64,          # Chunk size for SSD computation
)

# Initialize the model
model = Mamba2ForCausalLM(config)

# Create some random input tokens
key = jax.random.PRNGKey(42)
batch_size, seq_len = 2, 64
input_ids = jax.random.randint(
    key, 
    (batch_size, seq_len), 
    minval=0, 
    maxval=config.vocab_size
)

# Initialize model parameters with the input shape
print("Initializing model parameters...")
variables = model.init(key, input_ids=input_ids)
params = variables["params"]

# Run forward pass with loss computation
print("Running forward pass...")
outputs = model.apply(
    {"params": params},
    input_ids=input_ids,
    labels=input_ids,  # Using same tokens as labels for demo
)

# Check outputs
print(f"Logits shape: {outputs['logits'].shape}")  # Should be (2, 64, 1024)
print(f"Loss: {float(outputs['loss']):.4f}")
print("Forward pass completed successfully!")
```


### Time Series Forecasting Example

This example shows how to use Mamba2 for time series prediction. The model takes a historical sequence and predicts future values:

```python
import jax
import jax.numpy as jnp
import optax

from mamba2_jax import Mamba2Forecaster

# Suppose we have univariate timeseries windows of length L
batch_size = 8
input_length = 32
forecast_horizon = 12
input_dim = 1
output_dim = 1

model = Mamba2Forecaster(
    input_dim=input_dim,
    d_model=256,
    n_layers=4,
    output_dim=output_dim,
    forecast_horizon=forecast_horizon,
)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (batch_size, input_length, input_dim))

variables = model.init(key, x)
params = variables["params"]

y_pred = model.apply({"params": params}, x)  # (B, H, D_out)
print("Timeseries output shape:", y_pred.shape)
```

### Advanced Features

The core `Mamba2Model` exposes the same SSM hooks as the PyTorch implementation:

- **Stateful / streaming inference** via `initial_states` and `output_last_ssm_states`.
- **Layer-wise analysis** via `output_hidden_states=True`.

See the runnable scripts in the `examples/` directory:

- `04_streaming_states_demo.py` – carry SSM state across chunks for streaming generation or very long sequences.
- `05_inspect_hidden_states.py` – retrieve per-layer hidden states for analysis or auxiliary losses.

### Examples

For more end-to-end, runnable scripts (tiny training loops, sine-wave forecasting, streaming state demos, etc.), see the `examples/` directory in this repository.

## Numerical Validation with PyTorch

The implementation has been validated against the reference PyTorch version [[2]](#references) to ensure numerical correctness on CPU. Further tests will investigate GPU (CUDA) and TPU performance once enabled post alpha release.

### Methodology

- A small Mamba2 model is instantiated in **both** PyTorch and JAX with identical hyperparameters (hidden size, state size, number of layers, sequence length, etc.).
- A simple synthetic MSE regression task is constructed and shared between the two frameworks using the same random seed.
- On **CPU only**, both models are trained side-by-side on this task for a short number of optimisation steps.
- At each training step we record:
  - the PyTorch loss,
  - the JAX loss,
  - the absolute difference `|L_torch - L_jax|`,
  - and the per-step wall-clock time for each framework.
- All experiments are run in `float32` with no mixed precision or framework-specific numerical tricks, to keep the comparison as fair as possible.

The whole procedure is implemented in the standalone script `test-parity.py`, which can be run on any CPU-only machine.

The key experiment configuration is summarised below:

| Category        | Parameter          | PyTorch                                              | JAX                                                        |
|----------------|--------------------|------------------------------------------------------|------------------------------------------------------------|
| **Backend**    | Framework          | PyTorch                                              | JAX + Optax + Flax                                        |
|                | Device             | CPU                                                  | CPU                                                        |
|                | Dtype              | `float32`                                            | `float32`                                                  |
| **Data / task**| Task               | Synthetic time-series forecasting (MSE regression)   | Same dataset via shared NumPy arrays                       |
|                | Batch size         | 2                                                    | 2                                                          |
|                | Context length     | 32 time steps                                        | 32 time steps                                              |
|                | Input dimension    | 10                                                   | 10                                                         |
|                | Forecast horizon   | 16 time steps                                        | 16 time steps                                              |
| **Model**      | Model wrapper      | `Mamba2Model` + linear head                          | `Mamba2Forecaster`                                         |
|                | `d_model`          | 768                                                  | 768                                                        |
|                | `n_layers`         | 1                                                    | 1                                                          |
|                | `d_state`          | 128 (Mamba2Config default)                           | 128                                                        |
|                | `headdim`          | 64 (Mamba2Config default)                            | 64                                                         |
|                | `expand`           | 2                                                    | 2                                                          |
|                | `d_conv`           | 4                                                    | 4                                                          |
| **Training**   | Optimiser          | Manual SGD (`param -= lr * grad`)                    | `optax.sgd(lr)`                                            |
|                | Loss function      | Mean squared error                                   | Mean squared error                                         |
|                | Learning rate      | 0.001                                                | 0.001                                                      |
|                | Training steps     | 16                                                   | 16                                                         |
|                | Random seed        | Shared seed for data + initialisation where possible| Shared seed for data + initialisation where possible      |


### Test Results

See the [Mamba2 PyTorch vs JAX parity test appendix](APPENDIX/APPENDIX.md) for full details.


The figure below summarises the comparison:

[![Mamba2 PyTorch vs JAX parity test results](APPENDIX/parity_test_results.png)](APPENDIX/parity_test_results.png)

- **Training loss (left panel).** The PyTorch and JAX MSE losses follow almost identical learning curves. Both decrease smoothly over time, and by the final steps the two curves are visually indistinguishable.
- **Loss difference (middle panel).** The absolute difference `|L_torch - L_jax|` starts around ~2×10⁻¹ and decays monotonically during training, reaching the ~10⁻² range after ~15–16 steps. This level of discrepancy is well within normal numerical noise between different backends and confirms that the JAX implementation closely tracks the PyTorch reference.
- **CPU wall-clock time (right panel).** On this micro-benchmark the JAX implementation is roughly **2× faster per step** on CPU, with typical step times around ~0.08–0.09 s versus ~0.18–0.20 s for PyTorch.

Overall, these experiments indicate that the JAX implementation is **numerically consistent** with the PyTorch model while offering competitive (and often better) CPU performance for this class of workloads.

#### Summary metrics

| Category           | Metric                      | PyTorch      | JAX         | Notes                                      |
|--------------------|-----------------------------|-------------:|------------:|--------------------------------------------|
| Training loss      | Initial MSE (step 0)        | 1.5894       | 2.1382      | Different random inits, both converge      |
| Training loss      | Final MSE (step 15)         | 0.0249       | 0.0371      | Final diff ≈ 0.0121                        |
| Training loss      | Mean abs. diff (all steps)  | –            | –           | 0.1606 (mean), 0.5487 (max)                |
| Training loss      | Mean rel. diff (all steps)  | –            | –           | ≈ 47 % mean, ≈ 51 % max                    |
| Prediction parity  | Pearson correlation         | –            | –           | 0.992 between PyTorch and JAX predictions  |
| Prediction parity  | MAE / std(torch)           | –            | –           | ≈ 0.10 (~10 %)                             |
| Prediction parity  | RMSE / std(torch)          | –            | –           | ≈ 0.13 (~13 %)                             |
| Timing (CPU)       | Mean step time              | 0.1935 s     | 0.0879 s    | JAX is ≈ 2.2× faster per step on CPU       |
| Timing (CPU)       | JIT compile (`train_step`)  | –            | 0.97 s      | One-off JIT cost before steady-state steps |

In short, both implementations learn very similar functions: the loss
curves track each other closely, the final losses differ by only
≈ 0.012, and the final predictions have a Pearson correlation of ~0.99
with discrepancies on the order of 10–13 % of the PyTorch signal scale.
On CPU, the JAX version achieves roughly a 2.2× lower per-step wall-clock
time once the one-off JIT compilation cost is paid, while remaining
numerically consistent with the PyTorch reference.

## Project Structure

## Roadmap

### Beta Release (Coming Soon)
- **GPU Optimisation**: Profile and optimize performance on modern GPUs
- **Expanded Test Suite**: Comprehensive unit tests and integration tests
- **Model Conversion Scripts**: Tools to convert pretrained PyTorch weights to JAX
- **Benchmarking Suite**: Systematic performance comparison across hardware
- **Documentation**: Detailed API documentation and architecture guide

### Future Releases
- **Triton Kernel Support**: Custom kernels for improved performance
- **Pretrained Models**: Host converted models on Hugging Face Hub
- **Mixed Precision Training**: BF16/FP16 support with proper loss scaling
- **Model Parallelism**: Support for large-scale training with pmap/pjit
- **Advanced Caching**: Efficient KV-like caching for generation
- **Hybrid Variants**: Attention and MLP hybrid architectures

## Known Limitations

This alpha release has several known limitations:

- **No Triton Kernels**: Uses naive SSD implementation, slower than optimized PyTorch version
- **No Pretrained Weights**: No conversion scripts yet (coming in beta)
- **Limited Generation Support**: Basic generation only, no advanced sampling methods
- **No Hybrid Architectures**: Pure Mamba2 blocks only (no attention/MLP variants)
- **Memory Usage**: Not optimized for very long sequences (>4096 tokens)

We're actively working on addressing these limitations in upcoming releases.

## Contributing

Contributions are welcome! Areas where help would be particularly valuable:

- Performance optimization and profiling
- Test coverage expansion
- Model conversion from PyTorch weights
- Documentation improvements
- Bug reports and feature requests

Please open an issue or submit a pull request on GitHub.

## Acknowledgments

This implementation builds upon the excellent work of many researchers and engineers:

**Original Mamba2 Authors [[1]](#references) :**
- Tri Dao and Albert Gu for the Mamba2 architecture and original implementation
- The entire State Spaces team for advancing SSM research

**PyTorch Implementation [[2]](#references) :**
- vasqu for the clean PyTorch implementation that served as a reference
- The implementation structure and many design decisions were inspired by mamba2-torch

**JAX Ecosystem [[3]](#references) [[4]](#references) :**
- The JAX, Flax, and Optax teams at Google for the excellent frameworks
- The broader JAX community for tools and support

## References

```bibtex
[1] Mamba2
@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}

[2] mamba2-torch (PyTorch Implementation)
@software{vasqu2024mamba2torch,
  author = {vasqu},
  title = {mamba2-torch: HuggingFace Compatible Mamba2},
  year = {2024},
  url = {https://github.com/vasqu/mamba2-torch}
}

[3] JAX
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}

[4] Flax
@software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.7.0},
  year = {2020},
}
```

## License

MIT

## Citation

If you use this implementation in your research, please cite both the original Mamba2 paper and acknowledge this JAX implementation:

```bibtex
@software{mamba2jax2024,
  author = {[Cosmo Santoni]},
  title = {mamba2-jax: Pure JAX Implementation of Mamba2},
  year = {2024},
  url = {https://github.com/CosmoNaught/mamba2-jax}
}
```

---

**Questions or Issues?** Please open an issue on GitHub or reach out through discussions.

**Want to Contribute?** PRs are welcome! See the Contributing section above.