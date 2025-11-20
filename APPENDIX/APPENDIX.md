# Appendix: Mamba2 PyTorch vs JAX Parity Test

This appendix contains the raw configuration, per-step training logs, and
summary statistics for the CPU-based numerical parity test between the
PyTorch and JAX implementations.

---

## Configuration

| Setting            | Value                        |
|--------------------|------------------------------|
| Batch size         | 2                            |
| Context length     | 32                           |
| Input dimension    | 10                           |
| d_model            | 768                          |
| Number of layers   | 1                            |
| Forecast horizon   | 16                           |
| Training steps     | 16                           |
| Learning rate      | 0.001                        |
| Device             | CPU                          |
| PyTorch config     | Uses `Mamba2Config` defaults |
| JAX config         | Matched to PyTorch defaults  |

**Note:** PyTorch uses the `Mamba2Config` defaults (e.g. `d_state=128`,
`headdim=64`, etc.) and the JAX model is configured to match these
defaults as closely as possible.

---

## Training losses and timings (per step)

| Step | PyTorch loss | PyTorch time | JAX loss | JAX time |
|-----:|-------------:|-------------:|---------:|---------:|
|  0 | 1.589445 | 0.2359s | 2.138174 | 0.0929s |
|  1 | 1.174688 | 0.1950s | 1.619388 | 0.0851s |
|  2 | 0.870945 | 0.1906s | 1.227181 | 0.0845s |
|  3 | 0.648227 | 0.1907s | 0.931120 | 0.0813s |
|  4 | 0.484461 | 0.1908s | 0.707282 | 0.0857s |
|  5 | 0.363588 | 0.1939s | 0.537731 | 0.0917s |
|  6 | 0.273993 | 0.1904s | 0.409152 | 0.0894s |
|  7 | 0.207287 | 0.1894s | 0.311597 | 0.0863s |
|  8 | 0.157403 | 0.1911s | 0.237559 | 0.0840s |
|  9 | 0.119940 | 0.1898s | 0.181345 | 0.0852s |
| 10 | 0.091689 | 0.1898s | 0.138634 | 0.0856s |
| 11 | 0.070305 | 0.1899s | 0.106146 | 0.0869s |
| 12 | 0.054058 | 0.1892s | 0.081401 | 0.0878s |
| 13 | 0.041674 | 0.1897s | 0.062524 | 0.0874s |
| 14 | 0.032204 | 0.1893s | 0.048098 | 0.0972s |
| 15 | 0.024942 | 0.1901s | 0.037057 | 0.0961s |

### Aggregate timing

| Metric                             | PyTorch | JAX      |
|------------------------------------|--------:|---------:|
| Mean per-step wall-clock time      | 0.1935s | 0.0879s  |
| Approx. speedup (PyTorch / JAX)    |   –     | ≈ 2.2×   |
| JIT compile time for `train_step` |   –     | 0.9674s  |

Final training losses and difference:

| Metric                 | Value      |
|------------------------|-----------:|
| Final PyTorch loss     | 0.024942   |
| Final JAX loss         | 0.037057   |
| Final loss difference  | 0.01211438 |

---

## Statistical comparison of training losses

| Metric                   | Value        |
|--------------------------|-------------:|
| Mean absolute difference | 0.16059623   |
| Max absolute difference  | 0.54872906   |
| Mean relative difference | 47.081195 %  |
| Max relative difference  | 51.199437 %  |

These are computed across the matched loss curves
\(`L_torch(step)` vs `L_jax(step)`\).

---

## Prediction comparison

All metrics below are computed on the final model predictions for a
held-out batch (32 elements).

| Metric                        | Value        |
|-------------------------------|-------------:|
| Number of elements            | 32           |
| PyTorch mean                  | -0.06450354  |
| JAX mean                      | -0.07521452  |
| PyTorch std                   | 1.05231334   |
| JAX std                       | 1.03512262   |
| MAE \(|torch - jax|\)         | 0.10564238   |
| RMSE                          | 0.13409644   |
| MAE / std(torch)              | 0.100391  (~10.04 %) |
| RMSE / std(torch)             | 0.127430  (~12.74 %) |
| L2(diff) / L2(torch)          | 0.127191  (~12.72 %) |
| Pearson correlation           | 0.99193425  |

---

## Generated plots

The following diagnostic plots are produced by the parity test script:

| Plot                         | File                               |
|------------------------------|------------------------------------|
| Training losses, diff & time | `parity_test_results.png`         |
| Prediction parity scatter    | `prediction_parity.png`           |

These files are generated in the working directory when running
`test-parity.py`.
