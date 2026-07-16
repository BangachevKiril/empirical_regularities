# Simple Ordinary-CP Empirical Propagation

This folder contains a self-contained implementation of empirical ordinary cumulant
propagation. A stored order-`k` cumulant is represented as

```text
T[i1, ..., ik] = (1 / M) sum_t A1[i1, t] ... Ak[ik, t]
```

with `M = max(1, ceil(delta * m))`. Linear layers act exactly by `weight @ factor`
on every mode, with deterministic bias added only to the order-one cumulant.

The nonlinear path uses polynomial Wick coefficients, vector-partition diagram
catalogs, one independent sampled rank per diagram-block occurrence, shared rank
inside each occurrence, and compression back to exactly `M` CP columns. Repeated
visible indices are ordinary entries of the full tensor and are not masked.

The implementation provides both the literal two-stage nonlinear estimator
(`reference_two_stage_nonlinear=True`) and a fused streaming estimator
(`False`, the default). Non-polynomial activations are deliberately not claimed
here; use exact `numpy.polynomial.Polynomial` objects or built-in `square`/`cube`.

Each run also fills an analytic FLOP counter:

```python
result.diagnostics.total_analytic_flops
result.diagnostics.analytic_flops_by_stage
result.diagnostics.flop_count_convention
```

The counter includes initialization compression, every linear layer, moment
extraction before nonlinearities, nonlinear diagram propagation/compression,
and final mean extraction. It counts floating-point adds/muls/divs in the CP
arithmetic and excludes RNG, indexing, memory movement, allocation, and Python
overhead. For the fused nonlinear path, nonlinear counts are expected FLOPs over
diagram sampling.

An experimental ReLU path is available only when `allow_nonpolynomial=True` and
`hermite_degree_cap` is set. It uses distributional ReLU Wick coefficients and
does not carry the finite-polynomial theorem guarantee.

The folder also includes `mean_prop`, a K=1-style baseline that computes the
coordinatewise empirical mean and population variance, then propagates only
mean and diagonal variance. Its diagnostics include:

```python
result.diagnostics.total_analytic_flops
result.diagnostics.analytic_flops_by_stage
```

Pass `sample_budget=M` to consume only the first `min(M, m)` samples from a
larger sample tensor. The analytic initialization FLOPs are charged to the
consumed sample count, not to the available `m`.

The `cov_prop` module adds a custom covariance propagation baseline with a
budget `M`: if `M >= n`, it estimates a dense empirical covariance from `M`
samples and runs Gaussian covariance propagation; if `M < n`, it dispatches to
the order-2 CP path. Dense ReLU covariance moments are computed with
Gauss-Hermite bivariate moments, and the diagnostics use the same
`total_analytic_flops` / `analytic_flops_by_stage` fields. Like `mean_prop`,
it consumes only `min(M, m)` samples, so when `M < m` the initialization FLOP
count has no hidden dependence on `m`.

## K=4 Hybrid Path

`ordinary_cp_mlp_k4` is the direct-rank K=4 entry point. It takes an explicit
integer `rank=M` and uses the packaged retained diagram list by default:

```python
result = ordinary_cp_mlp_k4(
    mlp,
    samples,
    rank=128,
    config=K4OrdinaryCPConfig(
        polynomial_by_layer=(Polynomial([0.0, 0.0, 1.0]),),
        activate_final=False,
    ),
)
```

The K=4 state keeps the mean explicit, stores covariance densely exactly when
`M >= n`, stores covariance as CP when `M < n`, and keeps third/fourth cumulants
as rank-`M` CP tensors. Runtime diagrams are loaded from
`diagram_lists/k4_default_orbits.json`, expanded from 26 orbit rows to 82
labelled diagrams with counts `(3, 15, 35, 29)`. Custom retained catalog paths
or Python diagram lists can be supplied explicitly.

The old `ordinary_cp_mlp(..., k_max=4, delta=...)` compatibility path now
delegates to this direct-rank implementation with `M = ceil(delta * m)`.

## Data Generators

`simple_CP_empirical_prop.data_generators` contains reusable generators:

```python
from simple_CP_empirical_prop.data_generators import (
    Isotropic_Gaussian,
    ICA,
    Wishart,
    MLP_dataset,
    image_dataset,
)

gaussian = Isotropic_Gaussian(n=256, m=4096, seed=0)
ica_data = ICA(n=256, p=512, m=4096, seed=0)
wishart_data = Wishart(n=256, p=512, m=4096, seed=0)
mlp_data = MLP_dataset(n=256, L=4, m=4096, seed=0)
```

Every generator returns a `GeneratedDataset` with `.samples`, `.metadata`, and
`.tensors`. Recorded tensors include `Wishart.tensors["Sigma"]`,
`Wishart.tensors["V"]`, `ICA.tensors["mixing_matrix"]`, MLP weights, and
`image_dataset(..., projection="random").tensors["W"]`. Calling
`dataset.save(path)` writes `samples.pt`, `recorded_tensors.pt`, and
`metadata.json`.

`image_dataset` supports `name in {"MNIST", "CIFAR", "ImageNET"}`. It converts
loaded PIL images to RGB PNGs before vectorization. With `projection=None`, the
output dimension is `3 * k^2`; with `projection="random"`, `n` is the projected
dimension and the fixed Gaussian projection matrix is recorded as `W`.

Run the focused tests from the repository parent with:

```bash
PYTHONPATH="/Users/kirilbangachev/Documents/Empirical Regularities" \
python -m unittest discover \
  -s "/Users/kirilbangachev/Documents/Empirical Regularities/simple_CP_empirical_prop/tests"
```

Run the smoke example with:

```bash
PYTHONPATH="/Users/kirilbangachev/Documents/Empirical Regularities" \
python "/Users/kirilbangachev/Documents/Empirical Regularities/simple_CP_empirical_prop/smoke.py"
```
