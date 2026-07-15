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
