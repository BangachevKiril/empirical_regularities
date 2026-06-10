# ICA Known-A Mean Propagation Experiments

This note documents the ICA experiments comparing Monte Carlo sampling against
cumulant propagation when the ICA parameter matrix `A` is known.

## Setup

The ICA input distribution is

```text
x = A r
```

where `A` has shape `(n, p)` with independent `N(0, 1)` entries, and
`r in {-1, 1}^p` has independent Rademacher coordinates. The MLP is a
depth-`L` ReLU MLP with square `n x n` layers.

The runs in this note use:

```text
L = 4
p = 16
mlp_seed = 0
ica_seed = 0
true_samples = 2^24
sampling baseline m = 2^k for k = 1, ..., 15
cumulant propagation K = 1, 2, 3, 4
```

## Input Information By K

For the known-A ICA runs, the initialized input cumulants are:

```text
K1 = 0
K2 = A A^T
K3 = 0
```

The difference between propagation orders is the amount of higher-order
information retained.

`K=1` uses only the radial/trace component of `K2`, effectively average input
variance.

`K=2` uses the full covariance `K2 = A A^T`.

`K=3` uses the full `K2` plus one scalar radial component of the fourth
cumulant. With

```text
tau = -2 * sum_a ||A[:, a]||_2^4,
```

the scalar core is

```text
3 * tau / (n (n + 2)).
```

The code passes this as an `HTensor` with `r=2`, so it represents only a pure
radial fourth-order component, not the full fourth cumulant.

`K=4` uses the full fourth cumulant

```text
K4 = -2 * sum_a A[:, a]^{\otimes 4},
```

stored in factored form.

## Reproduction Commands

The `n=64` run:

```bash
python -m experiments.mlp_mean_concentration \
  --device cuda \
  --n 64 \
  --depth 4 \
  --p 16 \
  --input-distribution ica \
  --ica-seed 0 \
  --mlp-seed 0 \
  --true-samples 16777216 \
  --batch-size 8192 \
  --k-min 1 \
  --k-max 15 \
  --runs 10 \
  --cumulant-orders 1,2,3,4 \
  --cumulant-factor \
  --sample-cumulant-k-min 1 \
  --sample-cumulant-k-max 0 \
  --output-dir results/mlp_mean_ica_N01_known_sampling_n64_L4_p16_k15_3tau_k3
```

The `n=256` run:

```bash
python -m experiments.mlp_mean_concentration \
  --device cuda \
  --n 256 \
  --depth 4 \
  --p 16 \
  --input-distribution ica \
  --ica-seed 0 \
  --mlp-seed 0 \
  --true-samples 16777216 \
  --batch-size 8192 \
  --k-min 1 \
  --k-max 15 \
  --runs 10 \
  --cumulant-orders 1,2,3,4 \
  --cumulant-factor \
  --sample-cumulant-k-min 1 \
  --sample-cumulant-k-max 0 \
  --output-dir results/mlp_mean_ica_N01_known_sampling_n256_L4_p16_k15_3tau_k3
```

The single-panel error-vs-FLOPs plot:

```bash
python -m experiments.plot_error_vs_flops \
  --result-dir results/mlp_mean_ica_N01_known_sampling_n256_L4_p16_k15_3tau_k3 \
  --output results/mlp_mean_ica_N01_known_sampling_n256_L4_p16_k15_3tau_k3/error_vs_flops.svg \
  --title "ICA with known parameter A"
```

## Results

Squared error for known-A cumulant propagation:

| n | K=1 | K=2 | K=3 | K=4 | Sampling, m=2^15 |
|---:|---:|---:|---:|---:|---:|
| 64 | 16.3567 | 0.286543 | 0.0911393 | 0.0194075 | 0.00909959 |
| 256 | 26.4893 | 0.750080 | 0.216058 | 0.0543234 | 0.0412578 |

The generated n=256 plot is:

```text
results/mlp_mean_ica_N01_known_sampling_n256_L4_p16_k15_3tau_k3/error_vs_flops.png
```
