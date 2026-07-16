"""Custom covariance propagation with a dense/CP switch.

For a requested rank/sample budget ``M`` and width ``n``:

* if ``M >= n``, initialize a dense empirical mean/covariance from ``M``
  samples and run Gaussian covariance propagation;
* if ``M < n``, run the order-2 ordinary-CP propagation path as a low-rank
  covariance proxy.

The dense ReLU covariance update uses a one-dimensional Gauss-Hermite rule for
the bivariate Gaussian ReLU cross moment. The CP branch reuses the package's
order-2 Wick/diagram machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from numpy.polynomial.hermite import hermgauss
from torch import Tensor

from .config import OrdinaryCPConfig
from .hermite import activation_spec_from_name
from .propagate import propagate_linear_activation_stages


COV_PROP_FLOP_CONVENTION = (
    "Counts floating-point adds/muls/divs/sqrt in empirical dense covariance "
    "initialization, dense linear covariance propagation, Gauss-Hermite ReLU "
    "moment propagation, and the reused order-2 CP diagnostics when M < n. "
    "Excludes RNG, indexing, allocation, memory movement, Python overhead, and "
    "transcendental exp/erf implementation cost."
)


@dataclass
class CovPropDiagnostics:
    sample_count: int
    input_width: int
    rank: int
    mode: str
    used_sample_count: int | None = None
    available_sample_count: int | None = None
    sample_budget: int | None = None
    total_analytic_flops: float = 0.0
    analytic_flops_by_stage: dict[str, float] = field(default_factory=dict)
    flop_count_convention: str = COV_PROP_FLOP_CONVENTION
    notes: list[str] = field(default_factory=list)

    def add_flops(self, label: str, flops: float) -> None:
        value = float(flops)
        self.analytic_flops_by_stage[label] = (
            self.analytic_flops_by_stage.get(label, 0.0) + value
        )
        self.total_analytic_flops += value


@dataclass
class CovPropState:
    mean: Tensor
    covariance: Tensor


@dataclass
class CovPropResult:
    mean: Tensor
    covariance: Tensor | None
    layer_means: list[Tensor] | None
    layer_covariances: list[Tensor] | None
    diagnostics: CovPropDiagnostics


def _budgeted_samples(samples: Tensor, sample_budget: int | None) -> tuple[Tensor, int]:
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, width]")
    if sample_budget is None:
        used = samples.shape[0]
    else:
        if not isinstance(sample_budget, int) or sample_budget < 1:
            raise ValueError("sample_budget must be a positive integer")
        used = min(sample_budget, samples.shape[0])
    return samples[:used], used


def empirical_mean_covariance(samples: Tensor) -> CovPropState:
    """Return empirical mean and population covariance with denominator ``m``."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, width]")
    sample_count = samples.shape[0]
    mean = samples.mean(dim=0)
    centered = samples - mean
    covariance = centered.transpose(0, 1) @ centered / sample_count
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    return CovPropState(mean=mean, covariance=covariance)


def empirical_mean_covariance_flops(*, sample_count: int, width: int) -> float:
    """FLOPs for the implemented dense empirical mean/covariance estimator."""
    mean_flops = sample_count * width
    center_flops = sample_count * width
    covariance_matmul = width * width * (2 * sample_count - 1)
    covariance_scale = width * width
    symmetrize = 2 * width * width
    return float(mean_flops + center_flops + covariance_matmul + covariance_scale + symmetrize)


def cov_prop_linear(
    state: CovPropState,
    linear_or_weight: torch.nn.Linear | Tensor,
    bias: Tensor | None = None,
) -> CovPropState:
    """Propagate dense mean/covariance through a deterministic linear map."""
    if isinstance(linear_or_weight, torch.nn.Linear):
        weight = linear_or_weight.weight
        layer_bias = linear_or_weight.bias
    else:
        weight = linear_or_weight
        layer_bias = bias
    if weight.ndim != 2 or weight.shape[1] != state.mean.shape[0]:
        raise ValueError("weight must have shape [output_width, input_width]")
    if layer_bias is not None and (
        layer_bias.ndim != 1 or layer_bias.shape[0] != weight.shape[0]
    ):
        raise ValueError("bias must have shape [output_width]")

    work_weight = weight.to(dtype=state.mean.dtype, device=state.mean.device)
    mean = work_weight @ state.mean
    if layer_bias is not None:
        mean = mean + layer_bias.to(dtype=state.mean.dtype, device=state.mean.device)
    covariance = work_weight @ state.covariance @ work_weight.transpose(0, 1)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    return CovPropState(mean=mean, covariance=covariance)


def cov_prop_linear_flops(
    *,
    input_width: int,
    output_width: int,
    has_bias: bool,
) -> float:
    """Analytic FLOPs for dense ``mean = W mu`` and ``cov = W Sigma W.T``."""
    mean_matvec = output_width * (2 * input_width - 1)
    bias_add = output_width if has_bias else 0
    left_matmul = output_width * input_width * (2 * input_width - 1)
    right_matmul = output_width * output_width * (2 * input_width - 1)
    symmetrize = 2 * output_width * output_width
    return float(mean_matvec + bias_add + left_matmul + right_matmul + symmetrize)


def _normal_pdf_cdf(alpha: Tensor) -> tuple[Tensor, Tensor]:
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    pdf = torch.exp(-0.5 * alpha.square()) * inv_sqrt_2pi
    cdf = 0.5 * (1.0 + torch.erf(alpha * inv_sqrt_2))
    return pdf, cdf


def _relu_univariate_moments(
    mean: Tensor,
    variance: Tensor,
    *,
    variance_min: float,
) -> tuple[Tensor, Tensor, Tensor]:
    variance = variance.clamp_min(variance_min)
    sigma = variance.sqrt()
    alpha = mean / sigma
    pdf, cdf = _normal_pdf_cdf(alpha)
    relu_mean = sigma * pdf + mean * cdf
    relu_second = (mean.square() + variance) * cdf + mean * sigma * pdf
    relu_variance = (relu_second - relu_mean.square()).clamp_min(variance_min)
    return relu_mean, relu_second, relu_variance


def cov_prop_activation(
    state: CovPropState,
    activation: str,
    *,
    quadrature_degree: int = 40,
    variance_min: float = 1e-10,
    correlation_eps: float = 1e-6,
    quadrature_float64: bool = True,
) -> CovPropState:
    """Apply a coordinatewise activation under a joint Gaussian assumption."""
    if variance_min <= 0:
        raise ValueError("variance_min must be positive")
    if quadrature_degree < 2:
        raise ValueError("quadrature_degree must be at least 2")
    name = activation.lower()
    if name == "identity":
        return state
    if name != "relu":
        raise NotImplementedError(f"unsupported cov_prop activation {activation!r}")

    dtype = torch.float64 if quadrature_float64 else state.mean.dtype
    mean = state.mean.to(dtype=dtype)
    covariance = state.covariance.to(dtype=dtype)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    variance = covariance.diagonal().clamp_min(variance_min)
    sigma = variance.sqrt()
    relu_mean, relu_second_diag, relu_variance = _relu_univariate_moments(
        mean,
        variance,
        variance_min=variance_min,
    )

    denom = sigma[:, None] * sigma[None, :]
    rho = (covariance / denom.clamp_min(variance_min)).clamp(
        -1.0 + correlation_eps,
        1.0 - correlation_eps,
    )

    nodes_np, weights_np = hermgauss(quadrature_degree)
    z = torch.as_tensor(
        nodes_np * math.sqrt(2.0),
        device=mean.device,
        dtype=dtype,
    )
    weights = torch.as_tensor(
        weights_np / math.sqrt(math.pi),
        device=mean.device,
        dtype=dtype,
    )
    one_minus_rho2 = (1.0 - rho.square()).clamp_min(correlation_eps)
    cond_sigma = sigma[None, :] * one_minus_rho2.sqrt()
    second = torch.zeros_like(covariance)
    for node, weight in zip(z, weights):
        left = torch.relu(mean + sigma * node)[:, None]
        cond_mean = mean[None, :] + sigma[None, :] * rho * node
        alpha = cond_mean / cond_sigma
        pdf, cdf = _normal_pdf_cdf(alpha)
        cond_relu_mean = cond_sigma * pdf + cond_mean * cdf
        second = second + weight * left * cond_relu_mean

    independent = rho.abs() < correlation_eps
    second[independent] = (relu_mean[:, None] * relu_mean[None, :])[independent]
    index = torch.arange(mean.numel(), device=mean.device)
    second[index, index] = relu_second_diag
    second = 0.5 * (second + second.transpose(0, 1))
    covariance_out = second - relu_mean[:, None] * relu_mean[None, :]
    covariance_out = 0.5 * (covariance_out + covariance_out.transpose(0, 1))
    covariance_out[index, index] = relu_variance
    return CovPropState(mean=relu_mean, covariance=covariance_out)


def cov_prop_activation_flops(
    *,
    width: int,
    activation: str,
    quadrature_degree: int,
) -> float:
    """Approximate scalar arithmetic count for dense activation propagation."""
    name = activation.lower()
    if name == "identity":
        return 0.0
    if name != "relu":
        raise NotImplementedError(f"unsupported cov_prop activation {activation!r}")
    univariate = 22 * width
    correlation_setup = 8 * width * width
    # Per quadrature node/pair: conditional mean, alpha, pdf/cdf affine work,
    # conditional ReLU mean, product with left ReLU, and weighted accumulation.
    quadrature = quadrature_degree * 20 * width * width
    final_covariance = 4 * width * width
    return float(univariate + correlation_setup + quadrature + final_covariance)


def regular_cov_prop_stages(
    stages: Sequence[tuple[torch.nn.Linear, str | None]],
    samples: Tensor,
    *,
    sample_budget: int | None = None,
    quadrature_degree: int = 40,
    variance_min: float = 1e-10,
    quadrature_float64: bool = True,
    return_all: bool = False,
) -> CovPropResult:
    """Run dense Gaussian covariance propagation over explicit stages."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, width]")
    available_sample_count = samples.shape[0]
    samples, used_sample_count = _budgeted_samples(samples, sample_budget)
    diagnostics = CovPropDiagnostics(
        sample_count=used_sample_count,
        input_width=samples.shape[1],
        rank=sample_budget if sample_budget is not None else used_sample_count,
        mode="dense",
        used_sample_count=used_sample_count,
        available_sample_count=available_sample_count,
        sample_budget=sample_budget,
        notes=[
            "M >= n: initialized dense empirical mean/covariance.",
            "Linear covariance propagation is exact for the Gaussian state.",
            "ReLU covariance propagation uses Gauss-Hermite bivariate moments.",
            "Empirical initialization consumes only min(sample_budget, available samples).",
        ],
    )
    state = empirical_mean_covariance(samples)
    diagnostics.add_flops(
        "empirical_mean_covariance",
        empirical_mean_covariance_flops(
            sample_count=samples.shape[0],
            width=samples.shape[1],
        ),
    )
    layer_means: list[Tensor] | None = [state.mean] if return_all else None
    layer_covariances: list[Tensor] | None = [state.covariance] if return_all else None
    for layer_index, (linear, activation) in enumerate(stages):
        diagnostics.add_flops(
            f"layer_{layer_index}_linear",
            cov_prop_linear_flops(
                input_width=int(linear.weight.shape[1]),
                output_width=int(linear.weight.shape[0]),
                has_bias=linear.bias is not None,
            ),
        )
        state = cov_prop_linear(state, linear)
        if activation is not None:
            diagnostics.add_flops(
                f"layer_{layer_index}_activation_{activation}",
                cov_prop_activation_flops(
                    width=state.mean.numel(),
                    activation=activation,
                    quadrature_degree=quadrature_degree,
                ),
            )
            state = cov_prop_activation(
                state,
                activation,
                quadrature_degree=quadrature_degree,
                variance_min=variance_min,
                quadrature_float64=quadrature_float64,
            )
        if layer_means is not None:
            layer_means.append(state.mean)
        if layer_covariances is not None:
            layer_covariances.append(state.covariance)
    return CovPropResult(
        mean=state.mean,
        covariance=state.covariance,
        layer_means=layer_means,
        layer_covariances=layer_covariances,
        diagnostics=diagnostics,
    )


def custom_cov_prop_stages(
    stages: Sequence[tuple[torch.nn.Linear, str | None]],
    samples: Tensor,
    *,
    rank: int | None = None,
    relu_degree_cap: int = 2,
    cp_seed: int = 0,
    quadrature_degree: int = 40,
    variance_min: float = 1e-10,
    quadrature_float64: bool = True,
    return_all: bool = False,
) -> CovPropResult:
    """Run the dense/CP covariance propagation switch requested for experiments."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, width]")
    requested_rank = int(rank if rank is not None else samples.shape[0])
    if requested_rank < 1:
        raise ValueError("rank must be positive")
    available_sample_count = samples.shape[0]
    budgeted_samples, used_sample_count = _budgeted_samples(samples, requested_rank)
    width = samples.shape[1]
    if requested_rank >= width:
        return regular_cov_prop_stages(
            stages,
            samples,
            sample_budget=requested_rank,
            quadrature_degree=quadrature_degree,
            variance_min=variance_min,
            quadrature_float64=quadrature_float64,
            return_all=return_all,
        )

    cp_stages = []
    for linear, activation in stages:
        if activation is None:
            cp_stages.append((linear, None))
        else:
            cp_stages.append(
                (
                    linear,
                    activation_spec_from_name(
                        activation,
                        allow_nonpolynomial=True,
                        hermite_degree_cap=relu_degree_cap,
                    ),
                )
            )
    config = OrdinaryCPConfig(
        k_max=2,
        delta=requested_rank / used_sample_count,
        allow_nonpolynomial=True,
        hermite_degree_cap=relu_degree_cap,
        variance_min=variance_min,
        reference_two_stage_nonlinear=False,
        record_factor_statistics=False,
    )
    cp_result = propagate_linear_activation_stages(
        cp_stages,
        budgeted_samples,
        config=config,
        seed=cp_seed,
        return_all=return_all,
    )
    diagnostics = CovPropDiagnostics(
        sample_count=used_sample_count,
        input_width=width,
        rank=requested_rank,
        mode="cp",
        used_sample_count=used_sample_count,
        available_sample_count=available_sample_count,
        sample_budget=requested_rank,
        total_analytic_flops=cp_result.diagnostics.total_analytic_flops,
        analytic_flops_by_stage=dict(cp_result.diagnostics.analytic_flops_by_stage),
        notes=[
            "M < n: used order-2 ordinary-CP propagation as the low-rank covariance path.",
            "ReLU is represented by the configured Hermite degree cap.",
            "CP initialization consumes only min(M, available samples).",
        ],
    )
    covariance = None
    if cp_result.final_tower is not None and 2 in cp_result.final_tower:
        covariance = cp_result.final_tower[2].debug_to_dense(
            max_elements=max(width * width, 1)
        )
    return CovPropResult(
        mean=cp_result.mean,
        covariance=covariance,
        layer_means=None,
        layer_covariances=None,
        diagnostics=diagnostics,
    )
