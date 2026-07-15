"""Coordinatewise mean/variance propagation baseline.

This is a deliberately small K=1-style baseline. It does not store higher
ordinary cumulants or cross-covariances. The state is only

    mean[i] = E[X_i], variance[i] = Var[X_i].

Linear layers propagate the mean exactly and propagate only marginal variances
under a diagonal-covariance approximation:

    mean_out = W @ mean + b
    variance_out = (W ** 2) @ variance.

For ReLU, square, and cube activations, the activation step assumes each
coordinate is Gaussian with the tracked mean and variance and computes the
resulting coordinatewise mean and variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor


MEAN_PROP_FLOP_CONVENTION = (
    "Counts floating-point adds/muls/divs/sqrt in empirical mean/variance, "
    "linear mean/diagonal-variance propagation, and coordinatewise activation "
    "moment formulas. Excludes RNG, indexing, allocation, memory movement, "
    "Python overhead, and transcendental exp/erf implementation cost."
)


@dataclass
class MeanPropDiagnostics:
    sample_count: int
    input_width: int
    total_analytic_flops: float = 0.0
    analytic_flops_by_stage: dict[str, float] = field(default_factory=dict)
    flop_count_convention: str = MEAN_PROP_FLOP_CONVENTION
    notes: list[str] = field(default_factory=list)

    def add_flops(self, label: str, flops: float) -> None:
        value = float(flops)
        self.analytic_flops_by_stage[label] = (
            self.analytic_flops_by_stage.get(label, 0.0) + value
        )
        self.total_analytic_flops += value


@dataclass
class MeanPropState:
    mean: Tensor
    variance: Tensor


@dataclass
class MeanPropResult:
    mean: Tensor
    variance: Tensor
    layer_means: list[Tensor] | None
    layer_variances: list[Tensor] | None
    diagnostics: MeanPropDiagnostics


def empirical_mean_variance(samples: Tensor) -> MeanPropState:
    """Return coordinatewise empirical mean and population variance.

    The variance convention is ``mean((x - mean) ** 2)``, matching a second
    cumulant/marginal variance with denominator ``m`` rather than ``m - 1``.
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, width]")
    mean = samples.mean(dim=0)
    variance = (samples - mean).square().mean(dim=0)
    return MeanPropState(mean=mean, variance=variance)


def empirical_mean_variance_flops(*, sample_count: int, width: int) -> float:
    """FLOPs for the implemented two-pass empirical mean/variance computation."""
    # Mean: (m - 1) adds + one division per coordinate ~= m.
    # Variance: m subtractions, m squares, (m - 1) adds, one division ~= 3m.
    return float(4 * sample_count * width)


def mean_prop_linear(
    state: MeanPropState,
    linear_or_weight: torch.nn.Linear | Tensor,
    bias: Tensor | None = None,
) -> MeanPropState:
    """Propagate coordinatewise mean/variance through a deterministic linear map."""
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
    mean = weight @ state.mean
    if layer_bias is not None:
        mean = mean + layer_bias
    variance = weight.square() @ state.variance
    return MeanPropState(mean=mean, variance=variance)


def mean_prop_linear_flops(
    *,
    input_width: int,
    output_width: int,
    has_bias: bool,
) -> float:
    """Analytic FLOPs for ``mean_prop_linear``."""
    mean_matvec = output_width * (2 * input_width - 1)
    square_weight = output_width * input_width
    var_matvec = output_width * (2 * input_width - 1)
    bias_add = output_width if has_bias else 0
    return float(mean_matvec + square_weight + var_matvec + bias_add)


def _normal_pdf_cdf(alpha: Tensor) -> tuple[Tensor, Tensor]:
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    pdf = torch.exp(-0.5 * alpha.square()) * inv_sqrt_2pi
    cdf = 0.5 * (1.0 + torch.erf(alpha * inv_sqrt_2))
    return pdf, cdf


def mean_prop_activation(
    state: MeanPropState,
    activation: str,
    *,
    variance_min: float = 1e-10,
) -> MeanPropState:
    """Apply a coordinatewise activation under a Gaussian marginal assumption."""
    if variance_min <= 0:
        raise ValueError("variance_min must be positive")
    name = activation.lower()
    mean = state.mean
    variance = state.variance.clamp_min(variance_min)
    if name == "identity":
        return MeanPropState(mean=mean, variance=variance)
    if name == "relu":
        sigma = variance.sqrt()
        alpha = mean / sigma
        pdf, cdf = _normal_pdf_cdf(alpha)
        out_mean = sigma * pdf + mean * cdf
        raw_second = (mean.square() + variance) * cdf + mean * sigma * pdf
        out_variance = (raw_second - out_mean.square()).clamp_min(variance_min)
        return MeanPropState(mean=out_mean, variance=out_variance)
    if name == "square":
        out_mean = mean.square() + variance
        raw_fourth = mean.pow(4) + 6 * mean.square() * variance + 3 * variance.square()
        out_variance = (raw_fourth - out_mean.square()).clamp_min(variance_min)
        return MeanPropState(mean=out_mean, variance=out_variance)
    if name == "cube":
        out_mean = mean.pow(3) + 3 * mean * variance
        raw_sixth = (
            mean.pow(6)
            + 15 * mean.pow(4) * variance
            + 45 * mean.square() * variance.square()
            + 15 * variance.pow(3)
        )
        out_variance = (raw_sixth - out_mean.square()).clamp_min(variance_min)
        return MeanPropState(mean=out_mean, variance=out_variance)
    raise NotImplementedError(f"unsupported mean_prop activation {activation!r}")


def mean_prop_activation_flops(*, width: int, activation: str) -> float:
    """Approximate scalar arithmetic count for activation moment formulas."""
    name = activation.lower()
    if name == "identity":
        return 0.0
    if name == "relu":
        # sqrt, divide, alpha^2, scalar multiply, pdf scale, cdf affine,
        # output mean, raw second, subtract square.
        return float(22 * width)
    if name == "square":
        return float(12 * width)
    if name == "cube":
        return float(24 * width)
    raise NotImplementedError(f"unsupported mean_prop activation {activation!r}")


def mean_prop_stages(
    stages: Sequence[tuple[torch.nn.Linear, str | None]],
    samples: Tensor,
    *,
    variance_min: float = 1e-10,
    return_all: bool = False,
) -> MeanPropResult:
    """Run coordinatewise mean/variance propagation over explicit stages."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, width]")
    diagnostics = MeanPropDiagnostics(
        sample_count=samples.shape[0],
        input_width=samples.shape[1],
        notes=[
            "Tracks only coordinatewise mean and variance.",
            "Linear variance propagation assumes diagonal covariance.",
            "Activation moments assume Gaussian marginals.",
        ],
    )
    state = empirical_mean_variance(samples)
    diagnostics.add_flops(
        "empirical_mean_variance",
        empirical_mean_variance_flops(
            sample_count=samples.shape[0],
            width=samples.shape[1],
        ),
    )
    layer_means: list[Tensor] | None = [state.mean] if return_all else None
    layer_variances: list[Tensor] | None = [state.variance] if return_all else None
    for layer_index, (linear, activation) in enumerate(stages):
        diagnostics.add_flops(
            f"layer_{layer_index}_linear",
            mean_prop_linear_flops(
                input_width=int(linear.weight.shape[1]),
                output_width=int(linear.weight.shape[0]),
                has_bias=linear.bias is not None,
            ),
        )
        state = mean_prop_linear(state, linear)
        if activation is not None:
            diagnostics.add_flops(
                f"layer_{layer_index}_activation_{activation}",
                mean_prop_activation_flops(width=state.mean.numel(), activation=activation),
            )
            state = mean_prop_activation(
                state,
                activation,
                variance_min=variance_min,
            )
        if layer_means is not None:
            layer_means.append(state.mean)
        if layer_variances is not None:
            layer_variances.append(state.variance)
    return MeanPropResult(
        mean=state.mean,
        variance=state.variance,
        layer_means=layer_means,
        layer_variances=layer_variances,
        diagnostics=diagnostics,
    )


def mean_prop_mlp(
    mlp: object,
    samples: Tensor,
    *,
    activation: str = "relu",
    variance_min: float = 1e-10,
    activate_final: bool = False,
    return_all: bool = False,
) -> MeanPropResult:
    """Run mean propagation for an object exposing ``Ws`` linear layers."""
    if not hasattr(mlp, "Ws"):
        raise TypeError("mean_prop_mlp expects an object exposing Ws")
    linears = list(getattr(mlp, "Ws"))
    if not linears:
        raise ValueError("mlp must contain at least one linear layer")
    stages = []
    for idx, linear in enumerate(linears):
        is_last = idx == len(linears) - 1
        stage_activation = activation if (activate_final or not is_last) else None
        stages.append((linear, stage_activation))
    return mean_prop_stages(
        stages,
        samples,
        variance_min=variance_min,
        return_all=return_all,
    )
