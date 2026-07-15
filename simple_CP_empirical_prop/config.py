"""Configuration and result dataclasses for ordinary-CP propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from numpy.polynomial import Polynomial


def rank_from_delta(m: int, delta: float) -> int:
    """Return M = max(1, ceil(delta * m))."""
    if not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    return max(1, math.ceil(delta * m))


@dataclass(frozen=True)
class OrdinaryCPConfig:
    """Options for the self-contained ordinary-CP implementation."""

    k_max: int
    delta: float
    polynomial_by_layer: tuple[Polynomial, ...] | None = None
    allow_nonpolynomial: bool = False
    hermite_degree_cap: int | None = None
    quadrature_degree: int = 100
    quadrature_float64: bool = True
    mean_abs_clip: float | None = None
    variance_min: float = 1e-10
    variance_max: float | None = None
    shuffle_initial_groups: bool = True
    reference_two_stage_nonlinear: bool = False
    structural_rng_on_cpu: bool = True
    validate_every_operation: bool = False
    record_intermediate_towers: bool = False
    record_factor_statistics: bool = True
    dense_debug_max_elements: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.k_max, int) or self.k_max < 1:
            raise ValueError("k_max must be a positive integer")
        if not math.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("delta must be positive and finite")
        if self.variance_min <= 0:
            raise ValueError("variance_min must be positive")
        if self.variance_max is not None and self.variance_max <= self.variance_min:
            raise ValueError("variance_max must exceed variance_min")
        if self.allow_nonpolynomial:
            if self.hermite_degree_cap is None or self.hermite_degree_cap <= 0:
                raise ValueError(
                    "non-polynomial mode requires a positive hermite_degree_cap"
                )
        if self.quadrature_degree <= 0:
            raise ValueError("quadrature_degree must be positive")


@dataclass(frozen=True)
class MeanVariance:
    raw_mean: torch.Tensor
    raw_variance: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    sigma: torch.Tensor
    mean_clip_fraction: float
    variance_clip_fraction: float


@dataclass
class OrdinaryCPDiagnostics:
    seed: int
    sample_count: int
    rank: int
    k_max: int
    delta: float
    initialization_sources_by_order: dict[int, int] = field(default_factory=dict)
    unused_samples_by_order: dict[int, int] = field(default_factory=dict)
    diagrams_by_layer_and_order: dict[tuple[int, int], int] = field(default_factory=dict)
    mean_clip_fraction_by_layer: list[float] = field(default_factory=list)
    variance_clip_fraction_by_layer: list[float] = field(default_factory=list)
    raw_variance_min_by_layer: list[float] = field(default_factory=list)
    raw_variance_max_by_layer: list[float] = field(default_factory=list)
    factor_abs_max_by_stage: dict[str, float] = field(default_factory=dict)
    factor_rms_by_stage: dict[str, float] = field(default_factory=dict)
    total_analytic_flops: float = 0.0
    analytic_flops_by_stage: dict[str, float] = field(default_factory=dict)
    flop_count_convention: str = (
        "Counts floating-point adds/muls/divs in CP arithmetic. Excludes RNG, "
        "indexing, allocation, memory movement, and Python overhead. Fused "
        "nonlinear counts are expected FLOPs over diagram sampling."
    )
    notes: list[str] = field(default_factory=list)

    def add_flops(self, label: str, flops: float) -> None:
        value = float(flops)
        self.analytic_flops_by_stage[label] = (
            self.analytic_flops_by_stage.get(label, 0.0) + value
        )
        self.total_analytic_flops += value


@dataclass
class OrdinaryCPResult:
    mean: torch.Tensor
    final_tower: dict[int, Any]
    pre_towers: list[dict[int, Any]] | None
    act_towers: list[dict[int, Any]] | None
    diagnostics: OrdinaryCPDiagnostics
