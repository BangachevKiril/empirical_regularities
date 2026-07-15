"""Analytic FLOP estimates for the ordinary-CP implementation.

The estimates count floating-point additions, multiplications, and divisions
performed by the CP tensor arithmetic in this package. They deliberately do not
count integer RNG/indexing work, memory movement, Python control flow, or tensor
allocation. For the fused nonlinear path, the count is the expectation over the
diagram choice distribution, so it can be fractional.
"""

from __future__ import annotations

from fractions import Fraction

from .diagrams import DiagramSpec, build_diagram_catalog


FLOP_CONVENTION = (
    "Counts floating-point adds/muls/divs in CP arithmetic. Excludes RNG, "
    "indexing, allocation, memory movement, and Python overhead. Fused "
    "nonlinear counts are expected FLOPs over diagram sampling."
)


def matmul_flops(output_width: int, input_width: int, columns: int) -> float:
    """FLOPs for [output_width, input_width] @ [input_width, columns]."""
    if min(output_width, input_width, columns) < 1:
        return 0.0
    return float(output_width * columns * (2 * input_width - 1))


def initialization_flops(*, k_max: int, width: int, rank: int) -> float:
    """Initialization arithmetic: one coefficient scaling per output column."""
    return float(k_max * width * rank)


def linear_layer_flops(
    *,
    k_max: int,
    input_width: int,
    output_width: int,
    rank: int,
    has_bias: bool,
) -> float:
    """FLOPs for exact linear tower propagation."""
    factor_count = k_max * (k_max + 1) // 2
    flops = factor_count * matmul_flops(output_width, input_width, rank)
    if has_bias:
        flops += output_width * rank
    return float(flops)


def moment_extraction_flops(*, width: int, rank: int) -> float:
    """FLOPs for mean and diagonal variance extraction before a nonlinearity."""
    # Mean: rank - 1 adds plus one division per coordinate.
    # Variance: rank multiplies, rank - 1 adds, plus one division per coordinate.
    return float(width * rank + 2 * width * rank)


def final_mean_flops(*, width: int, rank: int) -> float:
    """FLOPs for extracting the output mean from the final order-one CP tensor."""
    return float(width * rank)


def compression_flops(*, width: int, rank: int) -> float:
    """FLOPs for source coefficient scaling in CP compression."""
    return float(width * rank)


def _fraction_abs(value: Fraction) -> float:
    return float(abs(value.numerator)) / float(value.denominator)


def diagram_leg_count(spec: DiagramSpec) -> int:
    """Number of floating factor multiplications per coordinate/column."""
    return sum(spec.block_orders)


def nonlinear_order_flops(
    catalog: tuple[DiagramSpec, ...],
    *,
    width: int,
    rank: int,
    reference_two_stage: bool,
) -> tuple[float, float]:
    """Return (diagram_sampling_flops, final_compression_flops) for one order."""
    nonzero = tuple(spec for spec in catalog if spec.coefficient != 0)
    if not nonzero:
        return 0.0, 0.0
    if reference_two_stage:
        leg_work = sum(diagram_leg_count(spec) for spec in nonzero)
    else:
        weights = [_fraction_abs(spec.coefficient) for spec in nonzero]
        total = sum(weights)
        leg_work = sum(w * diagram_leg_count(spec) for w, spec in zip(weights, nonzero)) / total
    return float(width * rank * leg_work), compression_flops(width=width, rank=rank)


def nonlinear_layer_flops(
    *,
    k_max: int,
    width: int,
    rank: int,
    hermite_degree: int,
    reference_two_stage: bool,
) -> tuple[float, dict[int, float]]:
    """Return total nonlinear FLOPs and a per-output-order breakdown."""
    by_order: dict[int, float] = {}
    total = 0.0
    for order in range(1, k_max + 1):
        catalog = build_diagram_catalog(order, k_max, hermite_degree)
        diagram_flops, compress_flops = nonlinear_order_flops(
            catalog,
            width=width,
            rank=rank,
            reference_two_stage=reference_two_stage,
        )
        order_total = diagram_flops + compress_flops
        by_order[order] = order_total
        total += order_total
    return total, by_order
