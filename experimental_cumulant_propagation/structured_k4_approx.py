from __future__ import annotations

from dataclasses import dataclass

from experimental_cumulant_propagation.structured_cp import (
    CPTerm,
    CompressionReport,
    EdgeCompressionConfig,
    HardHadamardEdge,
    compress_hard_edge,
    cp_contract_W,
)


@dataclass(frozen=True)
class StructuredK4ApproxReport:
    """Bookkeeping returned by the experimental approximate K4 path."""

    hard_edges_compressed: int
    compression_reports: tuple[CompressionReport, ...]
    total_cp_rank: int
    estimated_future_linear_cost: int

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(report.warning for report in self.compression_reports if report.warning is not None)


def contract_cp_terms_W(terms: tuple[CPTerm, ...], W) -> tuple[CPTerm, ...]:
    """Apply the exact CP linear step F(A1, ..., Ar) -> F(WA1, ..., WAr)."""

    return tuple(cp_contract_W(term, W) for term in terms)


def compress_hard_edges(
    hard_edges: tuple[HardHadamardEdge, ...],
    cfg: EdgeCompressionConfig | None = None,
    *,
    sample_budget_m: int | None = None,
) -> tuple[tuple[CPTerm, ...], StructuredK4ApproxReport]:
    """Compress all hard dense-edge/rank-endpoint products immediately."""

    cfg = cfg or EdgeCompressionConfig()
    compressed_terms = []
    reports = []
    for edge in hard_edges:
        term, report = compress_hard_edge(
            edge,
            cfg,
            num_hard_edges=len(hard_edges),
            sample_budget_m=sample_budget_m,
        )
        compressed_terms.append(term)
        reports.append(report)

    total_rank = sum(term.rank for term in compressed_terms)
    future_cost = sum(report.estimated_future_linear_cost for report in reports)
    return (
        tuple(compressed_terms),
        StructuredK4ApproxReport(
            hard_edges_compressed=len(hard_edges),
            compression_reports=tuple(reports),
            total_cp_rank=total_rank,
            estimated_future_linear_cost=future_cost,
        ),
    )


def structured_k4_approx_step(*args, **kwargs):
    """Placeholder for the full K=4 nonlinear recurrence driver.

    This mode is approximate. It should follow the existing K=4 cumulant
    recurrence, but every structured product must route through hard-edge
    detection and `compress_hard_edges` before a term crosses a layer boundary.
    Set `EdgeCompressionConfig(explicit_edge_rank=n)` when affordable to recover
    the exact hard-edge primitive.
    """

    del args, kwargs
    raise NotImplementedError(
        "The experimental CP4 primitives are implemented, but the full K=4 "
        "nonlinear recurrence has not yet been wired into this package."
    )
