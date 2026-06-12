from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Literal

import torch
from torch import Tensor


RankKind = Literal["dense", "rank_m", "mixed"]
EdgeMethod = Literal["svd", "eigh", "randomized_svd"]


@dataclass(frozen=True)
class CPTerm:
    """Ordered CP tensor term F(A1, ..., Ad) with factors shaped (n, J)."""

    degree: int
    factors: tuple[Tensor, ...]
    coef: Tensor | float = 1.0
    rank_kind: RankKind = "rank_m"
    symmetric: bool = False

    def __post_init__(self) -> None:
        if self.degree != len(self.factors):
            raise ValueError("degree must equal the number of CP factors")
        if self.degree < 1:
            raise ValueError("degree must be positive")
        if not self.factors:
            return
        n, rank = self.factors[0].shape
        for factor in self.factors:
            if factor.ndim != 2:
                raise ValueError("each CP factor must have shape (n, J)")
            if tuple(factor.shape) != (n, rank):
                raise ValueError("all CP factors must have the same shape")

    @property
    def n(self) -> int:
        return int(self.factors[0].shape[0])

    @property
    def rank(self) -> int:
        return int(self.factors[0].shape[1])


@dataclass(frozen=True)
class DenseSlice:
    pattern: tuple[int, ...]
    tensor: Tensor
    rank_kind: Literal["dense"] = "dense"


@dataclass(frozen=True)
class EdgeLowRank:
    P: Tensor
    S: Tensor
    Q: Tensor
    residual_fro: float
    residual_op: float | None

    @property
    def rank(self) -> int:
        return int(self.S.numel())


@dataclass(frozen=True)
class EdgeCompressionConfig:
    method: EdgeMethod = "svd"
    edge_rank_multiplier: float = 1.0
    min_edge_rank: int = 1
    max_edge_rank: int | None = None
    total_hard_edge_rank_budget: int | None = None
    allow_exact_when_affordable: bool = True
    residual_warning_tol: float | None = None
    explicit_edge_rank: int | None = None
    cost_budget_multiplier: float = 4.0


@dataclass(frozen=True)
class CompressionReport:
    n: int
    input_rank: int
    edge_rank: int
    output_rank: int
    residual_fro: float
    residual_op: float | None
    method: EdgeMethod
    estimated_future_linear_cost: int
    budget: int | None
    warning: str | None = None


@dataclass(frozen=True)
class HardHadamardEdge:
    """Temporary E_ab U_at V_bt ... term before low-rank edge projection."""

    edge: Tensor
    left_factor: Tensor
    right_factor: Tensor
    trailing_factors: tuple[Tensor, ...] = ()
    coef: Tensor | float = 1.0
    output_degree: int = 2

    def __post_init__(self) -> None:
        if self.edge.ndim != 2 or self.edge.shape[0] != self.edge.shape[1]:
            raise ValueError("edge must have shape (n, n)")
        n = int(self.edge.shape[0])
        if tuple(self.left_factor.shape) != tuple(self.right_factor.shape):
            raise ValueError("left and right factors must have matching shape")
        if self.left_factor.ndim != 2 or self.left_factor.shape[0] != n:
            raise ValueError("endpoint factors must have shape (n, J)")
        for factor in self.trailing_factors:
            if tuple(factor.shape) != tuple(self.left_factor.shape):
                raise ValueError("trailing factors must have shape (n, J)")
        if self.output_degree != 2 + len(self.trailing_factors):
            raise ValueError("output_degree must match endpoint plus trailing factors")

    @property
    def n(self) -> int:
        return int(self.edge.shape[0])

    @property
    def rank(self) -> int:
        return int(self.left_factor.shape[1])


def choose_edge_rank(
    n: int,
    current_rank_J: int,
    num_hard_edges: int = 1,
    cfg: EdgeCompressionConfig | None = None,
) -> int:
    """Rank policy from the spec, with optional explicit override."""

    cfg = cfg or EdgeCompressionConfig()
    if cfg.explicit_edge_rank is not None:
        return min(max(1, int(cfg.explicit_edge_rank)), int(n))

    base = int(cfg.edge_rank_multiplier * (1.0 + (int(n) * int(n)) / max(int(current_rank_J), 1)))
    rank = max(int(cfg.min_edge_rank), base)
    if cfg.max_edge_rank is not None:
        rank = min(rank, int(cfg.max_edge_rank))
    if cfg.total_hard_edge_rank_budget is not None:
        per_edge = int(cfg.total_hard_edge_rank_budget) // max(int(num_hard_edges), 1)
        rank = min(rank, max(1, per_edge))
    if cfg.allow_exact_when_affordable and int(current_rank_J) <= int(n):
        rank = int(n)
    return min(rank, int(n))


def low_rank_edge(edge: Tensor, rank: int, method: EdgeMethod = "svd") -> EdgeLowRank:
    """Return E ~= P diag(S) Q.T and residual diagnostics."""

    n = int(edge.shape[0])
    rank = min(max(1, int(rank)), n)

    if method == "eigh":
        if not torch.allclose(edge, edge.T, rtol=1e-5, atol=1e-7):
            raise ValueError("eigh edge compression requires a symmetric edge")
        values, vectors = torch.linalg.eigh(edge)
        order = torch.argsort(torch.abs(values), descending=True)
        keep = order[:rank]
        P = vectors[:, keep]
        S = values[keep]
        Q = vectors[:, keep]
    elif method in {"svd", "randomized_svd"}:
        # The randomized_svd API is reserved for later; exact SVD keeps tests
        # deterministic until a sketching implementation is selected.
        U, S_all, Vh = torch.linalg.svd(edge, full_matrices=False)
        P = U[:, :rank]
        S = S_all[:rank]
        Q = Vh[:rank, :].T
    else:
        raise ValueError(f"Unsupported edge compression method: {method}")

    if rank < n:
        if method == "eigh":
            dropped = values[order[rank:]]
            residual_fro = float(torch.linalg.vector_norm(dropped).item())
            residual_op = float(torch.max(torch.abs(dropped)).item()) if dropped.numel() else 0.0
        else:
            dropped = S_all[rank:]
            residual_fro = float(torch.linalg.vector_norm(dropped).item())
            residual_op = float(dropped[0].item()) if dropped.numel() else 0.0
    else:
        residual_fro = 0.0
        residual_op = 0.0

    return EdgeLowRank(P=P, S=S, Q=Q, residual_fro=residual_fro, residual_op=residual_op)


def compress_hard_edge(
    term: HardHadamardEdge,
    cfg: EdgeCompressionConfig | None = None,
    *,
    num_hard_edges: int = 1,
    sample_budget_m: int | None = None,
) -> tuple[CPTerm, CompressionReport]:
    """Approximate E_ab U_at V_bt ... by CP after low-rank compression of E."""

    cfg = cfg or EdgeCompressionConfig()
    n = term.n
    J = term.rank
    R = choose_edge_rank(n=n, current_rank_J=J, num_hard_edges=num_hard_edges, cfg=cfg)
    edge_lr = low_rank_edge(term.edge, R, method=cfg.method)

    root = torch.sqrt(torch.abs(edge_lr.S))
    sign_root = torch.sign(edge_lr.S) * root
    U = term.left_factor
    V = term.right_factor

    A = (edge_lr.P[:, :, None] * U[:, None, :] * root[None, :, None]).reshape(n, R * J)
    B = (edge_lr.Q[:, :, None] * V[:, None, :] * sign_root[None, :, None]).reshape(n, R * J)
    new_factors: list[Tensor] = [A, B]
    for trailing in term.trailing_factors:
        new_factors.append(trailing[:, None, :].expand(n, R, J).reshape(n, R * J))

    warning = None
    if cfg.residual_warning_tol is not None and edge_lr.residual_fro > cfg.residual_warning_tol:
        warning = (
            f"edge residual Frobenius norm {edge_lr.residual_fro:.3e} exceeds "
            f"{cfg.residual_warning_tol:.3e}"
        )

    future_cost = int((R * J) * n * n * term.output_degree)
    budget = None
    if sample_budget_m is not None:
        budget = int(cfg.cost_budget_multiplier * (int(sample_budget_m) * n * n + n**4))
        if future_cost > budget:
            suffix = f"future linear cost {future_cost} exceeds budget {budget}"
            warning = suffix if warning is None else f"{warning}; {suffix}"

    return (
        CPTerm(
            degree=term.output_degree,
            factors=tuple(new_factors),
            coef=term.coef,
            rank_kind="mixed",
            symmetric=False,
        ),
        CompressionReport(
            n=n,
            input_rank=J,
            edge_rank=R,
            output_rank=R * J,
            residual_fro=edge_lr.residual_fro,
            residual_op=edge_lr.residual_op,
            method=cfg.method,
            estimated_future_linear_cost=future_cost,
            budget=budget,
            warning=warning,
        ),
    )


def cp_contract_W(term: CPTerm, W: Tensor) -> CPTerm:
    """Exact dense linear contraction of an ordered CP term."""

    return CPTerm(
        degree=term.degree,
        factors=tuple(W @ factor for factor in term.factors),
        coef=term.coef,
        rank_kind=term.rank_kind,
        symmetric=term.symmetric,
    )


def _ordered_slice(term: CPTerm, pattern: tuple[int, ...]) -> CPTerm | DenseSlice:
    cursor = 0
    new_factors: list[Tensor] = []
    for block in pattern:
        factors = term.factors[cursor : cursor + block]
        cursor += block
        merged = factors[0]
        for factor in factors[1:]:
            merged = merged * factor
        new_factors.append(merged)
    if len(new_factors) == 1:
        vector = torch.sum(new_factors[0], dim=1) * term.coef
        return DenseSlice(pattern=pattern, tensor=vector)
    return CPTerm(
        degree=len(new_factors),
        factors=tuple(new_factors),
        coef=term.coef,
        rank_kind=term.rank_kind,
        symmetric=False,
    )


def cp_slice_D(term: CPTerm, pattern: tuple[int, ...]) -> list[CPTerm | DenseSlice]:
    """Structured repeated-index slice of a CP term."""

    if sum(pattern) != term.degree:
        raise ValueError("slice pattern must sum to term degree")
    if not term.symmetric:
        return [_ordered_slice(term, pattern)]

    outputs: list[CPTerm | DenseSlice] = []
    orders = list(permutations(range(term.degree)))
    coef = term.coef / float(len(orders))
    for order in orders:
        ordered = CPTerm(
            degree=term.degree,
            factors=tuple(term.factors[index] for index in order),
            coef=coef,
            rank_kind=term.rank_kind,
            symmetric=False,
        )
        outputs.append(_ordered_slice(ordered, pattern))
    return outputs


def cp_to_dense(term: CPTerm) -> Tensor:
    """Materialize a small CP term for tests. Do not use on large rank-m terms."""

    letters = "abcdefghijklmnopqrstuvwxyz"
    if term.degree > len(letters):
        raise ValueError("degree is too large for the dense test helper")
    if not term.symmetric:
        factor_terms = [f"{letters[i]}r" for i in range(term.degree)]
        expr = ",".join(factor_terms) + "->" + letters[: term.degree]
        return torch.einsum(expr, *term.factors) * term.coef

    orders = list(permutations(range(term.degree)))
    out = None
    for order in orders:
        ordered = CPTerm(
            degree=term.degree,
            factors=tuple(term.factors[index] for index in order),
            coef=term.coef / float(len(orders)),
            rank_kind=term.rank_kind,
            symmetric=False,
        )
        dense = cp_to_dense(ordered)
        out = dense if out is None else out + dense
    assert out is not None
    return out


def dense_hard_edge(term: HardHadamardEdge) -> Tensor:
    """Materialize the exact hard-edge tensor for tests."""

    factors = [term.left_factor, term.right_factor, *term.trailing_factors]
    cp = CPTerm(term.output_degree, tuple(factors), coef=1.0)
    dense = cp_to_dense(cp)
    edge_shape = (term.n, term.n) + (1,) * len(term.trailing_factors)
    return dense * term.edge.reshape(edge_shape) * term.coef


def batch_diagonal_quadratic_exact(
    L: Tensor,
    R: Tensor,
    E: Tensor,
    U: Tensor,
    V: Tensor,
) -> Tensor:
    """Exact Q[x,t] = sum_ab L[x,a] R[x,b] E[a,b] U[a,t] V[b,t]."""

    return torch.einsum("xa,xb,ab,at,bt->xt", L, R, E, U, V)


def batch_diagonal_quadratic_compressed(
    L: Tensor,
    R: Tensor,
    compressed: CPTerm,
    *,
    edge_rank: int,
) -> Tensor:
    """Evaluate the compressed hard-edge primitive after CP conversion."""

    if compressed.degree != 2:
        raise ValueError("batch diagonal quadratic expects a degree-2 compressed term")
    if compressed.rank % int(edge_rank) != 0:
        raise ValueError("compressed rank must be a multiple of edge_rank")
    A, B = compressed.factors
    original_rank = compressed.rank // int(edge_rank)
    values = torch.einsum("xa,xb,ar,br->xr", L, R, A, B)
    return values.reshape(L.shape[0], int(edge_rank), original_rank).sum(dim=1)
