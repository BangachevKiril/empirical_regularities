from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cumulant_propagation._arc_mlp_kprop.diagslice import DSTensor, zero_repeated
from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from experimental_cumulant_propagation.structured_cp import CPTerm, cp_contract_W, cp_to_dense


def _as_wick_tuple(wick: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(wick, tuple):
        if len(wick) != 4:
            raise ValueError("degree-4 wick tuple must contain four vectors")
        return wick
    return (wick, wick, wick, wick)


def _all_same_object(factors: tuple[Tensor, ...]) -> bool:
    first = factors[0]
    return all(factor is first for factor in factors[1:])


def _all_close_factors(factors: tuple[Tensor, ...]) -> bool:
    first = factors[0]
    return all(torch.equal(factor, first) for factor in factors[1:])


def _cp4_slice_dense(term: CPTerm, part: tuple[int, ...]) -> Tensor:
    """Materialize repeated slices of a CP4 term without forming an n^4 tensor."""

    if term.degree != 4:
        raise ValueError("StructuredTensor4 only supports degree-4 CP terms")
    if sum(part) != 4:
        raise ValueError("degree-4 slice pattern must sum to 4")

    A, B, C, D = term.factors
    coef = term.coef

    # The unknown-A direct empirical fourth moment has identical ordered modes,
    # and linear/ReLU-wick steps preserve this. The closed forms below are exact
    # and avoid the 24-permutation expansion on the hot path.
    if term.symmetric and (_all_same_object(term.factors) or _all_close_factors(term.factors)):
        X = A
        if part == (4,):
            return zero_repeated(torch.einsum("ir,ir,ir,ir->i", X, X, X, X) * coef)
        if part == (3, 1):
            return zero_repeated(torch.einsum("ir,ir,ir,jr->ij", X, X, X, X) * coef)
        if part == (2, 2):
            return zero_repeated(torch.einsum("ir,ir,jr,jr->ij", X, X, X, X) * coef)
        if part == (2, 1, 1):
            return zero_repeated(torch.einsum("ir,ir,jr,kr->ijk", X, X, X, X) * coef)

    dense = cp_to_dense(term)
    if part == (4,):
        return zero_repeated(torch.einsum("iiii->i", dense))
    if part == (3, 1):
        return zero_repeated(torch.einsum("iiij->ij", dense))
    if part == (2, 2):
        return zero_repeated(torch.einsum("iijj->ij", dense))
    if part == (2, 1, 1):
        return zero_repeated(torch.einsum("iijk->ijk", dense))
    if part == (1, 1, 1, 1):
        return zero_repeated(dense)
    raise ValueError(f"Unsupported degree-4 slice pattern: {part}")


@dataclass
class StructuredTensor4:
    """Degree-4 tensor with CP4 rank-m terms plus exact Pair4 corrections.

    This is an experimental bridge into the existing K=4 recurrence. The direct
    empirical fourth moment stays as ordered CP4 factors `(n, m)`, so dense
    linear layers cost `O(m n^2)` for that term rather than materializing Pair4
    factors `(n, n, m)`.

    Repeated slices are materialized only when the current exact recurrence asks
    for them. That makes this class a runnable integration point, while the
    newer hard-edge product builder can replace those dense slice calls later.
    """

    n: int
    cp_terms: tuple[CPTerm, ...] = ()
    pair_tensor: FactoredTensor4 | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    d: int = 4

    def __post_init__(self) -> None:
        if self.device is None:
            if self.cp_terms:
                self.device = self.cp_terms[0].factors[0].device
            elif self.pair_tensor is not None:
                self.device = self.pair_tensor.device
        if self.dtype is None:
            if self.cp_terms:
                self.dtype = self.cp_terms[0].factors[0].dtype
            elif self.pair_tensor is not None:
                self.dtype = self.pair_tensor.dtype
        if self.device is None or self.dtype is None:
            raise ValueError("device and dtype are required for an empty StructuredTensor4")
        for term in self.cp_terms:
            if term.degree != 4:
                raise ValueError("StructuredTensor4 only accepts degree-4 CP terms")
            if term.n != self.n:
                raise ValueError(f"CP term width {term.n} does not match n={self.n}")
        if self.pair_tensor is None:
            self.pair_tensor = FactoredTensor4(
                self.n,
                device=self.device,
                dtype=self.dtype,
                assume_symmetric=True,
            )
        elif self.pair_tensor.n != self.n:
            raise ValueError(f"Pair4 width {self.pair_tensor.n} does not match n={self.n}")
        self.repeated = DSTensor(d=4, n=self.n, slices=dict(), device=self.device, dtype=self.dtype)

    @property
    def ndim(self) -> int:
        return 4

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.n, self.n, self.n, self.n)

    @property
    def rank(self) -> int:
        return sum(term.rank for term in self.cp_terms) + int(self.pair_tensor.A.shape[2])

    @property
    def cp_rank(self) -> int:
        return sum(term.rank for term in self.cp_terms)

    @property
    def pair_rank(self) -> int:
        return int(self.pair_tensor.A.shape[2])

    def clone(self) -> "StructuredTensor4":
        pair_factors = self.pair_tensor.factors
        pair_clone = FactoredTensor4(
            self.n,
            factors=pair_factors,
            device=pair_factors[0].device,
            dtype=pair_factors[0].dtype,
            assume_symmetric=True,
        )
        return StructuredTensor4(
            n=self.n,
            cp_terms=tuple(
                CPTerm(
                    degree=term.degree,
                    factors=tuple(factor.clone() for factor in term.factors),
                    coef=term.coef.clone() if isinstance(term.coef, Tensor) else term.coef,
                    rank_kind=term.rank_kind,
                    symmetric=term.symmetric,
                )
                for term in self.cp_terms
            ),
            pair_tensor=pair_clone,
            device=pair_factors[0].device,
            dtype=self.dtype,
        )

    def clear_repeated(self) -> None:
        self.repeated = DSTensor(d=4, n=self.n, slices=dict(), device=self.device, dtype=self.dtype)
        self.pair_tensor.clear_repeated()

    def contract_W(self, W: Tensor) -> "StructuredTensor4":
        return StructuredTensor4(
            n=self.n,
            cp_terms=tuple(cp_contract_W(term, W) for term in self.cp_terms),
            pair_tensor=self.pair_tensor.contract_W(W),
            device=W.device,
            dtype=self.dtype,
        )

    def contract_wick_(self, wick: Tensor | tuple[Tensor, ...]) -> None:
        wicks = _as_wick_tuple(wick)
        new_terms = []
        for term in self.cp_terms:
            factors = tuple(factor * wicks[index][:, None] for index, factor in enumerate(term.factors))
            new_terms.append(
                CPTerm(
                    degree=4,
                    factors=factors,
                    coef=term.coef,
                    rank_kind=term.rank_kind,
                    symmetric=term.symmetric and all(torch.equal(wicks[0], w) for w in wicks[1:]),
                )
            )
        self.cp_terms = tuple(new_terms)
        if all(torch.equal(wicks[0], w) for w in wicks[1:]):
            self.pair_tensor.contract_wick_(wicks[0])
        else:
            raise NotImplementedError("mode-specific Pair4 wick scaling is not implemented")
        self.clear_repeated()

    def contract_wick(self, wick: Tensor | tuple[Tensor, ...]) -> "StructuredTensor4":
        new = self.clone()
        new.contract_wick_(wick)
        return new

    def add_factors_(self, factors: tuple[Tensor, Tensor]) -> None:
        self.pair_tensor.add_factors_(factors)
        self.clear_repeated()

    def add_factors(self, factors: tuple[Tensor, Tensor]) -> "StructuredTensor4":
        new = self.clone()
        new.add_factors_(factors)
        return new

    def __add__(self, other) -> "StructuredTensor4":
        new = self.clone()
        if isinstance(other, StructuredTensor4):
            new.cp_terms = new.cp_terms + other.cp_terms
            new.pair_tensor = new.pair_tensor + other.pair_tensor
        elif isinstance(other, FactoredTensor4):
            new.pair_tensor = new.pair_tensor + other
        else:
            return NotImplemented
        new.clear_repeated()
        return new

    def __radd__(self, other) -> "StructuredTensor4":
        return self.__add__(other)

    def to_tensor(self) -> Tensor:
        out = self.pair_tensor.to_tensor()
        for term in self.cp_terms:
            out = out + cp_to_dense(term)
        return out

    def get_dslice(self, part: tuple[int, ...]) -> Tensor:
        sorted_part = tuple(sorted(part, reverse=True))
        if sorted_part not in self.repeated.slices:
            value = self.pair_tensor.get_dslice(sorted_part).clone()
            for term in self.cp_terms:
                value = value + _cp4_slice_dense(term, sorted_part)
            self.repeated.slices[sorted_part] = value
        return self.repeated.get_slice(part)

    def get_repeated(self) -> DSTensor:
        slices = {}
        for part in ((4,), (3, 1), (2, 2), (2, 1, 1)):
            slices[part] = self.get_dslice(part)
        return DSTensor(d=4, n=self.n, slices=slices, device=self.device, dtype=self.dtype)
