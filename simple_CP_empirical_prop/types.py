"""CP tensor data structures."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, TypeAlias

import torch
from torch import Tensor


class FiniteCPSource(Protocol):
    @property
    def order(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def rank(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def gather_columns(self, indices: Tensor) -> tuple[Tensor, ...]: ...


@dataclass(frozen=True)
class CPTensor:
    """A rank-M CP tensor normalized by an average over columns.

    For factors (A_1, ..., A_r), every factor has shape [width, rank] and

        T[i_1, ..., i_r] = mean_t prod_u A_u[i_u, t].
    """

    factors: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("A CPTensor must have positive order")
        first = self.factors[0]
        if first.ndim != 2:
            raise ValueError("Every factor must have shape [width, rank]")
        width, rank = first.shape
        if rank < 1:
            raise ValueError("CP rank must be positive")
        for mode, factor in enumerate(self.factors):
            if factor.ndim != 2:
                raise ValueError(f"factor {mode} is not a matrix")
            if tuple(factor.shape) != (width, rank):
                raise ValueError("All factors must have identical [width, rank]")
            if factor.device != first.device:
                raise ValueError("All factors must be on the same device")
            if factor.dtype != first.dtype:
                raise ValueError("All factors must have the same dtype")

    @property
    def order(self) -> int:
        return len(self.factors)

    @property
    def width(self) -> int:
        return self.factors[0].shape[0]

    @property
    def rank(self) -> int:
        return self.factors[0].shape[1]

    @property
    def device(self) -> torch.device:
        return self.factors[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.factors[0].dtype

    def gather_columns(self, indices: Tensor) -> tuple[Tensor, ...]:
        idx = indices.to(self.device)
        return tuple(factor.index_select(1, idx) for factor in self.factors)

    def entry(self, indices: tuple[int, ...]) -> Tensor:
        if len(indices) != self.order:
            raise ValueError("number of indices must equal tensor order")
        values = torch.ones(self.rank, device=self.device, dtype=self.dtype)
        for factor, index in zip(self.factors, indices):
            values = values * factor[index, :]
        return values.mean()

    def entries(self, indices: Tensor) -> Tensor:
        if indices.ndim != 2 or indices.shape[1] != self.order:
            raise ValueError("indices must have shape [num_entries, order]")
        idx = indices.to(self.device)
        outs = []
        for row in idx:
            values = torch.ones(self.rank, device=self.device, dtype=self.dtype)
            for mode, factor in enumerate(self.factors):
                values = values * factor[row[mode], :]
            outs.append(values.mean())
        return torch.stack(outs)

    def mean_vector(self) -> Tensor:
        if self.order != 1:
            raise ValueError("mean_vector is only defined for order-one CP tensors")
        return self.factors[0].mean(dim=1)

    def diagonal_order2(self) -> Tensor:
        if self.order != 2:
            raise ValueError("diagonal_order2 is only defined for order-two CP tensors")
        return (self.factors[0] * self.factors[1]).mean(dim=1)

    def permute_modes(self, permutation: tuple[int, ...]) -> "CPTensor":
        if sorted(permutation) != list(range(self.order)):
            raise ValueError("permutation must contain every mode exactly once")
        return CPTensor(tuple(self.factors[i] for i in permutation))

    def to(self, *args: object, **kwargs: object) -> "CPTensor":
        return CPTensor(tuple(factor.to(*args, **kwargs) for factor in self.factors))

    def clone(self) -> "CPTensor":
        return CPTensor(tuple(factor.clone() for factor in self.factors))

    def validate(
        self,
        *,
        expected_order: int | None = None,
        expected_rank: int | None = None,
        expected_width: int | None = None,
    ) -> None:
        if expected_order is not None and self.order != expected_order:
            raise ValueError(f"expected order {expected_order}, got {self.order}")
        if expected_rank is not None and self.rank != expected_rank:
            raise ValueError(f"expected rank {expected_rank}, got {self.rank}")
        if expected_width is not None and self.width != expected_width:
            raise ValueError(f"expected width {expected_width}, got {self.width}")

    def debug_to_dense(self, max_elements: int = 100_000) -> Tensor:
        elements = self.width ** self.order
        if elements > max_elements:
            raise ValueError(
                f"refusing to allocate dense tensor with {elements} elements"
            )
        out = torch.zeros(
            (self.width,) * self.order, device=self.device, dtype=self.dtype
        )
        for column in range(self.rank):
            term = self.factors[0][:, column]
            for factor in self.factors[1:]:
                shape = (1,) * term.ndim + (self.width,)
                term = term.unsqueeze(-1) * factor[:, column].reshape(shape)
            out = out + term
        return out / self.rank


CPTower: TypeAlias = dict[int, CPTensor]


@dataclass(frozen=True)
class WeightedCPSource:
    """A scalar multiple of a finite CP source."""

    coefficient: float | Fraction
    source: FiniteCPSource


def zero_cp(
    *,
    width: int,
    order: int,
    rank: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> CPTensor:
    if order < 1:
        raise ValueError("order must be positive")
    first = torch.zeros(width, rank, device=device, dtype=dtype)
    rest = tuple(torch.ones(width, rank, device=device, dtype=dtype) for _ in range(order - 1))
    return CPTensor((first, *rest))


def validate_tower(
    tower: CPTower,
    *,
    k_max: int,
    rank: int | None = None,
    width: int | None = None,
    partial: bool = False,
) -> None:
    expected = set(range(1, k_max + 1))
    keys = set(tower)
    if partial:
        if not keys <= expected:
            raise ValueError("partial tower has keys outside 1..k_max")
    elif keys != expected:
        raise ValueError(f"tower keys must be exactly {sorted(expected)}")
    widths = set()
    for order, cp in tower.items():
        cp.validate(expected_order=order, expected_rank=rank, expected_width=width)
        widths.add(cp.width)
    if len(widths) > 1:
        raise ValueError("all tower tensors must have the same width")
