"""Diagram catalog construction for ordinary-CP nonlinear propagation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction

from .combinatorics import (
    is_connected,
    vec_part_coef,
    vector_partition_weight,
)


@dataclass(frozen=True)
class DiagramSpec:
    output_order: int
    hermite_degrees: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]
    coefficient: Fraction
    block_orders: tuple[int, ...]
    owners_by_block: tuple[tuple[int, ...], ...]


def _owners(block: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(vertex for vertex, count in enumerate(block) for _ in range(count))


def _block_support_size(block: tuple[int, ...]) -> int:
    return sum(1 for count in block if count)


def _allowed_blocks(
    k_vec: tuple[int, ...], k_max: int
) -> tuple[tuple[int, ...], ...]:
    ranges = [range(k + 1) for k in k_vec]
    blocks: list[tuple[int, ...]] = []
    for block in itertools.product(*ranges):
        block_order = sum(block)
        if block_order == 0 or block_order > k_max:
            continue
        support_size = _block_support_size(block)
        if support_size > 2:
            continue
        # The PDF's cDia[<=2](k) removes local singleton and local pair blocks,
        # because the reference Gaussian already matches the preactivation mean
        # and marginal variance. Local higher cumulants remain eligible.
        if support_size == 1 and block_order <= 2:
            continue
        blocks.append(tuple(block))
    return tuple(sorted(blocks, key=lambda b: (sum(b), b)))


def _constrained_vector_partitions(
    k_vec: tuple[int, ...], k_max: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Generate only mixed block-order-truncated vector partitions."""
    if not any(k_vec):
        return ((),)
    blocks = _allowed_blocks(k_vec, k_max)

    def rec(
        remaining: tuple[int, ...], min_index: int
    ) -> list[tuple[tuple[int, ...], ...]]:
        if not any(remaining):
            return [()]
        out: list[tuple[tuple[int, ...], ...]] = []
        for idx in range(min_index, len(blocks)):
            block = blocks[idx]
            if all(block[j] <= remaining[j] for j in range(len(k_vec))):
                new_remaining = tuple(remaining[j] - block[j] for j in range(len(k_vec)))
                for suffix in rec(new_remaining, idx):
                    out.append((block, *suffix))
        return out

    return tuple(rec(k_vec, 0))


@lru_cache(maxsize=None)
def build_diagram_catalog(
    output_order: int, k_max: int, hermite_degree: int
) -> tuple[DiagramSpec, ...]:
    """Build deterministic retained vector-partition diagrams."""
    if output_order < 1:
        raise ValueError("output_order must be positive")
    if k_max < 1:
        raise ValueError("k_max must be positive")
    if hermite_degree < 0:
        raise ValueError("hermite_degree must be nonnegative")
    specs: list[DiagramSpec] = []
    for k_vec in itertools.product(range(hermite_degree + 1), repeat=output_order):
        for part in _constrained_vector_partitions(tuple(k_vec), k_max):
            if not is_connected(part, d=output_order):
                continue
            block_orders = tuple(sum(block) for block in part)
            fiber_count = vec_part_coef(part, divide_fac=False)
            denominator = math.prod(math.factorial(k) for k in k_vec)
            coefficient = Fraction(fiber_count, denominator)
            direct = vector_partition_weight(part)
            if coefficient != direct:
                raise AssertionError("vector-partition coefficient mismatch")
            specs.append(
                DiagramSpec(
                    output_order=output_order,
                    hermite_degrees=tuple(k_vec),
                    blocks=tuple(part),
                    coefficient=coefficient,
                    block_orders=block_orders,
                    owners_by_block=tuple(_owners(block) for block in part),
                )
            )
    specs.sort(key=lambda s: (s.hermite_degrees, len(s.blocks), s.blocks))
    return tuple(specs)
