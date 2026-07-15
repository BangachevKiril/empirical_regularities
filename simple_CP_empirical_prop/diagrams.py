"""Diagram catalog construction for ordinary-CP nonlinear propagation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction

from .combinatorics import (
    is_connected,
    is_mixed,
    vec_part_coef,
    vector_partition_weight,
    vector_partitions,
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
        for part in vector_partitions(tuple(k_vec)):
            if not is_connected(part, d=output_order):
                continue
            if not is_mixed(part, m=2):
                continue
            block_orders = tuple(sum(block) for block in part)
            if any(order > k_max for order in block_orders):
                continue
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
