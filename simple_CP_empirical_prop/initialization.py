"""Empirical grouped ordinary-cumulant initialization."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from torch import Tensor

from .combinatorics import falling_factorial, set_partitions
from .compression import compress_cp_sources
from .rng import RNGStreams
from .types import CPTensor, CPTower, WeightedCPSource, validate_tower


@dataclass(frozen=True)
class GroupedCumulantTermSource:
    """Lazy conceptual rank-G source sharing one grouped-sample tensor."""

    grouped_samples: Tensor
    slot_by_mode: tuple[int, ...]

    @property
    def order(self) -> int:
        return len(self.slot_by_mode)

    @property
    def rank(self) -> int:
        return self.grouped_samples.shape[0]

    @property
    def width(self) -> int:
        return self.grouped_samples.shape[2]

    @property
    def device(self) -> torch.device:
        return self.grouped_samples.device

    @property
    def dtype(self) -> torch.dtype:
        return self.grouped_samples.dtype

    def gather_columns(self, indices: Tensor) -> tuple[Tensor, ...]:
        idx = indices.to(self.grouped_samples.device)
        return tuple(
            self.grouped_samples.index_select(0, idx)[:, slot, :]
            .transpose(0, 1)
            .contiguous()
            for slot in self.slot_by_mode
        )


def initialization_sources(
    grouped_samples: Tensor, order: int
) -> list[WeightedCPSource]:
    """Build lazy weighted grouped cumulant sources for one order."""
    out: list[WeightedCPSource] = []
    for partition in set_partitions(order):
        blocks = tuple(sorted(partition, key=lambda b: (min(b), len(b), b)))
        q = len(blocks)
        mu = (-1) ** (q - 1) * math.factorial(q - 1)
        coefficient = Fraction(mu, falling_factorial(order, q))
        for slots in itertools.permutations(range(order), q):
            slot_by_mode = [0] * order
            for block, slot in zip(blocks, slots):
                for mode in block:
                    slot_by_mode[mode] = slot
            out.append(
                WeightedCPSource(
                    coefficient,
                    GroupedCumulantTermSource(grouped_samples, tuple(slot_by_mode)),
                )
            )
    return out


def initialize_cumulant_cp(
    samples: Tensor,
    *,
    order: int,
    rank: int,
    group_generator: torch.Generator,
    compression_generator: torch.Generator,
    shuffle: bool = True,
) -> CPTensor:
    """Initialize one full ordinary cumulant tensor from grouped samples."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, width]")
    if order < 1:
        raise ValueError("order must be positive")
    m, width = samples.shape
    if m < order:
        raise ValueError("m must be at least the requested cumulant order")
    group_count = m // order
    used = group_count * order
    if shuffle:
        perm_cpu = torch.randperm(m, generator=group_generator, device="cpu")[:used]
        selected = samples.index_select(0, perm_cpu.to(samples.device))
    else:
        selected = samples[:used]
    grouped = selected.reshape(group_count, order, width)
    sources = initialization_sources(grouped, order)
    return compress_cp_sources(sources, rank=rank, generator=compression_generator)


def initialize_tower(
    samples: Tensor,
    *,
    k_max: int,
    rank: int,
    streams: RNGStreams,
    shuffle: bool = True,
) -> tuple[CPTower, dict[int, int], dict[int, int]]:
    """Initialize cumulant orders 1..k_max and return diagnostics."""
    tower: CPTower = {}
    source_counts: dict[int, int] = {}
    unused: dict[int, int] = {}
    for order in range(1, k_max + 1):
        group_count = samples.shape[0] // order
        unused[order] = samples.shape[0] - group_count * order
        group_gen = streams.generator("init-group", order)
        comp_gen = streams.generator("init-compress", order)
        cp = initialize_cumulant_cp(
            samples,
            order=order,
            rank=rank,
            group_generator=group_gen,
            compression_generator=comp_gen,
            shuffle=shuffle,
        )
        tower[order] = cp
        # Bell/injection expansion count, without materializing sample copies.
        source_counts[order] = len(initialization_sources(samples[:group_count * order].reshape(group_count, order, samples.shape[1]), order))
    validate_tower(tower, k_max=k_max, rank=rank, width=samples.shape[1])
    return tower, source_counts, unused
