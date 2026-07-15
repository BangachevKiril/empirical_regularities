"""Nonlinear ordinary-CP propagation by sampled Hermite diagrams."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import torch
from torch import Tensor

from .compression import compress_cp_sources
from .config import MeanVariance, OrdinaryCPConfig
from .diagrams import DiagramSpec, build_diagram_catalog
from .hermite import ActivationWickSpec, wick_vectors
from .rng import RNGStreams
from .types import CPTensor, CPTower, WeightedCPSource, validate_tower, zero_cp


def _coef_float(value: Fraction) -> float:
    return float(value.numerator) / float(value.denominator)


def sample_diagram_cp(
    spec: DiagramSpec,
    pre_tower: CPTower,
    wick_vectors_by_degree: dict[int, Tensor],
    *,
    rank: int,
    generator: torch.Generator,
) -> CPTensor:
    """Sample one complete diagram product as a rank-``rank`` CP tensor.

    Each block occurrence samples one independent CP rank index, and that same
    index is shared across all modes inside the occurrence.
    """
    if rank < 1:
        raise ValueError("rank must be positive")
    if not pre_tower:
        raise ValueError("pre_tower must be nonempty")
    first = next(iter(pre_tower.values()))
    width, device, dtype = first.width, first.device, first.dtype
    out = [
        wick_vectors_by_degree[k][:, None].expand(width, rank).clone()
        for k in spec.hermite_degrees
    ]
    for block, owners in zip(spec.blocks, spec.owners_by_block):
        block_order = sum(block)
        source = pre_tower[block_order]
        indices_cpu = torch.randint(
            low=0,
            high=source.rank,
            size=(rank,),
            generator=generator,
            device="cpu",
        )
        indices = indices_cpu.to(device)
        for mode, owner in enumerate(owners):
            selected = source.factors[mode].index_select(1, indices)
            out[owner] = out[owner] * selected
    return CPTensor(tuple(out))


def nonlinear_cp_reference(
    output_order: int,
    pre_tower: CPTower,
    wick_vectors_by_degree: dict[int, Tensor],
    catalog: Sequence[DiagramSpec],
    *,
    rank: int,
    streams: RNGStreams,
    layer_index: int,
) -> CPTensor:
    """Literal two-stage nonlinear propagation for one output order."""
    if not catalog:
        first = next(iter(pre_tower.values()))
        return zero_cp(
            width=first.width,
            order=output_order,
            rank=rank,
            device=first.device,
            dtype=first.dtype,
        )
    weighted: list[WeightedCPSource] = []
    for diagram_id, spec in enumerate(catalog):
        cp = sample_diagram_cp(
            spec,
            pre_tower,
            wick_vectors_by_degree,
            rank=rank,
            generator=streams.generator("diagram", layer_index, output_order, diagram_id),
        )
        weighted.append(WeightedCPSource(spec.coefficient, cp))
    return compress_cp_sources(
        weighted,
        rank=rank,
        generator=streams.generator("nonlinear-final-compress", layer_index, output_order),
    )


def nonlinear_cp_fused(
    output_order: int,
    pre_tower: CPTower,
    wick_vectors_by_degree: dict[int, Tensor],
    catalog: Sequence[DiagramSpec],
    *,
    rank: int,
    streams: RNGStreams,
    layer_index: int,
) -> CPTensor:
    """Fused diagram-sum sampler avoiding one full CP tensor per diagram."""
    first = next(iter(pre_tower.values()))
    width, device, dtype = first.width, first.device, first.dtype
    nonzero = [spec for spec in catalog if spec.coefficient != 0]
    if not nonzero:
        return zero_cp(width=width, order=output_order, rank=rank, device=device, dtype=dtype)
    abs_coefs = torch.tensor(
        [abs(_coef_float(spec.coefficient)) for spec in nonzero],
        dtype=torch.float64,
        device="cpu",
    )
    c0 = float(abs_coefs.sum().item())
    probs = abs_coefs / abs_coefs.sum()
    choice_gen = streams.generator("nonlinear-fused-diagram-choice", layer_index, output_order)
    choices = torch.multinomial(probs, rank, replacement=True, generator=choice_gen)
    out = [torch.empty(width, rank, device=device, dtype=dtype) for _ in range(output_order)]
    for local_id in torch.unique(choices, sorted=True).tolist():
        mask = choices == local_id
        cols_cpu = mask.nonzero(as_tuple=False).flatten()
        count = int(cols_cpu.numel())
        spec = nonzero[local_id]
        cp = sample_diagram_cp(
            spec,
            pre_tower,
            wick_vectors_by_degree,
            rank=count,
            generator=streams.generator(
                "nonlinear-fused-diagram", layer_index, output_order, local_id
            ),
        )
        sign = 1.0 if spec.coefficient > 0 else -1.0
        cols = cols_cpu.to(device)
        for mode, factor in enumerate(cp.factors):
            value = factor
            if mode == 0:
                value = value * torch.as_tensor(c0 * sign, device=device, dtype=dtype)
            out[mode].index_copy_(1, cols, value)
    return CPTensor(tuple(out))


def nonlinear_tower(
    pre_tower: CPTower,
    mean_var: MeanVariance,
    activation_spec: ActivationWickSpec,
    config: OrdinaryCPConfig,
    *,
    rank: int,
    streams: RNGStreams,
    layer_index: int,
) -> tuple[CPTower, dict[int, int]]:
    """Propagate all cumulant orders through one coordinatewise activation."""
    vectors = wick_vectors(activation_spec, mean_var.mean, mean_var.variance)
    tower: CPTower = {}
    diagram_counts: dict[int, int] = {}
    for order in range(1, config.k_max + 1):
        catalog = build_diagram_catalog(order, config.k_max, activation_spec.degree)
        diagram_counts[order] = len(catalog)
        if config.reference_two_stage_nonlinear:
            cp = nonlinear_cp_reference(
                order,
                pre_tower,
                vectors,
                catalog,
                rank=rank,
                streams=streams,
                layer_index=layer_index,
            )
        else:
            cp = nonlinear_cp_fused(
                order,
                pre_tower,
                vectors,
                catalog,
                rank=rank,
                streams=streams,
                layer_index=layer_index,
            )
        tower[order] = cp
    validate_tower(tower, k_max=config.k_max, rank=rank)
    return tower, diagram_counts
