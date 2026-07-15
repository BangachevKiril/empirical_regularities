"""Weighted CP source compression by unbiased importance sampling."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import torch

from .types import CPTensor, FiniteCPSource, WeightedCPSource, zero_cp


def _coef_float(value: float | Fraction) -> float:
    return float(value.numerator) / float(value.denominator) if isinstance(value, Fraction) else float(value)


def _validate_sources(sources: Sequence[WeightedCPSource]) -> tuple[int, int, torch.device, torch.dtype]:
    if not sources:
        raise ValueError("sources must be nonempty")
    first = sources[0].source
    order, width, device, dtype = first.order, first.width, first.device, first.dtype
    for weighted in sources:
        source = weighted.source
        if source.order != order:
            raise ValueError("all sources must have the same order")
        if source.width != width:
            raise ValueError("all sources must have the same width")
        if source.device != device:
            raise ValueError("all sources must be on the same device")
        if source.dtype != dtype:
            raise ValueError("all sources must have the same dtype")
        if source.rank < 1:
            raise ValueError("source rank must be positive")
    return order, width, device, dtype


def compress_cp_sources(
    sources: Sequence[WeightedCPSource],
    *,
    rank: int,
    generator: torch.Generator,
) -> CPTensor:
    """Compress a weighted CP sum to exactly ``rank`` columns.

    A weighted list represents sum_q c_q (1 / N_q) sum_j a_j^1 otimes ...
    otimes a_j^r. Each output column samples q with probability |c_q| / C0,
    samples j uniformly, copies all source factors, and multiplies only the
    first output factor by C0 * sign(c_q). This makes every output column an
    unbiased rank-one estimate of the full weighted sum.
    """
    if rank < 1:
        raise ValueError("rank must be positive")
    order, width, device, dtype = _validate_sources(sources)
    nonzero = [src for src in sources if _coef_float(src.coefficient) != 0.0]
    if not nonzero:
        return zero_cp(width=width, order=order, rank=rank, device=device, dtype=dtype)

    abs_coefs = torch.tensor(
        [abs(_coef_float(src.coefficient)) for src in nonzero],
        dtype=torch.float64,
        device="cpu",
    )
    c0 = float(abs_coefs.sum().item())
    if not math.isfinite(c0) or c0 <= 0.0:
        raise ValueError("sum of absolute coefficients must be finite and positive")
    probs = abs_coefs / abs_coefs.sum()
    source_choices = torch.multinomial(
        probs, num_samples=rank, replacement=True, generator=generator
    )

    factors = [
        torch.empty(width, rank, device=device, dtype=dtype) for _ in range(order)
    ]
    for source_pos in torch.unique(source_choices, sorted=True).tolist():
        mask = source_choices == source_pos
        output_cols = mask.nonzero(as_tuple=False).flatten()
        source = nonzero[source_pos].source
        column_choices = torch.randint(
            low=0,
            high=source.rank,
            size=(int(output_cols.numel()),),
            generator=generator,
            device="cpu",
        )
        gathered = source.gather_columns(column_choices)
        coef = _coef_float(nonzero[source_pos].coefficient)
        scale = c0 if coef > 0 else -c0
        device_cols = output_cols.to(device)
        for mode, selected in enumerate(gathered):
            value = selected
            if mode == 0:
                value = value * torch.as_tensor(scale, device=device, dtype=dtype)
            factors[mode].index_copy_(1, device_cols, value)
    return CPTensor(tuple(factors))


def reference_compress_cp_sources(
    sources: Sequence[WeightedCPSource],
    *,
    rank: int,
    generator: torch.Generator,
) -> CPTensor:
    """Loop-based compression oracle for tests and debugging."""
    if rank < 1:
        raise ValueError("rank must be positive")
    order, width, device, dtype = _validate_sources(sources)
    nonzero = [src for src in sources if _coef_float(src.coefficient) != 0.0]
    if not nonzero:
        return zero_cp(width=width, order=order, rank=rank, device=device, dtype=dtype)
    abs_coefs = torch.tensor([abs(_coef_float(s.coefficient)) for s in nonzero], dtype=torch.float64)
    probs = abs_coefs / abs_coefs.sum()
    c0 = float(abs_coefs.sum().item())
    out = [torch.empty(width, rank, device=device, dtype=dtype) for _ in range(order)]
    for t in range(rank):
        q = int(torch.multinomial(probs, 1, replacement=True, generator=generator).item())
        source = nonzero[q].source
        j = torch.randint(source.rank, (1,), generator=generator, device="cpu")
        gathered = source.gather_columns(j)
        sign = 1.0 if _coef_float(nonzero[q].coefficient) > 0 else -1.0
        for mode, selected in enumerate(gathered):
            value = selected[:, 0]
            if mode == 0:
                value = value * torch.as_tensor(c0 * sign, device=device, dtype=dtype)
            out[mode][:, t] = value
    return CPTensor(tuple(out))
