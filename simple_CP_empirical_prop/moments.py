"""Mean and marginal variance extraction from an ordinary-CP tower."""

from __future__ import annotations

import torch

from .config import MeanVariance, OrdinaryCPConfig
from .types import CPTower


def _fraction_changed(before: torch.Tensor, after: torch.Tensor) -> float:
    if before.numel() == 0:
        return 0.0
    return float((before != after).to(torch.float64).mean().item())


def mean_and_variance(tower: CPTower, config: OrdinaryCPConfig) -> MeanVariance:
    """Extract mean from T1 and variance from the diagonal of T2.

    T2 is already a second cumulant, so this function intentionally does not
    subtract ``mean**2``.
    """
    if 1 not in tower or 2 not in tower:
        raise ValueError("mean_and_variance requires orders one and two")
    raw_mean = tower[1].mean_vector()
    raw_variance = tower[2].diagonal_order2()
    mean = raw_mean
    if config.mean_abs_clip is not None:
        mean = mean.clamp(-config.mean_abs_clip, config.mean_abs_clip)
    variance = raw_variance.clamp_min(config.variance_min)
    if config.variance_max is not None:
        variance = variance.clamp_max(config.variance_max)
    return MeanVariance(
        raw_mean=raw_mean,
        raw_variance=raw_variance,
        mean=mean,
        variance=variance,
        sigma=variance.sqrt(),
        mean_clip_fraction=_fraction_changed(raw_mean, mean),
        variance_clip_fraction=_fraction_changed(raw_variance, variance),
    )
