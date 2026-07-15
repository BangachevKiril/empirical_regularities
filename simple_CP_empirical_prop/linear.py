"""Exact linear propagation of CP cumulant factors."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import CPTensor, CPTower


def linear_cp(cp: CPTensor, weight: Tensor, bias: Tensor | None = None) -> CPTensor:
    """Apply a deterministic affine map to one ordinary cumulant CP tensor.

    Factors are stored as [input_width, rank], so a linear layer with weight
    [output_width, input_width] acts by ``weight @ factor`` on every mode. The
    deterministic bias shifts only the first cumulant.
    """
    if weight.ndim != 2 or weight.shape[1] != cp.width:
        raise ValueError("weight must have shape [output_width, cp.width]")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must have shape [output_width]")
    factors = tuple(weight @ factor for factor in cp.factors)
    if cp.order == 1 and bias is not None:
        factors = (factors[0] + bias[:, None],)
    return CPTensor(factors)


def linear_tower(
    tower: CPTower,
    linear_or_weight: torch.nn.Linear | Tensor,
    bias: Tensor | None = None,
) -> CPTower:
    """Apply a linear layer to all cumulant orders in a tower."""
    if isinstance(linear_or_weight, torch.nn.Linear):
        weight = linear_or_weight.weight
        layer_bias = linear_or_weight.bias
    else:
        weight = linear_or_weight
        layer_bias = bias
    return {
        order: linear_cp(cp, weight, layer_bias if order == 1 else None)
        for order, cp in tower.items()
    }
