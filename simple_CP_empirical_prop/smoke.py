"""Runnable smoke example for ordinary-CP empirical propagation."""

from __future__ import annotations

import math
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PACKAGE_DIR)
if sys.path and os.path.abspath(sys.path[0]) == PACKAGE_DIR:
    sys.path.pop(0)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import torch
from numpy.polynomial import Polynomial

from simple_CP_empirical_prop import OrdinaryCPConfig, ordinary_cp_mlp


def main() -> None:
    seed = 0
    torch.manual_seed(seed)

    m = 64
    input_dim = 4
    hidden_dim = 6
    output_dim = 3

    samples = torch.randn(m, input_dim, dtype=torch.float64)
    mlp = torch.nn.Module()
    mlp.Ws = torch.nn.ModuleList(
        [
            torch.nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            torch.nn.Linear(hidden_dim, output_dim, dtype=torch.float64),
        ]
    )
    mlp.nonlin_names = ["square"]

    config = OrdinaryCPConfig(
        k_max=3,
        delta=0.25,
        polynomial_by_layer=(Polynomial([0.0, 0.0, 1.0]),),
        reference_two_stage_nonlinear=False,
    )
    result = ordinary_cp_mlp(
        mlp,
        samples,
        k_max=3,
        delta=0.25,
        config=config,
        seed=seed,
        return_all=False,
    )
    print("rank", result.final_tower[1].rank, "expected", max(1, math.ceil(0.25 * m)))
    print("mean", result.mean)
    print("diagrams", result.diagnostics.diagrams_by_layer_and_order)
    print("analytic flops", result.diagnostics.total_analytic_flops)
    print("flop breakdown", result.diagnostics.analytic_flops_by_stage)
    print("clip fractions", result.diagnostics.mean_clip_fraction_by_layer, result.diagnostics.variance_clip_fraction_by_layer)


if __name__ == "__main__":
    main()
