from __future__ import annotations

import torch

from data_generators.gaussian.data_generation import GaussianDataGenerator


def initialization_flops(
    *,
    n: int,
    p: int | float,
    k_max: int,
    rank4: int | None = None,
) -> int:
    del p, k_max, rank4
    return int(n)


def input_cumulants(
    *,
    data_generator: GaussianDataGenerator,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    del k_max
    eye = torch.eye(data_generator.n, device=device, dtype=dtype)
    return {
        1: torch.zeros(data_generator.n, device=device, dtype=dtype),
        2: float(data_generator.p) * eye,
    }
