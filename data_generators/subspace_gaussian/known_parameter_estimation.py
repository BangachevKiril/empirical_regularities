from __future__ import annotations

import torch

from data_generators.subspace_gaussian.data_generation import SubspaceGaussianDataGenerator


def initialization_flops(
    *,
    n: int,
    p: int,
    k_max: int,
    rank4: int | None = None,
) -> int:
    del k_max, rank4
    return 2 * int(n) * int(n) * int(p)


def input_cumulants(
    *,
    data_generator: SubspaceGaussianDataGenerator,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    del k_max
    A = data_generator.A.to(device=device, dtype=dtype)
    n = A.shape[0]
    return {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: A @ A.T,
    }
