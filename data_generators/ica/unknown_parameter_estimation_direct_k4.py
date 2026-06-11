from __future__ import annotations

import torch

from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from cumulant_propagation._arc_mlp_kprop.harmonic import HTensor
from data_generators.ica.data_generation import ICADataGenerator
from data_generators.ica.known_parameter_estimation import (
    k2_estimator_flops,
    sample_flops,
)
from data_generators.ica.unknown_parameter_estimation import (
    input_cumulants as old_input_cumulants,
    tau_initialization_flops,
    tau_raw_estimate,
)


def k4_estimator_flops(
    *,
    n: int,
    p: int,
    sample_count: int,
    rank4: int | None,
) -> int:
    del p, rank4
    # Build Z[a,b,i] = x_i[a] x_i[b] and scale one factor by coefficient / m.
    return 2 * int(sample_count) * int(n) * int(n) + 4 * int(n) * int(n)


def initialization_flops(
    *,
    n: int,
    p: int,
    sample_count: int,
    k_max: int,
    rank4: int | None = None,
) -> int:
    flops = sample_flops(n=n, p=p, sample_count=sample_count)
    if k_max == 1:
        return flops + 2 * int(sample_count) * int(n) + 4
    flops += k2_estimator_flops(n=n, sample_count=sample_count)
    if k_max == 3:
        flops += tau_initialization_flops(n=n, sample_count=sample_count)
    if k_max >= 4:
        flops += k4_estimator_flops(
            n=n,
            p=p,
            sample_count=sample_count,
            rank4=rank4,
        )
    return flops


def direct_k4_factors_from_samples(
    *,
    samples: torch.Tensor,
    coefficient: float,
    const_left: torch.Tensor,
    const_right: torch.Tensor,
    square_left: torch.Tensor,
    square_right: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    m = samples.shape[0]
    n = samples.shape[1]
    left = torch.empty((n, n, m + 2), device=samples.device, dtype=samples.dtype)
    right = torch.empty((n, n, m + 2), device=samples.device, dtype=samples.dtype)
    left[:, :, 0] = const_left
    right[:, :, 0] = const_right
    left[:, :, 1] = square_left
    right[:, :, 1] = square_right

    scale = float(coefficient) / float(m)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        pair_factors = torch.einsum("ma,mb->abm", samples[start:stop], samples[start:stop])
        right[:, :, start + 2 : stop + 2] = pair_factors
        left[:, :, start + 2 : stop + 2] = scale * pair_factors
    return left, right, int(m)


def input_cumulants(
    *,
    data_generator: ICADataGenerator,
    m: int,
    seed: int,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
    eig_tol: float,
    gram_chunk_size: int,
) -> tuple[dict[int, torch.Tensor | FactoredTensor4 | HTensor], int | None]:
    if k_max < 4:
        return old_input_cumulants(
            data_generator=data_generator,
            m=m,
            seed=seed,
            k_max=k_max,
            device=device,
            dtype=dtype,
            eig_tol=eig_tol,
            gram_chunk_size=gram_chunk_size,
        )

    samples = data_generator.sample(m, seed_=seed).to(device=device, dtype=dtype)
    n = samples.shape[1]
    p = data_generator.p
    eye = torch.eye(n, device=device, dtype=dtype)

    x2 = samples.T @ samples / float(m)
    second_a = float(p * (p - 1)) / float(m + p - 1)
    second_b = float(m) / float(m + p - 1)
    cumulants: dict[int, torch.Tensor | FactoredTensor4 | HTensor] = {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: second_a * eye + second_b * x2,
    }

    if k_max == 3:
        tau_raw = tau_raw_estimate(
            samples=samples,
            n=n,
            p=p,
            gram_chunk_size=gram_chunk_size,
        )
        core = (3.0 * tau_raw / (float(n) * float(n + 2))).to(device=device, dtype=dtype)
        cumulants[4] = HTensor(core=core, r=2, n=n)
        return cumulants, None

    lambda2 = float(m) / float(m + p - 1)
    gamma = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
    beta = lambda2 - float(p) * gamma
    alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

    left, right, rank4 = direct_k4_factors_from_samples(
        samples=samples,
        coefficient=-2.0 * gamma,
        const_left=-6.0 * alpha * eye,
        const_right=eye,
        square_left=-12.0 * beta * x2,
        square_right=eye,
        chunk_size=gram_chunk_size,
    )
    cumulants[4] = FactoredTensor4(
        n=n,
        factors=(left, right),
        device=eye.device,
        dtype=dtype,
        assume_symmetric=True,
    )
    return cumulants, rank4
