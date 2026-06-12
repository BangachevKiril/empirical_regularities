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
from experimental_cumulant_propagation.structured_cp import CPTerm
from experimental_cumulant_propagation.structured_tensor4 import StructuredTensor4


def k4_estimator_flops(
    *,
    n: int,
    p: int,
    sample_count: int,
    rank4: int | None,
) -> int:
    del p, rank4
    # Covariance-style corrections plus storing X.T as CP4. The sample
    # generation cost is accounted for separately by initialization_flops.
    return 2 * int(sample_count) * int(n) * int(n) + 8 * int(n) * int(n)


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
) -> tuple[dict[int, torch.Tensor | StructuredTensor4 | FactoredTensor4 | HTensor], int | None]:
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

    del eig_tol, gram_chunk_size
    samples = data_generator.sample(m, seed_=seed).to(device=device, dtype=dtype)
    n = samples.shape[1]
    p = data_generator.p
    eye = torch.eye(n, device=device, dtype=dtype)

    x2 = samples.T @ samples / float(m)
    second_a = float(p * (p - 1)) / float(m + p - 1)
    second_b = float(m) / float(m + p - 1)
    cumulants: dict[int, torch.Tensor | StructuredTensor4 | FactoredTensor4 | HTensor] = {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: second_a * eye + second_b * x2,
    }

    lambda2 = float(m) / float(m + p - 1)
    gamma = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
    beta = lambda2 - float(p) * gamma
    alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

    correction_left = torch.empty((n, n, 2), device=device, dtype=dtype)
    correction_right = torch.empty((n, n, 2), device=device, dtype=dtype)
    correction_left[:, :, 0] = -6.0 * alpha * eye
    correction_right[:, :, 0] = eye
    correction_left[:, :, 1] = -12.0 * beta * x2
    correction_right[:, :, 1] = eye
    correction_tensor = FactoredTensor4(
        n=n,
        factors=(correction_left, correction_right),
        device=eye.device,
        dtype=dtype,
        assume_symmetric=True,
    )

    X = samples.T.contiguous()
    sample_term = CPTerm(
        degree=4,
        factors=(X, X, X, X),
        coef=-2.0 * gamma / float(m),
        rank_kind="rank_m",
        symmetric=True,
    )
    cumulants[4] = StructuredTensor4(
        n=n,
        cp_terms=(sample_term,),
        pair_tensor=correction_tensor,
        device=eye.device,
        dtype=dtype,
    )
    return cumulants, int(m)
