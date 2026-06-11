from __future__ import annotations

import torch

from cumulant_propagation._arc_mlp_kprop.diagslice import DSTensor
from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from cumulant_propagation._arc_mlp_kprop.harmonic import HTensor
from data_generators.ica.data_generation import ICADataGenerator


def prior_initialization_flops(
    *,
    n: int,
    p: int,
    k_max: int,
    rank4: int | None = None,
) -> int:
    del p, rank4
    flops = int(n)
    if k_max >= 4:
        flops += 2 * int(n) * int(n)
    return flops


def initialization_flops(
    *,
    n: int,
    p: int,
    k_max: int,
    rank4: int | None = None,
) -> int:
    del rank4
    flops = 2 * int(n) * int(n) * int(p)
    if k_max == 3:
        flops += int(n) * int(p) + int(p) * max(int(n) - 1, 0)
        flops += int(p) + max(int(p) - 1, 0) + 2
    if k_max >= 4:
        flops += 2 * int(n) * int(n) * int(p)
    return flops


def sample_initialization_flops(
    *,
    n: int,
    p: int,
    sample_count: int,
    k_max: int,
    rank4: int | None = None,
) -> int:
    flops = sample_flops(n=n, p=p, sample_count=sample_count)
    flops += k2_estimator_flops(n=n, sample_count=sample_count)
    if k_max >= 4:
        pair_dim = int(p) * int(p)
        rank = int(rank4) if rank4 is not None else min(
            int(sample_count),
            int(p) * (int(p) + 1) // 2,
        )
        pair_features = int(sample_count) * pair_dim
        gram = 2 * int(sample_count) * pair_dim * pair_dim
        symmetric_eigh = int(round((10.0 / 3.0) * pair_dim**3))
        lift = rank * (2 * int(n) * int(p) * int(p) + 2 * int(n) * int(n) * int(p))
        scale_and_pack = 3 * rank * int(n) * int(n)
        flops += pair_features + gram + symmetric_eigh + lift + scale_and_pack
    return flops


def sample_flops(*, n: int, p: int, sample_count: int) -> int:
    return int(sample_count) * (2 * int(p) * int(n) + 2 * int(p))


def k2_estimator_flops(*, n: int, sample_count: int) -> int:
    return 2 * int(sample_count) * int(n) * int(n) + 3 * int(n) * int(n)


def prior_input_cumulants(
    *,
    n: int,
    p: int,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor | FactoredTensor4]:
    eye = torch.eye(n, device=device, dtype=dtype)
    cumulants = {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: float(p) * eye,
    }
    if k_max >= 4:
        zeros_2 = torch.zeros((n, n), device=eye.device, dtype=dtype)
        zeros_3 = torch.zeros((n, n, n), device=eye.device, dtype=dtype)
        pair_slice = torch.full((n, n), -2.0 * float(p), device=eye.device, dtype=dtype)
        pair_slice.diagonal().zero_()
        repeated = DSTensor(
            {
                (4,): torch.full((n,), -6.0 * float(p), device=eye.device, dtype=dtype),
                (3, 1): zeros_2,
                (2, 2): pair_slice,
                (2, 1, 1): zeros_3,
            },
            n=n,
            d=4,
            device=eye.device,
            dtype=dtype,
        )
        cumulants[4] = FactoredTensor4(
            n=n,
            factors=(
                (-6.0 * float(p) * eye)[:, :, None],
                eye[:, :, None],
            ),
            repeated=repeated,
            device=eye.device,
            dtype=dtype,
            assume_symmetric=True,
        )
    return cumulants


def scalar_fourth_core(A: torch.Tensor) -> torch.Tensor:
    """Normalized scalar d=4,r=2 ICA fourth-cumulant component for K=3 propagation."""
    n = A.shape[0]
    column_norms_squared = A.square().sum(dim=0)
    tau = -2.0 * column_norms_squared.square().sum()
    return 3.0 * tau / (float(n) * float(n + 2))


def input_cumulants(
    *,
    data_generator: ICADataGenerator,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor | FactoredTensor4 | HTensor]:
    A = data_generator.A.to(device=device, dtype=dtype)
    n = A.shape[0]
    cumulants = {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: A @ A.T,
    }
    if k_max == 3:
        cumulants[4] = HTensor(
            core=scalar_fourth_core(A),
            r=2,
            n=n,
        )
    if k_max >= 4:
        pair_factors = A[:, None, :] * A[None, :, :]
        cumulants[4] = FactoredTensor4(
            n=n,
            factors=(-2.0 * pair_factors, pair_factors),
            device=A.device,
            dtype=dtype,
            assume_symmetric=True,
        )
    return cumulants


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _sample_with_sources(
    data_generator: ICADataGenerator,
    *,
    m: int,
    seed: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _make_generator(data_generator.device, seed)
    signs = torch.randint(
        low=0,
        high=2,
        size=(m, data_generator.p),
        generator=generator,
        device=data_generator.device,
    )
    sources = signs.to(dtype=dtype).mul_(2).sub_(1)
    A = data_generator.A.to(dtype=dtype)
    samples = sources @ A.T
    return samples, sources


def _compressed_s4_factors_from_sources(
    *,
    A: torch.Tensor,
    sources: torch.Tensor,
    coefficient: float,
    eig_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    m, p = sources.shape
    pair_features = torch.einsum("ma,mb->mab", sources, sources).reshape(m, p * p)
    gram = pair_features.T @ pair_features
    gram = gram / float(m)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    keep = eigvals > eig_tol * eigvals.max().clamp_min(1.0)
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    basis = eigvecs.T.reshape(-1, p, p)
    basis = (basis + basis.transpose(1, 2)) / 2
    lifted = torch.einsum("ia,rab,jb->rij", A, basis, A)
    left = coefficient * eigvals[:, None, None] * lifted
    right = lifted
    return left.permute(1, 2, 0), right.permute(1, 2, 0), int(eigvals.numel())


def sample_average_input_cumulants(
    *,
    data_generator: ICADataGenerator,
    m: int,
    seed: int,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
    eig_tol: float,
) -> tuple[dict[int, torch.Tensor | FactoredTensor4], int | None]:
    samples, sources = _sample_with_sources(
        data_generator,
        m=m,
        seed=seed,
        dtype=dtype,
    )
    samples = samples.to(device=device, dtype=dtype)
    sources = sources.to(device=device, dtype=dtype)
    n = samples.shape[1]
    p = data_generator.p
    eye = torch.eye(n, device=device, dtype=dtype)
    x2 = samples.T @ samples / float(m)

    second_a = float(p * (p - 1)) / float(m + p - 1)
    second_b = float(m) / float(m + p - 1)
    cumulants: dict[int, torch.Tensor | FactoredTensor4] = {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: second_a * eye + second_b * x2,
    }
    rank4: int | None = None

    if k_max >= 4:
        q2 = float(m) / float(m + p - 1)
        q4 = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
        alpha = float(p) - 2.0 * float(p) * q2 + float(p * p) * q4
        beta = q2 - float(p) * q4
        gamma = q4

        const_left = (-6.0 * alpha * eye)[:, :, None]
        const_right = eye[:, :, None]
        square_left = (-12.0 * beta * x2)[:, :, None]
        square_right = eye[:, :, None]
        s4_left, s4_right, rank4 = _compressed_s4_factors_from_sources(
            A=data_generator.A.to(device=device, dtype=dtype),
            sources=sources,
            coefficient=-2.0 * gamma,
            eig_tol=eig_tol,
        )
        cumulants[4] = FactoredTensor4(
            n=n,
            factors=(
                torch.cat((const_left, square_left, s4_left), dim=2),
                torch.cat((const_right, square_right, s4_right), dim=2),
            ),
            device=eye.device,
            dtype=dtype,
            assume_symmetric=True,
        )

    return cumulants, rank4
