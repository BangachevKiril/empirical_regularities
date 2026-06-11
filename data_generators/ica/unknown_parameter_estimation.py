from __future__ import annotations

import math

import torch

from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from cumulant_propagation._arc_mlp_kprop.harmonic import HTensor
from data_generators.ica.data_generation import ICADataGenerator
from data_generators.ica.known_parameter_estimation import (
    k2_estimator_flops,
    sample_flops,
)


def tau_initialization_flops(*, n: int, sample_count: int) -> int:
    return 2 * int(sample_count) * int(n) * int(n) + 3 * int(n) * int(n)


def k4_estimator_flops(
    *,
    n: int,
    p: int,
    sample_count: int,
    rank4: int | None,
) -> int:
    subspace_rank = min(int(n), int(p), int(sample_count))
    pair_dim = subspace_rank * (subspace_rank + 1) // 2
    rank = int(rank4) if rank4 is not None else int(sample_count)
    covariance = 2 * int(sample_count) * int(n) * int(n)
    subspace_eigh = int(round((10.0 / 3.0) * int(n) ** 3))
    coordinates = 2 * int(sample_count) * int(n) * subspace_rank
    pair_features = int(sample_count) * pair_dim
    pair_gram = 2 * int(sample_count) * pair_dim * pair_dim
    pair_eigh = int(round((10.0 / 3.0) * pair_dim**3))
    lift = rank * (2 * int(n) * subspace_rank * subspace_rank + 2 * int(n) * int(n) * subspace_rank)
    scale_and_pack = 3 * rank * int(n) * int(n)
    return (
        covariance
        + subspace_eigh
        + coordinates
        + pair_features
        + pair_gram
        + pair_eigh
        + lift
        + scale_and_pack
    )


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


def _symmetric_pair_features(
    coordinates: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    features = coordinates[:, rows] * coordinates[:, cols]
    offdiag = rows != cols
    if bool(offdiag.any()):
        features[:, offdiag] *= math.sqrt(2.0)
    return features


def compressed_s4_factors_from_samples(
    *,
    samples: torch.Tensor,
    coefficient: float,
    eig_tol: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    m, n = samples.shape
    covariance = samples.T @ samples / float(m)
    subspace_eigvals, subspace = torch.linalg.eigh(covariance)
    subspace_keep = subspace_eigvals > eig_tol * subspace_eigvals.max().clamp_min(1.0)
    subspace = subspace[:, subspace_keep]
    q = int(subspace.shape[1])
    if q == 0:
        empty = torch.zeros((n, n, 0), device=samples.device, dtype=samples.dtype)
        return empty, empty, 0

    rows, cols = torch.triu_indices(q, q, device=samples.device)
    pair_dim = int(rows.numel())
    pair_gram = torch.zeros(
        (pair_dim, pair_dim),
        device=samples.device,
        dtype=samples.dtype,
    )
    chunk_size = max(1, int(chunk_size))
    for start in range(0, m, chunk_size):
        coordinates = samples[start : start + chunk_size] @ subspace
        pair_features = _symmetric_pair_features(coordinates, rows, cols)
        pair_gram += pair_features.T @ pair_features
    pair_gram = pair_gram / float(m)
    eigvals, eigvecs = torch.linalg.eigh(pair_gram)
    keep = eigvals > eig_tol * eigvals.max().clamp_min(1.0)
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    rank = int(eigvals.numel())
    if rank == 0:
        empty = torch.zeros((n, n, 0), device=samples.device, dtype=samples.dtype)
        return empty, empty, 0

    coordinate_basis = torch.zeros(
        (rank, q, q),
        device=samples.device,
        dtype=samples.dtype,
    )
    basis_coefficients = eigvecs.T
    offdiag = rows != cols
    coordinate_basis[:, rows[~offdiag], cols[~offdiag]] = basis_coefficients[:, ~offdiag]
    if bool(offdiag.any()):
        offdiag_coefficients = basis_coefficients[:, offdiag] / math.sqrt(2.0)
        coordinate_basis[:, rows[offdiag], cols[offdiag]] = offdiag_coefficients
        coordinate_basis[:, cols[offdiag], rows[offdiag]] = offdiag_coefficients
    lifted = torch.einsum("ia,rab,jb->rij", subspace, coordinate_basis, subspace)
    left = coefficient * eigvals[:, None, None] * lifted
    right = lifted
    return left.permute(1, 2, 0), right.permute(1, 2, 0), rank


def gram_scalar_statistics(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples64 = samples.to(dtype=torch.float64)
    squared_norms = samples64.square().sum(dim=1)
    U = squared_norms.sum()
    V = squared_norms.square().sum()
    gram_square_sum = (samples64.T @ samples64).square().sum()
    W = gram_square_sum - V
    return U, V, W


def tau_raw_estimate(
    *,
    samples: torch.Tensor,
    n: int,
    p: int,
    gram_chunk_size: int,
) -> torch.Tensor:
    m = samples.shape[0]
    device = samples.device
    dtype = torch.float64
    del gram_chunk_size
    U, V, W = gram_scalar_statistics(samples)
    Z = torch.stack(
        (
            torch.ones((), device=device, dtype=dtype),
            U,
            U.square(),
            V,
            W,
        )
    )

    n_f = float(n)
    p_f = float(p)
    m_f = float(m)
    R2 = m_f * p_f * (m_f + p_f - 1.0)
    R22 = m_f * p_f * (
        m_f**3 * p_f
        + 2.0 * m_f**2 * p_f**2
        - 2.0 * m_f**2 * p_f
        + m_f * p_f**3
        - 2.0 * m_f * p_f**2
        + 5.0 * m_f * p_f
        - 4.0 * m_f
        - 4.0 * p_f
        + 4.0
    )
    R3 = m_f * p_f * (
        m_f**2 + 3.0 * m_f * p_f - 3.0 * m_f + p_f**2 - 3.0 * p_f + 2.0
    )
    R4 = (
        m_f
        * p_f
        * (m_f + p_f - 1.0)
        * (m_f**2 + 5.0 * m_f * p_f - 5.0 * m_f + p_f**2 - 5.0 * p_f + 4.0)
    )

    def vec(values: tuple[float, float, float, float, float]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=dtype)

    c0 = vec(
        (
            1.0,
            n_f * m_f * p_f,
            n_f**2 * m_f**2 * p_f**2,
            m_f * n_f * p_f**2 * (n_f + 2.0),
            n_f * m_f * p_f**2 * (m_f - n_f - 2.0),
        )
    )
    c1 = vec((0.0, 0.0, 2.0 * n_f, 0.0, n_f * (n_f + 1.0)))
    ell1 = vec(
        (
            0.0,
            1.0,
            2.0 * n_f * m_f * p_f,
            2.0 * p_f * (n_f + 2.0),
            2.0 * p_f * (m_f - n_f - 2.0),
        )
    )
    ell2 = vec((0.0, 0.0, 4.0, 0.0, 2.0 * (n_f + 1.0)))

    M = (
        torch.outer(c0, c0)
        + (torch.outer(c0, c1) + torch.outer(c1, c0)) * R2
        + torch.outer(c1, c1) * R22
        + 2.0
        * n_f
        * (
            torch.outer(ell1, ell1) * R2
            + (torch.outer(ell1, ell2) + torch.outer(ell2, ell1)) * R3
            + torch.outer(ell2, ell2) * R4
        )
    )
    gamma = torch.zeros((5, 5), device=device, dtype=dtype)
    gamma[2, 2] = 8.0 * n_f**2 * R22 + 16.0 * n_f * R4
    gamma[2, 3] = 8.0 * m_f * n_f * p_f * (n_f + 2.0) * (
        m_f**2 * p_f
        + 2.0 * m_f * p_f**2
        - 2.0 * m_f
        + p_f**3
        - 2.0 * p_f**2
        - p_f
        + 2.0
    )
    gamma[2, 4] = 8.0 * m_f * n_f * p_f * (m_f - 1.0) * (
        m_f**2 * n_f
        + m_f**2 * p_f
        + m_f**2
        + 5.0 * m_f * n_f * p_f
        - 5.0 * m_f * n_f
        + 2.0 * m_f * p_f**2
        + 3.0 * m_f * p_f
        - 5.0 * m_f
        + 4.0 * n_f * p_f**2
        - 10.0 * n_f * p_f
        + 6.0 * n_f
        + p_f**3
        + 2.0 * p_f**2
        - 7.0 * p_f
        + 4.0
    )
    gamma[3, 3] = 8.0 * m_f * n_f * p_f * (n_f + 2.0) * (
        3.0 * m_f * p_f - 2.0 * m_f + p_f**3 - 3.0 * p_f + 2.0
    )
    gamma[3, 4] = (
        8.0
        * m_f
        * n_f
        * p_f**2
        * (m_f - 1.0)
        * (n_f + 2.0)
        * (m_f + 2.0 * p_f - 2.0)
    )
    gamma[4, 4] = 4.0 * m_f * n_f * p_f * (m_f - 1.0) * (
        m_f**2 * n_f * p_f
        + m_f**2 * n_f
        + m_f**2 * p_f
        + 3.0 * m_f**2
        + 2.0 * m_f * n_f * p_f**2
        + m_f * n_f * p_f
        - 5.0 * m_f * n_f
        + 2.0 * m_f * p_f**2
        + 9.0 * m_f * p_f
        - 15.0 * m_f
        + n_f * p_f**3
        - 2.0 * n_f * p_f**2
        - 3.0 * n_f * p_f
        + 4.0 * n_f
        + p_f**3
        + 2.0 * p_f**2
        - 19.0 * p_f
        + 16.0
    )
    gamma = gamma + gamma.T - torch.diag(gamma.diag())
    M = M + gamma

    tau0 = -2.0 * p_f * n_f * (n_f + 2.0)
    q = vec(
        (
            0.0,
            0.0,
            -16.0 * p_f * n_f * (n_f + 2.0) * m_f**2,
            -16.0 * p_f * n_f * (n_f + 2.0) * m_f,
            -16.0 * p_f * n_f * (n_f + 2.0) * m_f * (m_f - 1.0),
        )
    )
    b = (
        tau0 * (c0 + c1 * R2)
        - 8.0 * n_f * (n_f + 2.0) * (ell1 * m_f * p_f + ell2 * R2)
        + q
    )
    try:
        theta = torch.linalg.solve(M, b)
    except torch.linalg.LinAlgError:
        theta = torch.linalg.pinv(M) @ b
    return torch.dot(theta, Z)


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
    samples = data_generator.sample(m, seed_=seed).to(device=device, dtype=dtype)
    n = samples.shape[1]
    p = data_generator.p
    eye = torch.eye(n, device=device, dtype=dtype)
    cumulants: dict[int, torch.Tensor | FactoredTensor4 | HTensor] = {
        1: torch.zeros(n, device=device, dtype=dtype),
    }
    energy = samples.square().sum()
    if k_max == 1:
        sigma2 = float(p * (p - 1)) / float(m + p - 1)
        sigma2 = sigma2 + energy / float(n * (m + p - 1))
        cumulants[2] = sigma2 * eye
        return cumulants, None

    x2 = samples.T @ samples / float(m)
    second_a = float(p * (p - 1)) / float(m + p - 1)
    second_b = float(m) / float(m + p - 1)
    cumulants[2] = second_a * eye + second_b * x2
    rank4: int | None = None

    if k_max == 3:
        tau_raw = tau_raw_estimate(
            samples=samples,
            n=n,
            p=p,
            gram_chunk_size=gram_chunk_size,
        )
        core = (3.0 * tau_raw / (float(n) * float(n + 2))).to(device=device, dtype=dtype)
        cumulants[4] = HTensor(core=core, r=2, n=n)
    if k_max >= 4:
        lambda2 = float(m) / float(m + p - 1)
        gamma = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
        beta = lambda2 - float(p) * gamma
        alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

        const_left = (-6.0 * alpha * eye)[:, :, None]
        const_right = eye[:, :, None]
        square_left = (-12.0 * beta * x2)[:, :, None]
        square_right = eye[:, :, None]
        s4_left, s4_right, rank4 = compressed_s4_factors_from_samples(
            samples=samples,
            coefficient=-2.0 * gamma,
            eig_tol=eig_tol,
            chunk_size=gram_chunk_size,
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
