from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from experimental_cumulant_propagation.structured_cp import CPTerm, DenseSlice


@dataclass(frozen=True)
class CP4InputCumulants:
    """Unknown-A ICA cumulants with the empirical fourth moment in CP4 form."""

    mean: Tensor
    covariance: Tensor
    sample_fourth: CPTerm
    dense_corrections: tuple[DenseSlice, ...]
    rank4: int
    gamma: float
    beta: float
    alpha: float


def unknown_a_direct_cp4_initialization_flops(*, n: int, p: int, sample_count: int) -> int:
    """Input construction cost excluding sample generation.

    The direct CP4 sample term stores X.T once. The dominant empirical work is
    the covariance estimate, O(m n^2), plus O(n^2) dense correction setup.
    """

    del p
    return 2 * int(sample_count) * int(n) * int(n) + 8 * int(n) * int(n)


def direct_ica_cp4_cumulants_from_samples(
    *,
    samples: Tensor,
    p: int,
) -> CP4InputCumulants:
    """Build unknown-A ICA input cumulants using CP4 for the rank-m term.

    This mirrors `unknown_parameter_estimation_direct_k4.py` algebra, but keeps
    the empirical fourth moment as F_4(X.T, X.T, X.T, X.T) instead of Pair4
    factors shaped `(n, n, m)`.
    """

    if samples.ndim != 2:
        raise ValueError("samples must have shape (m, n)")
    m = int(samples.shape[0])
    n = int(samples.shape[1])
    dtype = samples.dtype
    device = samples.device

    eye = torch.eye(n, dtype=dtype, device=device)
    x2 = samples.T @ samples / float(m)

    second_a = float(p * (p - 1)) / float(m + p - 1)
    second_b = float(m) / float(m + p - 1)
    covariance = second_a * eye + second_b * x2

    lambda2 = float(m) / float(m + p - 1)
    gamma = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
    beta = lambda2 - float(p) * gamma
    alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

    X = samples.T.contiguous()
    sample_fourth = CPTerm(
        degree=4,
        factors=(X, X, X, X),
        coef=-2.0 * gamma / float(m),
        rank_kind="rank_m",
        symmetric=True,
    )
    corrections = (
        DenseSlice(pattern=(2, 2), tensor=-6.0 * alpha * eye),
        DenseSlice(pattern=(2, 2), tensor=-12.0 * beta * x2),
    )

    return CP4InputCumulants(
        mean=torch.zeros(n, dtype=dtype, device=device),
        covariance=covariance,
        sample_fourth=sample_fourth,
        dense_corrections=corrections,
        rank4=m,
        gamma=gamma,
        beta=beta,
        alpha=alpha,
    )
