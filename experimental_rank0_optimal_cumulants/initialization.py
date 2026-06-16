from __future__ import annotations

import torch

from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4


def rank0_input_cumulants(
    *,
    n: int,
    p: int,
    sample_count: int,
    covariance: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, object]:
    """Section 3.1 averaged ICA estimator, truncated to rank-0 K4 factors."""
    eye = torch.eye(n, device=device, dtype=dtype)
    actual_device = eye.device
    Sbar = covariance.to(device=actual_device, dtype=dtype)
    m = int(sample_count)

    a_star = float(p * (p - 1)) / float(m + p - 1)
    b_star = float(m) / float(m + p - 1)
    K2 = Sbar.mul(b_star)
    K2.diagonal().add_(a_star)

    lambda2 = float(m) / float(m + p - 1)
    gamma = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
    beta = lambda2 - float(p) * gamma
    alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

    analytic_left = Sbar.mul(-12.0 * beta)
    analytic_left.diagonal().add_(-6.0 * alpha)
    K4 = FactoredTensor4(
        n=n,
        factors=(analytic_left[:, :, None], eye[:, :, None]),
        device=actual_device,
        dtype=dtype,
        assume_symmetric=True,
    )

    return {
        1: torch.zeros(n, device=actual_device, dtype=dtype),
        2: K2,
        4: K4,
    }
