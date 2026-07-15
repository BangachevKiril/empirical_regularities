"""Polynomial Wick/Hermite coefficient adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from numpy.polynomial import Polynomial
from torch import Tensor


@dataclass(frozen=True)
class ActivationWickSpec:
    name: str
    degree: int
    coefficient_fn: Callable[[Tensor, Tensor, int], Tensor]
    exact_polynomial: bool


def _trimmed_degree(poly: Polynomial) -> int:
    coefs = list(poly.coef)
    while len(coefs) > 1 and abs(float(coefs[-1])) == 0.0:
        coefs.pop()
    return len(coefs) - 1


def normal_raw_moment(order: int, mean: Tensor, variance: Tensor) -> Tensor:
    """Return E[(mean + sqrt(variance) G)^order] for G standard normal."""
    if order < 0:
        raise ValueError("order must be nonnegative")
    result = torch.zeros_like(mean)
    for pairs in range(order // 2 + 1):
        singles = order - 2 * pairs
        coeff = math.factorial(order) / (
            (2.0**pairs) * math.factorial(pairs) * math.factorial(singles)
        )
        result = result + coeff * variance.pow(pairs) * mean.pow(singles)
    return result


def poly_wick_coef(poly: Polynomial, mean: Tensor, variance: Tensor, k: int) -> Tensor:
    """Return E[d^k/dx^k poly(G)] for G ~ N(mean, variance)."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k > _trimmed_degree(poly):
        return torch.zeros_like(mean)
    result = torch.zeros_like(mean)
    for power, coef in enumerate(poly.coef):
        if power < k or float(coef) == 0.0:
            continue
        deriv_coeff = float(coef) * math.factorial(power) / math.factorial(power - k)
        result = result + deriv_coeff * normal_raw_moment(power - k, mean, variance)
    return result


def polynomial_wick_spec(poly: Polynomial, *, name: str = "polynomial") -> ActivationWickSpec:
    degree = _trimmed_degree(poly)

    def coefficient_fn(mean: Tensor, variance: Tensor, k: int) -> Tensor:
        return poly_wick_coef(poly, mean, variance, k)

    return ActivationWickSpec(
        name=name,
        degree=degree,
        coefficient_fn=coefficient_fn,
        exact_polynomial=True,
    )


def _probabilists_hermite(order: int, x: Tensor) -> Tensor:
    if order == 0:
        return torch.ones_like(x)
    if order == 1:
        return x
    prev = torch.ones_like(x)
    cur = x
    for degree in range(1, order):
        prev, cur = cur, x * cur - degree * prev
    return cur


def relu_wick_coef(mean: Tensor, variance: Tensor, k: int) -> Tensor:
    """Return E[d^k/dx^k ReLU(Z)] for Z ~ N(mean, variance).

    For k >= 2 this uses distributional derivatives:
    ReLU'' = delta_0 and ReLU^(k) = delta_0^(k-2).
    """
    if k < 0:
        raise ValueError("k must be nonnegative")
    sigma = variance.sqrt()
    alpha = mean / sigma
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    pdf = torch.exp(-0.5 * alpha.square()) * inv_sqrt_2pi
    cdf = 0.5 * (1.0 + torch.erf(alpha * inv_sqrt_2))
    if k == 0:
        return sigma * pdf + mean * cdf
    if k == 1:
        return cdf
    derivative_order = k - 2
    return pdf * _probabilists_hermite(derivative_order, -alpha) / sigma.pow(
        derivative_order + 1
    )


def relu_wick_spec(hermite_degree_cap: int) -> ActivationWickSpec:
    if hermite_degree_cap <= 0:
        raise ValueError("hermite_degree_cap must be positive for ReLU")

    def coefficient_fn(mean: Tensor, variance: Tensor, k: int) -> Tensor:
        if k > hermite_degree_cap:
            return torch.zeros_like(mean)
        return relu_wick_coef(mean, variance, k)

    return ActivationWickSpec(
        name="relu_experimental",
        degree=hermite_degree_cap,
        coefficient_fn=coefficient_fn,
        exact_polynomial=False,
    )


def activation_spec_from_name(
    name: str,
    *,
    allow_nonpolynomial: bool = False,
    hermite_degree_cap: int | None = None,
) -> ActivationWickSpec:
    """Return built-in exact polynomial specs for square and cube.

    Non-polynomial activations are deliberately rejected unless the caller opts
    into an experimental degree cap. This self-contained implementation does
    not currently include quadrature for those activations.
    """
    lowered = name.lower()
    if lowered == "square":
        return polynomial_wick_spec(Polynomial([0.0, 0.0, 1.0]), name="square")
    if lowered == "cube":
        return polynomial_wick_spec(Polynomial([0.0, 0.0, 0.0, 1.0]), name="cube")
    if lowered == "relu" and allow_nonpolynomial and hermite_degree_cap is not None:
        return relu_wick_spec(hermite_degree_cap)
    if allow_nonpolynomial and hermite_degree_cap is not None:
        raise NotImplementedError("only experimental ReLU is implemented")
    raise ValueError(
        f"activation {name!r} is not an exact built-in polynomial; pass a Polynomial"
    )


def wick_vectors(spec: ActivationWickSpec, mean: Tensor, variance: Tensor) -> dict[int, Tensor]:
    """Cache Wick vectors 0..degree for one layer."""
    return {k: spec.coefficient_fn(mean, variance, k) for k in range(spec.degree + 1)}
