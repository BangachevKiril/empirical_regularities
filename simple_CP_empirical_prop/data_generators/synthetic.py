"""Synthetic data generators used by empirical propagation experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import GeneratedDataset


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu" if device is None else device)


def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def isotropic_gaussian(
    *,
    n: int,
    m: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GeneratedDataset:
    """Generate ``m`` samples in ``R^n`` from ``N(0, I_n)``."""
    _check_positive_int("n", n)
    _check_positive_int("m", m)
    dev = _device(device)
    gen = _generator(dev, seed)
    samples = torch.randn(m, n, generator=gen, device=dev, dtype=dtype)
    return GeneratedDataset(
        name="Isotropic_Gaussian",
        samples=samples,
        metadata={
            "n": n,
            "m": m,
            "seed": seed,
            "distribution": "N(0, I_n)",
        },
    )


def _standardized_laplace(
    *,
    shape: tuple[int, int],
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    eps = torch.finfo(dtype).eps
    u = torch.rand(shape, generator=generator, device=device, dtype=dtype)
    u = u.clamp(eps, 1.0 - eps)
    scale = 1.0 / math.sqrt(2.0)
    return torch.where(
        u < 0.5,
        scale * torch.log(2.0 * u),
        -scale * torch.log(2.0 * (1.0 - u)),
    )


def ica(
    *,
    n: int,
    p: int,
    m: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GeneratedDataset:
    """Generate an ICA mixture ``X = S A.T``.

    The hidden sources ``S`` are independent standardized Laplace coordinates,
    and the mixing matrix ``A`` has entries ``N(0, 1/p)``.
    """
    _check_positive_int("n", n)
    _check_positive_int("p", p)
    _check_positive_int("m", m)
    dev = _device(device)
    gen = _generator(dev, seed)
    mixing = torch.randn(n, p, generator=gen, device=dev, dtype=dtype) / math.sqrt(p)
    sources = _standardized_laplace(
        shape=(m, p),
        generator=gen,
        device=dev,
        dtype=dtype,
    )
    samples = sources @ mixing.transpose(0, 1)
    return GeneratedDataset(
        name="ICA",
        samples=samples,
        metadata={
            "n": n,
            "p": p,
            "m": m,
            "seed": seed,
            "source_distribution": "independent standardized Laplace",
            "mixing_entry_variance": 1.0 / p,
        },
        tensors={
            "mixing_matrix": mixing,
        },
    )


def wishart(
    *,
    n: int,
    p: int,
    m: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GeneratedDataset:
    """Generate samples from ``N(0, Sigma)`` with ``Sigma = V V.T``.

    ``V`` has shape ``[n, p]`` with entries ``N(0, 1/p)``.
    """
    _check_positive_int("n", n)
    _check_positive_int("p", p)
    _check_positive_int("m", m)
    dev = _device(device)
    gen = _generator(dev, seed)
    v = torch.randn(n, p, generator=gen, device=dev, dtype=dtype) / math.sqrt(p)
    sigma = v @ v.transpose(0, 1)
    latent = torch.randn(m, p, generator=gen, device=dev, dtype=dtype)
    samples = latent @ v.transpose(0, 1)
    return GeneratedDataset(
        name="Wishart",
        samples=samples,
        metadata={
            "n": n,
            "p": p,
            "m": m,
            "seed": seed,
            "V_entry_variance": 1.0 / p,
            "distribution": "N(0, V V.T)",
        },
        tensors={
            "V": v,
            "Sigma": sigma,
        },
    )


def mlp_dataset(
    *,
    n: int,
    L: int,
    m: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GeneratedDataset:
    """Generate samples by passing ``N(0, I_n)`` through a depth-``L`` ReLU MLP."""
    _check_positive_int("n", n)
    _check_positive_int("L", L)
    _check_positive_int("m", m)
    dev = _device(device)
    gen = _generator(dev, seed)
    std = math.sqrt(2.0 / n)
    samples = torch.randn(m, n, generator=gen, device=dev, dtype=dtype)
    weights: dict[str, Tensor] = {}
    for layer in range(L):
        weight = torch.randn(n, n, generator=gen, device=dev, dtype=dtype) * std
        weights[f"weight_{layer}"] = weight
        samples = torch.relu(samples @ weight.transpose(0, 1))
    return GeneratedDataset(
        name="MLP_dataset",
        samples=samples,
        metadata={
            "n": n,
            "L": L,
            "m": m,
            "seed": seed,
            "weight_entry_variance": 2.0 / n,
            "activation": "ReLU after every layer",
            "input_distribution": "N(0, I_n)",
        },
        tensors=weights,
    )


Isotropic_Gaussian = isotropic_gaussian
ICA = ica
Wishart = wishart
MLP_dataset = mlp_dataset
