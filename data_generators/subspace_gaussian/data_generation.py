from __future__ import annotations

import math

import torch
from torch import Tensor

from data_generators.base import DataGenerator


class SubspaceGaussianDataGenerator(DataGenerator):
    """Gaussian data supported on a random p-dimensional subspace.

    Draw A' with independent N(0, 1) entries, write its thin SVD as
    A' = U Sigma V^T, and set A = sqrt(n) U. Samples are x = A r for
    r with independent N(0, 1) coordinates.
    """

    def __init__(
        self,
        n: int,
        seed: int,
        p: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if p <= 0:
            raise ValueError("p must be positive.")
        if p > n:
            raise ValueError("p must be at most n for orthogonal columns.")
        self.p = int(p)
        self.A: Tensor
        self.sources: Tensor | None = None
        super().__init__(n, seed, device=device, dtype=dtype)

        raw = torch.randn(
            (self.n, self.p),
            generator=self._make_generator(self.seed),
            device=self.device,
            dtype=self.dtype,
        )
        U, _, _ = torch.linalg.svd(raw, full_matrices=False)
        self.A = math.sqrt(float(self.n)) * U[:, : self.p]

    def _sample(self, m: int, generator: torch.Generator) -> Tensor:
        sources = torch.randn(
            (m, self.p),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.sources = sources
        return sources @ self.A.T
