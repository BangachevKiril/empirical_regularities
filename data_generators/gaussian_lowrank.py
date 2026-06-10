from __future__ import annotations

import torch
from torch import Tensor

from .base import DataGenerator


class GaussianLowRankDataGenerator(DataGenerator):
    """Low-rank Gaussian data generator.

    Samples are rows of the returned dataset. For each sample,

        x_i = A r_i,

    where A has shape (n, p) with independent N(0, 1) entries and
    r_i has independent N(0, 1) coordinates.
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
        self.p = int(p)
        self.A: Tensor
        self.sources: Tensor | None = None
        super().__init__(n, seed, device=device, dtype=dtype)
        self.A = torch.randn(
            (self.n, self.p),
            generator=self._make_generator(self.seed),
            device=self.device,
            dtype=self.dtype,
        )

    def _sample(self, m: int, generator: torch.Generator) -> Tensor:
        sources = torch.randn(
            (m, self.p),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.sources = sources
        return sources @ self.A.T
