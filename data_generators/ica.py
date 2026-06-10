from __future__ import annotations

import torch
from torch import Tensor

from .base import DataGenerator


class ICADataGenerator(DataGenerator):
    """Independent component analysis data generator.

    Samples are rows of the returned dataset. For each sample,

        x_i = A r_i,

    where A has shape (n, p) with independent N(0, 1 / (np)) entries and
    r_i is uniform on {-1, 1}^p.
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
        scale = float(self.n * self.p) ** -0.5
        self.A = torch.randn(
            (self.n, self.p),
            generator=self._make_generator(self.seed),
            device=self.device,
            dtype=self.dtype,
        ).mul_(scale)

    def _sample(self, m: int, generator: torch.Generator) -> Tensor:
        signs = torch.randint(
            low=0,
            high=2,
            size=(m, self.p),
            generator=generator,
            device=self.device,
        )
        r = signs.to(dtype=self.dtype).mul_(2).sub_(1)
        self.sources = r
        return r @ self.A.T
