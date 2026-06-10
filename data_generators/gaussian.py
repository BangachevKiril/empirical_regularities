from __future__ import annotations

import torch
from torch import Tensor

from .base import DataGenerator


class GaussianDataGenerator(DataGenerator):
    """Isotropic Gaussian data generator.

    Samples are rows of the returned dataset, with independent N(0, p)
    coordinates in dimension n.
    """

    def __init__(
        self,
        n: int,
        seed: int = 0,
        p: float = 1,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if p <= 0:
            raise ValueError("p must be positive.")
        self.p = float(p)
        super().__init__(n, seed, device=device, dtype=dtype)

    def sample(self, m: int, seed_: int | None = None) -> Tensor:
        """Generate and store m isotropic Gaussian samples."""
        sample_seed = self.seed if seed_ is None else int(seed_)
        return super().sample(m, seed_=sample_seed)

    def _sample(self, m: int, generator: torch.Generator) -> Tensor:
        dataset = torch.randn(
            (m, self.n),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        if self.p != 1.0:
            dataset = dataset * (self.p**0.5)
        return dataset
