from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class DataGenerator(ABC):
    """Abstract template for generators that sample m-by-n datasets."""

    def __init__(
        self,
        n: int,
        seed: int,
        *aux_args: Any,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **aux_kwargs: Any,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive.")

        self.n = int(n)
        self.seed = int(seed)
        self.aux_args = aux_args
        self.aux_kwargs = dict(aux_kwargs)
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self.dataset: Tensor | None = None
        self.sample_seed: int | None = None

        torch.manual_seed(self.seed)

    def sample(self, m: int, seed_: int = 0) -> Tensor:
        """Generate and store m samples of dimension n."""
        if m <= 0:
            raise ValueError("m must be positive.")

        self.sample_seed = int(seed_)
        generator = self._make_generator(self.sample_seed)
        torch.manual_seed(self.sample_seed)

        dataset = self._sample(int(m), generator)
        self._validate_dataset(dataset, int(m))
        self.dataset = dataset
        return dataset

    @abstractmethod
    def _sample(self, m: int, generator: torch.Generator) -> Tensor:
        """Generate an m-by-n torch tensor containing the dataset."""

    def _make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def _validate_dataset(self, dataset: Tensor, m: int) -> None:
        if not isinstance(dataset, Tensor):
            raise TypeError("sample() must return a torch.Tensor.")
        if dataset.ndim != 2:
            raise ValueError("dataset must be a 2-dimensional tensor.")
        if dataset.shape != (m, self.n):
            raise ValueError(
                f"dataset must have shape ({m}, {self.n}); "
                f"got {tuple(dataset.shape)}."
            )

    def __len__(self) -> int:
        return 0 if self.dataset is None else self.dataset.shape[0]

    def __getitem__(self, index: int | slice) -> Tensor:
        if self.dataset is None:
            raise RuntimeError("No dataset has been sampled yet.")
        return self.dataset[index]
