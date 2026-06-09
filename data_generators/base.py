from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class DataGenerator(ABC):
    """Abstract template for generators that produce an m-by-n dataset."""

    def __init__(
        self,
        m: int,
        n: int,
        seed: int,
        *aux_args: Any,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **aux_kwargs: Any,
    ) -> None:
        if m <= 0:
            raise ValueError("m must be positive.")
        if n <= 0:
            raise ValueError("n must be positive.")

        self.m = int(m)
        self.n = int(n)
        self.seed = int(seed)
        self.aux_args = aux_args
        self.aux_kwargs = dict(aux_kwargs)
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        torch.manual_seed(self.seed)

        self.dataset = self.generate(*self.aux_args, **self.aux_kwargs)
        self._validate_dataset(self.dataset)

    @abstractmethod
    def generate(self, *aux_args: Any, **aux_kwargs: Any) -> Tensor:
        """Generate an m-by-n torch tensor containing the dataset."""

    def _validate_dataset(self, dataset: Tensor) -> None:
        if not isinstance(dataset, Tensor):
            raise TypeError("generate() must return a torch.Tensor.")
        if dataset.ndim != 2:
            raise ValueError("dataset must be a 2-dimensional tensor.")
        if dataset.shape != (self.m, self.n):
            raise ValueError(
                f"dataset must have shape ({self.m}, {self.n}); "
                f"got {tuple(dataset.shape)}."
            )

    def __len__(self) -> int:
        return self.m

    def __getitem__(self, index: int | slice) -> Tensor:
        return self.dataset[index]

