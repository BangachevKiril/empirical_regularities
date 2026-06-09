from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DeepReLUMLP(nn.Module):
    """Depth-L ReLU MLP with square n-by-n weight matrices.

    By default, weights are Gaussian with variance 2 / n, equivalently
    standard deviation sqrt(2 / n).

    The module computes

        Z_i = W_i X_{i-1}
        X_i = ReLU(Z_i)

    for i = 1, ..., L. Inputs use the mathematical column convention:
    `x` may have shape `(n,)` or `(n, batch)`.
    """

    def __init__(
        self,
        n: int,
        L: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        if n <= 0:
            raise ValueError("n must be positive.")
        if L <= 0:
            raise ValueError("L must be positive.")

        self.n = int(n)
        self.L = int(L)
        self.init_std = float((2.0 / n) ** 0.5 if init_std is None else init_std)
        self.init_variance = self.init_std**2

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty((self.n, self.n), **factory_kwargs))
                for _ in range(self.L)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.normal_(weight, mean=0.0, std=self.init_std)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in (1, 2):
            raise ValueError("x must have shape (n,) or (n, batch).")
        if x.shape[0] != self.n:
            raise ValueError(f"x.shape[0] must be n={self.n}; got {x.shape[0]}.")

        for weight in self.weights:
            x = F.relu(weight @ x)
        return x
