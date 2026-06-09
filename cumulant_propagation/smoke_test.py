from __future__ import annotations

import torch

from cumulant_propagation import propagate_cumulants
from inference_models import DeepReLUMLP


def _symmetrize3(tensor: torch.Tensor) -> torch.Tensor:
    return (
        tensor
        + tensor.permute(0, 2, 1)
        + tensor.permute(1, 0, 2)
        + tensor.permute(1, 2, 0)
        + tensor.permute(2, 0, 1)
        + tensor.permute(2, 1, 0)
    ) / 6


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 8
    torch.manual_seed(0)

    model = DeepReLUMLP(n=n, L=2, device=device, dtype=torch.float64)
    raw_cov = torch.randn(n, n, device=device, dtype=torch.float64)
    raw_k3 = torch.randn(n, n, n, device=device, dtype=torch.float64)
    cumulants = {
        1: 0.1 * torch.randn(n, device=device, dtype=torch.float64),
        2: raw_cov @ raw_cov.T / n + 1e-3 * torch.eye(n, device=device, dtype=torch.float64),
        3: 0.01 * _symmetrize3(raw_k3),
    }

    output = propagate_cumulants(model, cumulants, k_max=3, return_tensors=True)
    print(f"orders: {sorted(output)}")
    print(f"mean_shape: {tuple(output[1].shape)}")
    print(f"cov_shape: {tuple(output[2].shape)}")


if __name__ == "__main__":
    main()
