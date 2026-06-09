from __future__ import annotations

import torch

from inference_models import DeepReLUMLP


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepReLUMLP(n=128, L=4, device=device)
    x = torch.randn(128, 16, device=device)
    y = model(x)

    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"layers: {len(model.weights)}")
    print(f"init_std: {model.init_std}")
    print(f"init_variance: {model.init_variance}")
    print(f"output_shape: {tuple(y.shape)}")
    print(f"output_device: {y.device}")


if __name__ == "__main__":
    main()
