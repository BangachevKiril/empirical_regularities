from __future__ import annotations

import torch

from data_generators import ICADataGenerator


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = ICADataGenerator(n=16, seed=1234, p=8, device=device)
    dataset = generator.sample(m=64, seed_=4321)

    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"A_shape: {tuple(generator.A.shape)}")
    print(f"sources_shape: {tuple(generator.sources.shape)}")
    print(f"dataset_shape: {tuple(dataset.shape)}")
    print(f"dataset_device: {dataset.device}")


if __name__ == "__main__":
    main()
