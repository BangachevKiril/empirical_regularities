from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from simple_CP_empirical_prop.data_generators import (
    ICA,
    MLP_dataset,
    Isotropic_Gaussian,
    Wishart,
)


class DataGeneratorTests(unittest.TestCase):
    def test_isotropic_gaussian_shape_and_determinism(self) -> None:
        first = Isotropic_Gaussian(n=5, m=7, seed=123)
        second = Isotropic_Gaussian(n=5, m=7, seed=123)
        self.assertEqual(tuple(first.samples.shape), (7, 5))
        self.assertTrue(torch.allclose(first.samples, second.samples))
        self.assertEqual(first.metadata["distribution"], "N(0, I_n)")

    def test_ica_records_mixing_matrix(self) -> None:
        dataset = ICA(n=4, p=3, m=6, seed=4)
        self.assertEqual(tuple(dataset.samples.shape), (6, 4))
        self.assertEqual(tuple(dataset.tensors["mixing_matrix"].shape), (4, 3))
        self.assertEqual(dataset.metadata["source_distribution"], "independent standardized Laplace")

    def test_wishart_records_sigma(self) -> None:
        dataset = Wishart(n=4, p=5, m=6, seed=5)
        v = dataset.tensors["V"]
        sigma = dataset.tensors["Sigma"]
        self.assertEqual(tuple(dataset.samples.shape), (6, 4))
        self.assertTrue(torch.allclose(sigma, v @ v.transpose(0, 1)))

    def test_mlp_dataset_records_weights(self) -> None:
        dataset = MLP_dataset(n=4, L=3, m=5, seed=6)
        self.assertEqual(tuple(dataset.samples.shape), (5, 4))
        for layer in range(3):
            self.assertEqual(tuple(dataset.tensors[f"weight_{layer}"].shape), (4, 4))
        self.assertTrue((dataset.samples >= 0).all())

    def test_generated_dataset_save_records_metadata(self) -> None:
        dataset = Wishart(n=3, p=4, m=2, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            dataset.save(tmp)
            root = Path(tmp)
            self.assertTrue((root / "samples.pt").exists())
            self.assertTrue((root / "recorded_tensors.pt").exists())
            self.assertTrue((root / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
