from __future__ import annotations

import unittest

import torch
from numpy.polynomial import Polynomial

from simple_CP_empirical_prop import (
    K4OrdinaryCPConfig,
    OrdinaryCPConfig,
    exact_covariance_factor_view,
    initialize_k4_state,
    load_k4_retained_diagrams,
    ordinary_cp_mlp,
    polynomial_wick_spec,
    propagate_k4_stages,
)


class K4HybridTests(unittest.TestCase):
    def test_default_catalog_counts(self) -> None:
        catalog, meta = load_k4_retained_diagrams()
        self.assertEqual(meta["orbit_count"], 26)
        self.assertEqual({order: len(catalog[order]) for order in range(1, 5)}, {1: 3, 2: 15, 3: 35, 4: 29})
        self.assertEqual(meta["labelled_count_by_order"], {1: 3, 2: 15, 3: 35, 4: 29})

    def test_exact_covariance_factor_view_reconstructs_indefinite_matrix(self) -> None:
        cov = torch.tensor([[2.0, 0.25], [0.25, -1.0]], dtype=torch.float64)
        source = exact_covariance_factor_view(cov)
        reconstructed = source.factors[0] @ source.factors[1].T / source.rank
        self.assertTrue(torch.allclose(reconstructed, cov, atol=1e-12))

    def test_initialization_covariance_branch_split(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(10)
        samples = torch.randn(8, 3, generator=gen, dtype=torch.float64)
        dense = initialize_k4_state(samples, M=3, config=K4OrdinaryCPConfig(), seed=1)
        self.assertEqual(dense.covariance_mode, "dense")
        self.assertIsNotNone(dense.covariance_dense)
        self.assertEqual(dense.third.rank, 3)
        self.assertEqual(dense.fourth.rank, 3)

        cp = initialize_k4_state(samples, M=2, config=K4OrdinaryCPConfig(), seed=1)
        self.assertEqual(cp.covariance_mode, "cp")
        self.assertIsNotNone(cp.covariance_cp)
        self.assertEqual(cp.covariance_cp.rank, 2)
        self.assertEqual(cp.third.rank, 2)
        self.assertEqual(cp.fourth.rank, 2)

    def test_scalar_identity_preserves_explicit_mean_and_dense_covariance(self) -> None:
        samples = torch.tensor([[-1.0], [0.5], [2.0], [3.0]], dtype=torch.float64)
        linear = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            linear.weight.fill_(1.0)
        identity = polynomial_wick_spec(Polynomial([0.0, 1.0]), name="identity")
        result = propagate_k4_stages(
            [(linear, identity)],
            samples,
            rank=2,
            config=K4OrdinaryCPConfig(),
            seed=4,
            return_all=True,
        )
        self.assertIsNotNone(result.act_states)
        initial = result.act_states[0]
        final = result.act_states[-1]
        self.assertTrue(torch.allclose(final.mean, initial.mean, atol=1e-12))
        self.assertIsNotNone(initial.covariance_dense)
        self.assertIsNotNone(final.covariance_dense)
        self.assertTrue(torch.allclose(final.covariance_dense, initial.covariance_dense, atol=1e-12))

    def test_ordinary_cp_mlp_k4_compatibility_smoke(self) -> None:
        torch.manual_seed(0)
        samples = torch.randn(12, 2, dtype=torch.float64)
        mlp = torch.nn.Module()
        mlp.Ws = torch.nn.ModuleList(
            [
                torch.nn.Linear(2, 3, dtype=torch.float64),
                torch.nn.Linear(3, 1, dtype=torch.float64),
            ]
        )
        mlp.nonlin_names = ["square"]
        config = OrdinaryCPConfig(
            k_max=4,
            delta=0.5,
            polynomial_by_layer=(Polynomial([0.0, 0.0, 1.0]),),
        )
        result = ordinary_cp_mlp(
            mlp,
            samples,
            k_max=4,
            delta=0.5,
            config=config,
            seed=7,
        )
        self.assertEqual(tuple(result.mean.shape), (1,))
        self.assertTrue(torch.isfinite(result.mean).all())
        self.assertEqual(result.diagnostics.labelled_count_by_order, {1: 3, 2: 15, 3: 35, 4: 29})


if __name__ == "__main__":
    unittest.main()
