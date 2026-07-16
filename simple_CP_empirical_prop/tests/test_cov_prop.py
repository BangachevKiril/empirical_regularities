from __future__ import annotations

import math
import unittest

import torch

from simple_CP_empirical_prop import (
    CovPropState,
    cov_prop_activation,
    cov_prop_linear,
    custom_cov_prop_stages,
    empirical_mean_covariance,
)


class CovPropTests(unittest.TestCase):
    def test_empirical_mean_covariance_population_convention(self) -> None:
        samples = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        state = empirical_mean_covariance(samples)
        centered = samples - torch.tensor([3.0, 6.0])
        expected_cov = centered.T @ centered / 3.0
        self.assertTrue(torch.allclose(state.mean, torch.tensor([3.0, 6.0])))
        self.assertTrue(torch.allclose(state.covariance, expected_cov))

    def test_linear_mean_and_full_covariance(self) -> None:
        state = CovPropState(
            mean=torch.tensor([1.0, -2.0]),
            covariance=torch.tensor([[3.0, 0.5], [0.5, 5.0]]),
        )
        weight = torch.tensor([[2.0, -1.0], [0.5, 4.0]])
        bias = torch.tensor([10.0, -3.0])
        out = cov_prop_linear(state, weight, bias)
        self.assertTrue(torch.allclose(out.mean, weight @ state.mean + bias))
        self.assertTrue(torch.allclose(out.covariance, weight @ state.covariance @ weight.T))

    def test_relu_standard_normal_moments(self) -> None:
        state = CovPropState(
            mean=torch.zeros(2, dtype=torch.float64),
            covariance=torch.eye(2, dtype=torch.float64),
        )
        out = cov_prop_activation(state, "relu", quadrature_degree=80)
        expected_mean = 1.0 / math.sqrt(2.0 * math.pi)
        expected_variance = 0.5 - expected_mean**2
        self.assertTrue(
            torch.allclose(
                out.mean,
                torch.full((2,), expected_mean, dtype=torch.float64),
                atol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                out.covariance.diagonal(),
                torch.full((2,), expected_variance, dtype=torch.float64),
                atol=1e-12,
            )
        )
        self.assertAlmostEqual(out.covariance[0, 1].item(), 0.0, places=6)

    def test_custom_cov_prop_switches_modes(self) -> None:
        dense_samples = torch.randn(4, 3, generator=torch.Generator().manual_seed(1))
        dense_linear = torch.nn.Linear(3, 3, bias=False)
        dense = custom_cov_prop_stages([(dense_linear, None)], dense_samples, rank=4)
        self.assertEqual(dense.diagnostics.mode, "dense")

        cp_samples = torch.randn(2, 4, generator=torch.Generator().manual_seed(2))
        cp_linear = torch.nn.Linear(4, 4, bias=False)
        cp = custom_cov_prop_stages([(cp_linear, None)], cp_samples, rank=2)
        self.assertEqual(cp.diagnostics.mode, "cp")

    def test_custom_cov_prop_dense_budget_uses_only_budgeted_prefix(self) -> None:
        samples = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 6.0],
                [5.0, 10.0],
                [100.0, 200.0],
            ]
        )
        result = custom_cov_prop_stages([], samples, rank=3)
        expected = empirical_mean_covariance(samples[:3])
        self.assertEqual(result.diagnostics.mode, "dense")
        self.assertEqual(result.diagnostics.available_sample_count, 4)
        self.assertEqual(result.diagnostics.used_sample_count, 3)
        self.assertTrue(torch.allclose(result.mean, expected.mean))
        self.assertTrue(torch.allclose(result.covariance, expected.covariance))

    def test_custom_cov_prop_cp_budget_uses_only_budgeted_prefix_for_flops(self) -> None:
        samples = torch.randn(5, 4, generator=torch.Generator().manual_seed(3))
        result = custom_cov_prop_stages([], samples, rank=2)
        self.assertEqual(result.diagnostics.mode, "cp")
        self.assertEqual(result.diagnostics.available_sample_count, 5)
        self.assertEqual(result.diagnostics.used_sample_count, 2)
        self.assertEqual(result.diagnostics.sample_budget, 2)


if __name__ == "__main__":
    unittest.main()
