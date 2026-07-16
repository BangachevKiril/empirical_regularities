from __future__ import annotations

import math
import unittest

import torch

from simple_CP_empirical_prop import (
    MeanPropState,
    empirical_mean_variance,
    mean_prop_activation,
    mean_prop_linear,
    mean_prop_stages,
)


class MeanPropTests(unittest.TestCase):
    def test_empirical_mean_variance_population_convention(self) -> None:
        samples = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        state = empirical_mean_variance(samples)
        self.assertTrue(torch.allclose(state.mean, torch.tensor([3.0, 6.0])))
        self.assertTrue(torch.allclose(state.variance, torch.tensor([8.0 / 3.0, 32.0 / 3.0])))

    def test_linear_mean_and_diagonal_variance(self) -> None:
        state = MeanPropState(
            mean=torch.tensor([1.0, -2.0]),
            variance=torch.tensor([3.0, 5.0]),
        )
        weight = torch.tensor([[2.0, -1.0], [0.5, 4.0]])
        bias = torch.tensor([10.0, -3.0])
        out = mean_prop_linear(state, weight, bias)
        self.assertTrue(torch.allclose(out.mean, weight @ state.mean + bias))
        self.assertTrue(torch.allclose(out.variance, weight.square() @ state.variance))

    def test_relu_gaussian_moments_at_standard_normal(self) -> None:
        state = MeanPropState(
            mean=torch.tensor([0.0], dtype=torch.float64),
            variance=torch.tensor([1.0], dtype=torch.float64),
        )
        out = mean_prop_activation(state, "relu")
        expected_mean = 1.0 / math.sqrt(2.0 * math.pi)
        expected_variance = 0.5 - expected_mean**2
        self.assertAlmostEqual(out.mean.item(), expected_mean, places=12)
        self.assertAlmostEqual(out.variance.item(), expected_variance, places=12)

    def test_mean_prop_stages_flop_count(self) -> None:
        samples = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        linear = torch.nn.Linear(2, 1, bias=True)
        result = mean_prop_stages([(linear, None)], samples)
        self.assertEqual(
            result.diagnostics.analytic_flops_by_stage["empirical_mean_variance"],
            24.0,
        )
        # mean matvec 1*(2*2-1)=3, W**2 cost 2, variance matvec 3, bias 1.
        self.assertEqual(result.diagnostics.analytic_flops_by_stage["layer_0_linear"], 9.0)
        self.assertEqual(result.diagnostics.total_analytic_flops, 33.0)

    def test_mean_prop_sample_budget_uses_only_budgeted_prefix(self) -> None:
        samples = torch.tensor([[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]])
        linear = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
        result = mean_prop_stages([(linear, None)], samples, sample_budget=2)
        expected = empirical_mean_variance(samples[:2])
        self.assertEqual(result.diagnostics.available_sample_count, 3)
        self.assertEqual(result.diagnostics.used_sample_count, 2)
        self.assertEqual(
            result.diagnostics.analytic_flops_by_stage["empirical_mean_variance"],
            16.0,
        )
        self.assertTrue(torch.allclose(result.mean, linear.weight @ expected.mean))


if __name__ == "__main__":
    unittest.main()
