from __future__ import annotations

import math
import unittest

import torch

from simple_CP_empirical_prop.hermite import relu_wick_coef


class ReLUWickTests(unittest.TestCase):
    def test_relu_wick_basic_values_at_standard_normal(self) -> None:
        mean = torch.tensor([0.0], dtype=torch.float64)
        var = torch.tensor([1.0], dtype=torch.float64)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        self.assertTrue(torch.allclose(relu_wick_coef(mean, var, 0), torch.tensor([inv_sqrt_2pi], dtype=torch.float64)))
        self.assertTrue(torch.allclose(relu_wick_coef(mean, var, 1), torch.tensor([0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(relu_wick_coef(mean, var, 2), torch.tensor([inv_sqrt_2pi], dtype=torch.float64)))
        self.assertTrue(torch.allclose(relu_wick_coef(mean, var, 3), torch.tensor([0.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(relu_wick_coef(mean, var, 4), torch.tensor([-inv_sqrt_2pi], dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
