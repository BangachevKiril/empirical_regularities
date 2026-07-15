from __future__ import annotations

import unittest
import math
from fractions import Fraction

import torch
from numpy.polynomial import Polynomial

from simple_CP_empirical_prop import (
    CPTensor,
    OrdinaryCPConfig,
    WeightedCPSource,
    build_diagram_catalog,
    compress_cp_sources,
    linear_cp,
    mean_and_variance,
    ordinary_cp_mlp,
    rank_from_delta,
    sample_diagram_cp,
)
from simple_CP_empirical_prop.combinatorics import set_partitions
from simple_CP_empirical_prop.diagrams import DiagramSpec
from simple_CP_empirical_prop.hermite import poly_wick_coef
from simple_CP_empirical_prop.initialization import initialization_sources


class OrdinaryCPTests(unittest.TestCase):
    def test_cp_entry_dense_mean_and_diag(self) -> None:
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 7.0], [11.0, 13.0]])
        cp = CPTensor((a, b))
        self.assertTrue(torch.allclose(cp.entry((1, 0)), torch.tensor((3.0 * 5.0 + 4.0 * 7.0) / 2.0)))
        dense = cp.debug_to_dense()
        self.assertTrue(torch.allclose(dense[1, 0], cp.entry((1, 0))))
        expected_diag = torch.tensor([(1 * 5 + 2 * 7) / 2, (3 * 11 + 4 * 13) / 2])
        self.assertTrue(torch.allclose(cp.diagonal_order2(), expected_diag))
        mean = CPTensor((a,))
        self.assertTrue(torch.allclose(mean.mean_vector(), torch.tensor([1.5, 3.5])))

    def test_rank_from_delta_uses_ceil(self) -> None:
        self.assertEqual(rank_from_delta(5, 0.21), 2)
        self.assertEqual(rank_from_delta(5, 0.01), 1)

    def test_compression_scaling_one_factor_and_rank(self) -> None:
        source = CPTensor((torch.tensor([[1.0, 3.0]]),))
        gen = torch.Generator(device="cpu").manual_seed(0)
        out = compress_cp_sources(
            [WeightedCPSource(Fraction(-2, 1), source)],
            rank=8,
            generator=gen,
        )
        self.assertEqual(out.rank, 8)
        self.assertEqual(out.order, 1)
        self.assertLessEqual(set(out.factors[0].flatten().tolist()), {-2.0, -6.0})

    def test_order_two_grouped_initialization_identity(self) -> None:
        grouped = torch.tensor([[[1.0, 2.0], [4.0, -1.0]]])
        sources = initialization_sources(grouped, order=2)
        dense = torch.zeros(2, 2)
        for weighted in sources:
            factors = weighted.source.gather_columns(torch.tensor([0]))
            dense = dense + float(weighted.coefficient) * (factors[0][:, 0, None] * factors[1][:, 0])
        diff = grouped[0, 0] - grouped[0, 1]
        expected = 0.5 * diff[:, None] * diff[None, :]
        self.assertTrue(torch.allclose(dense, expected))

    def test_linear_bias_only_order_one(self) -> None:
        factors = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        )
        cp2 = CPTensor(factors)
        weight = torch.tensor([[2.0, -1.0], [0.5, 1.0], [1.0, 0.0]])
        bias = torch.tensor([10.0, 20.0, 30.0])
        out2 = linear_cp(cp2, weight, bias)
        self.assertTrue(torch.allclose(out2.factors[0], weight @ factors[0]))
        cp1 = CPTensor((factors[0],))
        out1 = linear_cp(cp1, weight, bias)
        self.assertTrue(torch.allclose(out1.factors[0], weight @ factors[0] + bias[:, None]))

    def test_mean_variance_does_not_subtract_mean_square(self) -> None:
        tower = {
            1: CPTensor((torch.tensor([[2.0, 4.0]]),)),
            2: CPTensor((torch.tensor([[3.0, 5.0]]), torch.tensor([[7.0, 11.0]]))),
        }
        mv = mean_and_variance(tower, OrdinaryCPConfig(k_max=2, delta=1.0))
        self.assertTrue(torch.allclose(mv.mean, torch.tensor([3.0])))
        expected_var = torch.tensor([(3.0 * 7.0 + 5.0 * 11.0) / 2.0])
        self.assertTrue(torch.allclose(mv.variance, expected_var))

    def test_polynomial_wick_square_and_cube(self) -> None:
        mean = torch.tensor([0.5, -1.0], dtype=torch.float64)
        var = torch.tensor([2.0, 0.25], dtype=torch.float64)
        square = Polynomial([0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(poly_wick_coef(square, mean, var, 0), mean**2 + var))
        self.assertTrue(torch.allclose(poly_wick_coef(square, mean, var, 1), 2 * mean))
        self.assertTrue(torch.allclose(poly_wick_coef(square, mean, var, 2), torch.full_like(mean, 2.0)))
        cube = Polynomial([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(poly_wick_coef(cube, mean, var, 0), mean**3 + 3 * mean * var))
        self.assertTrue(torch.allclose(poly_wick_coef(cube, mean, var, 1), 3 * (mean**2 + var)))
        self.assertTrue(torch.allclose(poly_wick_coef(cube, mean, var, 2), 6 * mean))
        self.assertTrue(torch.allclose(poly_wick_coef(cube, mean, var, 3), torch.full_like(mean, 6.0)))

    def test_catalog_empty_diagram_only_order_one(self) -> None:
        cat1 = build_diagram_catalog(1, 2, 0)
        cat2 = build_diagram_catalog(2, 2, 0)
        self.assertEqual(len(cat1), 1)
        self.assertEqual(cat1[0].blocks, ())
        self.assertEqual(len(cat2), 0)

    def test_catalog_excludes_local_singleton_and_pair_blocks(self) -> None:
        order_one_degree_two = build_diagram_catalog(1, 2, 2)
        self.assertEqual(len(order_one_degree_two), 1)
        self.assertEqual(order_one_degree_two[0].blocks, ())
        order_two_degree_one = build_diagram_catalog(2, 2, 1)
        self.assertEqual(len(order_two_degree_one), 1)
        self.assertEqual(order_two_degree_one[0].blocks, ((1, 1),))

    def test_diagram_occurrences_are_sampled_independently(self) -> None:
        spec = DiagramSpec(
            output_order=1,
            hermite_degrees=(0,),
            blocks=((1,), (1,)),
            coefficient=Fraction(1, 1),
            block_orders=(1, 1),
            owners_by_block=((0,), (0,)),
        )
        source = CPTensor((torch.tensor([[1.0, 3.0]]),))
        tower = {1: source}
        wick = {0: torch.ones(1)}
        cp = sample_diagram_cp(
            spec,
            tower,
            wick,
            rank=20_000,
            generator=torch.Generator(device="cpu").manual_seed(123),
        )
        self.assertAlmostEqual(cp.entry((0,)).item(), 4.0, delta=0.08)

    def test_end_to_end_square_smoke(self) -> None:
        torch.manual_seed(0)
        samples = torch.randn(16, 3, dtype=torch.float64)
        mlp = torch.nn.Module()
        mlp.Ws = torch.nn.ModuleList(
            [
                torch.nn.Linear(3, 4, dtype=torch.float64),
                torch.nn.Linear(4, 2, dtype=torch.float64),
            ]
        )
        mlp.nonlin_names = ["square"]
        config = OrdinaryCPConfig(
            k_max=2,
            delta=0.5,
            polynomial_by_layer=(Polynomial([0.0, 0.0, 1.0]),),
            reference_two_stage_nonlinear=True,
        )
        result = ordinary_cp_mlp(mlp, samples, k_max=2, delta=0.5, config=config, seed=7)
        self.assertEqual(tuple(result.mean.shape), (2,))
        self.assertEqual(result.final_tower[1].rank, 8)
        self.assertTrue(torch.isfinite(result.mean).all())

    def test_linear_only_analytic_flop_count(self) -> None:
        samples = torch.randn(8, 3, dtype=torch.float64)
        mlp = torch.nn.Module()
        mlp.Ws = torch.nn.ModuleList([torch.nn.Linear(3, 2, dtype=torch.float64)])
        config = OrdinaryCPConfig(k_max=2, delta=0.5)
        result = ordinary_cp_mlp(mlp, samples, k_max=2, delta=0.5, config=config, seed=1)
        # rank = 4. Initialization: K * width * rank = 2 * 3 * 4 = 24.
        # Linear: (1 + 2) factors, each [2,3] @ [3,4] costs 2*4*(2*3-1)=40,
        # plus order-one bias additions 2*4 = 8, so 128.
        # Final mean extraction: output_width * rank = 2 * 4 = 8.
        self.assertEqual(result.diagnostics.analytic_flops_by_stage["initialization"], 24.0)
        self.assertEqual(result.diagnostics.analytic_flops_by_stage["layer_0_linear"], 128.0)
        self.assertEqual(result.diagnostics.analytic_flops_by_stage["final_mean"], 8.0)
        self.assertEqual(result.diagnostics.total_analytic_flops, 160.0)

    def test_scalar_two_square_layers_need_fourth_cumulant(self) -> None:
        xs = [-1.0, 2.0]
        ps = [0.3, 0.7]

        def raw_moment(order: int) -> float:
            return sum(p * x**order for p, x in zip(ps, xs))

        def cumulant(order: int) -> float:
            total = 0.0
            for part in set_partitions(order):
                mu = (-1) ** (len(part) - 1) * math.factorial(len(part) - 1)
                prod = 1.0
                for block in part:
                    prod *= raw_moment(len(block))
                total += mu * prod
            return total

        def retained_square_update(tower: dict[int, float], k_max: int) -> dict[int, float]:
            mean = torch.tensor([tower[1]], dtype=torch.float64)
            var = torch.tensor([tower[2]], dtype=torch.float64)
            poly = Polynomial([0.0, 0.0, 1.0])
            wick = {k: poly_wick_coef(poly, mean, var, k).item() for k in range(3)}
            new = {}
            for order in range(1, k_max + 1):
                out = 0.0
                for spec in build_diagram_catalog(order, k_max, 2):
                    term = float(spec.coefficient)
                    for degree in spec.hermite_degrees:
                        term *= wick[degree]
                    for block_order in spec.block_orders:
                        term *= tower[block_order]
                    out += term
                new[order] = out
            return new

        true_final_mean = raw_moment(4)
        outputs = {}
        for k_max in (2, 3, 4):
            tower = {order: cumulant(order) for order in range(1, k_max + 1)}
            tower = retained_square_update(tower, k_max)
            tower = retained_square_update(tower, k_max)
            outputs[k_max] = tower[1]

        self.assertGreater(abs(outputs[2] - true_final_mean), 1.0)
        self.assertGreater(abs(outputs[3] - true_final_mean), 1.0)
        self.assertAlmostEqual(outputs[4], true_final_mean, places=12)


if __name__ == "__main__":
    unittest.main()
