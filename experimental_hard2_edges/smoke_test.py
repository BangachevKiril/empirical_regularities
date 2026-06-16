from __future__ import annotations

import argparse
from copy import deepcopy

import torch

import cumulant_propagation._arc_mlp_kprop.factor_k4 as factor_k4
from cumulant_propagation import propagate_cumulants
from cumulant_propagation._arc_mlp_kprop.harmonic import HTensor
from cumulant_propagation._arc_mlp_kprop.wick import relu_wick_coef
from data_generators import make_data_generator
from experimental_hard2_edges.optimized_hard2 import (
    build_optimized_factored_nonlin_kprop_k4,
    expected_grouped_hard2_simple_flops,
    expected_ungrouped_hard2_simple_flops,
    install,
    uninstall,
)
from experimental_rank0_optimized.rank0_optimized_sweep import (
    _stream_covariance_rank0,
    optimized_cumulant_propagation_mean,
    rank0_input_cumulants_optimized,
)
from experiments.mlp_mean_concentration import _cumulant_dtype
from experiments.run_k4_rank_truncation_sweep import _make_args
from inference_models import DeepReLUMLP


def _canonical_device(device: torch.device | str) -> torch.device:
    return torch.empty((), device=device).device


def _clone_tower(tower: dict[int, object]) -> dict[int, object]:
    return {degree: value.clone() for degree, value in tower.items()}


def _assert_object_close(actual: object, expected: object, *, rtol: float, atol: float) -> None:
    if hasattr(actual, "to_tensor") and hasattr(expected, "to_tensor"):
        torch.testing.assert_close(actual.to_tensor(), expected.to_tensor(), rtol=rtol, atol=atol)
        return
    if isinstance(actual, HTensor) and isinstance(expected, HTensor):
        torch.testing.assert_close(actual.core, expected.core, rtol=rtol, atol=atol)
        if isinstance(actual.metric, torch.Tensor) or isinstance(expected.metric, torch.Tensor):
            torch.testing.assert_close(actual.metric, expected.metric, rtol=rtol, atol=atol)
        return
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return
    raise TypeError(f"Unsupported comparison types: {type(actual)!r}, {type(expected)!r}")


def _make_problem(args: argparse.Namespace, *, dtype: torch.dtype):
    device = _canonical_device(args.device)
    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)
    covariance = _stream_covariance_rank0(
        data_generator,
        total_samples=args.sample_count,
        seed=args.cumulant_seed,
        batch_size=args.batch_size,
        dtype=dtype,
    )
    cumulants = rank0_input_cumulants_optimized(
        n=args.n,
        p=args.p,
        sample_count=args.sample_count,
        covariance=covariance,
        device=device,
        dtype=dtype,
    )
    return device, model, cumulants


def test_full_nonlinear_step(args: argparse.Namespace, *, dtype: torch.dtype) -> None:
    device, model, cumulants = _make_problem(args, dtype=dtype)
    pre0 = propagate_cumulants(
        model,
        cumulants,
        k_max=4,
        factor=True,
        return_tensors=False,
        up_to_layer="pre0",
        output_d_max=4,
        device=device,
        dtype=dtype,
    )

    original = factor_k4.factored_nonlin_kprop_k4
    optimized = build_optimized_factored_nonlin_kprop_k4()
    old_out = original(
        K_in=_clone_tower(pre0),
        nonlin_wick_coef=relu_wick_coef,
        augment=False,
        base=False,
        use_pK=True,
    )
    new_out = optimized(
        K_in=_clone_tower(pre0),
        nonlin_wick_coef=relu_wick_coef,
        augment=False,
        base=False,
        use_pK=True,
    )
    assert set(old_out) == set(new_out)
    for degree in old_out:
        _assert_object_close(new_out[degree], old_out[degree], rtol=args.rtol, atol=args.atol)
    print("full nonlinear step equality ok")


def test_end_to_end_mean(args: argparse.Namespace, *, dtype: torch.dtype) -> None:
    device, model, cumulants = _make_problem(args, dtype=dtype)
    old_mean = optimized_cumulant_propagation_mean(
        model=model,
        cumulants=deepcopy(cumulants),
        device=device,
        dtype=dtype,
    )
    install()
    try:
        new_mean = optimized_cumulant_propagation_mean(
            model=model,
            cumulants=deepcopy(cumulants),
            device=device,
            dtype=dtype,
        )
    finally:
        uninstall()
    torch.testing.assert_close(new_mean, old_mean, rtol=args.rtol, atol=args.atol)
    print("end-to-end mean equality ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cumulant-seed", type=int, default=123)
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    dtype = _cumulant_dtype(args.cumulant_dtype)
    test_full_nonlinear_step(args, dtype=dtype)
    test_end_to_end_mean(args, dtype=dtype)
    print(
        "hard2 flops simple "
        f"grouped={expected_grouped_hard2_simple_flops(args.n)} "
        f"old={expected_ungrouped_hard2_simple_flops(args.n)}"
    )


if __name__ == "__main__":
    main()
