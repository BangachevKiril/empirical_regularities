from __future__ import annotations

import argparse

import torch

from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from data_generators import make_data_generator
from experimental_rank0_optimal_cumulants.initialization import rank0_input_cumulants
from experimental_rank0_optimal_cumulants.reduced_k4 import patched
from experimental_rank0_optimal_cumulants.run_comparison import reduced_cumulant_propagation_mean
from experimental_rank0_optimized.rank0_optimized_sweep import _stream_covariance_rank0
from experiments.mlp_mean_concentration import _cumulant_dtype
from experiments.run_k4_rank_truncation_sweep import _make_args
from inference_models import DeepReLUMLP


def _canonical_device(device: torch.device | str) -> torch.device:
    return torch.empty((), device=device).device


def _dense_J(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eye = torch.eye(n, device=device, dtype=dtype)
    return (
        torch.einsum("ij,kl->ijkl", eye, eye)
        + torch.einsum("ik,jl->ijkl", eye, eye)
        + torch.einsum("il,jk->ijkl", eye, eye)
    )


def _dense_S_I_M(M: torch.Tensor) -> torch.Tensor:
    n = M.shape[0]
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    return (
        torch.einsum("ij,kl->ijkl", eye, M)
        + torch.einsum("ik,jl->ijkl", eye, M)
        + torch.einsum("il,jk->ijkl", eye, M)
        + torch.einsum("jk,il->ijkl", eye, M)
        + torch.einsum("jl,ik->ijkl", eye, M)
        + torch.einsum("kl,ij->ijkl", eye, M)
    )


def test_initializer(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(args.mlp_seed)
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)
    covariance = _stream_covariance_rank0(
        data_generator,
        total_samples=args.sample_count,
        seed=args.cumulant_seed,
        batch_size=args.batch_size,
        dtype=dtype,
    )
    cumulants = rank0_input_cumulants(
        n=args.n,
        p=args.p,
        sample_count=args.sample_count,
        covariance=covariance,
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(args.n, device=device, dtype=dtype)
    m = int(args.sample_count)
    a_star = float(args.p * (args.p - 1)) / float(m + args.p - 1)
    b_star = float(m) / float(m + args.p - 1)
    expected_k2 = a_star * eye + b_star * covariance
    torch.testing.assert_close(cumulants[2], expected_k2, rtol=args.rtol, atol=args.atol)

    lambda2 = float(m) / float(m + args.p - 1)
    gamma = float(m) / float(args.p**3 + (m - 1) * (3 * args.p - 2))
    beta = lambda2 - float(args.p) * gamma
    alpha = float(args.p) - 2.0 * float(args.p) * lambda2 + float(args.p * args.p) * gamma
    expected_k4 = -2.0 * alpha * _dense_J(args.n, device=device, dtype=dtype) - 2.0 * beta * _dense_S_I_M(covariance)
    torch.testing.assert_close(cumulants[4].to_tensor(), expected_k4, rtol=args.rtol, atol=args.atol)

    old = FactoredTensor4(
        n=args.n,
        factors=(
            torch.cat(((-6.0 * alpha * eye)[:, :, None], (-12.0 * beta * covariance)[:, :, None]), dim=2),
            torch.cat((eye[:, :, None], eye[:, :, None]), dim=2),
        ),
        device=device,
        dtype=dtype,
        assume_symmetric=True,
    )
    for part in [(4,), (3, 1), (2, 2), (2, 1, 1)]:
        torch.testing.assert_close(old.get_dslice(part), cumulants[4].get_dslice(part), rtol=args.rtol, atol=args.atol)
    print("initializer formula checks ok")


def test_end_to_end(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> None:
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
    cumulants = rank0_input_cumulants(
        n=args.n,
        p=args.p,
        sample_count=args.sample_count,
        covariance=covariance,
        device=device,
        dtype=dtype,
    )
    with patched():
        mean = reduced_cumulant_propagation_mean(
            model=model,
            cumulants=cumulants,
            device=device,
            dtype=dtype,
        )
    assert mean.shape == (args.n,)
    assert torch.isfinite(mean).all()
    print("reduced end-to-end mean ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cumulant-seed", type=int, default=123)
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    device = _canonical_device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)
    test_initializer(args, device=device, dtype=dtype)
    test_end_to_end(args, device=device, dtype=dtype)


if __name__ == "__main__":
    main()
