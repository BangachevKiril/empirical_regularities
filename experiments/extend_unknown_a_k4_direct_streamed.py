from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from data_generators import make_data_generator
from data_generators.ica.unknown_parameter_estimation import _symmetric_pair_features
from data_generators.ica.unknown_parameter_estimation_direct_k4 import (
    initialization_flops as direct_initialization_flops,
)
from experiments.mlp_mean_concentration import (
    CumulantResult,
    _cumulant_dtype,
    _sync_if_cuda,
    cumulant_propagation_mean,
    stream_mlp_mean,
    write_cumulant_csv,
)
from inference_models import DeepReLUMLP


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _read_base_direct_results(path: Path, *, max_sample_k: int) -> list[CumulantResult]:
    if not path.exists():
        return []
    results: list[CumulantResult] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = _optional_int(row.get("sample_k"))
            if row.get("method") != "unknown_a_direct" or sample_k is None:
                continue
            if int(row["cumulant_k_max"]) != 4 or sample_k > max_sample_k:
                continue
            results.append(
                CumulantResult(
                    method="unknown_a_direct",
                    cumulant_k_max=4,
                    sample_k=sample_k,
                    sample_count=_optional_int(row.get("sample_count")),
                    squared_error=float(row["squared_error"]),
                    elapsed_seconds=float(row.get("elapsed_seconds") or 0.0),
                    warmup_seconds=float(row.get("warmup_seconds") or 0.0),
                    rank4=_optional_int(row.get("rank4")),
                    initialization_flops=_optional_int(row.get("initialization_flops")),
                    propagation_flops=_optional_int(row.get("propagation_flops")),
                )
            )
    return sorted(results, key=lambda result: result.sample_k or -1)


def _stream_ica_samples(
    data_generator,
    *,
    total_samples: int,
    seed: int,
    batch_size: int,
    dtype: torch.dtype,
):
    generator = data_generator._make_generator(seed)
    samples_seen = 0
    with torch.inference_mode():
        while samples_seen < total_samples:
            current_batch = min(batch_size, total_samples - samples_seen)
            signs = torch.randint(
                low=0,
                high=2,
                size=(current_batch, data_generator.p),
                generator=generator,
                device=data_generator.device,
            )
            sources = signs.to(dtype=dtype).mul_(2).sub_(1)
            yield sources @ data_generator.A.to(dtype=dtype).T
            samples_seen += current_batch


def _stream_covariance(
    data_generator,
    *,
    total_samples: int,
    seed: int,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    n = data_generator.n
    covariance_sum = torch.zeros(
        (n, n),
        device=data_generator.device,
        dtype=dtype,
    )
    for samples in _stream_ica_samples(
        data_generator,
        total_samples=total_samples,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    ):
        covariance_sum.add_(samples.T @ samples)
    return covariance_sum / float(total_samples)


def _stream_compressed_s4_factors(
    data_generator,
    *,
    total_samples: int,
    seed: int,
    covariance: torch.Tensor,
    coefficient: float,
    eig_tol: float,
    batch_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    n = data_generator.n
    eigvals, subspace = torch.linalg.eigh(covariance)
    keep_subspace = eigvals > eig_tol * eigvals.max().clamp_min(1.0)
    subspace = subspace[:, keep_subspace]
    q = int(subspace.shape[1])
    if q == 0:
        empty = torch.zeros((n, n, 0), device=data_generator.device, dtype=dtype)
        return empty, empty, 0

    rows, cols = torch.triu_indices(q, q, device=data_generator.device)
    pair_dim = int(rows.numel())
    pair_gram = torch.zeros(
        (pair_dim, pair_dim),
        device=data_generator.device,
        dtype=dtype,
    )
    for samples in _stream_ica_samples(
        data_generator,
        total_samples=total_samples,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    ):
        coordinates = samples @ subspace
        pair_features = _symmetric_pair_features(coordinates, rows, cols)
        pair_gram.add_(pair_features.T @ pair_features)
    pair_gram.div_(float(total_samples))

    pair_eigvals, pair_eigvecs = torch.linalg.eigh(pair_gram)
    keep = pair_eigvals > eig_tol * pair_eigvals.max().clamp_min(1.0)
    pair_eigvals = pair_eigvals[keep]
    pair_eigvecs = pair_eigvecs[:, keep]
    rank = int(pair_eigvals.numel())
    if rank == 0:
        empty = torch.zeros((n, n, 0), device=data_generator.device, dtype=dtype)
        return empty, empty, 0

    coordinate_basis = torch.zeros(
        (rank, q, q),
        device=data_generator.device,
        dtype=dtype,
    )
    basis_coefficients = pair_eigvecs.T
    offdiag = rows != cols
    coordinate_basis[:, rows[~offdiag], cols[~offdiag]] = basis_coefficients[:, ~offdiag]
    if bool(offdiag.any()):
        offdiag_coefficients = basis_coefficients[:, offdiag] / math.sqrt(2.0)
        coordinate_basis[:, rows[offdiag], cols[offdiag]] = offdiag_coefficients
        coordinate_basis[:, cols[offdiag], rows[offdiag]] = offdiag_coefficients

    lifted = torch.einsum("ia,rab,jb->rij", subspace, coordinate_basis, subspace)
    left = coefficient * pair_eigvals[:, None, None] * lifted
    right = lifted
    return left.permute(1, 2, 0), right.permute(1, 2, 0), rank


def _streamed_k4_cumulants(
    data_generator,
    *,
    total_samples: int,
    seed: int,
    batch_size: int,
    dtype: torch.dtype,
    eig_tol: float,
) -> tuple[dict[int, object], int]:
    n = data_generator.n
    p = data_generator.p
    device = data_generator.device
    eye = torch.eye(n, device=device, dtype=dtype)
    covariance = _stream_covariance(
        data_generator,
        total_samples=total_samples,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    )

    second_a = float(p * (p - 1)) / float(total_samples + p - 1)
    second_b = float(total_samples) / float(total_samples + p - 1)
    cumulants: dict[int, object] = {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: second_a * eye + second_b * covariance,
    }

    lambda2 = float(total_samples) / float(total_samples + p - 1)
    gamma = float(total_samples) / float(p**3 + (total_samples - 1) * (3 * p - 2))
    beta = lambda2 - float(p) * gamma
    alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

    const_left = (-6.0 * alpha * eye)[:, :, None]
    const_right = eye[:, :, None]
    square_left = (-12.0 * beta * covariance)[:, :, None]
    square_right = eye[:, :, None]
    s4_left, s4_right, _compressed_rank = _stream_compressed_s4_factors(
        data_generator,
        total_samples=total_samples,
        seed=seed,
        covariance=covariance,
        coefficient=-2.0 * gamma,
        eig_tol=eig_tol,
        batch_size=batch_size,
        dtype=dtype,
    )
    cumulants[4] = FactoredTensor4(
        n=n,
        factors=(
            torch.cat((const_left, square_left, s4_left), dim=2),
            torch.cat((const_right, square_right, s4_right), dim=2),
        ),
        device=eye.device,
        dtype=dtype,
        assume_symmetric=True,
    )
    return cumulants, int(total_samples)


def _make_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        input_distribution="ica",
        n=args.n,
        p=args.p,
        ica_seed=args.ica_seed,
        gaussian_seed=0,
        lowrank_seed=0,
        subspace_seed=0,
    )


def _run_streamed_k4(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    data_generator,
    true_mean: torch.Tensor,
    sample_k: int,
    dtype: torch.dtype,
) -> CumulantResult:
    sample_count = 2**sample_k
    seed = args.cumulant_sample_seed_base + sample_k * args.seed_stride
    _sync_if_cuda(data_generator.device)
    start = time.time()
    cumulants, direct_rank = _streamed_k4_cumulants(
        data_generator,
        total_samples=sample_count,
        seed=seed,
        batch_size=args.cumulant_batch_size,
        dtype=dtype,
        eig_tol=args.sample_fourth_eig_tol,
    )
    if data_generator.device.type == "cuda":
        torch.cuda.empty_cache()
    mean = cumulant_propagation_mean(
        model=model,
        cumulants=cumulants,
        cumulant_k_max=4,
        factor=args.cumulant_factor,
        device=data_generator.device,
        dtype=dtype,
    )
    _sync_if_cuda(data_generator.device)
    squared_error = torch.sum((mean - true_mean) ** 2).item()
    elapsed = time.time() - start
    del cumulants, mean
    if data_generator.device.type == "cuda":
        torch.cuda.empty_cache()

    return CumulantResult(
        method="unknown_a_direct",
        cumulant_k_max=4,
        sample_k=sample_k,
        sample_count=sample_count,
        squared_error=squared_error,
        elapsed_seconds=elapsed,
        warmup_seconds=0.0,
        rank4=direct_rank,
        initialization_flops=direct_initialization_flops(
            n=args.n,
            p=args.p,
            sample_count=sample_count,
            k_max=4,
            rank4=direct_rank,
        ),
        propagation_flops=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--k-min", type=int, default=21)
    parser.add_argument("--k-max", type=int, default=27)
    parser.add_argument("--true-samples", type=int, default=4_194_304)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--cumulant-batch-size", type=int, default=8192)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--no-cumulant-factor",
        dest="cumulant_factor",
        action="store_false",
        help="disable factored cumulant propagation",
    )
    parser.set_defaults(cumulant_factor=True)
    parser.add_argument("--sample-fourth-eig-tol", type=float, default=1e-6)
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--cumulant-sample-seed-base", type=int, default=2_000_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-cumulant-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)
    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(
        args=_make_args(args),
        device=device,
        dtype=torch.float32,
    )

    print(f"device={device}", flush=True)
    print(f"n={args.n} depth={args.depth} p={args.p}", flush=True)
    print(f"streamed K4 direct-Z-equivalent sample_k={args.k_min}..{args.k_max}", flush=True)
    true_mean_result = stream_mlp_mean(
        model=model,
        data_generator=data_generator,
        total_samples=args.true_samples,
        batch_size=args.batch_size,
        seed_base=args.true_seed_base,
    )
    true_mean = true_mean_result.mean
    print(
        f"computed true mean from {args.true_samples} samples "
        f"(forward={true_mean_result.forward_seconds:.2f}s)",
        flush=True,
    )

    results = _read_base_direct_results(args.base_cumulant_csv, max_sample_k=args.k_min - 1)
    print(f"loaded {len(results)} base direct-Z rows", flush=True)
    write_cumulant_csv(results, args.output_dir / "cumulant_results.csv")

    existing_k = {result.sample_k for result in results}
    for sample_k in range(args.k_min, args.k_max + 1):
        if sample_k in existing_k:
            continue
        result = _run_streamed_k4(
            args=args,
            model=model,
            data_generator=data_generator,
            true_mean=true_mean,
            sample_k=sample_k,
            dtype=dtype,
        )
        results.append(result)
        results.sort(key=lambda item: item.sample_k or -1)
        write_cumulant_csv(results, args.output_dir / "cumulant_results.csv")
        print(
            f"unknown_a_direct K=4 m=2^{sample_k} "
            f"log_sq_error={result.log_squared_error: .6f} "
            f"run={result.elapsed_seconds:.2f}s rank4={result.rank4}",
            flush=True,
        )


if __name__ == "__main__":
    main()
