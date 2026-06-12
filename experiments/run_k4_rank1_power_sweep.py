from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from data_generators import make_data_generator
from experiments.mlp_mean_concentration import (
    _cumulant_dtype,
    _sync_if_cuda,
    cumulant_propagation_mean,
    stream_mlp_mean,
)
from experiments.run_k4_rank_truncation_sweep import (
    _cumulants_from_rank_cap,
    _make_args,
    _stream_covariance,
    _stream_ica_samples,
)
from inference_models import DeepReLUMLP


def _matrix_to_pair_vector(matrix: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    values = matrix[rows, cols].clone()
    offdiag = rows != cols
    if bool(offdiag.any()):
        values[offdiag] *= math.sqrt(2.0)
    return values


def _pair_vector_to_matrix(vector: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, q: int) -> torch.Tensor:
    matrix = torch.zeros((q, q), device=vector.device, dtype=vector.dtype)
    offdiag = rows != cols
    matrix[rows[~offdiag], cols[~offdiag]] = vector[~offdiag]
    if bool(offdiag.any()):
        offdiag_values = vector[offdiag] / math.sqrt(2.0)
        matrix[rows[offdiag], cols[offdiag]] = offdiag_values
        matrix[cols[offdiag], rows[offdiag]] = offdiag_values
    return matrix


def _apply_pair_gram(
    vector: torch.Tensor,
    *,
    data_generator,
    subspace: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    total_samples: int,
    seed: int,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    q = int(subspace.shape[1])
    matrix = _pair_vector_to_matrix(vector, rows, cols, q)
    accum = torch.zeros((q, q), device=subspace.device, dtype=dtype)
    for samples in _stream_ica_samples(
        data_generator,
        total_samples=total_samples,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    ):
        coordinates = samples @ subspace
        weights = (coordinates @ matrix * coordinates).sum(dim=1)
        accum.add_(torch.einsum("b,bi,bj->ij", weights, coordinates, coordinates))
    accum.div_(float(total_samples))
    return _matrix_to_pair_vector(accum, rows, cols)


def _stream_pair_top_eigen_power(
    data_generator,
    *,
    total_samples: int,
    seed: int,
    covariance: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    eig_tol: float,
    iterations: int,
    init_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    eigvals, subspace = torch.linalg.eigh(covariance)
    keep_subspace = eigvals > eig_tol * eigvals.max().clamp_min(1.0)
    subspace = subspace[:, keep_subspace]
    q = int(subspace.shape[1])
    if q == 0:
        empty_values = torch.empty((0,), device=data_generator.device, dtype=dtype)
        empty_vectors = torch.empty((0, 0), device=data_generator.device, dtype=dtype)
        empty_indices = torch.empty((0,), device=data_generator.device, dtype=torch.long)
        return subspace, empty_indices, empty_indices, empty_values, empty_vectors, 0.0

    rows, cols = torch.triu_indices(q, q, device=data_generator.device)
    pair_dim = int(rows.numel())
    generator = torch.Generator(device=data_generator.device)
    generator.manual_seed(int(init_seed))
    vector = torch.randn(pair_dim, device=data_generator.device, dtype=dtype, generator=generator)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(torch.finfo(dtype).eps)

    last_delta = math.inf
    for _ in range(int(iterations)):
        applied = _apply_pair_gram(
            vector,
            data_generator=data_generator,
            subspace=subspace,
            rows=rows,
            cols=cols,
            total_samples=total_samples,
            seed=seed,
            batch_size=batch_size,
            dtype=dtype,
        )
        norm = torch.linalg.vector_norm(applied).clamp_min(torch.finfo(dtype).eps)
        next_vector = applied / norm
        # Sign-insensitive convergence diagnostic.
        delta = min(
            torch.linalg.vector_norm(next_vector - vector).item(),
            torch.linalg.vector_norm(next_vector + vector).item(),
        )
        last_delta = delta
        vector = next_vector

    applied = _apply_pair_gram(
        vector,
        data_generator=data_generator,
        subspace=subspace,
        rows=rows,
        cols=cols,
        total_samples=total_samples,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    )
    eigenvalue = torch.dot(vector, applied).clamp_min(0.0)
    return subspace, rows, cols, eigenvalue[None], vector[:, None], last_delta


def _read_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row["sample_k"]) for row in csv.DictReader(handle)}


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "rank_label",
        "sample_k",
        "sample_count",
        "squared_error",
        "log_squared_error",
        "elapsed_seconds",
        "covariance_seconds",
        "power_seconds",
        "propagation_seconds",
        "power_iterations",
        "power_delta",
        "subspace_rank",
        "available_pair_dim",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--true-samples", type=int, default=2**30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--cumulant-batch-size", type=int, default=8192)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--sample-fourth-eig-tol", type=float, default=1e-6)
    parser.add_argument("--power-iterations", type=int, default=8)
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--cumulant-sample-seed-base", type=int, default=2_000_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--true-mean-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rank1_power_results.csv"
    done = _read_done(csv_path)

    device = torch.device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)
    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)

    if args.true_mean_path is not None and args.true_mean_path.exists():
        true_mean = torch.load(args.true_mean_path, map_location=device).to(device=device, dtype=torch.float64)
        print(f"loaded true mean from {args.true_mean_path}", flush=True)
    else:
        true_mean_result = stream_mlp_mean(
            model=model,
            data_generator=data_generator,
            total_samples=args.true_samples,
            batch_size=args.batch_size,
            seed_base=args.true_seed_base,
        )
        true_mean = true_mean_result.mean
        torch.save(true_mean.detach().cpu(), args.output_dir / "true_mean.pt")
        print(
            f"computed true mean from {args.true_samples} samples "
            f"(forward={true_mean_result.forward_seconds:.2f}s)",
            flush=True,
        )

    print(
        f"rank-1 power sweep n={args.n} p={args.p} L={args.depth} "
        f"k={args.k_min}..{args.k_max} iterations={args.power_iterations}",
        flush=True,
    )
    for sample_k in range(args.k_min, args.k_max + 1):
        if sample_k in done:
            continue
        sample_count = 2**sample_k
        seed = args.cumulant_sample_seed_base + sample_k * args.seed_stride
        _sync_if_cuda(device)
        start = time.time()
        cov_start = time.time()
        covariance = _stream_covariance(
            data_generator,
            total_samples=sample_count,
            seed=seed,
            batch_size=args.cumulant_batch_size,
            dtype=dtype,
        )
        _sync_if_cuda(device)
        covariance_seconds = time.time() - cov_start

        power_start = time.time()
        subspace, rows, cols, pair_eigvals, pair_eigvecs, power_delta = _stream_pair_top_eigen_power(
            data_generator,
            total_samples=sample_count,
            seed=seed,
            covariance=covariance,
            batch_size=args.cumulant_batch_size,
            dtype=dtype,
            eig_tol=args.sample_fourth_eig_tol,
            iterations=args.power_iterations,
            init_seed=seed + 17,
        )
        _sync_if_cuda(device)
        power_seconds = time.time() - power_start
        cumulants, actual_rank = _cumulants_from_rank_cap(
            data_generator=data_generator,
            total_samples=sample_count,
            covariance=covariance,
            subspace=subspace,
            rows=rows,
            cols=cols,
            pair_eigvals=pair_eigvals,
            pair_eigvecs=pair_eigvecs,
            rank_cap=1,
            dtype=dtype,
        )
        prop_start = time.time()
        mean = cumulant_propagation_mean(
            model=model,
            cumulants=cumulants,
            cumulant_k_max=4,
            factor=True,
            device=device,
            dtype=dtype,
        )
        _sync_if_cuda(device)
        propagation_seconds = time.time() - prop_start
        squared_error = torch.sum((mean - true_mean) ** 2).item()
        elapsed_seconds = time.time() - start
        row = {
            "method": "structured_cp4_rank1_power",
            "rank_label": "r=1 power",
            "sample_k": sample_k,
            "sample_count": sample_count,
            "squared_error": squared_error,
            "log_squared_error": math.log(squared_error),
            "elapsed_seconds": elapsed_seconds,
            "covariance_seconds": covariance_seconds,
            "power_seconds": power_seconds,
            "propagation_seconds": propagation_seconds,
            "power_iterations": args.power_iterations,
            "power_delta": power_delta,
            "subspace_rank": int(subspace.shape[1]),
            "available_pair_dim": int(rows.numel()),
        }
        _append_row(csv_path, row)
        print(
            f"k={sample_k} m={sample_count} log_sq_error={math.log(squared_error): .6f} "
            f"cov={covariance_seconds:.2f}s power={power_seconds:.2f}s "
            f"prop={propagation_seconds:.2f}s delta={power_delta:.3e}",
            flush=True,
        )
        del covariance, subspace, rows, cols, pair_eigvals, pair_eigvecs, cumulants, mean
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
