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
from experiments.mlp_mean_concentration import (
    _cumulant_dtype,
    _sync_if_cuda,
    cumulant_propagation_mean,
    stream_mlp_mean,
)
from inference_models import DeepReLUMLP


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
    covariance_sum = torch.zeros((n, n), device=data_generator.device, dtype=dtype)
    for samples in _stream_ica_samples(
        data_generator,
        total_samples=total_samples,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    ):
        covariance_sum.add_(samples.T @ samples)
    return covariance_sum / float(total_samples)


def _stream_pair_eigendecomposition(
    data_generator,
    *,
    total_samples: int,
    seed: int,
    covariance: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    eig_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eigvals, subspace = torch.linalg.eigh(covariance)
    keep_subspace = eigvals > eig_tol * eigvals.max().clamp_min(1.0)
    subspace = subspace[:, keep_subspace]
    q = int(subspace.shape[1])
    if q == 0:
        empty_values = torch.empty((0,), device=data_generator.device, dtype=dtype)
        empty_vectors = torch.empty((0, 0), device=data_generator.device, dtype=dtype)
        empty_indices = torch.empty((0,), device=data_generator.device, dtype=torch.long)
        return subspace, empty_indices, empty_indices, empty_values, empty_vectors

    rows, cols = torch.triu_indices(q, q, device=data_generator.device)
    pair_dim = int(rows.numel())
    pair_gram = torch.zeros((pair_dim, pair_dim), device=data_generator.device, dtype=dtype)
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
    order = torch.argsort(pair_eigvals, descending=True)
    return subspace, rows, cols, pair_eigvals[order], pair_eigvecs[:, order]


def _s4_factors_from_truncated_pair_eig(
    *,
    subspace: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    pair_eigvals: torch.Tensor,
    pair_eigvecs: torch.Tensor,
    coefficient: float,
    rank_cap: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    n = int(subspace.shape[0])
    q = int(subspace.shape[1])
    rank = min(int(rank_cap), int(pair_eigvals.numel()))
    dtype = subspace.dtype
    device = subspace.device
    if rank == 0:
        empty = torch.zeros((n, n, 0), device=device, dtype=dtype)
        return empty, empty, 0

    coordinate_basis = torch.zeros((rank, q, q), device=device, dtype=dtype)
    basis_coefficients = pair_eigvecs[:, :rank].T
    offdiag = rows != cols
    coordinate_basis[:, rows[~offdiag], cols[~offdiag]] = basis_coefficients[:, ~offdiag]
    if bool(offdiag.any()):
        offdiag_coefficients = basis_coefficients[:, offdiag] / math.sqrt(2.0)
        coordinate_basis[:, rows[offdiag], cols[offdiag]] = offdiag_coefficients
        coordinate_basis[:, cols[offdiag], rows[offdiag]] = offdiag_coefficients

    lifted = torch.einsum("ia,rab,jb->rij", subspace, coordinate_basis, subspace)
    left = coefficient * pair_eigvals[:rank, None, None] * lifted
    right = lifted
    return left.permute(1, 2, 0), right.permute(1, 2, 0), rank


def _cumulants_from_rank_cap(
    *,
    data_generator,
    total_samples: int,
    covariance: torch.Tensor,
    subspace: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    pair_eigvals: torch.Tensor,
    pair_eigvecs: torch.Tensor,
    rank_cap: int,
    dtype: torch.dtype,
) -> tuple[dict[int, object], int]:
    n = data_generator.n
    p = data_generator.p
    device = data_generator.device
    eye = torch.eye(n, device=device, dtype=dtype)

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
    s4_left, s4_right, actual_rank = _s4_factors_from_truncated_pair_eig(
        subspace=subspace,
        rows=rows,
        cols=cols,
        pair_eigvals=pair_eigvals,
        pair_eigvecs=pair_eigvecs,
        coefficient=-2.0 * gamma,
        rank_cap=rank_cap,
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
    return cumulants, actual_rank


def _read_done(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            done.add((int(row["sample_k"]), row["rank_label"]))
    return done


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "rank_label",
        "rank_cap",
        "actual_rank",
        "sample_k",
        "sample_count",
        "squared_error",
        "log_squared_error",
        "elapsed_seconds",
        "stream_setup_seconds",
        "propagation_seconds",
        "available_rank",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _rank_caps(n: int) -> list[tuple[str, int]]:
    return [("r=1", 1), ("r=n", int(n)), ("r=n^2", int(n) * int(n))]


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
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--cumulant-sample-seed-base", type=int, default=2_000_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rank_truncation_results.csv"
    done = _read_done(csv_path)

    device = torch.device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)
    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)

    print(f"device={device}", flush=True)
    print(f"n={args.n} p={args.p} L={args.depth} k={args.k_min}..{args.k_max}", flush=True)
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

    rank_caps = _rank_caps(args.n)
    for sample_k in range(args.k_min, args.k_max + 1):
        pending = [(label, cap) for label, cap in rank_caps if (sample_k, label) not in done]
        if not pending:
            continue
        sample_count = 2**sample_k
        seed = args.cumulant_sample_seed_base + sample_k * args.seed_stride
        _sync_if_cuda(device)
        setup_start = time.time()
        covariance = _stream_covariance(
            data_generator,
            total_samples=sample_count,
            seed=seed,
            batch_size=args.cumulant_batch_size,
            dtype=dtype,
        )
        subspace, rows, cols, pair_eigvals, pair_eigvecs = _stream_pair_eigendecomposition(
            data_generator,
            total_samples=sample_count,
            seed=seed,
            covariance=covariance,
            batch_size=args.cumulant_batch_size,
            dtype=dtype,
            eig_tol=args.sample_fourth_eig_tol,
        )
        _sync_if_cuda(device)
        setup_seconds = time.time() - setup_start
        available_rank = int(pair_eigvals.numel())
        print(
            f"k={sample_k} m={sample_count} available_rank={available_rank} "
            f"setup={setup_seconds:.2f}s",
            flush=True,
        )

        rows_out: list[dict[str, object]] = []
        for rank_label, rank_cap in pending:
            _sync_if_cuda(device)
            start = time.time()
            cumulants, actual_rank = _cumulants_from_rank_cap(
                data_generator=data_generator,
                total_samples=sample_count,
                covariance=covariance,
                subspace=subspace,
                rows=rows,
                cols=cols,
                pair_eigvals=pair_eigvals,
                pair_eigvecs=pair_eigvecs,
                rank_cap=rank_cap,
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
            rows_out.append(
                {
                    "method": "structured_cp4_truncated",
                    "rank_label": rank_label,
                    "rank_cap": rank_cap,
                    "actual_rank": actual_rank,
                    "sample_k": sample_k,
                    "sample_count": sample_count,
                    "squared_error": squared_error,
                    "log_squared_error": math.log(squared_error),
                    "elapsed_seconds": elapsed_seconds,
                    "stream_setup_seconds": setup_seconds,
                    "propagation_seconds": propagation_seconds,
                    "available_rank": available_rank,
                }
            )
            print(
                f"  {rank_label} actual_rank={actual_rank} "
                f"log_sq_error={math.log(squared_error): .6f} "
                f"prop={propagation_seconds:.2f}s",
                flush=True,
            )
            del cumulants, mean
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _append_rows(csv_path, rows_out)
        done.update((int(row["sample_k"]), str(row["rank_label"])) for row in rows_out)
        del covariance, subspace, rows, cols, pair_eigvals, pair_eigvecs
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
