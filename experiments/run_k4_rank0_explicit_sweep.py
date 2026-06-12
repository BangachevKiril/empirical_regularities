from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from data_generators import make_data_generator
from experiments.mlp_mean_concentration import (
    _cumulant_dtype,
    _sync_if_cuda,
    cumulant_propagation_mean,
    stream_mlp_mean,
)
from experiments.run_k4_rank_truncation_sweep import _make_args, _stream_covariance
from inference_models import DeepReLUMLP


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
        "propagation_seconds",
        "actual_rank",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rank0_input_cumulants(
    *,
    n: int,
    p: int,
    sample_count: int,
    covariance: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, object]:
    eye = torch.eye(n, device=device, dtype=dtype)
    m = int(sample_count)

    second_a = float(p * (p - 1)) / float(m + p - 1)
    second_b = float(m) / float(m + p - 1)
    cumulants: dict[int, object] = {
        1: torch.zeros(n, device=device, dtype=dtype),
        2: second_a * eye + second_b * covariance,
    }

    lambda2 = float(m) / float(m + p - 1)
    gamma = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
    beta = lambda2 - float(p) * gamma
    alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

    # Rank-0 / vector-factorized K=4 input: keep only the analytic correction
    # factors. The sample fourth-moment CP4 term that would be decomposed into
    # rank-1, rank-n, or exact dense/vector factors is intentionally omitted.
    const_left = (-6.0 * alpha * eye)[:, :, None]
    const_right = eye[:, :, None]
    square_left = (-12.0 * beta * covariance)[:, :, None]
    square_right = eye[:, :, None]
    cumulants[4] = FactoredTensor4(
        n=n,
        factors=(
            torch.cat((const_left, square_left), dim=2),
            torch.cat((const_right, square_right), dim=2),
        ),
        device=device,
        dtype=dtype,
        assume_symmetric=True,
    )
    return cumulants


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
    csv_path = args.output_dir / "rank0_results.csv"
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
        f"explicit rank-0/vector-factorized sweep n={args.n} p={args.p} "
        f"L={args.depth} k={args.k_min}..{args.k_max}",
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

        cumulants = rank0_input_cumulants(
            n=args.n,
            p=args.p,
            sample_count=sample_count,
            covariance=covariance,
            device=device,
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
        _append_row(
            csv_path,
            {
                "method": "structured_cp4_rank0_explicit",
                "rank_label": "r=0 explicit",
                "sample_k": sample_k,
                "sample_count": sample_count,
                "squared_error": squared_error,
                "log_squared_error": math.log(squared_error),
                "elapsed_seconds": elapsed_seconds,
                "covariance_seconds": covariance_seconds,
                "propagation_seconds": propagation_seconds,
                "actual_rank": 0,
            },
        )
        print(
            f"k={sample_k} m={sample_count} log_sq_error={math.log(squared_error): .6f} "
            f"cov={covariance_seconds:.2f}s prop={propagation_seconds:.2f}s",
            flush=True,
        )
        del covariance, cumulants, mean
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
