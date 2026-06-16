from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

import experiments.plot_k4_rank1_power_comparison as plot_base
from data_generators import make_data_generator
from experimental_hard2_edges.optimized_hard2 import (
    expected_grouped_hard2_simple_flops,
    expected_ungrouped_hard2_simple_flops,
    install,
)
from experimental_rank0_optimized.plot_sampling_comparison import _optimized_rank0_flops
from experimental_rank0_optimized.rank0_optimized_sweep import (
    _canonical_device,
    _stream_covariance_rank0,
    optimized_cumulant_propagation_mean,
    rank0_input_cumulants_optimized,
)
from experiments.mlp_mean_concentration import (
    _cumulant_dtype,
    _mlp_forward_flops,
    _sync_if_cuda,
    stream_mlp_mean,
)
from experiments.run_k4_rank_truncation_sweep import _make_args
from inference_models import DeepReLUMLP


def _read_done(path: Path, key: str = "sample_k") -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row[key]) for row in csv.DictReader(handle)}


def _append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _hard2_k4_flops(*, n: int, p: int, depth: int, sample_count: int) -> int:
    base = _optimized_rank0_flops(n=n, p=p, depth=depth, sample_count=sample_count)
    hard2_savings = (
        max(int(depth) - 1, 0)
        * (
            expected_ungrouped_hard2_simple_flops(n)
            - expected_grouped_hard2_simple_flops(n)
        )
    )
    return max(1, int(base) - int(hard2_savings))


def _load_or_compute_true_mean(
    *,
    path: Path,
    model: DeepReLUMLP,
    data_generator,
    true_samples: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
) -> torch.Tensor:
    if path.exists():
        true_mean = torch.load(path, map_location=device).to(device=device, dtype=torch.float64)
        print(f"loaded true mean from {path}", flush=True)
        return true_mean

    result = stream_mlp_mean(
        model=model,
        data_generator=data_generator,
        total_samples=true_samples,
        batch_size=batch_size,
        seed_base=seed_base,
    )
    true_mean = result.mean
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(true_mean.detach().cpu(), path)
    print(
        f"computed true mean from {true_samples} samples "
        f"(forward={result.forward_seconds:.2f}s)",
        flush=True,
    )
    return true_mean


def _run_k4(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    data_generator,
    true_mean: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    csv_path: Path,
) -> None:
    fieldnames = [
        "method",
        "sample_k",
        "sample_count",
        "squared_error",
        "log_squared_error",
        "flops",
        "elapsed_seconds",
        "covariance_seconds",
        "propagation_seconds",
    ]
    done = _read_done(csv_path)
    for sample_k in range(args.k_min, args.k_max + 1):
        if sample_k in done:
            continue
        sample_count = 2**sample_k
        seed = args.cumulant_sample_seed_base + sample_k * args.seed_stride
        _sync_if_cuda(device)
        start = time.time()
        cov_start = time.time()
        covariance = _stream_covariance_rank0(
            data_generator,
            total_samples=sample_count,
            seed=seed,
            batch_size=args.cumulant_batch_size,
            dtype=dtype,
        )
        _sync_if_cuda(device)
        covariance_seconds = time.time() - cov_start

        cumulants = rank0_input_cumulants_optimized(
            n=args.n,
            p=args.p,
            sample_count=sample_count,
            covariance=covariance,
            device=device,
            dtype=dtype,
        )
        prop_start = time.time()
        mean = optimized_cumulant_propagation_mean(
            model=model,
            cumulants=cumulants,
            device=device,
            dtype=dtype,
        )
        _sync_if_cuda(device)
        propagation_seconds = time.time() - prop_start
        squared_error = torch.sum((mean - true_mean) ** 2).item()
        elapsed_seconds = time.time() - start
        _append_row(
            csv_path,
            fieldnames,
            {
                "method": "hard2_k4",
                "sample_k": sample_k,
                "sample_count": sample_count,
                "squared_error": squared_error,
                "log_squared_error": math.log(squared_error),
                "flops": _hard2_k4_flops(
                    n=args.n,
                    p=args.p,
                    depth=args.depth,
                    sample_count=sample_count,
                ),
                "elapsed_seconds": elapsed_seconds,
                "covariance_seconds": covariance_seconds,
                "propagation_seconds": propagation_seconds,
            },
        )
        print(
            f"K=4 k={sample_k} m={sample_count} "
            f"log_sq_error={math.log(squared_error): .6f} "
            f"cov={covariance_seconds:.2f}s prop={propagation_seconds:.2f}s",
            flush=True,
        )
        del covariance, cumulants, mean
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_sampling(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    data_generator,
    true_mean: torch.Tensor,
    device: torch.device,
    csv_path: Path,
) -> None:
    fieldnames = [
        "method",
        "k",
        "m",
        "mean_squared_error",
        "log_mean_squared_error",
        "forward_flops",
        "elapsed_seconds",
        "forward_seconds",
    ]
    done = _read_done(csv_path, key="k")
    for sample_k in range(args.k_min, args.k_max + 1):
        if sample_k in done:
            continue
        sample_count = 2**sample_k
        seed_base = args.sampling_seed_base + sample_k * args.seed_stride
        _sync_if_cuda(device)
        result = stream_mlp_mean(
            model=model,
            data_generator=data_generator,
            total_samples=sample_count,
            batch_size=args.batch_size,
            seed_base=seed_base,
        )
        _sync_if_cuda(device)
        squared_error = torch.sum((result.mean - true_mean) ** 2).item()
        _append_row(
            csv_path,
            fieldnames,
            {
                "method": "sampling",
                "k": sample_k,
                "m": sample_count,
                "mean_squared_error": squared_error,
                "log_mean_squared_error": math.log(squared_error),
                "forward_flops": _mlp_forward_flops(
                    n=args.n,
                    depth=args.depth,
                    sample_count=sample_count,
                ),
                "elapsed_seconds": result.elapsed_seconds,
                "forward_seconds": result.forward_seconds,
            },
        )
        print(
            f"sampling k={sample_k} m={sample_count} "
            f"log_sq_error={math.log(squared_error): .6f} "
            f"forward={result.forward_seconds:.2f}s",
            flush=True,
        )
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _read_points(*, sampling_csv: Path, k4_csv: Path, k_min: int, k_max: int) -> list[plot_base.Point]:
    points: list[plot_base.Point] = []
    with sampling_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = int(row["k"])
            if k_min <= sample_k <= k_max:
                points.append(
                    plot_base.Point(
                        "sampling",
                        sample_k,
                        float(row["m"]),
                        float(row["forward_flops"]),
                        float(row["mean_squared_error"]),
                    )
                )
    with k4_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = int(row["sample_k"])
            if k_min <= sample_k <= k_max:
                points.append(
                    plot_base.Point(
                        "K=4",
                        sample_k,
                        float(row["sample_count"]),
                        float(row["flops"]),
                        float(row["squared_error"]),
                    )
                )
    return sorted(points, key=lambda point: (point.label, point.sample_k))


def _plot(*, args: argparse.Namespace, output_dir: Path, sampling_csv: Path, k4_csv: Path) -> None:
    plot_base.COLORS = {
        "sampling": "#1f77b4",
        "K=4": "#9467bd",
    }
    plot_base.SERIES_ORDER = ["sampling", "K=4"]
    points = _read_points(
        sampling_csv=sampling_csv,
        k4_csv=k4_csv,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_base._write_points(points, output_dir / "plot_points.csv")
    subtitle = (
        f"ICA unknown A, n={args.n}, p={args.p}, L={args.depth}, "
        f"m=2^{args.k_min}..2^{args.k_max}, truth=2^30"
    )
    plot_base._draw_svg(
        output=output_dir / "flops_vs_error.svg",
        title="Error vs FLOPs",
        subtitle=subtitle,
        xlabel="FLOPs (log scale; p-dependent data generation excluded)",
        ylabel="Squared error",
        points=points,
        x_value=lambda point: point.flops,
        y_value=lambda point: point.error,
        x_tick_kind="flops",
        y_tick_kind="error",
    )
    plot_base._draw_svg(
        output=output_dir / "samples_vs_error.svg",
        title="Error vs Samples",
        subtitle=subtitle,
        xlabel="Samples m",
        ylabel="Squared error",
        points=points,
        x_value=lambda point: point.samples,
        y_value=lambda point: point.error,
        x_tick_kind="samples",
        y_tick_kind="error",
    )
    plot_base._draw_svg(
        output=output_dir / "samples_vs_flops.svg",
        title="FLOPs vs Samples",
        subtitle=subtitle,
        xlabel="Samples m",
        ylabel="FLOPs",
        points=points,
        x_value=lambda point: point.samples,
        y_value=lambda point: point.flops,
        x_tick_kind="samples",
        y_tick_kind="flops",
    )
    print(f"wrote plots to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--true-samples", type=int, default=2**30)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--cumulant-batch-size", type=int, default=65536)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--sampling-seed-base", type=int, default=1_000_000_000)
    parser.add_argument("--cumulant-sample-seed-base", type=int, default=2_000_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--true-mean-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    install()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _canonical_device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)
    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)

    true_mean_path = args.true_mean_path or (args.output_dir / "true_mean.pt")
    true_mean = _load_or_compute_true_mean(
        path=true_mean_path,
        model=model,
        data_generator=data_generator,
        true_samples=args.true_samples,
        batch_size=args.batch_size,
        seed_base=args.true_seed_base,
        device=device,
    )

    k4_csv = args.output_dir / "k4_results.csv"
    sampling_csv = args.output_dir / "sampling_results.csv"
    _run_k4(
        args=args,
        model=model,
        data_generator=data_generator,
        true_mean=true_mean,
        device=device,
        dtype=dtype,
        csv_path=k4_csv,
    )
    _run_sampling(
        args=args,
        model=model,
        data_generator=data_generator,
        true_mean=true_mean,
        device=device,
        csv_path=sampling_csv,
    )
    _plot(
        args=args,
        output_dir=args.output_dir / "plots",
        sampling_csv=sampling_csv,
        k4_csv=k4_csv,
    )


if __name__ == "__main__":
    main()
