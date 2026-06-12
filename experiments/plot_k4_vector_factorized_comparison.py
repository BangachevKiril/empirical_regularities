from __future__ import annotations

import argparse
import csv
from pathlib import Path

import experiments.plot_k4_rank1_power_comparison as base
from experiments.plot_k4_rank0_rank1_comparison import (
    _rank0_no_data_flops,
    _rank0_propagation_flops,
)
from experiments.plot_k4_rank1_power_comparison import Point


def _exact_factorized_no_data_flops(*, n: int, depth: int, sample_count: int) -> int:
    n_i = int(n)
    m_i = int(sample_count)
    construction = 4 * m_i * n_i * n_i + 7 * n_i * n_i
    sample_rank_propagation = int(depth) * m_i * (16 * n_i**3 + 14 * n_i**2 + 2 * n_i)
    return construction + _rank0_propagation_flops(n=n_i, depth=depth) + sample_rank_propagation


def _read_exact_factorized(
    result_dir: Path,
    *,
    n: int,
    depth: int,
    k_min: int,
    k_max: int,
) -> list[Point]:
    points: list[Point] = []
    with (result_dir / "cumulant_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "unknown_a_direct" or int(row["cumulant_k_max"]) != 4:
                continue
            sample_k = int(row["sample_k"])
            if sample_k < k_min or sample_k > k_max:
                continue
            samples = base._positive_float(row.get("sample_count"))
            error = base._positive_float(row.get("squared_error"))
            if samples is None or error is None:
                continue
            points.append(
                Point(
                    "exact factorized K=4",
                    sample_k,
                    samples,
                    float(_exact_factorized_no_data_flops(n=n, depth=depth, sample_count=int(samples))),
                    error,
                )
            )
    return sorted(points, key=lambda point: point.sample_k)


def _read_vector_factorized(
    rank0_dir: Path,
    *,
    n: int,
    depth: int,
    k_min: int,
    k_max: int,
) -> list[Point]:
    points: list[Point] = []
    with (rank0_dir / "rank0_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = int(row["sample_k"])
            if sample_k < k_min or sample_k > k_max:
                continue
            samples = base._positive_float(row.get("sample_count"))
            error = base._positive_float(row.get("squared_error"))
            if samples is None or error is None:
                continue
            points.append(
                Point(
                    "vector-factorized K=4",
                    sample_k,
                    samples,
                    float(_rank0_no_data_flops(n=n, depth=depth, sample_count=int(samples))),
                    error,
                )
            )
    return sorted(points, key=lambda point: point.sample_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--true-samples", type=int, default=2**30)
    args = parser.parse_args()

    base.COLORS = {
        "sampling": "#1f77b4",
        "exact factorized K=4": "#2ca02c",
        "vector-factorized K=4": "#7f7f7f",
    }
    base.SERIES_ORDER = ["sampling", "exact factorized K=4", "vector-factorized K=4"]

    points = []
    points.extend(base._read_sampling(args.result_dir, k_min=args.k_min, k_max=args.k_max))
    points.extend(
        _read_exact_factorized(
            args.result_dir,
            n=args.n,
            depth=args.depth,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    )
    points.extend(
        _read_vector_factorized(
            args.vector_dir,
            n=args.n,
            depth=args.depth,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    )
    points = sorted(points, key=lambda point: (base.SERIES_ORDER.index(point.label), point.sample_k))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base._write_points(points, args.output_dir / "plot_points.csv")

    subtitle = (
        f"ICA unknown A, n={args.n}, p={args.p}, L={args.depth}, "
        f"m=2^{args.k_min}..2^{args.k_max}, truth=2^30"
    )
    base._draw_svg(
        output=args.output_dir / "error_vs_flops.svg",
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
    base._draw_svg(
        output=args.output_dir / "error_vs_samples.svg",
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
    base._draw_svg(
        output=args.output_dir / "flops_vs_samples.svg",
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
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
