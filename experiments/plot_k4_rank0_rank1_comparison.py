from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import experiments.plot_k4_rank1_power_comparison as base
from experiments.plot_k4_rank1_power_comparison import Point


def _rank0_propagation_flops(*, n: int, depth: int) -> int:
    n_i = int(n)
    depth_i = int(depth)
    # Same calibration procedure as the rank-1 plotter, but with the CP4 sample
    # component set to rank zero. Fitted to NamedFlopCounter propagation counts
    # at n=8,12,16,24,32 with depth=2.
    return int(
        round(
            depth_i
            * (
                41.32005590 * n_i**4
                + 1890.75995195 * n_i**3
                + 3120.63095111 * n_i**2
            )
        )
    )


def _rank0_no_data_flops(*, n: int, depth: int, sample_count: int) -> int:
    n_i = int(n)
    m_i = int(sample_count)
    covariance = 2 * m_i * n_i * n_i
    correction_setup = 8 * n_i * n_i
    propagation = _rank0_propagation_flops(n=n_i, depth=depth)
    return covariance + correction_setup + propagation


def _read_rank0(rank0_dir: Path, *, n: int, depth: int, k_min: int, k_max: int) -> list[Point]:
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
                    "rank-0 K=4",
                    sample_k,
                    samples,
                    float(_rank0_no_data_flops(n=n, depth=depth, sample_count=int(samples))),
                    error,
                )
            )
    return sorted(points, key=lambda point: point.sample_k)


def _read_rank1(rank1_dir: Path, *, n: int, depth: int, k_min: int, k_max: int) -> list[Point]:
    return [
        Point("rank-1 K=4", point.sample_k, point.samples, point.flops, point.error)
        for point in base._read_rank1_power(rank1_dir, n=n, depth=depth, k_min=k_min, k_max=k_max)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--rank0-dir", type=Path, required=True)
    parser.add_argument("--rank1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--true-samples", type=int, default=2**30)
    args = parser.parse_args()

    base.COLORS = {
        "sampling": "#1f77b4",
        "rank-0 K=4": "#7f7f7f",
        "rank-1 K=4": "#d62728",
    }
    base.SERIES_ORDER = ["sampling", "rank-0 K=4", "rank-1 K=4"]

    points = []
    points.extend(base._read_sampling(args.sampling_dir, k_min=args.k_min, k_max=args.k_max))
    points.extend(_read_rank0(args.rank0_dir, n=args.n, depth=args.depth, k_min=args.k_min, k_max=args.k_max))
    points.extend(_read_rank1(args.rank1_dir, n=args.n, depth=args.depth, k_min=args.k_min, k_max=args.k_max))
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
