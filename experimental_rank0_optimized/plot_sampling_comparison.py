from __future__ import annotations

import argparse
import csv
from pathlib import Path

import experiments.plot_k4_rank1_power_comparison as base
from experiments.plot_k4_rank1_power_comparison import Point


# Per-layer fit from NamedFlopCounter measurements of the isolated optimized
# path at n=8,12,16,24,32 with L=4. The measured total coefficients were
# (271.58959943, 6797.28838770, 11240.84700444), divided by 4 here.
OPT_PROP_COEFFS = (67.8973998575, 1699.322096925, 2810.21175111)


def _optimized_rank0_propagation_flops(*, n: int, depth: int) -> int:
    n_i = int(n)
    depth_i = int(depth)
    a, b, c = OPT_PROP_COEFFS
    return int(round(depth_i * (a * n_i**4 + b * n_i**3 + c * n_i**2)))


def _rank0_covariance_flops(*, n: int, p: int, sample_count: int) -> int:
    n_i = int(n)
    p_i = int(p)
    m_i = int(sample_count)
    direct_cost = m_i * (p_i * n_i + n_i * n_i)
    source_cost = m_i * p_i * p_i + n_i * p_i * p_i + n_i * n_i * p_i
    if source_cost >= direct_cost:
        return 2 * m_i * n_i * n_i
    return 2 * m_i * p_i * p_i + 2 * n_i * p_i * p_i + 2 * n_i * n_i * p_i


def _optimized_rank0_flops(*, n: int, p: int, depth: int, sample_count: int) -> int:
    n_i = int(n)
    construction = _rank0_covariance_flops(n=n, p=p, sample_count=sample_count) + 2 * n_i * n_i + 2 * n_i
    return construction + _optimized_rank0_propagation_flops(n=n, depth=depth)


def _read_optimized(rank0_dir: Path, *, n: int, p: int, depth: int, k_min: int, k_max: int) -> list[Point]:
    points: list[Point] = []
    with (rank0_dir / "rank0_optimized_results.csv").open(newline="") as handle:
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
                    "optimized rank-0 K=4",
                    sample_k,
                    samples,
                    float(_optimized_rank0_flops(n=n, p=p, depth=depth, sample_count=int(samples))),
                    error,
                )
            )
    return sorted(points, key=lambda point: point.sample_k)


def _read_known_a(known_a_dir: Path) -> list[Point]:
    points: list[Point] = []
    with (known_a_dir / "known_a_k4_point.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            samples = base._positive_float(row.get("sample_count"))
            flops = base._positive_float(row.get("total_flops"))
            error = base._positive_float(row.get("squared_error"))
            if samples is None or flops is None or error is None:
                continue
            points.append(
                Point(
                    "known A K=4",
                    int(row.get("sample_k") or 0),
                    samples,
                    flops,
                    error,
                )
            )
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--known-a-dir", type=Path, default=None)
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
        "optimized rank-0 K=4": "#9467bd",
        "known A K=4": "#111827",
    }
    base.SERIES_ORDER = ["sampling", "optimized rank-0 K=4", "known A K=4"]

    points = []
    points.extend(base._read_sampling(args.sampling_dir, k_min=args.k_min, k_max=args.k_max))
    points.extend(
        _read_optimized(
            args.optimized_dir,
            n=args.n,
            p=args.p,
            depth=args.depth,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    )
    if args.known_a_dir is not None:
        points.extend(_read_known_a(args.known_a_dir))
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
