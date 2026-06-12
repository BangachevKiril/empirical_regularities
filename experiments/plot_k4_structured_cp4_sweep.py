from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {
    "sampling": "#1f77b4",
    "baseline direct K=4": "#2ca02c",
    "structured CP4": "#d62728",
    "structured CP4 r=1": "#9467bd",
    "structured CP4 r=n": "#ff7f0e",
    "structured CP4 r=n^2": "#8c564b",
}
SERIES_ORDER = [
    "sampling",
    "baseline direct K=4",
    "structured CP4",
    "structured CP4 r=1",
    "structured CP4 r=n",
    "structured CP4 r=n^2",
]


@dataclass(frozen=True)
class Point:
    label: str
    sample_k: int
    samples: float
    flops: float
    error: float


def _positive_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _direct_z_no_data_flops(*, n: int, depth: int, sample_count: int) -> int:
    n_i = int(n)
    m_i = int(sample_count)
    construction = 4 * m_i * n_i * n_i + 7 * n_i * n_i
    propagation = int(depth) * m_i * (16 * n_i**3 + 14 * n_i**2 + 2 * n_i)
    return construction + propagation


def _structured_cp4_no_data_flops(*, n: int, depth: int, sample_count: int) -> int:
    n_i = int(n)
    m_i = int(sample_count)
    # Excludes p-dependent ICA sample generation. The first term builds K2 plus
    # the CP4 empirical input/corrections; the second is four CP factors through
    # each dense linear layer. Dense n-only work is left as a conservative n^4
    # layer budget so small-m points do not look artificially free.
    construction = 4 * m_i * n_i * n_i + 11 * n_i * n_i
    cp_linear = int(depth) * m_i * (8 * n_i * n_i + 4 * n_i)
    dense_budget = int(depth) * 16 * n_i**4
    return construction + cp_linear + dense_budget


def _structured_cp4_rank_no_data_flops(
    *,
    n: int,
    depth: int,
    sample_count: int,
    actual_rank: int,
) -> int:
    n_i = int(n)
    m_i = int(sample_count)
    r_i = int(actual_rank)
    construction = 4 * m_i * n_i * n_i + 11 * n_i * n_i
    cp_linear = int(depth) * r_i * (8 * n_i * n_i + 4 * n_i)
    dense_budget = int(depth) * 16 * n_i**4
    return construction + cp_linear + dense_budget


def _read_sampling(result_dir: Path, *, k_min: int, k_max: int) -> list[Point]:
    points: list[Point] = []
    with (result_dir / "results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = int(row["k"])
            if sample_k < k_min or sample_k > k_max:
                continue
            samples = _positive_float(row.get("m"))
            flops = _positive_float(row.get("forward_flops"))
            error = _positive_float(row.get("mean_squared_error"))
            if samples is None or flops is None or error is None:
                continue
            points.append(Point("sampling", sample_k, samples, flops, error))
    return sorted(points, key=lambda point: point.sample_k)


def _read_direct_and_structured(
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
            samples = _positive_float(row.get("sample_count"))
            error = _positive_float(row.get("squared_error"))
            if samples is None or error is None:
                continue
            sample_count = int(samples)
            points.append(
                Point(
                    "baseline direct K=4",
                    sample_k,
                    samples,
                    float(_direct_z_no_data_flops(n=n, depth=depth, sample_count=sample_count)),
                    error,
                )
            )
            points.append(
                Point(
                    "structured CP4",
                    sample_k,
                    samples,
                    float(_structured_cp4_no_data_flops(n=n, depth=depth, sample_count=sample_count)),
                    error,
                )
            )
    return sorted(points, key=lambda point: (point.label, point.sample_k))


def _read_rank_truncation(
    truncation_dir: Path | None,
    *,
    n: int,
    depth: int,
    k_min: int,
    k_max: int,
) -> list[Point]:
    if truncation_dir is None:
        return []
    csv_path = truncation_dir / "rank_truncation_results.csv"
    if not csv_path.exists():
        return []
    label_map = {
        "r=1": "structured CP4 r=1",
        "r=n": "structured CP4 r=n",
        "r=n^2": "structured CP4 r=n^2",
    }
    points: list[Point] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rank_label = row.get("rank_label")
            if rank_label not in label_map:
                continue
            sample_k = int(row["sample_k"])
            if sample_k < k_min or sample_k > k_max:
                continue
            samples = _positive_float(row.get("sample_count"))
            error = _positive_float(row.get("squared_error"))
            actual_rank = _positive_float(row.get("actual_rank"))
            if samples is None or error is None or actual_rank is None:
                continue
            points.append(
                Point(
                    label_map[rank_label],
                    sample_k,
                    samples,
                    float(
                        _structured_cp4_rank_no_data_flops(
                            n=n,
                            depth=depth,
                            sample_count=int(samples),
                            actual_rank=int(actual_rank),
                        )
                    ),
                    error,
                )
            )
    return sorted(points, key=lambda point: (point.label, point.sample_k))


def _write_points(points: list[Point], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "sample_k", "samples", "flops", "error"])
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "label": point.label,
                    "sample_k": point.sample_k,
                    "samples": int(point.samples),
                    "flops": int(point.flops),
                    "error": point.error,
                }
            )


def _log_bounds(values: list[float]) -> tuple[float, float]:
    logs = [math.log10(value) for value in values if value > 0.0 and math.isfinite(value)]
    if not logs:
        raise ValueError("need positive finite values")
    lo, hi = min(logs), max(logs)
    padding = 0.08 * max(hi - lo, 1.0)
    return lo - padding, hi + padding


def _power10_ticks(bounds: tuple[float, float]) -> list[tuple[float, str]]:
    lo, hi = bounds
    return [(10.0**exp, f"10^{exp}") for exp in range(math.ceil(lo), math.floor(hi) + 1)]


def _power2_ticks(bounds: tuple[float, float]) -> list[tuple[float, str]]:
    lo, hi = bounds
    min_exp = max(0, math.ceil(lo / math.log10(2.0)))
    max_exp = math.floor(hi / math.log10(2.0))
    step = max(1, math.ceil((max_exp - min_exp + 1) / 8))
    ticks = [(2.0**exp, f"2^{exp}") for exp in range(min_exp, max_exp + 1, step)]
    if ticks and ticks[-1][0] < 2.0**max_exp:
        ticks.append((2.0**max_exp, f"2^{max_exp}"))
    return ticks


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _draw_svg(
    *,
    output: Path,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    points: list[Point],
    x_value,
    y_value,
    x_tick_kind: str,
    y_tick_kind: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    x_bounds = _log_bounds([x_value(point) for point in points])
    y_bounds = _log_bounds([y_value(point) for point in points])
    x_ticks = _power2_ticks(x_bounds) if x_tick_kind == "samples" else _power10_ticks(x_bounds)
    y_ticks = _power2_ticks(y_bounds) if y_tick_kind == "samples" else _power10_ticks(y_bounds)

    width, height = 1080, 720
    left, right = 118, 46
    top, bottom = 122, 90
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        lo, hi = x_bounds
        return left + (math.log10(value) - lo) / (hi - lo) * plot_width

    def sy(value: float) -> float:
        lo, hi = y_bounds
        return top + (hi - math.log10(value)) / (hi - lo) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }",
        ".title { font-size: 26px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #526071; }",
        ".axis-label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #526071; }",
        ".legend { font-size: 13px; fill: #263447; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.25; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.2f}" y="36" text-anchor="middle">{escape(title)}</text>',
        f'<text class="subtitle" x="{width / 2:.2f}" y="58" text-anchor="middle">{escape(subtitle)}</text>',
        f'<line class="axis" x1="{left:.2f}" y1="{top + plot_height:.2f}" x2="{left + plot_width:.2f}" y2="{top + plot_height:.2f}"/>',
        f'<line class="axis" x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_height:.2f}"/>',
    ]
    for value, label in x_ticks:
        x = sx(value)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + plot_height:.2f}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{top + plot_height:.2f}" x2="{x:.2f}" y2="{top + plot_height + 6:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 25:.2f}" text-anchor="middle">{escape(label)}</text>',
            ]
        )
    for value, label in y_ticks:
        y = sy(value)
        parts.extend(
            [
                f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" x2="{left + plot_width:.2f}" y2="{y:.2f}"/>',
                f'<line class="axis" x1="{left - 6:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 12:.2f}" y="{y + 4:.2f}" text-anchor="end">{escape(label)}</text>',
            ]
        )

    for index, label in enumerate(SERIES_ORDER):
        series = sorted([point for point in points if point.label == label], key=lambda point: point.sample_k)
        if not series:
            continue
        color = COLORS[label]
        coords = [(sx(x_value(point)), sy(y_value(point))) for point in series]
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{_polyline(coords)}"/>'
        )
        for x, y in coords:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
        lx = left + 18 + (index % 3) * 285
        ly = top - 46 + (index // 3) * 23
        parts.extend(
            [
                f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{lx + 34:.2f}" y2="{ly:.2f}" stroke="{color}" stroke-width="2.6"/>',
                f'<circle cx="{lx + 17:.2f}" cy="{ly:.2f}" r="3.6" fill="{color}"/>',
                f'<text class="legend" x="{lx + 44:.2f}" y="{ly + 4:.2f}">{escape(label)}</text>',
            ]
        )

    parts.extend(
        [
            f'<text class="axis-label" x="{left + plot_width / 2:.2f}" y="{height - 30:.2f}" text-anchor="middle">{escape(xlabel)}</text>',
            f'<text class="axis-label" transform="translate(28 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--truncation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--true-samples", type=int, default=2**30)
    parser.add_argument(
        "--include-labels",
        default=None,
        help="Comma-separated series labels to keep, e.g. 'sampling,structured CP4 r=1'.",
    )
    args = parser.parse_args()

    points = _read_sampling(args.result_dir, k_min=args.k_min, k_max=args.k_max)
    points.extend(
        _read_direct_and_structured(
            args.result_dir,
            n=args.n,
            depth=args.depth,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    )
    points.extend(
        _read_rank_truncation(
            args.truncation_dir,
            n=args.n,
            depth=args.depth,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    )
    if args.include_labels:
        keep = {label.strip() for label in args.include_labels.split(",") if label.strip()}
        points = [point for point in points if point.label in keep]
    points = sorted(points, key=lambda point: (SERIES_ORDER.index(point.label), point.sample_k))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_points(points, args.output_dir / "plot_points.csv")

    subtitle = (
        f"ICA unknown A, n={args.n}, p={args.p}, L={args.depth}, "
        f"m=2^{args.k_min}..2^{args.k_max}, truth=2^30"
    )
    _draw_svg(
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
    _draw_svg(
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
    _draw_svg(
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
