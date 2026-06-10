from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {
    8: "#1b9e77",
    16: "#d95f02",
    32: "#7570b3",
    64: "#e7298a",
    128: "#66a61e",
    256: "#e6ab02",
}


@dataclass(frozen=True)
class ErrorSeries:
    p: int
    color: str
    cumulants: list[tuple[int, float, float]]
    sampling: list[tuple[float, float]]


@dataclass(frozen=True)
class RankSeries:
    p: int
    color: str
    rows: list[tuple[int, float]]


def _read_cumulants(path: Path, p: int) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "known_distribution" or row["sample_count"]:
                continue
            k = int(row["cumulant_k_max"])
            flops = float(row["total_flops"])
            error = float(row["squared_error"]) / float(p)
            if math.isfinite(flops) and math.isfinite(error) and flops > 0 and error > 0:
                points.append((k, flops, error))
    return sorted(points)


def _read_sampling(path: Path, p: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            flops = float(row["forward_flops"])
            error = float(row["mean_squared_error"]) / float(p)
            if math.isfinite(flops) and math.isfinite(error) and flops > 0 and error > 0:
                points.append((flops, error))
    return points


def _read_error_series(result_dir: Path, p: int) -> ErrorSeries:
    cumulants = _read_cumulants(result_dir / "cumulant_results.csv", p)
    sampling = _read_sampling(result_dir / "results.csv", p)
    if len(cumulants) < 2:
        raise ValueError(f"Expected at least 2 finite cumulant points for p={p} in {result_dir}")
    if len(cumulants) != 4:
        observed = ",".join(str(k) for k, _, _ in cumulants)
        print(
            f"warning: p={p} has finite cumulant points only for K={observed}; "
            "non-finite points will be omitted.",
            flush=True,
        )
    if not sampling:
        raise ValueError(f"No sampling points for p={p} in {result_dir}")
    return ErrorSeries(
        p=p,
        color=COLORS.get(p, "#333333"),
        cumulants=cumulants,
        sampling=sampling,
    )


def _read_rank_series(result_dir: Path, p: int) -> RankSeries:
    rows: list[tuple[int, float]] = []
    with (result_dir / "results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["layer"]), float(row["effective_rank"])))
    if not rows:
        raise ValueError(f"No rank rows for p={p} in {result_dir}")
    return RankSeries(p=p, color=COLORS.get(p, "#333333"), rows=rows)


def _log_bounds(values: list[float], padding_fraction: float = 0.08) -> tuple[float, float]:
    logs = [math.log10(value) for value in values if value > 0]
    lo, hi = min(logs), max(logs)
    padding = padding_fraction * max(hi - lo, 1.0)
    return lo - padding, hi + padding


def _linear_bounds(values: list[float], padding_fraction: float = 0.08) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    padding = padding_fraction * max(hi - lo, 1.0)
    return lo - padding, hi + padding


def _log_ticks(bounds: tuple[float, float]) -> list[tuple[float, str]]:
    lo, hi = bounds
    return [(10.0**e, f"10^{e}") for e in range(math.ceil(lo), math.floor(hi) + 1)]


def _linear_ticks(bounds: tuple[float, float], count: int) -> list[float]:
    lo, hi = bounds
    if count <= 1 or lo == hi:
        return [lo]
    step = (hi - lo) / float(count - 1)
    return [lo + i * step for i in range(count)]


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _parse_dir_override(value: str) -> tuple[int, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("override must be formatted as p:path")
    p_text, path_text = value.split(":", 1)
    if not p_text or not path_text:
        raise argparse.ArgumentTypeError("override must include both p and path")
    return int(p_text), Path(path_text)


def write_error_svg(series: list[ErrorSeries], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    x_values = [
        flops
        for item in series
        for flops in (
            [point[1] for point in item.cumulants]
            + [point[0] for point in item.sampling]
        )
    ]
    y_values = [
        error
        for item in series
        for error in (
            [point[2] for point in item.cumulants]
            + [point[1] for point in item.sampling]
        )
    ]
    x_bounds = _log_bounds(x_values)
    y_bounds = _log_bounds(y_values)

    width, height = 1120, 760
    left, right = 112, 250
    top, bottom = 72, 98
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
        ".axis-label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #526071; }",
        ".legend { font-size: 13px; fill: #263447; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.25; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.2f}" y="38" text-anchor="middle">ICA known A, L=8, n=256: error / p versus FLOPs</text>',
        f'<line class="axis" x1="{left:.2f}" y1="{top + plot_height:.2f}" x2="{left + plot_width:.2f}" y2="{top + plot_height:.2f}"/>',
        f'<line class="axis" x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_height:.2f}"/>',
    ]

    for value, label in _log_ticks(x_bounds):
        x = sx(value)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + plot_height:.2f}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{top + plot_height:.2f}" x2="{x:.2f}" y2="{top + plot_height + 6:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 26:.2f}" text-anchor="middle">{escape(label)}</text>',
            ]
        )
    for value, label in _log_ticks(y_bounds):
        y = sy(value)
        parts.extend(
            [
                f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" x2="{left + plot_width:.2f}" y2="{y:.2f}"/>',
                f'<line class="axis" x1="{left - 6:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 12:.2f}" y="{y + 4:.2f}" text-anchor="end">{escape(label)}</text>',
            ]
        )

    for item in series:
        cumulant_points = [(sx(flops), sy(error)) for _, flops, error in item.cumulants]
        sampling_points = [(sx(flops), sy(error)) for flops, error in item.sampling]
        segment: list[tuple[float, float]] = []
        previous_k: int | None = None
        for (k, _, _), point in zip(item.cumulants, cumulant_points):
            if previous_k is None or k == previous_k + 1:
                segment.append(point)
            else:
                if len(segment) > 1:
                    parts.append(
                        f'<polyline fill="none" stroke="{item.color}" stroke-width="3" points="{_polyline(segment)}"/>'
                    )
                segment = [point]
            previous_k = k
        if len(segment) > 1:
            parts.append(
                f'<polyline fill="none" stroke="{item.color}" stroke-width="3" points="{_polyline(segment)}"/>'
            )
        parts.append(
            f'<polyline fill="none" stroke="{item.color}" stroke-width="2.4" stroke-dasharray="8 7" points="{_polyline(sampling_points)}"/>'
        )
        for (k, _, _), (x, y) in zip(item.cumulants, cumulant_points):
            parts.extend(
                [
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.8" fill="{item.color}" stroke="white" stroke-width="1.5"/>',
                    f'<text class="tick" x="{x + 7:.2f}" y="{y - 7:.2f}">K={k}</text>',
                ]
            )

    legend_x = left + plot_width + 38
    legend_y = top + 20
    parts.append(
        f'<text class="legend" x="{legend_x:.2f}" y="{legend_y:.2f}" style="font-weight:700">p values</text>'
    )
    for index, item in enumerate(series):
        y = legend_y + 28 * (index + 1)
        parts.extend(
            [
                f'<line x1="{legend_x:.2f}" y1="{y:.2f}" x2="{legend_x + 34:.2f}" y2="{y:.2f}" stroke="{item.color}" stroke-width="3"/>',
                f'<line x1="{legend_x + 78:.2f}" y1="{y:.2f}" x2="{legend_x + 112:.2f}" y2="{y:.2f}" stroke="{item.color}" stroke-width="2.4" stroke-dasharray="8 7"/>',
                f'<text class="legend" x="{legend_x + 124:.2f}" y="{y + 4:.2f}">p={item.p}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="tick" x="{legend_x:.2f}" y="{legend_y + 28 * (len(series) + 2):.2f}">solid: cum. prop. K=1..4</text>',
            f'<text class="tick" x="{legend_x:.2f}" y="{legend_y + 28 * (len(series) + 2) + 20:.2f}">dashed: sampling</text>',
            f'<text class="tick" x="{legend_x:.2f}" y="{legend_y + 28 * (len(series) + 2) + 40:.2f}">non-finite K points omitted</text>',
            f'<text class="axis-label" x="{left + plot_width / 2:.2f}" y="{height - 32:.2f}" text-anchor="middle">FLOPs</text>',
            f'<text class="axis-label" transform="translate(32 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">squared error / p</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts))


def write_rank_svg(series: list[RankSeries], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(layer) for item in series for layer, _ in item.rows]
    ys = [rank for item in series for _, rank in item.rows]
    x_bounds = (min(xs), max(xs))
    y_bounds = _linear_bounds([0.0, *ys], padding_fraction=0.06)

    width, height = 1040, 700
    left, right = 96, 210
    top, bottom = 68, 92
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        lo, hi = x_bounds
        return left + (value - lo) / (hi - lo) * plot_width

    def sy(value: float) -> float:
        lo, hi = y_bounds
        return top + (hi - value) / (hi - lo) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }",
        ".title { font-size: 26px; font-weight: 700; }",
        ".axis-label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #526071; }",
        ".legend { font-size: 13px; fill: #263447; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.25; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.2f}" y="38" text-anchor="middle">ICA rank proxy by layer, L=8, n=256</text>',
        f'<line class="axis" x1="{left:.2f}" y1="{top + plot_height:.2f}" x2="{left + plot_width:.2f}" y2="{top + plot_height:.2f}"/>',
        f'<line class="axis" x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_height:.2f}"/>',
    ]
    for layer in range(int(x_bounds[0]), int(x_bounds[1]) + 1):
        x = sx(float(layer))
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + plot_height:.2f}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{top + plot_height:.2f}" x2="{x:.2f}" y2="{top + plot_height + 6:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 25:.2f}" text-anchor="middle">{layer}</text>',
            ]
        )
    for value in _linear_ticks(y_bounds, 7):
        y = sy(value)
        parts.extend(
            [
                f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" x2="{left + plot_width:.2f}" y2="{y:.2f}"/>',
                f'<line class="axis" x1="{left - 6:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 12:.2f}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>',
            ]
        )

    for item in series:
        points = [(sx(float(layer)), sy(rank)) for layer, rank in item.rows]
        parts.append(
            f'<polyline fill="none" stroke="{item.color}" stroke-width="3" points="{_polyline(points)}"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{item.color}" stroke="white" stroke-width="1.2"/>'
            )

    legend_x = left + plot_width + 36
    legend_y = top + 20
    parts.append(
        f'<text class="legend" x="{legend_x:.2f}" y="{legend_y:.2f}" style="font-weight:700">p values</text>'
    )
    for index, item in enumerate(series):
        y = legend_y + 28 * (index + 1)
        parts.extend(
            [
                f'<line x1="{legend_x:.2f}" y1="{y:.2f}" x2="{legend_x + 34:.2f}" y2="{y:.2f}" stroke="{item.color}" stroke-width="3"/>',
                f'<circle cx="{legend_x + 17:.2f}" cy="{y:.2f}" r="4.5" fill="{item.color}" stroke="white" stroke-width="1.2"/>',
                f'<text class="legend" x="{legend_x + 46:.2f}" y="{y + 4:.2f}">p={item.p}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="axis-label" x="{left + plot_width / 2:.2f}" y="{height - 32:.2f}" text-anchor="middle">layer</text>',
            f'<text class="axis-label" transform="translate(30 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">effective rank R_i</text>',
            f'<text class="tick" x="{left + plot_width / 2:.2f}" y="{height - 10:.2f}" text-anchor="middle">R_i = (sum_j sigma_j)^2 / sum_j sigma_j^2; batch size 8192</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ICA p-sweep error and rank proxy.")
    parser.add_argument("--p-values", default="8,16,32,64,128,256")
    parser.add_argument(
        "--mean-template",
        default="results/mlp_mean_ica_N01_known_sampling_n256_L8_p{p}_k15_3tau_k3",
    )
    parser.add_argument(
        "--rank-template",
        default="results/mlp_effective_rank_ica_N01_n256_L8_p{p}",
    )
    parser.add_argument(
        "--mean-dir",
        action="append",
        type=_parse_dir_override,
        default=[],
        help="Override one mean result directory, formatted as p:path.",
    )
    parser.add_argument(
        "--rank-dir",
        action="append",
        type=_parse_dir_override,
        default=[],
        help="Override one rank result directory, formatted as p:path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ica_p_sweep_n256_L8"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_values = [int(part) for part in args.p_values.split(",") if part.strip()]
    mean_overrides = dict(args.mean_dir)
    rank_overrides = dict(args.rank_dir)
    error_series = [
        _read_error_series(
            mean_overrides.get(p, Path(args.mean_template.format(p=p))),
            p,
        )
        for p in p_values
    ]
    rank_series = [
        _read_rank_series(
            rank_overrides.get(p, Path(args.rank_template.format(p=p))),
            p,
        )
        for p in p_values
    ]
    write_error_svg(error_series, args.output_dir / "error_over_p_vs_flops.svg")
    write_rank_svg(rank_series, args.output_dir / "rank_proxy_vs_layer.svg")
    print(f"wrote {args.output_dir / 'error_over_p_vs_flops.svg'}", flush=True)
    print(f"wrote {args.output_dir / 'rank_proxy_vs_layer.svg'}", flush=True)


if __name__ == "__main__":
    main()
