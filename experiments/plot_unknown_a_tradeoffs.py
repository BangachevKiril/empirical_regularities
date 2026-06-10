from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {
    "sampling": "#1f77b4",
    1: "#2ca02c",
    2: "#9467bd",
    3: "#ff7f0e",
    4: "#d62728",
}


@dataclass(frozen=True)
class Point:
    label: str
    sample_count: float
    flops: float
    squared_error: float | None
    color: str


def _positive_float(value: str) -> float | None:
    if not value:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _read_sampling(result_dir: Path) -> list[Point]:
    points: list[Point] = []
    with (result_dir / "results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_count = _positive_float(row["m"])
            flops = _positive_float(row["forward_flops"])
            error = _positive_float(row["mean_squared_error"])
            if sample_count is None or flops is None or error is None:
                continue
            points.append(
                Point(
                    label="sampling",
                    sample_count=sample_count,
                    flops=flops,
                    squared_error=error,
                    color=COLORS["sampling"],
                )
            )
    return points


def _cumulant_flops(row: dict[str, str]) -> float | None:
    total = _positive_float(row.get("total_flops", ""))
    if total is not None:
        return total
    initialization = _positive_float(row.get("initialization_flops", ""))
    propagation = _positive_float(row.get("propagation_flops", ""))
    if initialization is not None and propagation is not None:
        return initialization + propagation
    return initialization


def _read_unknown_a(result_dir: Path, method: str) -> dict[int, list[Point]]:
    groups: dict[int, list[Point]] = {1: [], 2: [], 3: [], 4: []}
    with (result_dir / "cumulant_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method or not row["sample_count"]:
                continue
            k_max = int(row["cumulant_k_max"])
            if k_max not in groups:
                continue
            sample_count = _positive_float(row["sample_count"])
            flops = _cumulant_flops(row)
            error = _positive_float(row["squared_error"])
            if sample_count is None or flops is None or error is None:
                continue
            groups[k_max].append(
                Point(
                    label=f"unknown A K={k_max}",
                    sample_count=sample_count,
                    flops=flops,
                    squared_error=error,
                    color=COLORS[k_max],
                )
            )
    return {k: sorted(points, key=lambda point: point.sample_count) for k, points in groups.items()}


def _log_bounds(values: list[float], padding_fraction: float = 0.08) -> tuple[float, float]:
    logs = [math.log10(value) for value in values if value > 0.0 and math.isfinite(value)]
    if not logs:
        raise ValueError("Need at least one positive value for log bounds.")
    lo, hi = min(logs), max(logs)
    padding = padding_fraction * max(hi - lo, 1.0)
    return lo - padding, hi + padding


def _log_ticks(bounds: tuple[float, float]) -> list[tuple[float, str]]:
    lo, hi = bounds
    return [(10.0**exp, f"10^{exp}") for exp in range(math.ceil(lo), math.floor(hi) + 1)]


def _sample_ticks(bounds: tuple[float, float]) -> list[tuple[float, str]]:
    lo, hi = bounds
    min_exp = max(0, math.ceil(lo / math.log10(2.0)))
    max_exp = math.floor(hi / math.log10(2.0))
    ticks: list[tuple[float, str]] = []
    for exp in range(min_exp, max_exp + 1, 2):
        ticks.append((2.0**exp, f"2^{exp}"))
    if not ticks or ticks[-1][0] < 2.0**max_exp:
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
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    series: list[tuple[str, str, list[Point], str, str]],
    x_value,
    y_value,
) -> None:
    width, height = 1040, 700
    left, right = 116, 44
    top, bottom = 126, 90
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
    ]

    legend_items = [(label, color, style) for label, color, _, _, style in series]
    item_widths = [58.0 + max(104.0, 7.1 * len(label)) for label, _, _ in legend_items]
    gap = 18.0
    legend_width = sum(item_widths) + gap * (len(item_widths) - 1)
    cursor = (width - legend_width) / 2
    legend_y = 88.0
    parts.append(
        f'<rect x="{cursor - 14:.2f}" y="{legend_y - 23:.2f}" width="{legend_width + 28:.2f}" height="40" rx="5" fill="#ffffff" stroke="#c8d1df" stroke-width="1"/>'
    )
    for (label, color, style), item_width in zip(legend_items, item_widths):
        dash = ' stroke-dasharray="9 6"' if style == "dash" else ""
        parts.extend(
            [
                f'<line x1="{cursor:.2f}" y1="{legend_y:.2f}" x2="{cursor + 38:.2f}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="3"{dash}/>',
                f'<circle cx="{cursor + 19:.2f}" cy="{legend_y:.2f}" r="4.2" fill="{color}" stroke="white" stroke-width="1"/>',
                f'<text class="legend" x="{cursor + 50:.2f}" y="{legend_y + 4:.2f}">{escape(label)}</text>',
            ]
        )
        cursor += item_width + gap

    parts.extend(
        [
            f'<line class="axis" x1="{left:.2f}" y1="{top + plot_height:.2f}" x2="{left + plot_width:.2f}" y2="{top + plot_height:.2f}"/>',
            f'<line class="axis" x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_height:.2f}"/>',
        ]
    )
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

    parts.extend(
        [
            f'<text class="axis-label" x="{left + plot_width / 2:.2f}" y="{height - 28:.2f}" text-anchor="middle">{escape(xlabel)}</text>',
            f'<text class="axis-label" transform="translate(30 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>',
        ]
    )

    for _, color, points, marker, style in series:
        drawn_points = [
            (sx(x_value(point)), sy(y_value(point)))
            for point in points
            if y_value(point) is not None
        ]
        dash = ' stroke-dasharray="9 6"' if style == "dash" else ""
        if len(drawn_points) >= 2:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash} points="{_polyline(drawn_points)}"/>'
            )
        for x, y in drawn_points:
            if marker == "square":
                parts.append(
                    f'<rect x="{x - 4.2:.2f}" y="{y - 4.2:.2f}" width="8.4" height="8.4" fill="{color}" stroke="white" stroke-width="1"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" stroke="white" stroke-width="1"/>'
                )

    parts.append("</svg>")
    output.write_text("\n".join(parts))


def write_plots(
    *,
    result_dir: Path,
    output_dir: Path,
    title_prefix: str,
    method: str,
) -> list[Path]:
    sampling = _read_sampling(result_dir)
    cumulants_by_k = _read_unknown_a(result_dir, method)
    cumulant_groups = [(k, points) for k, points in sorted(cumulants_by_k.items()) if points]
    if not sampling:
        raise ValueError(f"No sampling rows found in {result_dir / 'results.csv'}")
    if not cumulant_groups:
        raise ValueError(f"No {method!r} sample-cumulant rows found in {result_dir / 'cumulant_results.csv'}")

    all_points = [*sampling, *(point for _, points in cumulant_groups for point in points)]
    subtitle = "Log-log axes; cumulant FLOPs use total_flops when present, otherwise initialization_flops."
    series = [
        ("sampling", COLORS["sampling"], sampling, "circle", "solid"),
        *[
            (f"unknown A K={k}", COLORS[k], points, "square", "dash")
            for k, points in cumulant_groups
        ],
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    error_values = [point.squared_error for point in all_points if point.squared_error is not None]
    flops_bounds = _log_bounds([point.flops for point in all_points])
    samples_bounds = _log_bounds([point.sample_count for point in all_points], padding_fraction=0.04)
    error_bounds = _log_bounds(error_values)

    output = output_dir / "unknown_a_error_vs_flops.svg"
    _draw_svg(
        output=output,
        title=f"{title_prefix}: Error vs FLOPs",
        subtitle=subtitle,
        xlabel="FLOPs",
        ylabel="squared error",
        x_bounds=flops_bounds,
        y_bounds=error_bounds,
        x_ticks=_log_ticks(flops_bounds),
        y_ticks=_log_ticks(error_bounds),
        series=series,
        x_value=lambda point: point.flops,
        y_value=lambda point: point.squared_error,
    )
    outputs.append(output)

    output = output_dir / "unknown_a_samples_vs_flops.svg"
    _draw_svg(
        output=output,
        title=f"{title_prefix}: Samples vs FLOPs",
        subtitle=subtitle,
        xlabel="FLOPs",
        ylabel="sample count m",
        x_bounds=flops_bounds,
        y_bounds=samples_bounds,
        x_ticks=_log_ticks(flops_bounds),
        y_ticks=_sample_ticks(samples_bounds),
        series=series,
        x_value=lambda point: point.flops,
        y_value=lambda point: point.sample_count,
    )
    outputs.append(output)

    output = output_dir / "unknown_a_error_vs_samples.svg"
    _draw_svg(
        output=output,
        title=f"{title_prefix}: Error vs Samples",
        subtitle="Log-log axes; m is the number of samples used by each estimator.",
        xlabel="sample count m",
        ylabel="squared error",
        x_bounds=samples_bounds,
        y_bounds=error_bounds,
        x_ticks=_sample_ticks(samples_bounds),
        y_ticks=_log_ticks(error_bounds),
        series=series,
        x_value=lambda point: point.sample_count,
        y_value=lambda point: point.squared_error,
    )
    outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot unknown-A ICA sample-cumulant tradeoffs.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--title-prefix", required=True)
    parser.add_argument("--method", default="unknown_a")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.result_dir
    outputs = write_plots(
        result_dir=args.result_dir,
        output_dir=output_dir,
        title_prefix=args.title_prefix,
        method=args.method,
    )
    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
