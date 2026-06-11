from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {
    "sampling": "#1f77b4",
    "K4 old": "#d62728",
    "K4 direct-Z": "#2ca02c",
}


@dataclass(frozen=True)
class Point:
    label: str
    samples: float
    flops: float
    error: float


def _positive_float(value: str | None) -> float | None:
    if not value:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _row_flops(row: dict[str, str]) -> float | None:
    total = _positive_float(row.get("total_flops"))
    if total is not None:
        return total
    initialization = _positive_float(row.get("initialization_flops"))
    propagation = _positive_float(row.get("propagation_flops"))
    if initialization is not None and propagation is not None:
        return initialization + propagation
    return initialization


def _read_sampling(result_dir: Path) -> list[Point]:
    points: list[Point] = []
    with (result_dir / "results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            samples = _positive_float(row.get("m"))
            flops = _positive_float(row.get("forward_flops"))
            error = _positive_float(row.get("mean_squared_error"))
            if samples is None or flops is None or error is None:
                continue
            points.append(Point(label="sampling", samples=samples, flops=flops, error=error))
    return sorted(points, key=lambda point: point.samples)


def _read_k4(result_dir: Path) -> list[Point]:
    method_labels = {
        "unknown_a": "K4 old",
        "unknown_a_old": "K4 old",
        "unknown_a_direct": "K4 direct-Z",
    }
    points: list[Point] = []
    with (result_dir / "cumulant_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") not in method_labels:
                continue
            if int(row["cumulant_k_max"]) != 4 or not row.get("sample_count"):
                continue
            samples = _positive_float(row.get("sample_count"))
            flops = _row_flops(row)
            error = _positive_float(row.get("squared_error"))
            if samples is None or flops is None or error is None:
                continue
            points.append(
                Point(
                    label=method_labels[row["method"]],
                    samples=samples,
                    flops=flops,
                    error=error,
                )
            )
    return sorted(points, key=lambda point: (point.label, point.samples))


def _log_bounds(values: list[float]) -> tuple[float, float]:
    logs = [math.log10(value) for value in values if value > 0.0 and math.isfinite(value)]
    if not logs:
        raise ValueError("Need positive finite values for log bounds.")
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
    step = max(1, math.ceil((max_exp - min_exp + 1) / 7))
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

    width, height = 1040, 700
    left, right = 118, 42
    top, bottom = 120, 88
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

    parts.extend(
        [
            f'<text class="axis-label" x="{left + plot_width / 2:.2f}" y="{height - 28:.2f}" text-anchor="middle">{escape(xlabel)}</text>',
            f'<text class="axis-label" transform="translate(30 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>',
        ]
    )

    legend_x = left + 18
    legend_y = 86
    for index, label in enumerate(["sampling", "K4 old", "K4 direct-Z"]):
        color = COLORS[label]
        x = legend_x + index * 220
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{legend_y:.2f}" x2="{x + 38:.2f}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x + 19:.2f}" cy="{legend_y:.2f}" r="4.2" fill="{color}" stroke="white" stroke-width="1"/>',
                f'<text class="legend" x="{x + 50:.2f}" y="{legend_y + 4:.2f}">{escape(label)}</text>',
            ]
        )

    for label in ["sampling", "K4 old", "K4 direct-Z"]:
        series = [point for point in points if point.label == label]
        if not series:
            continue
        color = COLORS[label]
        drawn = [(sx(x_value(point)), sy(y_value(point))) for point in series]
        if len(drawn) >= 2:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{_polyline(drawn)}"/>'
            )
        for x, y in drawn:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.3" fill="{color}" stroke="white" stroke-width="1"/>'
            )

    parts.append("</svg>")
    output.write_text("\n".join(parts))


def write_plots(*, result_dir: Path, output_dir: Path, title_prefix: str) -> list[Path]:
    sampling = _read_sampling(result_dir)
    k4_points = _read_k4(result_dir)
    points = [*sampling, *k4_points]
    labels = {point.label for point in points}
    missing = {"sampling", "K4 old", "K4 direct-Z"} - labels
    if missing:
        raise ValueError(f"Missing required series: {', '.join(sorted(missing))}")

    subtitle = "Log-log axes; FLOPs use total_flops when present, otherwise initialization_flops."
    outputs = [
        output_dir / "k4_factorization_error_vs_flops.svg",
        output_dir / "k4_factorization_error_vs_samples.svg",
        output_dir / "k4_factorization_samples_vs_flops.svg",
    ]
    _draw_svg(
        output=outputs[0],
        title=f"{title_prefix}: Error vs FLOPs",
        subtitle=subtitle,
        xlabel="FLOPs",
        ylabel="squared error",
        points=points,
        x_value=lambda point: point.flops,
        y_value=lambda point: point.error,
        x_tick_kind="flops",
        y_tick_kind="error",
    )
    _draw_svg(
        output=outputs[1],
        title=f"{title_prefix}: Error vs Samples",
        subtitle=subtitle,
        xlabel="samples m",
        ylabel="squared error",
        points=points,
        x_value=lambda point: point.samples,
        y_value=lambda point: point.error,
        x_tick_kind="samples",
        y_tick_kind="error",
    )
    _draw_svg(
        output=outputs[2],
        title=f"{title_prefix}: Samples vs FLOPs",
        subtitle=subtitle,
        xlabel="FLOPs",
        ylabel="samples m",
        points=points,
        x_value=lambda point: point.flops,
        y_value=lambda point: point.samples,
        x_tick_kind="flops",
        y_tick_kind="samples",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="Unknown-A ICA K4 factorization")
    args = parser.parse_args()

    outputs = write_plots(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        title_prefix=args.title_prefix,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
