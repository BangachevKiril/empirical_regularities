from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {
    "sampling": "#1f77b4",
    1: "#2ca02c",
    2: "#9467bd",
    3: "#ff7f0e",
    4: "#d62728",
}


def _read_sampling(result_dir: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with (result_dir / "results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            flops = float(row["forward_flops"])
            error = float(row["mean_squared_error"])
            if math.isfinite(flops) and math.isfinite(error) and flops > 0 and error > 0:
                points.append((flops, error))
    return points


def _read_known_a_points(result_dir: Path) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    with (result_dir / "cumulant_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "known_distribution" or row["sample_count"]:
                continue
            k = int(row["cumulant_k_max"])
            flops = float(row["total_flops"])
            error = float(row["squared_error"])
            if math.isfinite(flops) and math.isfinite(error) and flops > 0 and error > 0:
                points.append((k, flops, error))
    return sorted(points)


def _log_bounds(values: list[float], padding_fraction: float = 0.08) -> tuple[float, float]:
    logs = [math.log10(value) for value in values if value > 0]
    lo, hi = min(logs), max(logs)
    padding = padding_fraction * max(hi - lo, 1.0)
    return lo - padding, hi + padding


def _ticks(bounds: tuple[float, float]) -> list[tuple[float, str]]:
    lo, hi = bounds
    return [(10.0**e, f"10^{e}") for e in range(math.ceil(lo), math.floor(hi) + 1)]


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg(result_dir: Path, output: Path, title: str) -> None:
    sampling = _read_sampling(result_dir)
    cumulants = _read_known_a_points(result_dir)
    if not sampling:
        raise ValueError(f"No sampling data found in {result_dir / 'results.csv'}")
    if not cumulants:
        raise ValueError(f"No known-A cumulant data found in {result_dir / 'cumulant_results.csv'}")

    x_values = [x for x, _ in sampling] + [x for _, x, _ in cumulants]
    y_values = [y for _, y in sampling] + [y for _, _, y in cumulants]
    x_bounds = _log_bounds(x_values)
    y_bounds = _log_bounds(y_values)

    width, height = 980, 660
    left, right = 112, 44
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
        ".axis-label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #526071; }",
        ".legend { font-size: 13px; fill: #263447; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.25; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.2f}" y="38" text-anchor="middle">{escape(title)}</text>',
    ]

    legend_items = [("sampling baseline", COLORS["sampling"])] + [
        (f"cum. prop. K={k}", COLORS.get(k, "#111111")) for k, _, _ in cumulants
    ]
    item_widths = [58.0 + max(102.0, 7.1 * len(label)) for label, _ in legend_items]
    gap = 18.0
    legend_width = sum(item_widths) + gap * (len(item_widths) - 1)
    cursor = (width - legend_width) / 2
    legend_y = 78.0
    parts.append(
        f'<rect x="{cursor - 14:.2f}" y="{legend_y - 23:.2f}" width="{legend_width + 28:.2f}" height="40" rx="5" fill="#ffffff" stroke="#c8d1df" stroke-width="1"/>'
    )
    for (label, color), item_width in zip(legend_items, item_widths):
        parts.extend(
            [
                f'<line x1="{cursor:.2f}" y1="{legend_y:.2f}" x2="{cursor + 38:.2f}" y2="{legend_y:.2f}" style="stroke:{color};stroke-width:3"/>',
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

    for value, label in _ticks(x_bounds):
        x = sx(value)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + plot_height:.2f}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{top + plot_height:.2f}" x2="{x:.2f}" y2="{top + plot_height + 6:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 26:.2f}" text-anchor="middle">{escape(label)}</text>',
            ]
        )
    for value, label in _ticks(y_bounds):
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
            f'<text class="axis-label" x="{left + plot_width / 2:.2f}" y="{height - 28:.2f}" text-anchor="middle">FLOPs</text>',
            f'<text class="axis-label" transform="translate(30 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">squared error</text>',
        ]
    )

    sample_points = [(sx(flops), sy(error)) for flops, error in sampling]
    parts.append(
        f'<polyline fill="none" stroke="{COLORS["sampling"]}" stroke-width="3" points="{_polyline(sample_points)}"/>'
    )
    for x, y in sample_points:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{COLORS["sampling"]}" stroke="white" stroke-width="1"/>')

    for k, flops, error in cumulants:
        x, y = sx(flops), sy(error)
        color = COLORS.get(k, "#111111")
        parts.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="{color}" stroke="white" stroke-width="1.5"/>',
                f'<text class="tick" x="{x + 9:.2f}" y="{y - 8:.2f}">K={k}</text>',
            ]
        )

    parts.append("</svg>")
    output.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot squared error versus FLOPs.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="ICA with known parameter A")
    args = parser.parse_args()
    write_svg(args.result_dir, args.output, args.title)


if __name__ == "__main__":
    main()
