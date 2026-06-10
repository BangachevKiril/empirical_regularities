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
class PanelSpec:
    result_dir: Path
    title: str


@dataclass(frozen=True)
class PanelData:
    title: str
    sampling: list[tuple[float, float]]
    cumulants: list[tuple[int, float, float]]


def _read_sampling(result_dir: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with (result_dir / "results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            flops = float(row["forward_flops"])
            error = float(row["mean_squared_error"])
            if math.isfinite(flops) and math.isfinite(error) and flops > 0 and error > 0:
                points.append((flops, error))
    return points


def _read_cumulant_points(result_dir: Path) -> list[tuple[int, float, float]]:
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


def _read_panel(spec: PanelSpec) -> PanelData:
    sampling = _read_sampling(spec.result_dir)
    cumulants = _read_cumulant_points(spec.result_dir)
    if not sampling:
        raise ValueError(f"No sampling data found in {spec.result_dir / 'results.csv'}")
    if not cumulants:
        raise ValueError(f"No exact cumulant data found in {spec.result_dir / 'cumulant_results.csv'}")
    return PanelData(title=spec.title, sampling=sampling, cumulants=cumulants)


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


def _parse_panel(value: str) -> PanelSpec:
    if "::" not in value:
        raise argparse.ArgumentTypeError("panel must be formatted as result_dir::title")
    result_dir, title = value.split("::", 1)
    if not result_dir or not title:
        raise argparse.ArgumentTypeError("panel must include both result_dir and title")
    return PanelSpec(result_dir=Path(result_dir), title=title)


def write_svg(
    *,
    panel_specs: list[PanelSpec],
    output: Path,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = [_read_panel(spec) for spec in panel_specs]
    if len(panels) < 2:
        raise ValueError("At least two panels are required.")

    y_values = [
        error
        for panel in panels
        for error in (
            [point[1] for point in panel.sampling]
            + [point[2] for point in panel.cumulants]
        )
    ]
    y_bounds = _log_bounds(y_values)
    x_bounds_by_panel = [
        _log_bounds(
            [flops for flops, _ in panel.sampling]
            + [flops for _, flops, _ in panel.cumulants]
        )
        for panel in panels
    ]

    panel_count = len(panels)
    width = 1480 if panel_count == 2 else 1880
    height = 690
    left_margin, right_margin = 108, 46
    top, bottom = 140, 94
    panel_gap = 58
    plot_width = (width - left_margin - right_margin - panel_gap * (panel_count - 1)) / panel_count
    plot_height = height - top - bottom
    panel_lefts = [
        left_margin + index * (plot_width + panel_gap)
        for index in range(panel_count)
    ]

    def sx(value: float, panel_index: int) -> float:
        lo, hi = x_bounds_by_panel[panel_index]
        return panel_lefts[panel_index] + (math.log10(value) - lo) / (hi - lo) * plot_width

    def sy(value: float) -> float:
        lo, hi = y_bounds
        return top + (hi - math.log10(value)) / (hi - lo) * plot_height

    legend_items = [("sampling baseline", COLORS["sampling"])] + [
        (f"cum. prop. K={k}", COLORS.get(k, "#111111"))
        for k, _, _ in panels[0].cumulants
    ]
    item_widths = [58.0 + max(102.0, 7.2 * len(label)) for label, _ in legend_items]
    gap = 18.0
    legend_width = sum(item_widths) + gap * (len(item_widths) - 1)
    legend_x = (width - legend_width) / 2
    legend_y = 82.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }",
        ".title { font-size: 27px; font-weight: 700; }",
        ".panel-title { font-size: 18px; font-weight: 700; }",
        ".axis-label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #526071; }",
        ".legend { font-size: 13px; fill: #263447; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.25; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.2f}" y="38" text-anchor="middle">{escape(title)}</text>',
        f'<rect x="{legend_x - 14:.2f}" y="{legend_y - 23:.2f}" width="{legend_width + 28:.2f}" height="40" rx="5" fill="#ffffff" stroke="#c8d1df" stroke-width="1"/>',
    ]

    cursor = legend_x
    for (label, color), item_width in zip(legend_items, item_widths):
        parts.extend(
            [
                f'<line x1="{cursor:.2f}" y1="{legend_y:.2f}" x2="{cursor + 38:.2f}" y2="{legend_y:.2f}" style="stroke:{color};stroke-width:3"/>',
                f'<circle cx="{cursor + 19:.2f}" cy="{legend_y:.2f}" r="4.2" fill="{color}" stroke="white" stroke-width="1"/>',
                f'<text class="legend" x="{cursor + 50:.2f}" y="{legend_y + 4:.2f}">{escape(label)}</text>',
            ]
        )
        cursor += item_width + gap

    y_ticks = _ticks(y_bounds)
    for panel_index, panel in enumerate(panels):
        panel_left = panel_lefts[panel_index]
        panel_right = panel_left + plot_width
        panel_center = panel_left + plot_width / 2
        x_axis_y = top + plot_height
        parts.extend(
            [
                f'<text class="panel-title" x="{panel_center:.2f}" y="{top - 24:.2f}" text-anchor="middle">{escape(panel.title)}</text>',
                f'<line class="axis" x1="{panel_left:.2f}" y1="{x_axis_y:.2f}" x2="{panel_right:.2f}" y2="{x_axis_y:.2f}"/>',
                f'<line class="axis" x1="{panel_left:.2f}" y1="{top:.2f}" x2="{panel_left:.2f}" y2="{x_axis_y:.2f}"/>',
            ]
        )

        for value, label in _ticks(x_bounds_by_panel[panel_index]):
            x = sx(value, panel_index)
            parts.extend(
                [
                    f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{x_axis_y:.2f}"/>',
                    f'<line class="axis" x1="{x:.2f}" y1="{x_axis_y:.2f}" x2="{x:.2f}" y2="{x_axis_y + 6:.2f}"/>',
                    f'<text class="tick" x="{x:.2f}" y="{x_axis_y + 26:.2f}" text-anchor="middle">{escape(label)}</text>',
                ]
            )

        for value, label in y_ticks:
            y = sy(value)
            parts.extend(
                [
                    f'<line class="grid" x1="{panel_left:.2f}" y1="{y:.2f}" x2="{panel_right:.2f}" y2="{y:.2f}"/>',
                    f'<line class="axis" x1="{panel_left - 6:.2f}" y1="{y:.2f}" x2="{panel_left:.2f}" y2="{y:.2f}"/>',
                ]
            )
            if panel_index == 0:
                parts.append(
                    f'<text class="tick" x="{panel_left - 12:.2f}" y="{y + 4:.2f}" text-anchor="end">{escape(label)}</text>'
                )

        sample_points = [(sx(flops, panel_index), sy(error)) for flops, error in panel.sampling]
        parts.append(
            f'<polyline fill="none" stroke="{COLORS["sampling"]}" stroke-width="3" points="{_polyline(sample_points)}"/>'
        )
        for x, y in sample_points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="{COLORS["sampling"]}" stroke="white" stroke-width="1"/>'
            )

        for k, flops, error in panel.cumulants:
            x, y = sx(flops, panel_index), sy(error)
            color = COLORS.get(k, "#111111")
            parts.extend(
                [
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.3" fill="{color}" stroke="white" stroke-width="1.5"/>',
                    f'<text class="tick" x="{x + 8:.2f}" y="{y - 8:.2f}">K={k}</text>',
                ]
            )

        parts.append(
            f'<text class="axis-label" x="{panel_center:.2f}" y="{height - 30:.2f}" text-anchor="middle">FLOPs</text>'
        )

    parts.append(
        f'<text class="axis-label" transform="translate(32 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">squared error</text>'
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts))


def _panel_specs_from_args(args: argparse.Namespace) -> list[PanelSpec]:
    if args.panel:
        return args.panel
    specs = [
        PanelSpec(args.left_result_dir, args.left_title),
        PanelSpec(args.right_result_dir, args.right_title),
    ]
    if args.third_result_dir is not None:
        specs.append(PanelSpec(args.third_result_dir, args.third_title))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot error-vs-FLOPs panels with a shared y-axis.")
    parser.add_argument(
        "--panel",
        action="append",
        type=_parse_panel,
        help="Panel specification formatted as result_dir::title. May be repeated.",
    )
    parser.add_argument("--left-result-dir", type=Path)
    parser.add_argument("--right-result-dir", type=Path)
    parser.add_argument("--third-result-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-title", default="Gaussian exact cumulants")
    parser.add_argument("--right-title", default="ICA with known parameter A")
    parser.add_argument("--third-title", default="Gaussian-Lowrank known A")
    parser.add_argument("--title", default="MLP mean error versus FLOPs")
    args = parser.parse_args()
    if not args.panel and (args.left_result_dir is None or args.right_result_dir is None):
        parser.error("either provide repeated --panel values or both --left-result-dir and --right-result-dir")
    write_svg(
        panel_specs=_panel_specs_from_args(args),
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
