from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RankSeries:
    label: str
    color: str
    rows: list[tuple[int, float]]


def read_rank_series(path: Path, label: str, color: str) -> RankSeries:
    rows: list[tuple[int, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["layer"]), float(row["effective_rank"])))
    return RankSeries(label=label, color=color, rows=rows)


def write_comparison_csv(series: list[RankSeries], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted({layer for rank_series in series for layer, _ in rank_series.rows})
    by_label = {
        rank_series.label: dict(rank_series.rows)
        for rank_series in series
    }

    with csv_path.open("w", newline="") as handle:
        fieldnames = ["layer", *[rank_series.label for rank_series in series]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer in layers:
            writer.writerow(
                {
                    "layer": layer,
                    **{
                        rank_series.label: by_label[rank_series.label][layer]
                        for rank_series in series
                    },
                }
            )


def _nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if min_value == max_value:
        return [min_value]
    step = (max_value - min_value) / max(count - 1, 1)
    return [min_value + i * step for i in range(count)]


def write_svg(series: list[RankSeries], svg_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [float(layer) for rank_series in series for layer, _ in rank_series.rows]
    ys = [rank for rank_series in series for _, rank in rank_series.rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)
    y_max += 0.10 * max(y_max - y_min, 1.0)

    width, height = 960, 640
    margin_left, margin_right = 96, 36
    margin_top, margin_bottom = 68, 104
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    x_ticks = [float(layer) for layer in range(int(x_min), int(x_max) + 1)]
    y_ticks = _nice_ticks(y_min, y_max, 6)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }",
        ".title { font-size: 25px; font-weight: 700; }",
        ".label { font-size: 16px; font-weight: 600; }",
        ".tick { font-size: 13px; fill: #526071; }",
        ".legend { font-size: 14px; font-weight: 600; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.4; }",
        ".line { fill: none; stroke-width: 3; }",
        ".point { stroke: white; stroke-width: 1.5; }",
        "</style>",
        '<rect width="960" height="640" fill="#ffffff"/>',
        '<text class="title" x="480" y="38" text-anchor="middle">MLP Activation Effective Rank</text>',
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
    ]

    for tick in x_ticks:
        x = sx(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}"/>',
                f'<text class="tick" x="{x:.2f}" y="{height - margin_bottom + 25}" text-anchor="middle">{int(tick)}</text>',
            ]
        )

    for tick in y_ticks:
        y = sy(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>',
                f'<line class="axis" x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.1f}</text>',
            ]
        )

    for rank_series in series:
        points = [(sx(float(layer)), sy(rank)) for layer, rank in rank_series.rows]
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline class="line" stroke="{rank_series.color}" points="{polyline}"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle class="point" fill="{rank_series.color}" cx="{x:.2f}" cy="{y:.2f}" r="5"/>'
            )

    legend_x, legend_y = 680, 88
    for index, rank_series in enumerate(series):
        y = legend_y + 28 * index
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" stroke="{rank_series.color}" stroke-width="3"/>',
                f'<circle fill="{rank_series.color}" stroke="white" stroke-width="1.5" cx="{legend_x + 15}" cy="{y}" r="5"/>',
                f'<text class="legend" x="{legend_x + 42}" y="{y + 5}">{rank_series.label}</text>',
            ]
        )

    parts.extend(
        [
            '<text class="label" x="480" y="592" text-anchor="middle">layer i</text>',
            '<text class="label" transform="translate(28 304) rotate(-90)" text-anchor="middle">effective rank R_i</text>',
            '<text class="tick" x="480" y="624" text-anchor="middle">R_i = (sum_j sigma_j)^2 / sum_j sigma_j^2; batch size 8192</text>',
            "</svg>",
        ]
    )
    svg_path.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ica-results",
        type=Path,
        default=Path("results/mlp_effective_rank/results.csv"),
    )
    parser.add_argument(
        "--gaussian-results",
        type=Path,
        default=Path("results/mlp_effective_rank_gaussian/results.csv"),
    )
    parser.add_argument(
        "--lowrank-results",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mlp_effective_rank_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = [
        read_rank_series(args.gaussian_results, "Gaussian", "#d62728"),
        *(
            [read_rank_series(args.lowrank_results, "Gaussian-Lowrank", "#2ca02c")]
            if args.lowrank_results is not None
            else []
        ),
        read_rank_series(args.ica_results, "ICA", "#1f77b4"),
    ]

    comparison_csv_path = args.output_dir / "results.csv"
    svg_path = args.output_dir / "plot_effective_rank_comparison.svg"
    write_comparison_csv(series, comparison_csv_path)
    write_svg(series, svg_path)

    print(f"wrote {comparison_csv_path}", flush=True)
    print(f"wrote {svg_path}", flush=True)


if __name__ == "__main__":
    main()
