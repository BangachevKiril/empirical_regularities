import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {
    "forward": "#1f77b4",
    1: "#2ca02c",
    2: "#9467bd",
    3: "#ff7f0e",
    4: "#d62728",
}


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    xs: list[float]
    flops: list[float]
    errors: list[float]


@dataclass(frozen=True)
class Dataset:
    name: str
    series: list[Series]


def _cumulant_label(propagation: str, cumulant_k: int) -> str:
    if propagation.startswith("K="):
        return f"cum. prop. {propagation}"
    return f"cum. prop. {propagation} K={cumulant_k}"


def _read_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _read_experiment(
    output_dir: Path,
    name: str,
    *,
    cumulant_source: str = "sample",
    fixed_method: str = "known_distribution",
) -> Dataset:
    summary_path = output_dir / "results.csv"
    cumulant_path = output_dir / "cumulant_results.csv"

    with summary_path.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    sample_ks = [_read_float(row, "k") for row in summary_rows]
    forward = Series(
        label="forward passes",
        color=COLORS["forward"],
        xs=sample_ks,
        flops=[_read_float(row, "forward_flops") for row in summary_rows],
        errors=[_read_float(row, "mean_squared_error") for row in summary_rows],
    )

    series = [forward]
    with cumulant_path.open(newline="") as handle:
        cumulant_rows = list(csv.DictReader(handle))

    if cumulant_source == "sample":
        grouped: dict[int, list[dict[str, str]]] = {}
        for row in cumulant_rows:
            if not row["sample_count"]:
                continue
            cumulant_k = int(row["cumulant_k_max"])
            if cumulant_k not in COLORS:
                continue
            grouped.setdefault(cumulant_k, []).append(row)

        for cumulant_k in sorted(grouped):
            rows = sorted(grouped[cumulant_k], key=lambda row: int(row["sample_count"]))
            propagation = rows[0]["propagation"]
            series.append(
                Series(
                    label=_cumulant_label(propagation, cumulant_k),
                    color=COLORS[cumulant_k],
                    xs=[math.log2(int(row["sample_count"])) for row in rows],
                    flops=[_read_float(row, "total_flops") for row in rows],
                    errors=[_read_float(row, "squared_error") for row in rows],
                )
            )
    elif cumulant_source == "fixed":
        rows_by_k: dict[int, dict[str, str]] = {}
        for row in cumulant_rows:
            if row["method"] != fixed_method or row["sample_count"]:
                continue
            cumulant_k = int(row["cumulant_k_max"])
            if cumulant_k in COLORS:
                rows_by_k[cumulant_k] = row
        for cumulant_k in sorted(rows_by_k):
            row = rows_by_k[cumulant_k]
            series.append(
                Series(
                    label=_cumulant_label(row["propagation"], cumulant_k),
                    color=COLORS[cumulant_k],
                    xs=sample_ks,
                    flops=[_read_float(row, "total_flops")] * len(sample_ks),
                    errors=[_read_float(row, "squared_error")] * len(sample_ks),
                )
            )
    else:
        raise ValueError(f"Unsupported cumulant_source: {cumulant_source!r}.")
    return Dataset(name=name, series=series)


def _log_bounds(values: list[float]) -> tuple[float, float]:
    logs = [math.log10(value) for value in values if value > 0.0]
    lo, hi = min(logs), max(logs)
    padding = 0.08 * max(hi - lo, 1.0)
    return lo - padding, hi + padding


def _ticks_for_log_bounds(lo: float, hi: float) -> list[int]:
    return list(range(math.ceil(lo), math.floor(hi) + 1))


def _sample_label(k: int) -> str:
    return f"{2 ** k:,}"


def _format_power(exponent: int) -> str:
    return f"10^{exponent}"


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _draw_panel(
    parts: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    ylabel: str,
    dataset: Dataset,
    value_kind: str,
    y_log_bounds: tuple[float, float],
    y_ticks: list[int],
) -> None:
    x_min, x_max = 1.0, 20.0
    y_min, y_max = y_log_bounds

    def sx(x: float) -> float:
        return x0 + (x - x_min) / (x_max - x_min) * width

    def sy(value: float) -> float:
        log_value = math.log10(max(value, 1e-300))
        return y0 + (y_max - log_value) / (y_max - y_min) * height

    x_ticks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]

    parts.append(f'<text class="panel-title" x="{x0:.2f}" y="{y0 - 18:.2f}">{escape(title)}</text>')
    parts.append(f'<line class="axis" x1="{x0:.2f}" y1="{y0 + height:.2f}" x2="{x0 + width:.2f}" y2="{y0 + height:.2f}"/>')
    parts.append(f'<line class="axis" x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + height:.2f}"/>')

    for tick in x_ticks:
        x = sx(float(tick))
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y0 + height:.2f}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{y0 + height:.2f}" x2="{x:.2f}" y2="{y0 + height + 7:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="{y0 + height + 29:.2f}" text-anchor="middle">{_sample_label(tick)}</text>',
            ]
        )

    for exponent in y_ticks:
        y = sy(10.0**exponent)
        parts.extend(
            [
                f'<line class="grid" x1="{x0:.2f}" y1="{y:.2f}" x2="{x0 + width:.2f}" y2="{y:.2f}"/>',
                f'<line class="axis" x1="{x0 - 7:.2f}" y1="{y:.2f}" x2="{x0:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{x0 - 14:.2f}" y="{y + 4:.2f}" text-anchor="end">{_format_power(exponent)}</text>',
            ]
        )

    for series in dataset.series:
        values = series.flops if value_kind == "flops" else series.errors
        points = [(sx(x), sy(value)) for x, value in zip(series.xs, values)]
        parts.append(
            f'<polyline points="{_polyline(points)}" style="fill:none;stroke:{series.color};stroke-width:3.2"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.9" style="fill:{series.color};stroke:#ffffff;stroke-width:1.2"/>'
            )

    parts.append(
        f'<text class="axis-label" transform="translate({x0 - 82:.2f} {y0 + height / 2:.2f}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>'
    )



def _draw_global_legend(parts: list[str], *, width: float, series: list[Series]) -> None:
    item_widths = [58.0 + max(108.0, 7.4 * len(item.label)) for item in series]
    gap = 34.0
    content_width = sum(item_widths) + gap * (len(series) - 1)
    legend_x = (width - content_width) / 2.0
    legend_y = 86.0
    parts.append(
        f'<rect x="{legend_x - 18:.2f}" y="{legend_y - 25:.2f}" width="{content_width + 36:.2f}" height="44" rx="5" fill="#ffffff" stroke="#c8d1df" stroke-width="1"/>'
    )
    cursor = legend_x
    for item, item_width in zip(series, item_widths):
        parts.extend(
            [
                f'<line x1="{cursor:.2f}" y1="{legend_y:.2f}" x2="{cursor + 44:.2f}" y2="{legend_y:.2f}" style="stroke:{item.color};stroke-width:3.2"/>',
                f'<text class="legend" x="{cursor + 56:.2f}" y="{legend_y + 5:.2f}">{escape(item.label)}</text>',
            ]
        )
        cursor += item_width + gap


def write_svg(
    ica_dir: Path,
    gaussian_dir: Path,
    svg_path: Path,
    *,
    known_ica_dir: Path | None = None,
) -> None:
    ica = _read_experiment(ica_dir, "ICA Sample Cumulants")
    gaussian = _read_experiment(gaussian_dir, "Gaussian Exact Cumulants")
    datasets = [ica]
    if known_ica_dir is not None:
        datasets.append(
            _read_experiment(
                known_ica_dir,
                "ICA Known A",
                cumulant_source="fixed",
                fixed_method="known_distribution",
            )
        )
    datasets.append(gaussian)

    all_flops = [value for dataset in datasets for series in dataset.series for value in series.flops]
    all_errors = [value for dataset in datasets for series in dataset.series for value in series.errors]
    flops_bounds = _log_bounds(all_flops)
    error_bounds = _log_bounds(all_errors)
    flops_ticks = _ticks_for_log_bounds(*flops_bounds)
    error_ticks = _ticks_for_log_bounds(*error_bounds)

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    height = 1140
    panel_width = 850 if len(datasets) == 2 else 760
    panel_height = 350
    margin_left = 110
    gutter = 270 if len(datasets) == 2 else 250
    width = int(margin_left + len(datasets) * panel_width + (len(datasets) - 1) * gutter + 110)
    xs = [margin_left + index * (panel_width + gutter) for index in range(len(datasets))]
    top_y, bottom_y = 155, 665

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }",
        ".column-title { font-size: 31px; font-weight: 700; }",
        ".panel-title { font-size: 19px; font-weight: 700; }",
        ".axis-label { font-size: 16px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #526071; }",
        ".legend { font-size: 13px; fill: #263447; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.35; }",
        ".caption { font-size: 13px; fill: #5b687a; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    _draw_global_legend(parts, width=float(width), series=datasets[0].series)

    for x, dataset in zip(xs, datasets):
        parts.append(
            f'<text class="column-title" x="{x + panel_width / 2:.2f}" y="44" text-anchor="middle">{escape(dataset.name)}</text>'
        )
        _draw_panel(
            parts,
            x0=x,
            y0=top_y,
            width=panel_width,
            height=panel_height,
            title="FLOPs vs Samples",
            ylabel="FLOPs, log10 scale",
            dataset=dataset,
            value_kind="flops",
            y_log_bounds=flops_bounds,
            y_ticks=flops_ticks,
        )
        _draw_panel(
            parts,
            x0=x,
            y0=bottom_y,
            width=panel_width,
            height=panel_height,
            title="Error vs Samples",
            ylabel="squared error, log10 scale",
            dataset=dataset,
            value_kind="errors",
            y_log_bounds=error_bounds,
            y_ticks=error_ticks,
        )
        parts.append(
            f'<text class="axis-label" x="{x + panel_width / 2:.2f}" y="{height - 62}" text-anchor="middle">number of samples m, log2 scale</text>'
        )

    parts.append(
        f'<text class="caption" x="{width / 2:.2f}" y="{height - 25}" text-anchor="middle">Top panels share one FLOP y-scale; bottom panels share one squared-error y-scale.</text>'
    )
    parts.append("</svg>")
    svg_path.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ica-dir", type=Path, required=True)
    parser.add_argument(
        "--known-ica-dir",
        type=Path,
        default=None,
        help="Optional ICA result dir whose fixed known_distribution rows are plotted as a third panel.",
    )
    parser.add_argument("--gaussian-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_svg(
        args.ica_dir,
        args.gaussian_dir,
        args.output,
        known_ica_dir=args.known_ica_dir,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
