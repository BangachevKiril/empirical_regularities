from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from data_generators import ICADataGenerator
from inference_models import DeepReLUMLP


@dataclass(frozen=True)
class LayerRankResult:
    layer: int
    effective_rank: float
    nuclear_norm: float
    frobenius_norm_squared: float
    top_singular_value: float
    numerical_rank: int


def effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    singular_values = singular_values.to(dtype=torch.float64)
    nuclear_norm = torch.sum(singular_values)
    frobenius_norm_squared = torch.sum(singular_values * singular_values)
    if frobenius_norm_squared.item() == 0.0:
        return 0.0
    return ((nuclear_norm * nuclear_norm) / frobenius_norm_squared).item()


def numerical_rank_from_singular_values(
    singular_values: torch.Tensor,
    matrix_shape: tuple[int, int],
    tolerance_dtype: torch.dtype,
) -> int:
    if singular_values.numel() == 0:
        return 0
    max_singular_value = torch.max(singular_values).item()
    if max_singular_value == 0.0:
        return 0
    eps = torch.finfo(tolerance_dtype).eps
    tolerance = max(matrix_shape) * eps * max_singular_value
    return int(torch.sum(singular_values > tolerance).item())


def summarize_layer(layer: int, matrix: torch.Tensor) -> tuple[LayerRankResult, list[float]]:
    tolerance_dtype = matrix.dtype
    matrix = matrix.to(dtype=torch.float64)
    singular_values = torch.linalg.svdvals(matrix)
    singular_values_cpu = singular_values.detach().cpu()
    nuclear_norm = torch.sum(singular_values).item()
    frobenius_norm_squared = torch.sum(singular_values * singular_values).item()
    result = LayerRankResult(
        layer=layer,
        effective_rank=effective_rank_from_singular_values(singular_values),
        nuclear_norm=nuclear_norm,
        frobenius_norm_squared=frobenius_norm_squared,
        top_singular_value=torch.max(singular_values).item(),
        numerical_rank=numerical_rank_from_singular_values(
            singular_values,
            (matrix.shape[0], matrix.shape[1]),
            tolerance_dtype,
        ),
    )
    return result, [float(value) for value in singular_values_cpu.tolist()]


def make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def sample_inputs(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.input_distribution == "ica":
        data_generator = ICADataGenerator(
            n=args.n,
            seed=args.ica_seed,
            p=args.p,
            device=device,
            dtype=torch.float32,
        )
        return data_generator.sample(args.samples, seed_=args.sample_seed)

    if args.input_distribution == "gaussian":
        return torch.randn(
            (args.samples, args.n),
            generator=make_torch_generator(device, args.gaussian_seed),
            device=device,
            dtype=torch.float32,
        )

    raise ValueError(f"Unknown input distribution: {args.input_distribution}.")


def run_experiment(args: argparse.Namespace) -> tuple[list[LayerRankResult], list[list[float]]]:
    device = torch.device(args.device)
    torch.manual_seed(args.mlp_seed)

    model = DeepReLUMLP(
        n=args.n,
        L=args.depth,
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    samples = sample_inputs(args, device)
    activations = samples.T.contiguous()

    results: list[LayerRankResult] = []
    singular_values_by_layer: list[list[float]] = []

    with torch.inference_mode():
        result, singular_values = summarize_layer(0, activations.T.contiguous())
        results.append(result)
        singular_values_by_layer.append(singular_values)
        print(
            f"layer={result.layer} effective_rank={result.effective_rank:.6f} "
            f"numerical_rank={result.numerical_rank}",
            flush=True,
        )

        for layer, weight in enumerate(model.weights, start=1):
            activations = F.relu(weight @ activations)
            result, singular_values = summarize_layer(
                layer,
                activations.T.contiguous(),
            )
            results.append(result)
            singular_values_by_layer.append(singular_values)
            print(
                f"layer={result.layer} effective_rank={result.effective_rank:.6f} "
                f"numerical_rank={result.numerical_rank}",
                flush=True,
            )

    return results, singular_values_by_layer


def write_results_csv(results: list[LayerRankResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer",
                "effective_rank",
                "nuclear_norm",
                "frobenius_norm_squared",
                "top_singular_value",
                "numerical_rank",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "layer": result.layer,
                    "effective_rank": result.effective_rank,
                    "nuclear_norm": result.nuclear_norm,
                    "frobenius_norm_squared": result.frobenius_norm_squared,
                    "top_singular_value": result.top_singular_value,
                    "numerical_rank": result.numerical_rank,
                }
            )


def write_singular_values_csv(
    singular_values_by_layer: list[list[float]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "index", "singular_value"])
        for layer, singular_values in enumerate(singular_values_by_layer):
            for index, singular_value in enumerate(singular_values, start=1):
                writer.writerow([layer, index, singular_value])


def _nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if min_value == max_value:
        return [min_value]
    step = (max_value - min_value) / max(count - 1, 1)
    return [min_value + i * step for i in range(count)]


def write_svg(
    results: list[LayerRankResult],
    svg_path: Path,
    *,
    input_distribution: str,
    samples: int,
) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [float(result.layer) for result in results]
    ys = [result.effective_rank for result in results]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)
    y_padding = 0.10 * max(y_max - y_min, 1.0)
    y_max += y_padding

    width, height = 960, 640
    margin_left, margin_right = 96, 36
    margin_top, margin_bottom = 68, 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    points = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    x_ticks = [float(layer) for layer in range(int(x_min), int(x_max) + 1)]
    y_ticks = _nice_ticks(y_min, y_max, 6)

    title_suffix = "Gaussian Input" if input_distribution == "gaussian" else "ICA Input"

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }",
        ".title { font-size: 25px; font-weight: 700; }",
        ".label { font-size: 16px; font-weight: 600; }",
        ".tick { font-size: 13px; fill: #526071; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.4; }",
        ".line { fill: none; stroke: #1f77b4; stroke-width: 3; }",
        ".point { fill: #d62728; stroke: white; stroke-width: 1.5; }",
        "</style>",
        '<rect width="960" height="640" fill="#ffffff"/>',
        f'<text class="title" x="480" y="38" text-anchor="middle">MLP Activation Effective Rank ({title_suffix})</text>',
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

    parts.append(f'<polyline class="line" points="{polyline}"/>')
    for x, y in points:
        parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="5"/>')

    parts.extend(
        [
            '<text class="label" x="480" y="604" text-anchor="middle">layer i</text>',
            '<text class="label" transform="translate(28 320) rotate(-90)" text-anchor="middle">effective rank R_i</text>',
            f'<text class="tick" x="480" y="630" text-anchor="middle">R_i = (sum_j sigma_j)^2 / sum_j sigma_j^2; batch size {samples}</text>',
            "</svg>",
        ]
    )
    svg_path.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--p", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument(
        "--input-distribution",
        choices=["ica", "gaussian"],
        default="ica",
    )
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--gaussian-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mlp_effective_rank"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}", flush=True)
    print(
        f"n={args.n} depth={args.depth} p={args.p} samples={args.samples}",
        flush=True,
    )
    print(
        f"input_distribution={args.input_distribution} "
        f"ica_seed={args.ica_seed} sample_seed={args.sample_seed} "
        f"gaussian_seed={args.gaussian_seed} mlp_seed={args.mlp_seed}",
        flush=True,
    )

    results, singular_values_by_layer = run_experiment(args)
    results_csv_path = args.output_dir / "results.csv"
    singular_values_csv_path = args.output_dir / "singular_values.csv"
    svg_path = args.output_dir / "plot_effective_rank_vs_layer.svg"
    write_results_csv(results, results_csv_path)
    write_singular_values_csv(singular_values_by_layer, singular_values_csv_path)
    write_svg(
        results,
        svg_path,
        input_distribution=args.input_distribution,
        samples=args.samples,
    )

    print(f"wrote {results_csv_path}", flush=True)
    print(f"wrote {singular_values_csv_path}", flush=True)
    print(f"wrote {svg_path}", flush=True)


if __name__ == "__main__":
    main()
