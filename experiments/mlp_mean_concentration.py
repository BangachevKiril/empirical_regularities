from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from data_generators import ICADataGenerator
from inference_models import DeepReLUMLP


@dataclass(frozen=True)
class ExperimentResult:
    k: int
    m: int
    squared_error: float

    @property
    def log_k(self) -> float:
        return math.log(self.k)

    @property
    def log_m(self) -> float:
        return math.log(self.m)

    @property
    def log_squared_error(self) -> float:
        return math.log(self.squared_error)


def stream_mlp_mean(
    *,
    model: DeepReLUMLP,
    data_generator: ICADataGenerator,
    total_samples: int,
    batch_size: int,
    seed_base: int,
) -> torch.Tensor:
    output_sum = torch.zeros(
        data_generator.n,
        device=data_generator.device,
        dtype=torch.float64,
    )
    samples_seen = 0
    batch_index = 0

    with torch.inference_mode():
        while samples_seen < total_samples:
            current_batch = min(batch_size, total_samples - samples_seen)
            samples = data_generator.sample(current_batch, seed_=seed_base + batch_index)
            outputs = model(samples.T.contiguous())
            output_sum += outputs.to(dtype=torch.float64).sum(dim=1)
            samples_seen += current_batch
            batch_index += 1

    return output_sum / total_samples


def run_experiment(args: argparse.Namespace) -> list[ExperimentResult]:
    device = torch.device(args.device)
    torch.manual_seed(args.mlp_seed)

    model = DeepReLUMLP(
        n=args.n,
        L=args.depth,
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    data_generator = ICADataGenerator(
        n=args.n,
        seed=args.ica_seed,
        p=args.p,
        device=device,
        dtype=torch.float32,
    )

    start = time.time()
    true_mean = stream_mlp_mean(
        model=model,
        data_generator=data_generator,
        total_samples=args.true_samples,
        batch_size=args.batch_size,
        seed_base=args.true_seed_base,
    )
    print(
        f"computed true mean from {args.true_samples} samples "
        f"in {time.time() - start:.2f}s",
        flush=True,
    )

    results: list[ExperimentResult] = []
    for k in range(args.k_min, args.k_max + 1):
        m = 2**k
        mean_estimate = stream_mlp_mean(
            model=model,
            data_generator=data_generator,
            total_samples=m,
            batch_size=args.batch_size,
            seed_base=args.estimate_seed_base + k * args.seed_stride,
        )
        squared_error = torch.sum((mean_estimate - true_mean) ** 2).item()
        result = ExperimentResult(k=k, m=m, squared_error=squared_error)
        results.append(result)
        print(
            f"k={k:2d} m={m:6d} "
            f"log_sq_error={result.log_squared_error: .6f}",
            flush=True,
        )

    return results


def write_csv(results: list[ExperimentResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "m",
                "squared_error",
                "log_k",
                "log_m",
                "log_squared_error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "k": result.k,
                    "m": result.m,
                    "squared_error": result.squared_error,
                    "log_k": result.log_k,
                    "log_m": result.log_m,
                    "log_squared_error": result.log_squared_error,
                }
            )


def _nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    step = (max_value - min_value) / max(count - 1, 1)
    return [min_value + i * step for i in range(count)]


def write_svg(results: list[ExperimentResult], svg_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [result.log_k for result in results]
    ys = [result.log_squared_error for result in results]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_padding = 0.08 * max(y_max - y_min, 1.0)
    y_min -= y_padding
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

    x_ticks = [math.log(k) for k in [1, 2, 4, 8, 16]]
    y_ticks = _nice_ticks(y_min, y_max, 6)

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
        '<text class="title" x="480" y="38" text-anchor="middle">MLP Output Mean Concentration</text>',
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
    ]

    for tick in x_ticks:
        x = sx(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}"/>',
                f'<text class="tick" x="{x:.2f}" y="{height - margin_bottom + 25}" text-anchor="middle">{tick:.2f}</text>',
            ]
        )

    for tick in y_ticks:
        y = sy(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>',
                f'<line class="axis" x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.2f}</text>',
            ]
        )

    parts.append(f'<polyline class="line" points="{polyline}"/>')
    for x, y in points:
        parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="4.5"/>')

    parts.extend(
        [
            '<text class="label" x="480" y="604" text-anchor="middle">log(k), where m = 2^k</text>',
            '<text class="label" transform="translate(28 320) rotate(-90)" text-anchor="middle">log ||mu_k - mu||_2^2</text>',
            '<text class="tick" x="480" y="630" text-anchor="middle">natural logs; true mean estimated using 1,000,000 samples</text>',
            "</svg>",
        ]
    )
    svg_path.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--p", type=int, default=32)
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--true-samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--estimate-seed-base", type=int, default=1_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mlp_mean_concentration"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}", flush=True)
    print(f"n={args.n} depth={args.depth} p={args.p}", flush=True)
    print(f"true_samples={args.true_samples} batch_size={args.batch_size}", flush=True)

    results = run_experiment(args)
    csv_path = args.output_dir / "results.csv"
    svg_path = args.output_dir / "plot_log_error_vs_log_k.svg"
    write_csv(results, csv_path)
    write_svg(results, svg_path)

    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {svg_path}", flush=True)


if __name__ == "__main__":
    main()

