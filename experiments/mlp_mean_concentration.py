from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from cumulant_propagation import propagate_cumulants
from data_generators import ICADataGenerator
from inference_models import DeepReLUMLP


@dataclass(frozen=True)
class RunResult:
    k: int
    m: int
    run: int
    seed_base: int
    squared_error: float

    @property
    def log_squared_error(self) -> float:
        return math.log(self.squared_error)


@dataclass(frozen=True)
class SummaryResult:
    k: int
    m: int
    runs: int
    mean_squared_error: float
    std_squared_error: float
    mean_log_squared_error: float
    std_log_squared_error: float


@dataclass(frozen=True)
class CumulantResult:
    cumulant_k_max: int
    input_variance: float
    squared_error: float
    elapsed_seconds: float

    @property
    def log_squared_error(self) -> float:
        return math.log(self.squared_error)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


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


def cumulant_propagation_mean(
    *,
    model: DeepReLUMLP,
    p: int,
    cumulant_k_max: int,
    device: torch.device,
) -> torch.Tensor:
    cumulants = {
        1: torch.zeros(model.n, device=device, dtype=torch.float64),
        2: float(p) * torch.eye(model.n, device=device, dtype=torch.float64),
    }
    propagated = propagate_cumulants(
        model,
        cumulants,
        k_max=cumulant_k_max,
        return_tensors=True,
        device=device,
        dtype=torch.float64,
    )
    return propagated[1].to(dtype=torch.float64)


def summarize_results(run_results: list[RunResult]) -> list[SummaryResult]:
    by_k: dict[int, list[RunResult]] = {}
    for result in run_results:
        by_k.setdefault(result.k, []).append(result)

    summaries: list[SummaryResult] = []
    for k in sorted(by_k):
        results = by_k[k]
        squared_errors = [result.squared_error for result in results]
        log_squared_errors = [result.log_squared_error for result in results]
        summaries.append(
            SummaryResult(
                k=k,
                m=results[0].m,
                runs=len(results),
                mean_squared_error=_mean(squared_errors),
                std_squared_error=_sample_std(squared_errors),
                mean_log_squared_error=_mean(log_squared_errors),
                std_log_squared_error=_sample_std(log_squared_errors),
            )
        )
    return summaries


def run_experiment(
    args: argparse.Namespace,
) -> tuple[list[RunResult], list[SummaryResult], CumulantResult]:
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

    start = time.time()
    cumulant_mean = cumulant_propagation_mean(
        model=model,
        p=args.p,
        cumulant_k_max=args.cumulant_k_max,
        device=device,
    )
    cumulant_elapsed = time.time() - start
    cumulant_squared_error = torch.sum((cumulant_mean - true_mean) ** 2).item()
    cumulant_result = CumulantResult(
        cumulant_k_max=args.cumulant_k_max,
        input_variance=float(args.p),
        squared_error=cumulant_squared_error,
        elapsed_seconds=cumulant_elapsed,
    )
    print(
        f"cumulant propagation k_max={args.cumulant_k_max} "
        f"log_sq_error={cumulant_result.log_squared_error: .6f} "
        f"in {cumulant_elapsed:.2f}s",
        flush=True,
    )

    run_results: list[RunResult] = []
    for k in range(args.k_min, args.k_max + 1):
        m = 2**k
        for run in range(args.runs):
            seed_base = (
                args.estimate_seed_base
                + k * args.seed_stride
                + run * args.run_seed_stride
            )
            mean_estimate = stream_mlp_mean(
                model=model,
                data_generator=data_generator,
                total_samples=m,
                batch_size=args.batch_size,
                seed_base=seed_base,
            )
            squared_error = torch.sum((mean_estimate - true_mean) ** 2).item()
            run_results.append(
                RunResult(
                    k=k,
                    m=m,
                    run=run,
                    seed_base=seed_base,
                    squared_error=squared_error,
                )
            )
        summary = summarize_results([result for result in run_results if result.k == k])[0]
        print(
            f"k={k:2d} m={m:6d} "
            f"mean_log_sq_error={summary.mean_log_squared_error: .6f} "
            f"std={summary.std_log_squared_error: .6f}",
            flush=True,
        )

    return run_results, summarize_results(run_results), cumulant_result


def write_run_csv(results: list[RunResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "m",
                "run",
                "seed_base",
                "squared_error",
                "log_squared_error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "k": result.k,
                    "m": result.m,
                    "run": result.run,
                    "seed_base": result.seed_base,
                    "squared_error": result.squared_error,
                    "log_squared_error": result.log_squared_error,
                }
            )


def write_summary_csv(results: list[SummaryResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "m",
                "runs",
                "mean_squared_error",
                "std_squared_error",
                "mean_log_squared_error",
                "std_log_squared_error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "k": result.k,
                    "m": result.m,
                    "runs": result.runs,
                    "mean_squared_error": result.mean_squared_error,
                    "std_squared_error": result.std_squared_error,
                    "mean_log_squared_error": result.mean_log_squared_error,
                    "std_log_squared_error": result.std_log_squared_error,
                }
            )


def write_cumulant_csv(result: CumulantResult, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cumulant_k_max",
                "input_variance",
                "squared_error",
                "log_squared_error",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "cumulant_k_max": result.cumulant_k_max,
                "input_variance": result.input_variance,
                "squared_error": result.squared_error,
                "log_squared_error": result.log_squared_error,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )


def _nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    step = (max_value - min_value) / max(count - 1, 1)
    return [min_value + i * step for i in range(count)]


def write_svg(
    results: list[SummaryResult],
    svg_path: Path,
    *,
    cumulant_result: CumulantResult | None = None,
) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [float(result.k) for result in results]
    ys = [result.mean_log_squared_error for result in results]
    lower_ys = [
        result.mean_log_squared_error - result.std_log_squared_error
        for result in results
    ]
    upper_ys = [
        result.mean_log_squared_error + result.std_log_squared_error
        for result in results
    ]
    x_min, x_max = min(xs), max(xs)
    comparison_ys = [*lower_ys, *upper_ys]
    if cumulant_result is not None:
        comparison_ys.append(cumulant_result.log_squared_error)
    y_min, y_max = min(comparison_ys), max(comparison_ys)
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
    upper_points = [(sx(x), sy(y)) for x, y in zip(xs, upper_ys)]
    lower_points = [(sx(x), sy(y)) for x, y in zip(reversed(xs), reversed(lower_ys))]
    shade_points = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in [*upper_points, *lower_points]
    )

    x_ticks = [float(k) for k in range(int(x_min), int(x_max) + 1)]
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
        ".shade { fill: #1f77b4; opacity: 0.18; }",
        ".line { fill: none; stroke: #1f77b4; stroke-width: 3; }",
        ".cumulant-line { fill: none; stroke: #2ca02c; stroke-width: 3; stroke-dasharray: 10 7; }",
        ".point { fill: #d62728; stroke: white; stroke-width: 1.5; }",
        ".legend { font-size: 14px; fill: #263447; }",
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
                f'<text class="tick" x="{x:.2f}" y="{height - margin_bottom + 25}" text-anchor="middle">{int(tick)}</text>',
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

    parts.append(f'<polygon class="shade" points="{shade_points}"/>')
    parts.append(f'<polyline class="line" points="{polyline}"/>')
    for x, y in points:
        parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="4.5"/>')
    if cumulant_result is not None:
        y = sy(cumulant_result.log_squared_error)
        parts.extend(
            [
                f'<line class="cumulant-line" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>',
                '<line class="line" x1="654" y1="76" x2="704" y2="76"/>',
                '<text class="legend" x="714" y="81">sampling mean</text>',
                '<line class="cumulant-line" x1="654" y1="102" x2="704" y2="102"/>',
                f'<text class="legend" x="714" y="107">cumulant propagation, orders &lt;= {cumulant_result.cumulant_k_max}</text>',
            ]
        )

    parts.extend(
        [
            '<text class="label" x="480" y="604" text-anchor="middle">k, where m = 2^k</text>',
            '<text class="label" transform="translate(28 320) rotate(-90)" text-anchor="middle">log squared error</text>',
            '<text class="tick" x="480" y="630" text-anchor="middle">natural logs; shaded band is +/- one standard deviation across sampling runs</text>',
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
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cumulant-k-max", type=int, default=2)
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--estimate-seed-base", type=int, default=1_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--run-seed-stride", type=int, default=1_000_000)
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
    print(f"runs_per_k={args.runs}", flush=True)

    run_results, summary_results, cumulant_result = run_experiment(args)
    run_csv_path = args.output_dir / "run_results.csv"
    summary_csv_path = args.output_dir / "results.csv"
    cumulant_csv_path = args.output_dir / "cumulant_results.csv"
    svg_path = args.output_dir / "plot_log_error_vs_k.svg"
    write_run_csv(run_results, run_csv_path)
    write_summary_csv(summary_results, summary_csv_path)
    write_cumulant_csv(cumulant_result, cumulant_csv_path)
    write_svg(summary_results, svg_path, cumulant_result=cumulant_result)

    print(f"wrote {run_csv_path}", flush=True)
    print(f"wrote {summary_csv_path}", flush=True)
    print(f"wrote {cumulant_csv_path}", flush=True)
    print(f"wrote {svg_path}", flush=True)


if __name__ == "__main__":
    main()
