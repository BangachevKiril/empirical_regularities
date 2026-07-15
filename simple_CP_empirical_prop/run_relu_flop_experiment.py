"""ReLU flop-vs-error experiment for simple ordinary-CP propagation.

This script runs the experiment requested in the Codex thread:

- Kaiming ReLU network with width n=256, L=4, weight variance 2/n.
- Streaming m=2^24 Gaussian inputs for a high-sample "true" last-layer mean.
- Sampling baselines for m_k = 2^k, k in {3, 6, 9, 12, 15, 18}.
- Ordinary-CP estimates from m=2^18 samples for K=2,3,4 and
  M in {2^3, 2^6, 2^9, 2^12, 2^15, 2^18}.

K=1 is recorded as unsupported for hidden nonlinearities because the algorithm
needs a preactivation variance to evaluate ReLU Wick coefficients. ReLU is
non-polynomial, so the Hermite degree cap is an explicit experiment parameter.
"""

from __future__ import annotations

import sys

SCRIPT_PATH0 = sys.path.pop(0) if sys.path else ""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PACKAGE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import torch

from simple_CP_empirical_prop import (
    OrdinaryCPConfig,
    activation_spec_from_name,
    propagate_linear_activation_stages,
)


def build_kaiming_relu_linears(
    *,
    n: int,
    layers: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> list[torch.nn.Linear]:
    gen = torch.Generator(device=device).manual_seed(seed)
    linears: list[torch.nn.Linear] = []
    std = math.sqrt(2.0 / n)
    for _ in range(layers):
        linear = torch.nn.Linear(n, n, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(n, n, generator=gen, device=device, dtype=dtype) * std)
        linears.append(linear)
    return linears


@torch.no_grad()
def sample_relu_means(
    linears: list[torch.nn.Linear],
    *,
    sample_count: int,
    batch_size: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> list[torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    sums = [torch.zeros(n, device=device, dtype=torch.float64) for _ in linears]
    done = 0
    start = time.time()
    while done < sample_count:
        current = min(batch_size, sample_count - done)
        x = torch.randn(current, n, generator=gen, device=device, dtype=dtype)
        for layer, linear in enumerate(linears):
            x = torch.relu(linear(x))
            sums[layer] += x.to(torch.float64).sum(dim=0)
        done += current
        if done == sample_count or done % (batch_size * 8) == 0:
            elapsed = time.time() - start
            print(f"[sampling] {done}/{sample_count} samples in {elapsed:.1f}s", flush=True)
    return [(s / sample_count).cpu() for s in sums]


def baseline_forward_flops(*, sample_count: int, n: int, layers: int) -> float:
    matmul = sample_count * layers * n * (2 * n - 1)
    relu = sample_count * layers * n
    return float(matmul + relu)


def rms_error(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    return float((estimate.to(torch.float64) - truth.to(torch.float64)).square().mean().sqrt().item())


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" />'


def write_svg_plot(path: Path, payload: dict[str, Any]) -> None:
    series: dict[str, list[tuple[float, float]]] = {"sampling": []}
    for row in payload["baseline"]:
        series["sampling"].append((row["flops"], row["last_layer_rms_error"]))
    for k, rows in payload["ordinary_cp"].items():
        pts = [
            (row["flops"], row["last_layer_rms_error"])
            for row in rows
            if row.get("supported", True)
        ]
        series[f"K={k}"] = pts
    all_points = [pt for pts in series.values() for pt in pts if pt[0] > 0 and pt[1] > 0]
    if not all_points:
        return
    xmin = min(math.log10(x) for x, _ in all_points)
    xmax = max(math.log10(x) for x, _ in all_points)
    ymin = min(math.log10(y) for _, y in all_points)
    ymax = max(math.log10(y) for _, y in all_points)
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymax += 1
    xpad = 0.08 * (xmax - xmin)
    ypad = 0.10 * (ymax - ymin)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    width, height = 980, 640
    left, right, top, bottom = 95, 35, 35, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (math.log10(x) - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - math.log10(y)) / (ymax - ymin) * plot_h

    colors = {
        "sampling": "#111111",
        "K=1": "#777777",
        "K=2": "#1f77b4",
        "K=3": "#2ca02c",
        "K=4": "#d62728",
    }
    lines: list[str] = []
    circles: list[str] = []
    for name, pts in series.items():
        pts = sorted(pts)
        screen = [(sx(x), sy(y)) for x, y in pts]
        lines.append(_svg_polyline(screen, colors.get(name, "#444444")))
        for x, y in screen:
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{colors.get(name, "#444444")}" />'
            )

    x_ticks = [10**p for p in range(math.floor(xmin), math.ceil(xmax) + 1)]
    y_ticks = [10**p for p in range(math.floor(ymin), math.ceil(ymax) + 1)]
    tick_svg: list[str] = []
    for val in x_ticks:
        x = sx(val)
        tick_svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e4e4e4" />')
        tick_svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle">1e{int(round(math.log10(val)))}</text>')
    for val in y_ticks:
        y = sy(val)
        tick_svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e4e4e4" />')
        tick_svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">1e{int(round(math.log10(val)))}</text>')

    legend: list[str] = []
    for idx, name in enumerate(["sampling", "K=1", "K=2", "K=3", "K=4"]):
        y = 62 + idx * 24
        color = colors[name]
        if name == "K=1":
            label = "K=1 unsupported"
        else:
            label = name
        legend.append(f'<line x1="740" y1="{y}" x2="780" y2="{y}" stroke="{color}" stroke-width="3" />')
        legend.append(f'<text x="790" y="{y + 5}" font-size="14">{label}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #222; }}
</style>
<text x="{left}" y="24" font-size="20" font-weight="700">Kaiming ReLU n=256 L=4: FLOPs vs Last-Layer Mean RMS Error</text>
{''.join(tick_svg)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.2" />
{''.join(lines)}
{''.join(circles)}
<text x="{left + plot_w / 2}" y="{height - 28}" font-size="16" text-anchor="middle">analytic FLOPs (log scale)</text>
<text x="24" y="{top + plot_h / 2}" font-size="16" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle">RMS error vs 2^24 truth (log scale)</text>
{''.join(legend)}
</svg>
"""
    path.write_text(svg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "relu_flop_error_results.json"
    svg_path = output_dir / "relu_flop_error.svg"

    print(f"[setup] device={device} dtype={dtype}", flush=True)
    linears = build_kaiming_relu_linears(
        n=args.n,
        layers=args.layers,
        device=device,
        dtype=dtype,
        seed=args.network_seed,
    )
    if args.resume and json_path.exists():
        payload = json.loads(json_path.read_text())
        print(f"[resume] loaded {json_path}", flush=True)
    else:
        payload = {
            "config": vars(args),
            "device": str(device),
            "dtype": str(dtype),
            "baseline": [],
            "ordinary_cp": {str(k): [] for k in args.k_values},
            "notes": [
                "K=1 ordinary-CP with hidden nonlinearities is unsupported because ReLU Wick coefficients need preactivation variance.",
                "FLOPs count matmul/ReLU for sampling and CP analytic diagnostics for empirical propagation.",
            ],
        }
    for k in args.k_values:
        payload.setdefault("ordinary_cp", {}).setdefault(str(k), [])

    truth_count = 2 ** args.true_power
    print(f"[truth] streaming 2^{args.true_power} = {truth_count} samples", flush=True)
    true_means = sample_relu_means(
        linears,
        sample_count=truth_count,
        batch_size=2 ** args.true_batch_power,
        n=args.n,
        device=device,
        dtype=dtype,
        seed=args.truth_seed,
    )
    true_last = true_means[-1]
    payload["true_last_mean_first8"] = [float(x) for x in true_last[:8]]
    payload["true_layer_mean_norms"] = [float(m.norm().item()) for m in true_means]
    save_json(json_path, payload)

    for power in args.baseline_powers:
        if any(row.get("sample_power") == power for row in payload["baseline"]):
            print(f"[baseline] skipping existing m=2^{power}", flush=True)
            continue
        count = 2**power
        print(f"[baseline] m=2^{power} ({count})", flush=True)
        means = sample_relu_means(
            linears,
            sample_count=count,
            batch_size=min(2 ** args.baseline_batch_power, count),
            n=args.n,
            device=device,
            dtype=dtype,
            seed=args.baseline_seed + power,
        )
        payload["baseline"].append(
            {
                "sample_power": power,
                "sample_count": count,
                "flops": baseline_forward_flops(sample_count=count, n=args.n, layers=args.layers),
                "last_layer_rms_error": rms_error(means[-1], true_last),
            }
        )
        save_json(json_path, payload)
        write_svg_plot(svg_path, payload)

    cp_sample_count = 2 ** args.cp_sample_power
    print(f"[cp] generating shared m=2^{args.cp_sample_power} samples", flush=True)
    cp_gen = torch.Generator(device=device).manual_seed(args.cp_sample_seed)
    cp_samples = torch.randn(cp_sample_count, args.n, generator=cp_gen, device=device, dtype=dtype)
    for k_max in args.k_values:
        if k_max < 2:
            if not payload["ordinary_cp"][str(k_max)]:
                payload["ordinary_cp"][str(k_max)].append(
                    {
                        "supported": False,
                        "reason": "hidden nonlinearities require K >= 2 to estimate preactivation variance",
                    }
                )
            save_json(json_path, payload)
            write_svg_plot(svg_path, payload)
            continue
        relu_degree_cap = min(k_max, args.relu_degree_cap)
        relu_spec = activation_spec_from_name(
            "relu",
            allow_nonpolynomial=True,
            hermite_degree_cap=relu_degree_cap,
        )
        stages = [(linear, relu_spec) for linear in linears]
        for m_power in args.m_powers:
            if any(
                row.get("supported", True) and row.get("rank_power") == m_power
                for row in payload["ordinary_cp"][str(k_max)]
            ):
                print(f"[cp] skipping existing K={k_max} M=2^{m_power}", flush=True)
                continue
            rank = 2**m_power
            print(f"[cp] K={k_max} M=2^{m_power} ({rank})", flush=True)
            config = OrdinaryCPConfig(
                k_max=k_max,
                delta=rank / cp_sample_count,
                allow_nonpolynomial=True,
                hermite_degree_cap=relu_degree_cap,
                reference_two_stage_nonlinear=False,
                record_factor_statistics=False,
            )
            start = time.time()
            with torch.no_grad():
                result = propagate_linear_activation_stages(
                    stages,
                    cp_samples,
                    config=config,
                    seed=args.cp_algorithm_seed + 1000 * k_max + m_power,
                    return_all=False,
                )
            elapsed = time.time() - start
            payload["ordinary_cp"][str(k_max)].append(
                {
                    "supported": True,
                    "k_max": k_max,
                    "rank_power": m_power,
                    "rank": rank,
                    "sample_count": cp_sample_count,
                    "relu_degree_cap": relu_degree_cap,
                    "flops": result.diagnostics.total_analytic_flops,
                    "flop_breakdown": result.diagnostics.analytic_flops_by_stage,
                    "last_layer_rms_error": rms_error(result.mean.detach().cpu(), true_last),
                    "elapsed_seconds": elapsed,
                    "diagram_counts": {
                        f"{layer},{order}": count
                        for (layer, order), count in result.diagnostics.diagrams_by_layer_and_order.items()
                    },
                }
            )
            save_json(json_path, payload)
            write_svg_plot(svg_path, payload)
            del result
            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_json(json_path, payload)
    write_svg_plot(svg_path, payload)
    print(f"[done] wrote {json_path}", flush=True)
    print(f"[done] wrote {svg_path}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--true-power", type=int, default=24)
    parser.add_argument("--true-batch-power", type=int, default=18)
    parser.add_argument("--baseline-powers", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18])
    parser.add_argument("--baseline-batch-power", type=int, default=18)
    parser.add_argument("--cp-sample-power", type=int, default=18)
    parser.add_argument("--m-powers", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18])
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--relu-degree-cap", type=int, default=2)
    parser.add_argument("--network-seed", type=int, default=123)
    parser.add_argument("--truth-seed", type=int, default=10_000)
    parser.add_argument("--baseline-seed", type=int, default=20_000)
    parser.add_argument("--cp-sample-seed", type=int, default=30_000)
    parser.add_argument("--cp-algorithm-seed", type=int, default=40_000)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(PACKAGE_DIR) / "experiment_outputs" / "relu_flop_error"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
