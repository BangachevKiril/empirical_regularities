"""K=1 mean-prop ReLU flop-vs-error experiment."""

from __future__ import annotations

import sys

if sys.path:
    sys.path.pop(0)

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PACKAGE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from simple_CP_empirical_prop.cov_prop import custom_cov_prop_stages
from simple_CP_empirical_prop.mean_prop import mean_prop_stages
from simple_CP_empirical_prop.run_relu_flop_experiment import (
    baseline_forward_flops,
    build_kaiming_relu_linears,
    rms_error,
    sample_relu_means,
    save_json,
)


def _svg_star(cx: float, cy: float, radius: float, color: str) -> str:
    points: list[str] = []
    inner = radius * 0.45
    for idx in range(10):
        angle = -math.pi / 2.0 + idx * math.pi / 5.0
        r = radius if idx % 2 == 0 else inner
        points.append(f"{cx + r * math.cos(angle):.2f},{cy + r * math.sin(angle):.2f}")
    return (
        f'<polygon points="{" ".join(points)}" fill="{color}" '
        f'stroke="{color}" stroke-width="1.2" />'
    )


def write_svg_plot(path: Path, payload: dict[str, Any]) -> None:
    width_n = payload.get("config", {}).get("n", "?")
    layer_count = payload.get("config", {}).get("layers", "?")
    series = {
        "sampling": [
            (row["flops"], row["last_layer_rms_error"]) for row in payload["baseline"]
        ],
        "mean_prop K=1": [
            (row["flops"], row["last_layer_rms_error"]) for row in payload["mean_prop"]
        ],
        "cov_prop": [
            (row["flops"], row["last_layer_rms_error"]) for row in payload.get("cov_prop", [])
        ],
    }
    points = [pt for pts in series.values() for pt in pts if pt[0] > 0 and pt[1] > 0]
    if not points:
        return
    xmin = min(math.log10(x) for x, _ in points)
    xmax = max(math.log10(x) for x, _ in points)
    ymin = min(math.log10(y) for _, y in points)
    ymax = max(math.log10(y) for _, y in points)
    xpad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ypad = 0.10 * (ymax - ymin if ymax > ymin else 1.0)
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

    colors = {"sampling": "#111111", "mean_prop K=1": "#d62728", "cov_prop": "#1f77b4"}
    lines: list[str] = []
    markers: list[str] = []
    for name, pts in series.items():
        pts = sorted(pts)
        if not pts:
            continue
        coords = [(sx(x), sy(y)) for x, y in pts]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        lines.append(
            f'<polyline points="{point_text}" fill="none" stroke="{colors[name]}" stroke-width="2.5" />'
        )
        if name == "cov_prop":
            for row in sorted(payload.get("cov_prop", []), key=lambda r: r["flops"]):
                x = sx(row["flops"])
                y = sy(row["last_layer_rms_error"])
                if row.get("mode") == "cp":
                    markers.append(_svg_star(x, y, 6.5, colors[name]))
                else:
                    markers.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.4" fill="{colors[name]}" />'
                    )
        else:
            for x, y in coords:
                markers.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{colors[name]}" />'
                )

    ticks: list[str] = []
    for val in [10**p for p in range(math.floor(xmin), math.ceil(xmax) + 1)]:
        x = sx(val)
        ticks.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e4e4e4" />')
        ticks.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle">1e{int(round(math.log10(val)))}</text>')
    for val in [10**p for p in range(math.floor(ymin), math.ceil(ymax) + 1)]:
        y = sy(val)
        ticks.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e4e4e4" />')
        ticks.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">1e{int(round(math.log10(val)))}</text>')

    legend = """
<line x1="740" y1="62" x2="780" y2="62" stroke="#111111" stroke-width="3" />
<text x="790" y="67" font-size="14">sampling</text>
<line x1="740" y1="88" x2="780" y2="88" stroke="#d62728" stroke-width="3" />
<text x="790" y="93" font-size="14">mean_prop K=1</text>
<line x1="740" y1="114" x2="780" y2="114" stroke="#1f77b4" stroke-width="3" />
<polygon points="756.00,107.50 758.19,111.99 763.14,112.69 759.57,116.18 760.41,121.10 756.00,118.78 751.59,121.10 752.43,116.18 748.86,112.69 753.81,111.99" fill="#1f77b4" stroke="#1f77b4" stroke-width="1.2" />
<circle cx="777" cy="114" r="4.4" fill="#1f77b4" />
<text x="790" y="119" font-size="14">cov_prop (star = CP)</text>
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<style>text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #222; }}</style>
<text x="{left}" y="24" font-size="20" font-weight="700">Kaiming ReLU n={width_n} L={layer_count}: K=1 Mean Prop and Cov Prop</text>
{''.join(ticks)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.2" />
{''.join(lines)}
{''.join(markers)}
<text x="{left + plot_w / 2}" y="{height - 28}" font-size="16" text-anchor="middle">analytic FLOPs (log scale)</text>
<text x="24" y="{top + plot_h / 2}" font-size="16" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle">RMS error vs 2^24 truth (log scale)</text>
{legend}
</svg>
"""
    path.write_text(svg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "relu_mean_prop_results.json"
    svg_path = output_dir / "relu_mean_prop.svg"

    linears = build_kaiming_relu_linears(
        n=args.n,
        layers=args.layers,
        device=device,
        dtype=dtype,
        seed=args.network_seed,
    )
    payload: dict[str, Any] = {
        "config": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "baseline": [],
        "mean_prop": [],
        "cov_prop": [],
        "notes": [
            "K=1 mean_prop tracks only coordinatewise mean and variance.",
            "Linear variance propagation uses a diagonal-covariance approximation.",
            "ReLU moments use Gaussian marginal formulas.",
            "custom cov_prop uses dense empirical covariance propagation when M >= n and order-2 CP propagation when M < n.",
            "mean_prop and cov_prop materialize/consume only min(M, m) samples; FLOPs use that consumed sample count.",
        ],
    }

    truth_count = 2 ** args.true_power
    print(f"[truth] 2^{args.true_power}", flush=True)
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
    save_json(json_path, payload)

    stages = [(linear, "relu") for linear in linears]
    prop_available_count = 2 ** args.prop_sample_power
    payload["prop_available_sample_count"] = prop_available_count
    for power in args.sample_powers:
        count = 2**power
        prop_used_count = min(count, prop_available_count)
        print(f"[sampling] m=2^{power}", flush=True)
        sample_means = sample_relu_means(
            linears,
            sample_count=count,
            batch_size=min(2 ** args.batch_power, count),
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
                "last_layer_rms_error": rms_error(sample_means[-1], true_last),
            }
        )

        gen = torch.Generator(device=device).manual_seed(args.mean_prop_seed + power)
        samples = torch.randn(prop_used_count, args.n, generator=gen, device=device, dtype=dtype)
        with torch.no_grad():
            result = mean_prop_stages(
                stages,
                samples,
                sample_budget=count,
                variance_min=args.variance_min,
            )
        payload["mean_prop"].append(
            {
                "sample_power": power,
                "available_sample_count": prop_available_count,
                "sample_budget": count,
                "sample_count": result.diagnostics.sample_count,
                "used_sample_count": result.diagnostics.used_sample_count,
                "materialized_sample_count": prop_used_count,
                "flops": result.diagnostics.total_analytic_flops,
                "flop_breakdown": result.diagnostics.analytic_flops_by_stage,
                "last_layer_rms_error": rms_error(result.mean.detach().cpu(), true_last),
            }
        )
        save_json(json_path, payload)
        write_svg_plot(svg_path, payload)
        del samples, result
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[cov_prop] M=2^{power}", flush=True)
        cov_gen = torch.Generator(device=device).manual_seed(args.cov_prop_seed + power)
        cov_samples = torch.randn(prop_used_count, args.n, generator=cov_gen, device=device, dtype=dtype)
        start = time.time()
        with torch.no_grad():
            cov_result = custom_cov_prop_stages(
                stages,
                cov_samples,
                rank=count,
                relu_degree_cap=args.cov_prop_relu_degree_cap,
                cp_seed=args.cov_prop_algorithm_seed + power,
                quadrature_degree=args.cov_prop_quadrature_degree,
                variance_min=args.variance_min,
            )
        elapsed = time.time() - start
        payload["cov_prop"].append(
            {
                "sample_power": power,
                "available_sample_count": prop_available_count,
                "sample_budget": count,
                "sample_count": cov_result.diagnostics.sample_count,
                "used_sample_count": cov_result.diagnostics.used_sample_count,
                "materialized_sample_count": prop_used_count,
                "rank": count,
                "mode": cov_result.diagnostics.mode,
                "used_cp": cov_result.diagnostics.mode == "cp",
                "relu_degree_cap": args.cov_prop_relu_degree_cap,
                "quadrature_degree": args.cov_prop_quadrature_degree,
                "flops": cov_result.diagnostics.total_analytic_flops,
                "flop_breakdown": cov_result.diagnostics.analytic_flops_by_stage,
                "last_layer_rms_error": rms_error(cov_result.mean.detach().cpu(), true_last),
                "elapsed_seconds": elapsed,
            }
        )
        save_json(json_path, payload)
        write_svg_plot(svg_path, payload)
        del cov_samples, cov_result
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
    parser.add_argument("--sample-powers", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18])
    parser.add_argument("--batch-power", type=int, default=18)
    parser.add_argument("--prop-sample-power", type=int, default=18)
    parser.add_argument("--network-seed", type=int, default=123)
    parser.add_argument("--truth-seed", type=int, default=10_000)
    parser.add_argument("--baseline-seed", type=int, default=20_000)
    parser.add_argument("--mean-prop-seed", type=int, default=50_000)
    parser.add_argument("--cov-prop-seed", type=int, default=60_000)
    parser.add_argument("--cov-prop-algorithm-seed", type=int, default=70_000)
    parser.add_argument("--cov-prop-relu-degree-cap", type=int, default=2)
    parser.add_argument("--cov-prop-quadrature-degree", type=int, default=40)
    parser.add_argument("--variance-min", type=float, default=1e-10)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(PACKAGE_DIR) / "experiment_outputs" / "relu_mean_prop_k1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
