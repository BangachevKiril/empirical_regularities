"""End-to-end ordinary-CP propagation drivers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from numpy.polynomial import Polynomial

from .config import OrdinaryCPConfig, OrdinaryCPDiagnostics, OrdinaryCPResult, rank_from_delta
from .flops import (
    final_mean_flops,
    initialization_flops,
    linear_layer_flops,
    moment_extraction_flops,
    nonlinear_layer_flops,
)
from .hermite import ActivationWickSpec, activation_spec_from_name, polynomial_wick_spec
from .initialization import initialize_tower
from .linear import linear_tower
from .moments import mean_and_variance
from .nonlinear import nonlinear_tower
from .rng import RNGStreams
from .types import CPTower, validate_tower


def _record_factor_stats(
    diagnostics: OrdinaryCPDiagnostics,
    label: str,
    tower: CPTower,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    with torch.no_grad():
        factors = [factor for cp in tower.values() for factor in cp.factors]
        if not factors:
            return
        abs_max = max(float(factor.abs().max().item()) for factor in factors)
        rms_num = sum(float((factor.to(torch.float64) ** 2).mean().item()) for factor in factors)
        diagnostics.factor_abs_max_by_stage[label] = abs_max
        diagnostics.factor_rms_by_stage[label] = (rms_num / len(factors)) ** 0.5


def _check_supported_mlp(mlp: object) -> None:
    for attr in ("layernorm", "batch_layernorm"):
        if bool(getattr(mlp, attr, False)):
            raise NotImplementedError(f"{attr} is not supported by ordinary CP")


def _linears_and_names_from_mlp(mlp: object) -> tuple[list[torch.nn.Linear], list[str]]:
    if hasattr(mlp, "Ws"):
        linears = list(getattr(mlp, "Ws"))
        names_obj = getattr(mlp, "nonlin_names", [])
        if isinstance(names_obj, str):
            names = [names_obj] * max(0, len(linears) - 1)
        else:
            names = list(names_obj)
        if len(names) < max(0, len(linears) - 1):
            names.extend(["square"] * (len(linears) - 1 - len(names)))
        return linears, names
    if isinstance(mlp, torch.nn.Sequential):
        linears: list[torch.nn.Linear] = []
        names: list[str] = []
        pending_linear = False
        for module in mlp:
            if isinstance(module, torch.nn.Linear):
                linears.append(module)
                pending_linear = True
            elif pending_linear:
                if isinstance(module, torch.nn.ReLU):
                    names.append("relu")
                else:
                    raise NotImplementedError(
                        f"unsupported sequential activation {module.__class__.__name__}"
                    )
                pending_linear = False
            else:
                raise NotImplementedError(f"unsupported module {module.__class__.__name__}")
        return linears, names[: max(0, len(linears) - 1)]
    raise TypeError("mlp must expose Ws or be a torch.nn.Sequential")


def _activation_for_layer(
    layer_index: int,
    names: Sequence[str],
    config: OrdinaryCPConfig,
) -> ActivationWickSpec:
    if config.polynomial_by_layer is not None:
        if layer_index >= len(config.polynomial_by_layer):
            raise ValueError("not enough polynomials for hidden layers")
        poly = config.polynomial_by_layer[layer_index]
        if not isinstance(poly, Polynomial):
            poly = Polynomial(poly)
        return polynomial_wick_spec(poly, name=f"poly_layer_{layer_index}")
    name = names[layer_index] if layer_index < len(names) else "square"
    return activation_spec_from_name(
        name,
        allow_nonpolynomial=config.allow_nonpolynomial,
        hermite_degree_cap=config.hermite_degree_cap,
    )


def propagate_linear_activation_stages(
    stages: Sequence[tuple[torch.nn.Linear, ActivationWickSpec | None]],
    samples: torch.Tensor,
    *,
    config: OrdinaryCPConfig,
    seed: int = 0,
    return_all: bool = False,
) -> OrdinaryCPResult:
    """Propagate through explicit (linear, optional activation) stages."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, input_dim]")
    if config.k_max == 4:
        from .k4 import K4OrdinaryCPConfig, propagate_k4_stages

        k4_config = K4OrdinaryCPConfig(
            variance_min=config.variance_min,
            variance_max=config.variance_max,
            mean_abs_clip=config.mean_abs_clip,
            activate_final=True,
            reference_two_stage_nonlinear=True,
            allow_nonpolynomial=config.allow_nonpolynomial,
            hermite_degree_cap=config.hermite_degree_cap,
        )
        return propagate_k4_stages(
            stages,
            samples,
            rank=rank_from_delta(samples.shape[0], config.delta),
            config=k4_config,
            seed=seed,
            return_all=return_all,
        )
    if any(activation is not None for _, activation in stages) and config.k_max < 2:
        raise ValueError("hidden nonlinearities require k_max >= 2")
    rank = rank_from_delta(samples.shape[0], config.delta)
    if samples.shape[0] < config.k_max:
        raise ValueError("m must be at least k_max for grouped initialization")
    streams = RNGStreams(seed, device="cpu")
    diagnostics = OrdinaryCPDiagnostics(
        seed=seed,
        sample_count=samples.shape[0],
        rank=rank,
        k_max=config.k_max,
        delta=config.delta,
    )
    tower, source_counts, unused = initialize_tower(
        samples,
        k_max=config.k_max,
        rank=rank,
        streams=streams,
        shuffle=config.shuffle_initial_groups,
    )
    diagnostics.initialization_sources_by_order.update(source_counts)
    diagnostics.unused_samples_by_order.update(unused)
    diagnostics.add_flops(
        "initialization",
        initialization_flops(k_max=config.k_max, width=samples.shape[1], rank=rank),
    )
    _record_factor_stats(diagnostics, "init", tower, enabled=config.record_factor_statistics)

    pre_towers: list[CPTower] | None = [] if return_all else None
    act_towers: list[CPTower] | None = [tower] if return_all else None

    for layer_index, (linear, activation) in enumerate(stages):
        diagnostics.add_flops(
            f"layer_{layer_index}_linear",
            linear_layer_flops(
                k_max=config.k_max,
                input_width=int(linear.weight.shape[1]),
                output_width=int(linear.weight.shape[0]),
                rank=rank,
                has_bias=linear.bias is not None,
            ),
        )
        pre_tower = linear_tower(tower, linear)
        validate_tower(pre_tower, k_max=config.k_max, rank=rank)
        _record_factor_stats(
            diagnostics,
            f"layer_{layer_index}_pre",
            pre_tower,
            enabled=config.record_factor_statistics,
        )
        if pre_towers is not None:
            pre_towers.append(pre_tower)
        if activation is None:
            tower = pre_tower
            continue
        mean_var = mean_and_variance(pre_tower, config)
        diagnostics.add_flops(
            f"layer_{layer_index}_moments",
            moment_extraction_flops(width=pre_tower[1].width, rank=rank),
        )
        diagnostics.mean_clip_fraction_by_layer.append(mean_var.mean_clip_fraction)
        diagnostics.variance_clip_fraction_by_layer.append(mean_var.variance_clip_fraction)
        diagnostics.raw_variance_min_by_layer.append(float(mean_var.raw_variance.min().item()))
        diagnostics.raw_variance_max_by_layer.append(float(mean_var.raw_variance.max().item()))
        _, nonlinear_by_order = nonlinear_layer_flops(
            k_max=config.k_max,
            width=pre_tower[1].width,
            rank=rank,
            hermite_degree=activation.degree,
            reference_two_stage=config.reference_two_stage_nonlinear,
        )
        for order, flops in nonlinear_by_order.items():
            diagnostics.add_flops(
                f"layer_{layer_index}_nonlinear_order_{order}",
                flops,
            )
        tower, diagram_counts = nonlinear_tower(
            pre_tower,
            mean_var,
            activation,
            config,
            rank=rank,
            streams=streams,
            layer_index=layer_index,
        )
        for order, count in diagram_counts.items():
            diagnostics.diagrams_by_layer_and_order[(layer_index, order)] = count
        _record_factor_stats(
            diagnostics,
            f"layer_{layer_index}_act",
            tower,
            enabled=config.record_factor_statistics,
        )
        if act_towers is not None:
            act_towers.append(tower)
    diagnostics.add_flops(
        "final_mean",
        final_mean_flops(width=tower[1].width, rank=rank),
    )
    return OrdinaryCPResult(
        mean=tower[1].mean_vector(),
        final_tower=tower,
        pre_towers=pre_towers,
        act_towers=act_towers,
        diagnostics=diagnostics,
    )


def ordinary_cp_mlp(
    mlp: object,
    samples: torch.Tensor,
    *,
    k_max: int,
    delta: float,
    config: OrdinaryCPConfig | None = None,
    seed: int = 0,
    return_all: bool = False,
) -> OrdinaryCPResult:
    """Estimate the output mean of an MLP using ordinary-CP cumulant propagation."""
    if config is None:
        config = OrdinaryCPConfig(k_max=k_max, delta=delta)
    elif config.k_max != k_max or config.delta != delta:
        raise ValueError("explicit k_max/delta conflict with config")
    _check_supported_mlp(mlp)
    if k_max == 4:
        from .k4 import K4OrdinaryCPConfig, ordinary_cp_mlp_k4

        k4_config = K4OrdinaryCPConfig(
            variance_min=config.variance_min,
            variance_max=config.variance_max,
            mean_abs_clip=config.mean_abs_clip,
            activate_final=False,
            reference_two_stage_nonlinear=True,
            polynomial_by_layer=config.polynomial_by_layer,
            allow_nonpolynomial=config.allow_nonpolynomial,
            hermite_degree_cap=config.hermite_degree_cap,
        )
        return ordinary_cp_mlp_k4(
            mlp,
            samples,
            rank=rank_from_delta(samples.shape[0], delta),
            config=k4_config,
            seed=seed,
            activate_final=False,
            return_all=return_all,
        )
    linears, names = _linears_and_names_from_mlp(mlp)
    if not linears:
        raise ValueError("mlp must contain at least one linear layer")
    hidden_count = max(0, len(linears) - 1)
    if hidden_count and config.k_max < 2:
        raise ValueError("hidden nonlinearities require k_max >= 2")
    if config.polynomial_by_layer is not None and len(config.polynomial_by_layer) < hidden_count:
        raise ValueError("polynomial_by_layer must cover every hidden layer")
    stages: list[tuple[torch.nn.Linear, ActivationWickSpec | None]] = []
    for layer_index, linear in enumerate(linears):
        is_last = layer_index == len(linears) - 1
        activation = None if is_last else _activation_for_layer(layer_index, names, config)
        stages.append((linear, activation))
    return propagate_linear_activation_stages(
        stages,
        samples,
        config=config,
        seed=seed,
        return_all=return_all,
    )
