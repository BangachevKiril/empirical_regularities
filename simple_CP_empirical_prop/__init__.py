"""Self-contained ordinary-CP empirical cumulant propagation.

The package stores every order-k ordinary cumulant as

    T[i_1, ..., i_k] = (1 / M) sum_t prod_u A_u[i_u, t].

It is intentionally isolated from the rest of the repository so the algorithm
can be tested and benchmarked independently.
"""

from .compression import compress_cp_sources, reference_compress_cp_sources
from .config import (
    MeanVariance,
    OrdinaryCPConfig,
    OrdinaryCPDiagnostics,
    OrdinaryCPResult,
    rank_from_delta,
)
from .diagrams import DiagramSpec, build_diagram_catalog
from .flops import FLOP_CONVENTION
from .hermite import (
    ActivationWickSpec,
    activation_spec_from_name,
    polynomial_wick_spec,
    relu_wick_coef,
    relu_wick_spec,
)
from .initialization import initialize_cumulant_cp, initialize_tower
from .linear import linear_cp, linear_tower
from .mean_prop import (
    MEAN_PROP_FLOP_CONVENTION,
    MeanPropDiagnostics,
    MeanPropResult,
    MeanPropState,
    empirical_mean_variance,
    mean_prop_activation,
    mean_prop_linear,
    mean_prop_mlp,
    mean_prop_stages,
)
from .moments import mean_and_variance
from .nonlinear import sample_diagram_cp, nonlinear_tower
from .propagate import ordinary_cp_mlp, propagate_linear_activation_stages
from .rng import RNGStreams, derive_seed
from .types import CPTensor, CPTower, WeightedCPSource, validate_tower, zero_cp

__all__ = [
    "ActivationWickSpec",
    "CPTensor",
    "CPTower",
    "DiagramSpec",
    "FLOP_CONVENTION",
    "MeanVariance",
    "MEAN_PROP_FLOP_CONVENTION",
    "MeanPropDiagnostics",
    "MeanPropResult",
    "MeanPropState",
    "OrdinaryCPConfig",
    "OrdinaryCPDiagnostics",
    "OrdinaryCPResult",
    "RNGStreams",
    "WeightedCPSource",
    "activation_spec_from_name",
    "build_diagram_catalog",
    "compress_cp_sources",
    "derive_seed",
    "empirical_mean_variance",
    "initialize_cumulant_cp",
    "initialize_tower",
    "linear_cp",
    "linear_tower",
    "mean_and_variance",
    "mean_prop_activation",
    "mean_prop_linear",
    "mean_prop_mlp",
    "mean_prop_stages",
    "nonlinear_tower",
    "ordinary_cp_mlp",
    "polynomial_wick_spec",
    "relu_wick_coef",
    "relu_wick_spec",
    "propagate_linear_activation_stages",
    "rank_from_delta",
    "reference_compress_cp_sources",
    "sample_diagram_cp",
    "validate_tower",
    "zero_cp",
]
