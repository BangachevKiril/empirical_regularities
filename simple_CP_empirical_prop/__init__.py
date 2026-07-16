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
from .cov_prop import (
    COV_PROP_FLOP_CONVENTION,
    CovPropDiagnostics,
    CovPropResult,
    CovPropState,
    cov_prop_activation,
    cov_prop_linear,
    custom_cov_prop_stages,
    empirical_mean_covariance,
    regular_cov_prop_stages,
)
from .data_generators import (
    ICA,
    MLP_dataset,
    GeneratedDataset,
    Image_dataset,
    Isotropic_Gaussian,
    Wishart,
    ica,
    image_dataset,
    isotropic_gaussian,
    mlp_dataset,
    wishart,
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
from .k4 import (
    CPBlockSource,
    K4CumulantState,
    K4OrdinaryCPConfig,
    K4OrdinaryCPDiagnostics,
    K4OrdinaryCPResult,
    RetainedDiagramSpec,
    exact_covariance_factor_view,
    exact_dense_diagram,
    initialize_k4_state,
    linear_k4_state,
    load_k4_retained_diagrams,
    mean_variance_from_k4_state,
    nonlinear_k4_state,
    ordinary_cp_mlp_k4,
    propagate_k4_stages,
    sample_labelled_diagram_columns,
    validate_k4_state,
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
    "COV_PROP_FLOP_CONVENTION",
    "CPBlockSource",
    "CPTensor",
    "CPTower",
    "CovPropDiagnostics",
    "CovPropResult",
    "CovPropState",
    "DiagramSpec",
    "FLOP_CONVENTION",
    "GeneratedDataset",
    "ICA",
    "Image_dataset",
    "Isotropic_Gaussian",
    "K4CumulantState",
    "K4OrdinaryCPConfig",
    "K4OrdinaryCPDiagnostics",
    "K4OrdinaryCPResult",
    "MLP_dataset",
    "MeanVariance",
    "MEAN_PROP_FLOP_CONVENTION",
    "MeanPropDiagnostics",
    "MeanPropResult",
    "MeanPropState",
    "OrdinaryCPConfig",
    "OrdinaryCPDiagnostics",
    "OrdinaryCPResult",
    "RNGStreams",
    "RetainedDiagramSpec",
    "WeightedCPSource",
    "Wishart",
    "activation_spec_from_name",
    "build_diagram_catalog",
    "compress_cp_sources",
    "cov_prop_activation",
    "cov_prop_linear",
    "custom_cov_prop_stages",
    "derive_seed",
    "empirical_mean_covariance",
    "empirical_mean_variance",
    "exact_covariance_factor_view",
    "exact_dense_diagram",
    "ica",
    "image_dataset",
    "initialize_k4_state",
    "initialize_cumulant_cp",
    "initialize_tower",
    "isotropic_gaussian",
    "linear_cp",
    "linear_k4_state",
    "linear_tower",
    "load_k4_retained_diagrams",
    "mean_and_variance",
    "mean_variance_from_k4_state",
    "mean_prop_activation",
    "mean_prop_linear",
    "mean_prop_mlp",
    "mean_prop_stages",
    "mlp_dataset",
    "nonlinear_k4_state",
    "nonlinear_tower",
    "ordinary_cp_mlp",
    "ordinary_cp_mlp_k4",
    "polynomial_wick_spec",
    "propagate_k4_stages",
    "relu_wick_coef",
    "relu_wick_spec",
    "propagate_linear_activation_stages",
    "rank_from_delta",
    "regular_cov_prop_stages",
    "reference_compress_cp_sources",
    "sample_diagram_cp",
    "sample_labelled_diagram_columns",
    "validate_k4_state",
    "validate_tower",
    "wishart",
    "zero_cp",
]
