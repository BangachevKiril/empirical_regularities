"""Dataset generators for empirical regularity experiments."""

from .base import GeneratedDataset
from .images import Image_dataset, image_dataset
from .synthetic import (
    ICA,
    MLP_dataset,
    Isotropic_Gaussian,
    Wishart,
    ica,
    isotropic_gaussian,
    mlp_dataset,
    wishart,
)

__all__ = [
    "GeneratedDataset",
    "ICA",
    "Image_dataset",
    "Isotropic_Gaussian",
    "MLP_dataset",
    "Wishart",
    "ica",
    "image_dataset",
    "isotropic_gaussian",
    "mlp_dataset",
    "wishart",
]
