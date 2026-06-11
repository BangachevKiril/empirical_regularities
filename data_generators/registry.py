from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import ModuleType

import torch

from data_generators.base import DataGenerator
from data_generators.gaussian import GaussianDataGenerator
from data_generators.gaussian import known_parameter_estimation as gaussian_known
from data_generators.gaussian import unknown_parameter_estimation as gaussian_unknown
from data_generators.gaussian_lowrank import GaussianLowRankDataGenerator
from data_generators.gaussian_lowrank import known_parameter_estimation as gaussian_lowrank_known
from data_generators.gaussian_lowrank import unknown_parameter_estimation as gaussian_lowrank_unknown
from data_generators.ica import ICADataGenerator
from data_generators.ica import known_parameter_estimation as ica_known
from data_generators.ica import unknown_parameter_estimation as ica_unknown
from data_generators.subspace_gaussian import SubspaceGaussianDataGenerator
from data_generators.subspace_gaussian import known_parameter_estimation as subspace_gaussian_known
from data_generators.subspace_gaussian import unknown_parameter_estimation as subspace_gaussian_unknown


@dataclass(frozen=True)
class DataGenerationProcess:
    name: str
    generator_cls: type[DataGenerator]
    known_estimation: ModuleType
    unknown_estimation: ModuleType
    seed_arg: str


PROCESSES: dict[str, DataGenerationProcess] = {
    "ica": DataGenerationProcess(
        name="ica",
        generator_cls=ICADataGenerator,
        known_estimation=ica_known,
        unknown_estimation=ica_unknown,
        seed_arg="ica_seed",
    ),
    "gaussian": DataGenerationProcess(
        name="gaussian",
        generator_cls=GaussianDataGenerator,
        known_estimation=gaussian_known,
        unknown_estimation=gaussian_unknown,
        seed_arg="gaussian_seed",
    ),
    "gaussian_lowrank": DataGenerationProcess(
        name="gaussian_lowrank",
        generator_cls=GaussianLowRankDataGenerator,
        known_estimation=gaussian_lowrank_known,
        unknown_estimation=gaussian_lowrank_unknown,
        seed_arg="lowrank_seed",
    ),
    "subspace_gaussian": DataGenerationProcess(
        name="subspace_gaussian",
        generator_cls=SubspaceGaussianDataGenerator,
        known_estimation=subspace_gaussian_known,
        unknown_estimation=subspace_gaussian_unknown,
        seed_arg="subspace_seed",
    ),
}


def get_process(name: str) -> DataGenerationProcess:
    try:
        return PROCESSES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PROCESSES))
        raise ValueError(f"Unknown data generation process {name!r}; expected one of {valid}.") from exc


def make_data_generator(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> DataGenerator:
    process = get_process(args.input_distribution)
    seed = getattr(args, process.seed_arg)
    return process.generator_cls(
        n=args.n,
        seed=seed,
        p=args.p,
        device=device,
        dtype=dtype,
    )


def estimation_module(*, data_generation: str, parameter_estimation: str) -> ModuleType:
    process = get_process(data_generation)
    if parameter_estimation == "known":
        return process.known_estimation
    if parameter_estimation == "unknown":
        return process.unknown_estimation
    raise ValueError(f"Unknown parameter_estimation mode: {parameter_estimation!r}.")
