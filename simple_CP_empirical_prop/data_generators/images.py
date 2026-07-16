"""Image dataset generators with optional random projection."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import torch

from .base import GeneratedDataset


_CANONICAL_IMAGE_NAMES = {
    "MNIST": "MNIST",
    "CIFAR": "CIFAR",
    "CIFAR10": "CIFAR",
    "IMAGENET": "ImageNET",
    "IMAGENET1K": "ImageNET",
    "IMAGE_NET": "ImageNET",
}

_DEFAULT_IMAGE_SIZE = {
    "MNIST": 28,
    "CIFAR": 32,
    "ImageNET": 224,
}


def _canonical_name(name: str) -> str:
    key = name.replace("-", "_").replace(" ", "_").upper()
    if key not in _CANONICAL_IMAGE_NAMES:
        raise ValueError("name must be one of MNIST, CIFAR, ImageNET")
    return _CANONICAL_IMAGE_NAMES[key]


def _infer_square_size_from_dim(n: int) -> int:
    if n % 3 != 0:
        raise ValueError("unprojected image dimension n must equal 3 * k^2")
    k_float = math.sqrt(n // 3)
    k = int(k_float)
    if 3 * k * k != n:
        raise ValueError("unprojected image dimension n must equal 3 * k^2")
    return k


def _load_torchvision_dataset(
    *,
    name: str,
    root: str | Path,
    train: bool,
    split: str | None,
    download: bool,
) -> Any:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "image_dataset requires torchvision. Install torchvision or use the "
            "synthetic generators."
        ) from exc

    root_path = Path(root)
    if name == "MNIST":
        return datasets.MNIST(str(root_path), train=train, download=download)
    if name == "CIFAR":
        return datasets.CIFAR10(str(root_path), train=train, download=download)

    image_split = split or ("train" if train else "val")
    try:
        return datasets.ImageNet(str(root_path), split=image_split)
    except Exception:
        folder = root_path / image_split
        if not folder.exists():
            folder = root_path
        return datasets.ImageFolder(str(folder))


def _sample_indices(
    *,
    dataset_size: int,
    m: int,
    seed: int | None,
    shuffle: bool,
) -> list[int]:
    if dataset_size < 1:
        raise ValueError("image dataset is empty")
    if m < 1:
        raise ValueError("m must be a positive integer")
    gen = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    if m <= dataset_size:
        if shuffle:
            return torch.randperm(dataset_size, generator=gen)[:m].tolist()
        return list(range(m))
    return torch.randint(dataset_size, (m,), generator=gen).tolist()


def _pil_to_png_rgb_vector(image: Any, *, image_size: int) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("image_dataset requires Pillow for PNG conversion") from exc
    import numpy as np

    pil = image.convert("RGB")
    if pil.size != (image_size, image_size):
        resample = getattr(Image, "Resampling", Image).BILINEAR
        pil = pil.resize((image_size, image_size), resample=resample)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        array = np.asarray(reopened.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array.reshape(-1) / 255.0)


def _vectors_from_dataset(
    dataset: Any,
    *,
    indices: list[int],
    image_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    vectors = []
    for idx in indices:
        item = dataset[int(idx)]
        image = item[0] if isinstance(item, tuple) else item
        vectors.append(_pil_to_png_rgb_vector(image, image_size=image_size))
    return torch.stack(vectors, dim=0).to(device=device, dtype=dtype)


def image_dataset(
    *,
    name: str,
    projection: str | None,
    n: int | None,
    m: int,
    root: str | Path,
    seed: int | None = None,
    train: bool = True,
    split: str | None = None,
    download: bool = False,
    shuffle: bool = True,
    image_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GeneratedDataset:
    """Load images as PNG vectors, optionally applying one fixed Gaussian projection.

    With ``projection=None``, output samples live in ``R^(3 k^2)``. If ``n`` is
    supplied in that mode, it must equal ``3 k^2``. With
    ``projection="random"``, ``n`` is the requested projected dimension, and the
    returned tensors include the fixed matrix ``W`` with shape ``[n, 3 k^2]``.
    """
    canonical = _canonical_name(name)
    if m < 1:
        raise ValueError("m must be a positive integer")
    if projection not in (None, "random"):
        raise ValueError('projection must be None or "random"')

    if projection is None and n is not None:
        k = _infer_square_size_from_dim(n)
    else:
        k = image_size or _DEFAULT_IMAGE_SIZE[canonical]
    input_dim = 3 * k * k
    if projection is None:
        output_dim = input_dim
        if n is not None and n != output_dim:
            raise ValueError("for projection=None, n must equal 3 * k^2")
    else:
        if n is None or n < 1:
            raise ValueError('projection="random" requires positive output dimension n')
        output_dim = n

    dev = torch.device("cpu" if device is None else device)
    dataset = _load_torchvision_dataset(
        name=canonical,
        root=root,
        train=train,
        split=split,
        download=download,
    )
    indices = _sample_indices(
        dataset_size=len(dataset),
        m=m,
        seed=seed,
        shuffle=shuffle,
    )
    vectors = _vectors_from_dataset(
        dataset,
        indices=indices,
        image_size=k,
        dtype=dtype,
        device=dev,
    )

    tensors: dict[str, torch.Tensor] = {}
    if projection == "random":
        gen = None if seed is None else torch.Generator(device=dev).manual_seed(seed + 1)
        w = torch.randn(output_dim, input_dim, generator=gen, device=dev, dtype=dtype) / math.sqrt(input_dim)
        samples = vectors @ w.transpose(0, 1)
        tensors["W"] = w
    else:
        samples = vectors

    return GeneratedDataset(
        name="image_dataset",
        samples=samples,
        metadata={
            "name": canonical,
            "projection": projection,
            "n": output_dim,
            "m": m,
            "seed": seed,
            "root": str(root),
            "train": train,
            "split": split,
            "download": download,
            "image_size": k,
            "input_dim": input_dim,
            "pixel_scaling": "PNG RGB values divided by 255",
            "indices": indices,
            "projection_entry_variance": None if projection is None else 1.0 / input_dim,
        },
        tensors=tensors,
    )


Image_dataset = image_dataset
