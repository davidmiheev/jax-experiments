"""Datasets, downloaded once into ``data/`` (gitignored).

MNIST comes from Google's CVDF mirror of LeCun's IDX files (the same files
torchvision fetches); each download is checked against the SHA-256 recorded
here, and normalised as ml-ladder's B2 does (mean 0.1307, std 0.3081).
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081  # ml-ladder B2 / torchvision convention


def _fetch(name: str) -> Path:
    path = DATA / "mnist" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL + name, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MNIST_FILES[name]:
        raise ValueError(f"{name}: sha256 {digest} does not match the recorded {MNIST_FILES[name]}")
    return path


def _idx(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def mnist(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """(images [n, 784] normalised float32, labels [n] int64)."""
    prefix = "train" if split == "train" else "t10k"
    images = _idx(_fetch(f"{prefix}-images-idx3-ubyte.gz")).reshape(-1, 784).astype(np.float32) / 255.0
    labels = _idx(_fetch(f"{prefix}-labels-idx1-ubyte.gz")).astype(np.int64)
    return (images - MNIST_MEAN) / MNIST_STD, labels


def balanced_subset(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Indices with ``per_class`` examples of every class, shuffled."""
    picks = [rng.choice(np.nonzero(labels == c)[0], per_class, replace=False) for c in np.unique(labels)]
    return rng.permutation(np.concatenate(picks))
