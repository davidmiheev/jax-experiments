"""enwik8 and the copy task, sampled exactly as ml-ladder's ``tools/gpu/c_context_scaling.py``.

Batches are byte-identical to ml-ladder's for the same seed: the same split
(the first 90M bytes for training, the next 5M for validation), the same
``np.random.default_rng(seed).integers(...)`` calls in the same order, and the
same non-overlapping validation windows (256 KiB, stride context + 1). With
weights ported from ml-ladder's own initialisation, the JAX run then starts
from the same point on the same data.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from jaxexp.data import DATA

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
ENWIK8_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
TRAIN_BYTES, VAL_BYTES = 90_000_000, 5_000_000  # the standard split, as ml-ladder
COPY_VOCAB = 32  # ml-ladder's copy task


def enwik8(data_dir: Path | None = None) -> np.ndarray:
    """All 100M bytes as int32 (vocabulary 256), downloaded once and checked by SHA-256."""
    path = (data_dir or DATA / "text") / "enwik8"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ENWIK8_URL, timeout=600) as response:
            path.write_bytes(zipfile.ZipFile(io.BytesIO(response.read())).read("enwik8"))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ENWIK8_SHA256:
        raise ValueError(f"enwik8 sha256 {digest} does not match {ENWIK8_SHA256}")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)


def split(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return data[:TRAIN_BYTES], data[TRAIN_BYTES : TRAIN_BYTES + VAL_BYTES]


class Batches:
    """ml-ladder's ``sample_batch``: uniform random windows of context + 1 bytes."""

    def __init__(self, train: np.ndarray, context: int, batch: int, seed: int) -> None:
        self.train, self.context, self.batch = train, context, batch
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(0, len(self.train) - self.context - 1, size=self.batch)
        chunk = np.stack([self.train[s : s + self.context + 1] for s in starts])
        return chunk[:, :-1], chunk[:, 1:]


def validation_windows(val: np.ndarray, context: int, eval_tokens: int = 262_144) -> np.ndarray:
    """Non-overlapping windows of context + 1 bytes: the same bytes for every model at a context."""
    n = eval_tokens // context
    starts = np.arange(n) * (context + 1)
    return np.stack([val[s : s + context + 1] for s in starts])


class CopyBatches:
    """ml-ladder's copy task: n random symbols, then the same n again; predict the next symbol."""

    def __init__(self, n: int, batch: int, seed: int) -> None:
        self.n, self.batch = n, batch
        self.rng = np.random.default_rng(seed)

    def __call__(self, size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        half = self.rng.integers(0, COPY_VOCAB, size=(size or self.batch, self.n))
        seq = np.concatenate([half, half], axis=1).astype(np.int32)
        return seq[:, :-1], seq[:, 1:]
