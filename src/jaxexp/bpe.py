"""O6: ml-ladder's byte-exact BPE tokenizer, and a corpus large enough to be seen once.

``train_bpe`` and ``encode`` are ml-ladder's ``train_bpe`` / ``bpe_encode``
(``tools/gpu/c_context_scaling.py``, section 8 of Part C at scale): HuggingFace ``tokenizers``
BPE over the bytes read as latin-1 (one character per byte), all 256 bytes in the base
alphabet, split at word, number, punctuation and whitespace boundaries, 4096 tokens, fitted
on the first 20 MB of enwik8's training split. Every token spells a known number of bytes
(``token_bytes``), so bits per *byte* is exact.

enwik8's training split is 31.1M tokens, too few for a compute-optimal sweep of models
above ~1.5M parameters (at ~20 tokens per parameter). ``training_corpus("enwik9")`` is
enwik9 with the bytes enwik8's validation split comes from removed: enwik8 is enwik9's first
100M bytes, so the training stream is enwik9[0:90M] + enwik9[100M:1G], about 990M bytes, and
the validation bytes are exactly ml-ladder's.

Encoding is cached as uint16 next to the data and guarded by a file lock, so parallel jobs on
one machine tokenize once.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from jaxexp import text

ENWIK9_URL = "https://mattmahoney.net/dc/enwik9.zip"
VOCAB, FIT_BYTES = 4096, 20_000_000
SPLIT_PATTERN = r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+"


def train_bpe(text_latin1: str, vocab_size: int = VOCAB):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=[chr(i) for i in range(256)],
                                  show_progress=False)
    tokenizer.train_from_iterator(_newline_chunks(text_latin1), trainer)
    return tokenizer


def _newline_chunks(text_latin1: str, size: int = 1 << 20):
    start = 0
    while start < len(text_latin1):
        end = text_latin1.find("\n", start + size)
        end = len(text_latin1) if end < 0 else end + 1
        yield text_latin1[start:end]
        start = end


def encode(tokenizer, text_latin1: str) -> np.ndarray:
    ids = [np.asarray(e.ids, dtype=np.uint16) for e in tokenizer.encode_batch(list(_newline_chunks(text_latin1)))]
    return np.concatenate(ids)


def token_bytes(tokenizer) -> np.ndarray:
    return np.array([len(tokenizer.id_to_token(i)) for i in range(tokenizer.get_vocab_size())], dtype=np.int64)


def latin1(byte_values: np.ndarray) -> str:
    return byte_values.astype(np.uint8).tobytes().decode("latin-1")


def enwik9(data_dir: Path) -> np.ndarray:
    """All 1e9 bytes; checked against enwik8 (its first 1e8 bytes) rather than a recorded hash."""
    path = data_dir / "enwik9"
    with _locked(data_dir / "enwik9.lock"):
        if not path.exists():
            with urllib.request.urlopen(ENWIK9_URL, timeout=1800) as response:
                path.write_bytes(zipfile.ZipFile(io.BytesIO(response.read())).read("enwik9"))
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    first = np.asarray(raw[:100_000_000])
    if hashlib.sha256(first.tobytes()).hexdigest() != text.ENWIK8_SHA256:
        raise ValueError("enwik9's first 100M bytes are not enwik8")
    return raw


class _locked:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w")  # noqa: SIM115

    def __enter__(self):
        fcntl.flock(self.file, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def corpus(data_dir: Path, cache_dir: Path, which: str = "enwik9") -> dict:
    """{"tokenizer", "train", "val", "token_bytes", "meta"}: the tokenizer fitted on enwik8's first
    20 MB of training bytes, the training stream (``which``: "enwik8" = its 90M training bytes,
    "enwik9" = enwik9 without enwik8's validation and test bytes) and enwik8's first 2 MB of
    validation bytes, all encoded (cached)."""
    from tokenizers import Tokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw8 = text.enwik8(data_dir)
    train8, val8 = text.split(raw8)
    with _locked(cache_dir / "bpe.lock"):  # one lock for the tokenizer and every cache
        tok_path = cache_dir / "bpe-4096.json"
        if tok_path.exists():
            tokenizer = Tokenizer.from_file(str(tok_path))
        else:
            tokenizer = train_bpe(latin1(train8[:FIT_BYTES]))
            tokenizer.save(str(tok_path))
        lengths = token_bytes(tokenizer)
        val_path = cache_dir / "bpe-val.npy"
        if not val_path.exists():
            val_text = latin1(raw8[text.TRAIN_BYTES : text.TRAIN_BYTES + 2_000_000])
            val = encode(tokenizer, val_text)
            assert int(lengths[val].sum()) == len(val_text), "BPE lost or duplicated validation bytes"
            np.save(val_path, val)
        train_path = cache_dir / f"bpe-train-{which}.npy"
        if not train_path.exists():
            if which == "enwik8":
                parts = [train8]
            elif which == "enwik9":
                raw9 = enwik9(data_dir)
                # skip enwik8's validation and test bytes (90M..100M)
                parts = [raw9[: text.TRAIN_BYTES], raw9[100_000_000:]]
            else:
                raise ValueError(f"unknown corpus {which!r}")
            pieces, total_bytes = [], 0
            for part in parts:
                for start in range(0, len(part), 100_000_000):  # 100 MB at a time
                    chunk = latin1(np.asarray(part[start : start + 100_000_000]))
                    ids = encode(tokenizer, chunk)
                    assert int(lengths[ids].sum()) == len(chunk), "BPE lost or duplicated training bytes"
                    pieces.append(ids)
                    total_bytes += len(chunk)
            np.save(train_path, np.concatenate(pieces))
    train, val = np.load(train_path, mmap_mode="r"), np.load(cache_dir / "bpe-val.npy")
    meta = {"vocab": tokenizer.get_vocab_size(), "fit_bytes": FIT_BYTES, "corpus": which,
            "train_tokens": int(len(train)), "val_tokens": int(len(val)),
            "bytes_per_token_val": float(2_000_000 / len(val))}
    return {"tokenizer": tokenizer, "train": train, "val": val, "token_bytes": lengths, "meta": meta}
