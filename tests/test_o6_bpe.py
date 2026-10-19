"""O6: the tokenizer is ml-ladder's, and its tokens spell every byte exactly once."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from jaxexp import bpe  # noqa: E402

LADDER = Path(os.environ.get("ML_LADDER", Path(__file__).resolve().parents[2] / "ml-ladder"))
SCRIPT = LADDER / "tools" / "gpu" / "c_context_scaling.py"


def _sample_text() -> str:
    rng = np.random.default_rng(0)
    words = ["the", "cat", "sat", "on", "a", "mat", "[[link]]", "1999", "&amp;", "<page>", "naïve"]
    parts = []
    for _ in range(4000):
        w = words[rng.integers(len(words))]
        parts.append(w + (" " if rng.random() < 0.8 else "\n"))
    return "".join(parts).encode("utf-8").decode("latin-1")  # bytes as latin-1, as on enwik8


def test_tokens_spell_every_byte_once():
    text = _sample_text()
    tok = bpe.train_bpe(text, vocab_size=300)
    ids = bpe.encode(tok, text)
    lengths = bpe.token_bytes(tok)
    assert lengths.min() >= 1 and 256 < tok.get_vocab_size() <= 300  # a tiny corpus can run out of merges
    assert int(lengths[ids].sum()) == len(text)
    # decoding the token strings back gives the text
    assert "".join(tok.id_to_token(int(i)) for i in ids) == text


@pytest.mark.skipif(not SCRIPT.exists(), reason="needs ../ml-ladder")
def test_same_tokenizer_as_ml_ladder():
    spec = importlib.util.spec_from_file_location("c_context_scaling", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = _sample_text()
    ours, theirs = bpe.train_bpe(text, vocab_size=300), module.train_bpe(text, 300)
    assert ours.get_vocab() == theirs.get_vocab()
    np.testing.assert_array_equal(bpe.encode(ours, text), module.bpe_encode(theirs, text))
