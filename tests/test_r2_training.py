"""R2 groundwork: recurrences equal PyTorch's, the schedule and sampling equal ml-ladder's,
and a few training steps equal ml-ladder's own ``train()``.

Tests that need torch or an ml-ladder checkout skip without them, as in R1.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxexp import recurrent as R
from jaxexp import text
from jaxexp import train as TR

ML_LADDER = Path(os.environ.get("ML_LADDER", Path(__file__).resolve().parents[2] / "ml-ladder"))


def test_schedule_matches_ml_ladder_formula():
    total, peak = 1000, 2e-3
    assert TR.lr_at(0, total, peak) == pytest.approx(peak / 50)  # warmup = 5% = 50 steps
    assert TR.lr_at(49, total, peak) == pytest.approx(peak)
    assert TR.lr_at(total - 1, total, peak) == pytest.approx(0.1 * peak, rel=1e-4)
    mid = 50 + (total - 50) // 2
    progress = (mid - 50) / (total - 50)
    assert TR.lr_at(mid, total, peak) == pytest.approx(peak * (0.1 + 0.45 * (1 + math.cos(math.pi * progress))))


def test_copy_batches_are_ml_ladder_sequences():
    batches = text.CopyBatches(n=8, batch=4, seed=0)
    inputs, targets = batches()
    seq = np.concatenate([inputs, targets[:, -1:]], axis=1)
    np.testing.assert_array_equal(seq[:, :8], seq[:, 8:])
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(seq[:, :8], rng.integers(0, text.COPY_VOCAB, size=(4, 8)))


def test_loss_scale_skips_overflow_and_backs_off():
    """A step whose *scaled* gradient overflows float32 is skipped and halves the scale (GradScaler).
    The unscaled problem is finite: logits ~1e37, gradient ~1e37; times the 2^16 scale it is inf."""
    params = {"w": jnp.ones((3,))}

    def apply(p, ids):
        return p["w"].sum() / 3 * 1e37 * jnp.arange(4.0) * (ids[..., None] == 1)

    opt = TR.optimizer(0.0)
    step = TR.make_step(apply, opt, amp=True)
    state = opt.init(params)
    ids = jnp.zeros((2, 3), jnp.int32)
    p1, state, scale1, _, finite = step(
        params, state, TR.LossScale(enabled=True).state(), ids.at[0, 0].set(1), ids, jnp.float32(1e-3)
    )
    assert not bool(finite)
    assert float(scale1[0]) == 2.0**15
    np.testing.assert_array_equal(p1["w"], params["w"])
    # The same step without poison is taken, and the scale holds.
    p2, _, scale2, _, finite2 = step(
        params, opt.init(params), TR.LossScale(enabled=True).state(), ids, ids, jnp.float32(1e-3)
    )
    assert bool(finite2) and float(scale2[0]) == 2.0**16


@pytest.fixture(scope="module")
def ladder():
    torch = pytest.importorskip("torch")
    if not (ML_LADDER / "tools" / "gpu" / "c_context_scaling.py").exists():
        pytest.skip(f"no ml-ladder checkout at {ML_LADDER}")
    sys.path[:0] = [str(ML_LADDER / "src"), str(ML_LADDER)]
    from tools.gpu import c_context_scaling

    return torch, c_context_scaling


@pytest.mark.parametrize("kind", ["gru", "lstm", "rnn"])
def test_recurrence_matches_torch(ladder, kind):
    torch, cc = ladder
    torch.manual_seed(0)
    model = cc.recurrent_lm(kind, 32, 2, vocab=50).eval()
    params = R.from_torch_state({k: v.detach().numpy() for k, v in model.state_dict().items()})
    ids = torch.randint(0, 50, (4, 40), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        reference = model(ids)[0].numpy()
    ours = np.asarray(R.apply(params, jnp.asarray(ids.numpy()), kind))
    assert np.linalg.norm(ours - reference) / np.linalg.norm(reference) < 1e-6


def test_training_steps_match_ml_ladder(ladder):
    """Ten steps of ml-ladder's train() and of jaxexp.train from the same weights and batches."""
    torch, cc = ladder
    torch.manual_seed(0)
    model = cc.recurrent_lm("gru", 32, 2)
    params = R.from_torch_state({k: v.detach().numpy() for k, v in model.state_dict().items()})
    data = np.random.default_rng(5).integers(0, 256, size=20_000).astype(np.int32)

    rng = np.random.default_rng(0)

    def sample_batch():
        starts = rng.integers(0, len(data) - 33, size=4)
        chunk = torch.from_numpy(np.stack([data[s : s + 33] for s in starts]).astype(np.int64))
        return chunk[:, :-1], chunk[:, 1:]

    info = cc.train(
        model,
        sample_batch,
        steps=10,
        peak_lr=2e-3,
        schedule="warmup-cosine",
        device=torch.device("cpu"),
        weight_decay=0.0,
        log_every=1,
    )
    result = TR.train(
        params,
        lambda p, ids: R.apply(p, ids, "gru"),
        text.Batches(data, 32, 4, 0),
        steps=10,
        peak_lr=2e-3,
        weight_decay=0.0,
        log_every=1,
    )
    theirs = np.array([loss for _, loss in info["train_history"]])
    ours = np.array([loss for _, loss in result.history])
    np.testing.assert_allclose(ours, theirs, rtol=5e-6)


def test_fp16_compute_tracks_fp32():
    """The fp16 matmul path stays close to fp32 at init (no overflow, sensible rounding)."""
    params = R.init_like_torch(jax.random.key(0), "gru", 64, 2)
    ids = jax.random.randint(jax.random.key(1), (4, 32), 0, 256)
    full = R.apply(params, ids, "gru")
    half = R.apply(params, ids, "gru", compute_dtype=jnp.float16)
    assert bool(jnp.all(jnp.isfinite(half)))
    assert float(jnp.linalg.norm(half - full) / jnp.linalg.norm(full)) < 5e-3
