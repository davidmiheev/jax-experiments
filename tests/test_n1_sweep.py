"""N1: K copies trained as one vmapped program equal K sequential runs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxexp import recurrent as R
from jaxexp import sweep as S
from jaxexp import text
from jaxexp import train as TR
from jaxexp import transformer as T

DATA = np.random.default_rng(5).integers(0, 256, size=50_000).astype(np.int32)


def _gru(seed, amp):
    params = R.init_like_torch(jax.random.key(seed), "gru", 32, 2)
    dtype = jnp.float16 if amp else None
    return params, lambda p, ids: R.apply(p, ids, "gru", compute_dtype=dtype)


def _transformer(seed, amp):
    # Explicit attention: XLA:CPU cannot lower the fused path's fp16 x fp16 -> fp32 backward
    # (the GPU runs use fused; the two agree bit for bit in fp16, test_r1_parity).
    numerics = T.Numerics(attention="explicit", compute_dtype=jnp.float16 if amp else None)
    model = T.CharTransformer(256, 64, 4, 2, 32, positional="rope", numerics=numerics, rngs=nnx.Rngs(seed))
    return TR.nnx_model(model)


@pytest.mark.parametrize("build", [_gru, _transformer], ids=["gru", "transformer"])
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "fp16"])
def test_vmapped_copies_equal_sequential_runs(build, amp):
    """Two seeds x two learning rates: copies of the same seed share one batch stream, as in
    ml-ladder's sweeps, and each copy must match its own sequential run."""
    grid = [(0, 2e-3), (0, 8e-3), (1, 2e-3), (1, 8e-3)]
    steps = 12
    sequential = []
    for seed, lr in grid:
        params, apply = build(seed, amp)
        result = TR.train(params, apply, text.Batches(DATA, 32, 4, seed), steps=steps, peak_lr=lr,
                          weight_decay=0.1, amp=amp, log_every=1)
        sequential.append(result)
    samplers = {seed: text.Batches(DATA, 32, 4, seed) for seed in (0, 1)}
    params_list = [build(seed, amp)[0] for seed, _ in grid]
    apply = build(0, amp)[1]
    vmapped, timing = S.train_many(params_list, apply, [samplers[s] for s, _ in grid], steps=steps,
                                   peak_lrs=[lr for _, lr in grid], weight_decay=0.1, amp=amp, log_every=1)
    assert timing["copies"] == 4
    # Compared over all parameters at once, and by the function they compute. Per tensor, the
    # attention key bias differs by up to 5%: its true gradient is exactly zero (a constant added
    # to every key leaves the softmax unchanged), so Adam turns round-off into full-size steps,
    # in either program, without changing the model's outputs.
    tol = 3e-3 if amp else 1e-6
    held_out = jnp.asarray(text.Batches(DATA, 32, 4, 99)()[0])

    def flat(tree):
        return np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in jax.tree.leaves(tree)])

    for seq, vm in zip(sequential, vmapped, strict=True):
        np.testing.assert_allclose([l for _, l in vm.history], [l for _, l in seq.history], rtol=tol)
        a, b = flat(vm.params), flat(seq.params)
        assert np.linalg.norm(a - b) <= tol * np.linalg.norm(b)
        out_vm, out_seq = (np.asarray(apply(r.params, held_out), np.float64) for r in (vm, seq))
        assert np.linalg.norm(out_vm - out_seq) <= tol * np.linalg.norm(out_seq)
        assert vm.skipped_steps == seq.skipped_steps and vm.final_scale == seq.final_scale


def test_learning_rates_differ_per_copy():
    """Same seed and data, different peak LRs: the copies must diverge (the LR really is per copy)."""
    params, apply = _gru(0, False)
    sampler = text.Batches(DATA, 32, 4, 0)
    results, _ = S.train_many([params, params], apply, [sampler, sampler], steps=5, peak_lrs=[1e-4, 1e-2],
                              weight_decay=0.0)
    assert results[0].history[-1][1] != results[1].history[-1][1]
