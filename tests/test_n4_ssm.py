"""N4: the parallel scan is the recurrence, and the LRU / selective-SSM models are causal and ~4M."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxexp import ssm


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_associative_scan_is_the_recurrence(dtype):
    key = jax.random.key(0)
    shape = (3, 37, 5)
    a = jax.random.uniform(key, shape, minval=0.5, maxval=0.99).astype(dtype)
    b = jax.random.normal(jax.random.fold_in(key, 1), shape).astype(dtype)
    if dtype == jnp.complex64:
        a = a * jnp.exp(1j * jax.random.uniform(jax.random.fold_in(key, 2), shape))
        b = b + 1j * jax.random.normal(jax.random.fold_in(key, 3), shape)
    h, loop = np.zeros((3, 5), np.complex128), []
    for t in range(shape[1]):
        h = np.asarray(a[:, t]) * h + np.asarray(b[:, t])
        loop.append(h)
    loop = np.stack(loop, 1)
    for mode in ("associative", "sequential"):
        np.testing.assert_allclose(np.asarray(ssm.linear_scan(a, b, mode)), loop, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", ["lru", "mamba"])
def test_scan_modes_agree_in_value_and_gradient(kind):
    params = ssm.KINDS[kind][1](jax.random.key(0), d=32, layers=2)
    ids = jax.random.randint(jax.random.key(1), (2, 40), 0, 256)
    out = {m: ssm.apply(params, ids, kind, scan=m) for m in ("associative", "sequential")}
    np.testing.assert_allclose(out["associative"], out["sequential"], rtol=1e-4, atol=1e-4)

    def loss(p, mode):
        return ssm.apply(p, ids, kind, scan=mode).mean()

    ga, gs = (jax.grad(loss)(params, m) for m in ("associative", "sequential"))
    for x, y in zip(jax.tree.leaves(ga), jax.tree.leaves(gs), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-3, atol=1e-6)
    # Remat recomputes the same function.
    np.testing.assert_allclose(ssm.apply(params, ids, kind, remat=True), out["associative"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", ["lru", "mamba"])
def test_causal(kind):
    params = ssm.KINDS[kind][1](jax.random.key(0), d=32, layers=2)
    ids = jax.random.randint(jax.random.key(1), (2, 30), 0, 256)
    t = 17
    changed = ids.at[:, t].set((ids[:, t] + 1) % 256)
    before, after = ssm.apply(params, ids, kind), ssm.apply(params, changed, kind)
    np.testing.assert_allclose(before[:, :t], after[:, :t], rtol=1e-6, atol=1e-6)
    assert not np.allclose(before[:, t], after[:, t])


def test_default_sizes_are_about_4m_like_r2s_transformer():
    lru = ssm.parameter_count(ssm.init_lru(jax.random.key(0)))
    mamba = ssm.parameter_count(ssm.init_mamba(jax.random.key(0)))
    assert 3.6e6 < lru < 4.6e6 and 3.6e6 < mamba < 4.6e6, (lru, mamba)


def test_initial_ranges():
    lru = ssm.init_lru(jax.random.key(0))
    modulus = np.abs(np.asarray(ssm.lru_lambda(lru["blocks"][0])))
    assert modulus.min() >= 0.9 - 1e-6 and modulus.max() <= 0.999 + 1e-6
    mamba = ssm.init_mamba(jax.random.key(0))
    dt = np.asarray(jax.nn.softplus(mamba["blocks"][0]["dt_proj"]["b"]))
    assert dt.min() >= 1e-3 * (1 - 1e-4) and dt.max() <= 1e-1 * (1 + 1e-4)


@pytest.mark.parametrize("t,chunk", [(40, 16), (64, 64), (37, 8)])  # padding, one chunk, ragged
def test_chunked_ssd_is_the_recurrence(t, chunk):
    key = jax.random.key(0)
    ks = jax.random.split(key, 4)
    bsz, h, p, n = 2, 3, 4, 5
    x = jax.random.normal(ks[0], (bsz, t, h, p))
    dta = -jax.random.uniform(ks[1], (bsz, t, h), minval=0.01, maxval=0.5)
    b, c = jax.random.normal(ks[2], (bsz, t, n)), jax.random.normal(ks[3], (bsz, t, n))
    np.testing.assert_allclose(ssm.ssd(x, dta, b, c, chunk=chunk), ssm.ssd_sequential(x, dta, b, c), rtol=1e-4, atol=1e-4)


def test_mamba2_modes_causality_and_size():
    params = ssm.init_mamba2(jax.random.key(0), d=64, layers=2, head_dim=16, state=8)
    ids = jax.random.randint(jax.random.key(1), (2, 50), 0, 256)
    chunked = ssm.apply(params, ids, "mamba2", scan="chunked")
    np.testing.assert_allclose(chunked, ssm.apply(params, ids, "mamba2", scan="sequential"), rtol=1e-4, atol=1e-4)
    changed = ids.at[:, 30].set((ids[:, 30] + 1) % 256)
    np.testing.assert_allclose(ssm.apply(params, changed, "mamba2")[:, :30], chunked[:, :30], rtol=1e-6, atol=1e-6)
    assert 3.6e6 < ssm.parameter_count(ssm.init_mamba2(jax.random.key(0))) < 4.6e6


def test_mamba2_gradients_are_finite_and_match_the_recurrence():
    """_segsum's -inf entries must not leak NaN into the backward pass."""
    params = ssm.init_mamba2(jax.random.key(0), d=64, layers=2, head_dim=16, state=8)
    ids = jax.random.randint(jax.random.key(1), (2, 40), 0, 256)
    ga, gs = (jax.grad(lambda p, m=m: ssm.apply(p, ids, "mamba2", scan=m).mean())(params) for m in ("chunked", "sequential"))
    for x, y in zip(jax.tree.leaves(ga), jax.tree.leaves(gs), strict=True):
        assert np.all(np.isfinite(np.asarray(x)))
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-3, atol=1e-6)
