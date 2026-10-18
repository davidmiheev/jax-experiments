"""O3: the Lion and Muon arms do what they claim, and run inside N1's vmapped sweep."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import optax
import pytest
from flax import nnx

from jaxexp import sweep as S
from jaxexp import text
from jaxexp import train as TR
from jaxexp import transformer as T

DATA = np.random.default_rng(5).integers(0, 256, size=50_000).astype(np.int32)


def _model(seed=0):
    model = T.CharTransformer(256, 64, 4, 2, 32, positional="learned", numerics=T.TORCH, rngs=nnx.Rngs(seed))
    return TR.nnx_model(model)


def _first_update(opt, params, lr, schedule=1.0):
    grads = jax.tree.map(lambda p: jax.random.normal(jax.random.key(p.size), p.shape), params)
    state = opt.init(params)
    inject = state[1]
    state = (state[0], inject._replace(hyperparams={**inject.hyperparams, "learning_rate": jnp.float32(lr),
                                                    **({"schedule": jnp.float32(schedule)} if "schedule" in inject.hyperparams else {})}))
    updates, _ = opt.update(grads, state, params)
    return grads, updates


def _by_name(tree):
    return {tu.keystr(path): leaf for path, leaf in tu.tree_flatten_with_path(tree)[0]}


def test_lion_first_step_is_the_sign():
    """Lion's update is lr * sign(...): every element moves by exactly lr (no weight decay)."""
    params, _ = _model()
    _, updates = _first_update(TR.optimizer(0.0, "lion"), params, lr=3e-4)
    for leaf in jax.tree.leaves(updates):
        np.testing.assert_allclose(np.abs(np.asarray(leaf)), 3e-4, rtol=1e-6)


def test_muon_orthogonalises_block_matrices_and_adamw_does_the_rest():
    params, _ = _model()
    names = _by_name(params)
    labels = _by_name(TR.muon_labels(params))
    muon_keys = {k for k, v in labels.items() if v == "muon"}
    expected = {k for k in names if "blocks" in k and "kernel" in k}
    assert muon_keys == expected and len(expected) == 2 * 4  # 2 blocks x (qkv, projection, up, down)

    grads, updates = _first_update(TR.optimizer(0.0, "muon", adam_peak=2e-3), params, lr=0.02, schedule=0.5)
    ups = _by_name(updates)
    for key in expected:  # Newton-Schulz: singular values pushed towards 1 (Muon's quintic lands in ~[0.7, 1.2]),
        u = np.asarray(ups[key], np.float64)  # times optax's width scaling sqrt(max(1, fan_out / fan_in))
        sv = np.linalg.svd(u / (0.02 * np.sqrt(max(1.0, u.shape[1] / u.shape[0]))), compute_uv=False)
        assert sv.max() < 1.6 and sv.min() > 0.3, (key, sv.min(), sv.max())
    # Everything else: plain AdamW's first step at adam_peak * schedule = 1e-3, i.e.
    # -1e-3 * (sign(g) + 0.1 p) with the protocol's weight decay 0.1 (no Nesterov factor).
    gs = _by_name(grads)
    clip = min(1.0, 1.0 / float(optax.global_norm(grads)))  # the chain clips at global norm 1.0 first
    for key, leaf in ups.items():
        if key not in expected:
            g = np.asarray(gs[key], np.float64) * clip
            want = -1e-3 * (g / (np.abs(g) + 1e-8) + 0.1 * np.asarray(names[key]))  # Adam's eps = 1e-8
            np.testing.assert_allclose(np.asarray(leaf), want, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("kind", ["lion", "muon"])
def test_optimizers_run_vmapped_and_match_sequential(kind):
    grid, steps = [(0, 1e-3), (0, 4e-3)], 8
    opt = TR.optimizer(0.1, kind, adam_peak=2e-3 if kind == "muon" else None)
    sequential = []
    for seed, lr in grid:
        params, apply = _model(seed)
        sequential.append(TR.train(params, apply, text.Batches(DATA, 32, 4, seed), steps=steps, peak_lr=lr,
                                   weight_decay=0.1, opt=opt, log_every=1))
    sampler = text.Batches(DATA, 32, 4, 0)
    params, apply = _model(0)
    vmapped, _ = S.train_many([params, params], apply, [sampler, sampler], steps=steps,
                              peak_lrs=[lr for _, lr in grid], weight_decay=0.1, opt=opt, log_every=1)
    for seq, vm in zip(sequential, vmapped, strict=True):
        np.testing.assert_allclose([l for _, l in vm.history], [l for _, l in seq.history], rtol=1e-5)
    assert vmapped[0].history[-1][1] != vmapped[1].history[-1][1]
