"""N3: the differentiable AdamW is ml-ladder's, and its LR gradient is the true derivative."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jaxexp import hypergrad as H
from jaxexp import train as TR
from jaxexp import transformer as T


def _tiny(dtype=None):
    # Fused attention for the float64 checks: explicit attention accumulates scores in float32 by
    # design (R2), which would put float32 noise into a float64 loss.
    numerics = T.Numerics(attention="fused" if dtype == jnp.float64 else "explicit")
    model = T.CharTransformer(256, 32, 2, 2, 16, numerics=numerics, rngs=nnx.Rngs(0))
    params, apply = TR.nnx_model(model)
    if dtype is not None:
        params = jax.tree.map(lambda a: a.astype(dtype), params)
    return params, apply


def _windows(k, batch=4, t=16, seed=0):
    return jax.random.randint(jax.random.key(seed), (k, batch, t + 1), 0, 256)


def test_adamw_step_equals_the_training_optimizer():
    params, apply = _tiny()
    opt = TR.optimizer(0.1, "adamw")
    opt_state = opt.init(params)
    mine, state = params, H.adamw_init(params)
    _, loss_fn = H.make_unroll(apply)
    windows = _windows(5)
    for k, lr in enumerate([1e-3, 3e-3, 2e-3, 5e-4, 4e-3]):
        grads = jax.grad(loss_fn)(params, windows[k])
        inject = opt_state[1]
        opt_state = (opt_state[0], inject._replace(hyperparams={**inject.hyperparams, "learning_rate": lr}))
        updates, opt_state = opt.update(grads, opt_state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
        mine, state = H.adamw_step(mine, state, jax.grad(loss_fn)(mine, windows[k]), lr)
    # Entries whose gradient is exactly zero (the attention key bias: softmax ignores a shift shared
    # by all keys; embedding rows of bytes absent from the batch) get Adam-amplified round-off of
    # size lr, different in each implementation; compare the rest.
    first = jax.grad(loss_fn)(_tiny()[0], windows[0])
    for a, b, g in zip(jax.tree.leaves(params), jax.tree.leaves(mine), jax.tree.leaves(first), strict=True):
        live = np.abs(np.asarray(g)) > 1e-7
        np.testing.assert_allclose(np.asarray(a)[live], np.asarray(b)[live], rtol=1e-5, atol=1e-6)


def _mlp():
    """A small attention-free byte model in float64: smooth to round-off, for finite differences."""
    k = jax.random.split(jax.random.key(0), 3)
    params = {"emb": jax.random.normal(k[0], (256, 16)) * 0.1, "w": jax.random.normal(k[1], (16, 32)) * 0.3,
              "b": jnp.zeros(32), "out": jax.random.normal(k[2], (32, 256)) * 0.2}
    params = jax.tree.map(lambda a: a.astype(jnp.float64), params)
    return params, lambda p, ids: jnp.tanh(p["emb"][ids] @ p["w"] + p["b"]) @ p["out"]


def test_hypergradient_is_the_derivative():
    """d L_val(theta_K) / d log lr, by reverse mode through K=4 AdamW steps (clipping, moments,
    bias correction, weight decay), against central finite differences in float64."""
    with jax.enable_x64(True):
        params, apply = _mlp()
        meta = H.make_meta_loss(apply)
        windows, val = _windows(4), _windows(1, seed=9)[0]
        state = H.adamw_init(params)
        f = jax.jit(lambda log_lr: meta(log_lr, params, state, windows, val)[0])
        x = jnp.log(jnp.float64(3e-3))
        g = jax.grad(lambda v: meta(v, params, state, windows, val)[0])(x)
        h = 1e-5
        np.testing.assert_allclose(float(g), float((f(x + h) - f(x - h)) / (2 * h)), rtol=1e-7)
        # Per-step learning rates: a [K] gradient whose sum is the shared-LR gradient.
        gk = jax.grad(lambda v: meta(v, params, state, windows, val)[0])(jnp.full((4,), x))
        assert gk.shape == (4,)
        np.testing.assert_allclose(float(gk.sum()), float(g), rtol=1e-10)


def test_transformer_hypergradient_reverse_equals_forward_mode():
    """On the transformer, finite differences are limited by non-smoothness at the 1e-9 level
    (they agree to 0.13% at the best step size); reverse and forward mode must agree exactly."""
    with jax.enable_x64(True):
        params, apply = _tiny(jnp.float64)
        meta = H.make_meta_loss(apply)
        windows, val = _windows(4), _windows(1, seed=9)[0]
        state = H.adamw_init(params)
        f = lambda log_lr: meta(log_lr, params, state, windows, val)[0]  # noqa: E731
        x = jnp.log(jnp.float64(3e-3))
        reverse = jax.grad(f)(x)
        forward = jax.jvp(f, (x,), (jnp.float64(1.0),))[1]
        np.testing.assert_allclose(float(reverse), float(forward), rtol=1e-6)


def test_remat_changes_nothing():
    """In float64; in float32 the two differ by ~4e-4 relative, Adam amplifying round-off."""
    with jax.enable_x64(True):
        params, apply = _tiny(jnp.float64)
        windows, val = _windows(3), _windows(1, seed=5)[0]
        state = H.adamw_init(params)
        x = jnp.log(jnp.float64(2e-3))
        ref = jax.value_and_grad(lambda v: H.make_meta_loss(apply, remat=False)(v, params, state, windows, val)[0])(x)
        got = jax.value_and_grad(lambda v: H.make_meta_loss(apply, remat=True)(v, params, state, windows, val)[0])(x)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-9)
