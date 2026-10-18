"""N3: the learning rate's gradient, through K steps of training.

Training is a function of its learning rates: ``theta_K = Step(...Step(theta_0, lr_1)..., lr_K)``.
So the validation loss after K steps, ``L_val(theta_K)``, has a gradient with respect to
every ``lr_k`` (a *hypergradient*), and ``jax.grad`` computes it by differentiating through
the optimiser itself: Adam's moments, the bias correction, the clipping, the weight decay.

Everything here is a pure function of arrays:

- ``adamw_init`` / ``adamw_step``: ml-ladder's AdamW (betas 0.9 / 0.95, eps 1e-8, decoupled
  weight decay, gradient clipping at global norm 1.0), written out so that it can be
  differentiated with respect to the LR; ``tests/test_n3_hypergrad.py`` checks it equals
  ``jaxexp.train.optimizer("adamw")`` step for step.
- ``unroll``: K training steps in ``lax.scan``. ``remat=True`` checkpoints each step, so
  reverse mode stores one (params, m, v) per step, not every step's activations.
- ``meta_loss``: validation loss after K steps, as a function of the log learning rates.
  The LR of step k is ``exp(log_lr[k])`` (one value broadcasts to all K).

fp32 throughout: the second-order pass through fp16 loss scaling is not attempted.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax


B1, B2, EPS, CLIP = 0.9, 0.95, 1e-8, 1.0


def _sqrt(x):
    """sqrt with a zero derivative at 0 instead of an infinite one. A parameter whose gradient is
    exactly zero (the attention key bias) has v = 0, and reverse mode would multiply 1/sqrt(0)
    by zero: NaN. The forward value is unchanged."""
    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def adamw_init(params) -> dict:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, params), "count": jnp.zeros((), jnp.int32)}


def adamw_step(params, state: dict, grads, lr, weight_decay: float = 0.1):
    """One AdamW step at learning rate ``lr`` (a traced scalar). Returns (params, state)."""
    norm = optax.tree.norm(grads)
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, CLIP / norm), grads)
    count = state["count"] + 1
    m = jax.tree.map(lambda m, g: B1 * m + (1 - B1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: B2 * v + (1 - B2) * g * g, state["v"], grads)
    c1 = 1 - B1 ** count.astype(jnp.float32)
    c2 = 1 - B2 ** count.astype(jnp.float32)
    params = jax.tree.map(
        lambda p, m, v: p - lr * ((m / c1) / (_sqrt(v / c2) + EPS) + weight_decay * p), params, m, v
    )
    return params, {"m": m, "v": v, "count": count}


def make_unroll(apply: Callable, *, weight_decay: float = 0.1, remat: bool = True):
    """``unroll(params, state, batches, lrs) -> (params, state, train_losses)`` for ``batches``
    of shape [K, B, T + 1] (inputs and targets as one window) and ``lrs`` of shape [K]."""

    def loss_fn(params, window):
        logits = apply(params, window[:, :-1])
        # Cross-entropy in float32, or float64 when the model runs in it (the tests' derivative checks).
        logits = logits.astype(jnp.promote_types(logits.dtype, jnp.float32))
        return optax.softmax_cross_entropy_with_integer_labels(logits, window[:, 1:]).mean()

    def body(carry, xs):
        params, state = carry
        window, lr = xs
        loss, grads = jax.value_and_grad(loss_fn)(params, window)
        params, state = adamw_step(params, state, grads, lr, weight_decay)
        return (params, state), loss

    if remat:
        body = jax.checkpoint(body)

    def unroll(params, state, batches, lrs):
        (params, state), losses = jax.lax.scan(body, (params, state), (batches, lrs))
        return params, state, losses

    return unroll, loss_fn


def make_meta_loss(apply: Callable, *, weight_decay: float = 0.1, remat: bool = True):
    """``meta_loss(log_lr, params, state, batches, val_window) -> (val_loss, (params, state, train_losses))``,
    differentiable in ``log_lr`` ([K] or a scalar broadcast to K steps)."""
    unroll, loss_fn = make_unroll(apply, weight_decay=weight_decay, remat=remat)

    def meta_loss(log_lr, params, state, batches, val_window):
        lrs = jnp.broadcast_to(jnp.exp(log_lr), (batches.shape[0],))
        params, state, losses = unroll(params, state, batches, lrs)
        return loss_fn(params, val_window), (params, state, losses)

    return meta_loss
