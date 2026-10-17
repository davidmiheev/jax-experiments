"""ml-ladder's Part C training protocol (``tools/gpu/c_context_scaling.train``) in JAX.

- AdamW, betas (0.9, 0.95), eps 1e-8; weight decay on every parameter (one
  param group, as there), 0.1 for transformers and 0 for recurrences.
- Gradient clipping to global norm 1.0, after unscaling.
- ``lr_at``: 5% linear warmup ((step + 1) / warmup), then cosine to 10% of
  peak. The rate is set from the step index every step (``inject_hyperparams``),
  as torch sets it, so a skipped fp16 step does not shift the schedule.
- fp16: matmuls in float16 (the model's ``compute_dtype``) with dynamic loss
  scaling as ``torch.amp.GradScaler``'s defaults: initial scale 2^16, x2 after
  2000 finite steps, x0.5 and the step skipped on overflow. Cross-entropy is
  always computed in float32 from the logits.
- Per-position evaluation: mean cross-entropy at every position of fixed
  validation windows, in nats (``bits = nats / ln 2``).

A model is a pair ``(params, apply)`` with ``apply(params, ids) -> logits``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax


def lr_at(step: int, total: int, peak: float, schedule: str = "warmup-cosine", warmup_fraction: float = 0.05):
    """ml-ladder's schedule, verbatim."""
    if schedule == "constant":
        return peak
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def optimizer(weight_decay: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adamw)(learning_rate=0.0, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay),
    )


def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Per-token cross-entropy in nats, in float32."""
    return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)


@dataclass
class LossScale:
    """torch.amp.GradScaler's dynamic loss scale; ``enabled=False`` is a no-op (fp32)."""

    enabled: bool
    scale: float = 2.0**16
    growth_interval: int = 2000
    good_steps: int = 0
    skipped: int = 0

    def state(self):
        return (jnp.float32(self.scale if self.enabled else 1.0), jnp.int32(self.good_steps))


def make_step(apply: Callable, opt: optax.GradientTransformation, amp: bool):
    def loss_fn(params, inputs, targets, scale):
        loss = cross_entropy(apply(params, inputs), targets).mean()
        return loss * scale, loss

    @jax.jit
    def step(params, opt_state, scale_state, inputs, targets, lr):
        scale, good = scale_state
        grads, loss = jax.grad(loss_fn, has_aux=True)(params, inputs, targets, scale)
        grads = jax.tree.map(lambda g: g / scale, grads)
        finite = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)]))
        inject = opt_state[1]
        opt_state = (opt_state[0], inject._replace(hyperparams={**inject.hyperparams, "learning_rate": lr}))
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        keep = lambda new, old: jax.tree.map(lambda a, b: jnp.where(finite, a, b), new, old)  # noqa: E731
        params, opt_state = keep(new_params, params), keep(new_opt_state, opt_state)
        if amp:
            grow = finite & (good + 1 >= 2000)
            scale = jnp.where(finite, jnp.where(grow, scale * 2.0, scale), scale * 0.5)
            good = jnp.where(finite & ~grow, good + 1, 0)
        return params, opt_state, (scale, good), loss, finite

    return step


@dataclass
class TrainResult:
    params: object
    history: list = field(default_factory=list)  # (step, loss) every log_every steps
    train_seconds: float = 0.0
    compile_seconds: float = 0.0
    skipped_steps: int = 0
    final_scale: float = 1.0


def train(
    params,
    apply: Callable,
    sample_batch: Callable[[], tuple[np.ndarray, np.ndarray]],
    *,
    steps: int,
    peak_lr: float,
    weight_decay: float,
    amp: bool = False,
    schedule: str = "warmup-cosine",
    log_every: int = 250,
    label: str = "",
) -> TrainResult:
    opt = optimizer(weight_decay)
    opt_state = opt.init(params)
    step_fn = make_step(apply, opt, amp)
    scale_state = LossScale(enabled=amp).state()
    result = TrainResult(params)
    started = time.time()
    for i in range(steps):
        inputs, targets = sample_batch()
        lr = jnp.float32(lr_at(i, steps, peak_lr, schedule))
        params, opt_state, scale_state, loss, finite = step_fn(
            params, opt_state, scale_state, jnp.asarray(inputs, jnp.int32), jnp.asarray(targets, jnp.int32), lr
        )
        if i == 0:
            jax.block_until_ready(loss)
            result.compile_seconds = time.time() - started
        if not bool(finite):
            result.skipped_steps += 1
        if i % log_every == 0 or i == steps - 1:
            result.history.append((i, float(loss)))
            if label:
                print(f"  [{label}] step {i:5d} loss {float(loss):.4f} ({time.time() - started:.0f}s)", flush=True)
    jax.block_until_ready(params)
    result.params = params
    result.train_seconds = time.time() - started
    result.final_scale = float(scale_state[0])
    return result


def per_position_loss(params, apply: Callable, windows: np.ndarray, batch_size: int) -> np.ndarray:
    """Mean cross-entropy (nats) at every position of the validation windows."""
    loss_at = jax.jit(lambda p, x, y: cross_entropy(apply(p, x), y).sum(0))
    total, count = None, 0
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        summed = np.asarray(loss_at(params, jnp.asarray(chunk[:, :-1]), jnp.asarray(chunk[:, 1:])), np.float64)
        total = summed if total is None else total + summed
        count += len(chunk)
    return total / count


def nnx_model(model) -> tuple[object, Callable]:
    """(state, apply) for an NNX module, so the loop above treats it like a pytree model."""
    from flax import nnx

    graphdef, state = nnx.split(model)

    def apply(state, ids, **kwargs):
        return nnx.merge(graphdef, state)(ids, **kwargs)

    return state, apply
