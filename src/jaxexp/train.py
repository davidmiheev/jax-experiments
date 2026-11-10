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
    """ml-ladder's schedule, verbatim; plus ``warmup-constant`` (N2: the same warmup, then flat, so
    that steps-to-a-target-loss does not depend on the run's total length)."""
    if schedule == "constant":
        return peak
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "warmup-constant":
        return peak
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def muon_labels(params):
    """"muon" for the hidden 2D weight matrices inside the transformer blocks (attention qkv and
    projection, MLP up and down); "adam" for embeddings, the output head, biases and LayerNorm."""
    import jax.tree_util as tu

    def label(path, leaf):
        key = tu.keystr(path)
        return "muon" if ("blocks" in key and "kernel" in key and leaf.ndim == 2) else "adam"

    return tu.tree_map_with_path(label, params)


def _muon(learning_rate, schedule, adam_peak, weight_decay, adam_weight_decay):
    # Assembled from parts rather than optax.contrib.muon, whose nesterov=True also turns its
    # AdamW branch into NAdam (first step 1.47x larger): here the non-matrix parameters get
    # exactly the protocol's AdamW, at their own peak on the same schedule.
    from optax.contrib import MuonDimensionNumbers
    from optax.transforms import MaskedNode

    def dims(updates):  # every unmasked leaf is a (reduction, output) = (0, 1) matrix
        masked = lambda x: isinstance(x, MaskedNode)  # noqa: E731
        return jax.tree.map(lambda x: x if masked(x) else MuonDimensionNumbers(), updates, is_leaf=masked)

    muon = optax.chain(
        optax.contrib.scale_by_muon(beta=0.95, nesterov=True, weight_dimension_numbers=dims),  # 5 NS steps
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )
    adam = optax.adamw(adam_peak * schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=adam_weight_decay)
    return optax.partition({"muon": muon, "adam": adam}, muon_labels)


def optimizer(weight_decay: float, kind: str = "adamw", adam_peak: float | None = None) -> optax.GradientTransformation:
    """Clipping at global norm 1.0, then one of:

    - ``adamw``: ml-ladder's AdamW, betas (0.9, 0.95), eps 1e-8 (R2's protocol, unchanged);
    - ``lion``: Lion, betas (0.9, 0.99), decoupled weight decay ``weight_decay`` (times the LR);
    - ``muon``: Muon (Nesterov momentum 0.95, 5 Newton-Schulz steps, optax's width scaling
      sqrt(max(1, fan_out / fan_in)), decoupled weight decay ``weight_decay`` times the LR) on the
      blocks' 2D matrices; ml-ladder's AdamW (0.9, 0.95; wd 0.1) at peak ``adam_peak`` on the rest.

    The peak LR is injected per step as the hyperparameter ``learning_rate``; Muon's AdamW part
    follows the same schedule through the injected multiplier ``schedule``."""
    if kind == "adamw":
        inner = optax.inject_hyperparams(optax.adamw)(
            learning_rate=0.0, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay
        )
    elif kind == "lion":
        inner = optax.inject_hyperparams(optax.lion)(learning_rate=0.0, b1=0.9, b2=0.99, weight_decay=weight_decay)
    elif kind == "muon":
        assert adam_peak is not None, "muon needs adam_peak for its AdamW part"
        inner = optax.inject_hyperparams(_muon, static_args=("adam_peak", "weight_decay", "adam_weight_decay"))(
            learning_rate=0.0, schedule=0.0, adam_peak=adam_peak, weight_decay=weight_decay, adam_weight_decay=0.1
        )
    else:
        raise ValueError(f"unknown optimizer {kind!r}")
    return optax.chain(optax.clip_by_global_norm(1.0), inner)


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
        # ``lr`` is the learning rate, or a dict of injected hyperparameters (e.g. Muon's ``schedule``).
        values = lr if isinstance(lr, dict) else {"learning_rate": lr}
        hyper = {**inject.hyperparams, **{k: v for k, v in values.items() if k in inject.hyperparams}}
        opt_state = (opt_state[0], inject._replace(hyperparams=hyper))
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
    measurements: list = field(default_factory=list)  # (step, on_log(step, params)) at logged steps


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
    opt: optax.GradientTransformation | None = None,
    on_log: Callable | None = None,
    measure_every: int | None = None,
) -> TrainResult:
    """``on_log(step, params) -> dict``, if given, runs every ``measure_every`` steps (default:
    ``log_every``) and at the last step, on the parameters before that step's update; its results
    are kept in ``TrainResult.measurements`` and its time is excluded from ``train_seconds``."""
    opt = opt or optimizer(weight_decay)
    opt_state = opt.init(params)
    step_fn = make_step(apply, opt, amp)
    scale_state = LossScale(enabled=amp).state()
    result = TrainResult(params)
    started = time.time()
    measure_every = measure_every or log_every
    measuring = 0.0
    for i in range(steps):
        if on_log is not None and (i % measure_every == 0 or i == steps - 1):
            t0 = time.time()
            result.measurements.append((i, on_log(i, params)))
            measuring += time.time() - t0
        inputs, targets = sample_batch()
        lr = {"learning_rate": jnp.float32(lr_at(i, steps, peak_lr, schedule)),
              "schedule": jnp.float32(lr_at(i, steps, 1.0, schedule))}
        params, opt_state, scale_state, loss, finite = step_fn(
            params, opt_state, scale_state, jnp.asarray(inputs, jnp.int32), jnp.asarray(targets, jnp.int32), lr
        )
        if i == 0:
            jax.block_until_ready(loss)
            result.compile_seconds = time.time() - started - measuring
        if not bool(finite):
            result.skipped_steps += 1
        if i % log_every == 0 or i == steps - 1:
            result.history.append((i, float(loss)))
            if label:
                print(f"  [{label}] step {i:5d} loss {float(loss):.4f} ({time.time() - started:.0f}s)", flush=True)
    jax.block_until_ready(params)
    result.params = params
    result.train_seconds = time.time() - started - measuring
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
