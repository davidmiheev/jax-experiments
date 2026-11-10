"""N1: train K copies of a model as one program, ``jit(vmap(step))``.

Each copy has its own parameters, optimiser state, loss scale, learning-rate
schedule and batch stream. Copies are stacked on a leading axis, and the
training step of ``jaxexp.train`` is vmapped over all of it, so one compiled
step trains every copy at once. A sweep over learning rates and seeds becomes
one program instead of K separate runs.

Batch streams are deduplicated by identity: copies given the *same* sampler
object (same seed, different learning rate, as in ml-ladder's sweeps) share
one draw per step, so each copy sees exactly the batches its sequential run
would.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxexp import train as TR


def stack(trees: Sequence) -> object:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def unstack(tree, k: int) -> list:
    return [jax.tree.map(lambda x, i=i: x[i], tree) for i in range(k)]


def make_sweep_step(apply: Callable, opt, amp: bool):
    """The training step of ``jaxexp.train.make_step``, vmapped over copies and compiled once."""
    return jax.jit(jax.vmap(TR.make_step(apply, opt, amp)))


def train_many(
    params_list: Sequence,
    apply: Callable,
    samplers: Sequence[Callable[[], tuple[np.ndarray, np.ndarray]]],
    *,
    steps: int,
    peak_lrs: Sequence[float],
    weight_decay: float,
    amp: bool = False,
    schedule: str = "warmup-cosine",
    log_every: int = 250,
    label: str = "",
    opt=None,
) -> tuple[list[TR.TrainResult], dict]:
    """Train ``len(params_list)`` copies together. Copy k starts from ``params_list[k]``, draws
    from ``samplers[k]`` and follows ml-ladder's schedule with peak ``peak_lrs[k]``. ``opt``
    (default: ml-ladder's AdamW) is shared; only its learning rate differs between copies.

    Returns one ``TrainResult`` per copy (its own loss history, skipped steps and final loss
    scale) and the shared timing: ``train_seconds`` and ``compile_seconds`` are for the whole
    program, which trains all K copies."""
    k = len(params_list)
    assert len(samplers) == k and len(peak_lrs) == k
    opt = opt or TR.optimizer(weight_decay)
    params = stack(params_list)
    opt_state = jax.vmap(opt.init)(params)
    one = TR.LossScale(enabled=amp).state()
    scale_state = (jnp.full((k,), one[0]), jnp.full((k,), one[1]))
    step_fn = make_sweep_step(apply, opt, amp)

    unique = list({id(s): s for s in samplers}.values())
    which = [next(j for j, u in enumerate(unique) if u is s) for s in samplers]
    histories = [[] for _ in range(k)]
    skipped = np.zeros(k, np.int64)
    started, compile_seconds = time.time(), 0.0
    for i in range(steps):
        drawn = [u() for u in unique]
        inputs = jnp.asarray(np.stack([drawn[j][0] for j in which]), jnp.int32)
        targets = jnp.asarray(np.stack([drawn[j][1] for j in which]), jnp.int32)
        lr = {
            "learning_rate": jnp.asarray([TR.lr_at(i, steps, p, schedule) for p in peak_lrs], jnp.float32),
            "schedule": jnp.full((k,), TR.lr_at(i, steps, 1.0, schedule), jnp.float32),
        }
        params, opt_state, scale_state, loss, finite = step_fn(params, opt_state, scale_state, inputs, targets, lr)
        if i == 0:
            jax.block_until_ready(loss)
            compile_seconds = time.time() - started
        skipped += ~np.asarray(finite)
        if i % log_every == 0 or i == steps - 1:
            losses = np.asarray(loss)
            for c in range(k):
                histories[c].append((i, float(losses[c])))
            if label:
                shown = " ".join(f"{x:.3f}" for x in losses)
                print(f"  [{label}] step {i:5d} loss {shown} ({time.time() - started:.0f}s)", flush=True)
    jax.block_until_ready(params)
    train_seconds = time.time() - started
    finals = unstack(params, k)
    scales = np.asarray(scale_state[0])
    results = [
        TR.TrainResult(
            finals[c],
            history=histories[c],
            train_seconds=train_seconds,
            compile_seconds=compile_seconds,
            skipped_steps=int(skipped[c]),
            final_scale=float(scales[c]),
        )
        for c in range(k)
    ]
    return results, {"train_seconds": train_seconds, "compile_seconds": compile_seconds, "copies": k}
