"""Timing and compiled-program accounting shared by the experiments.

Wall-clock is noisy, so every cost is also read from the compiled XLA program,
which is deterministic: ``cost_analysis()`` FLOPs and ``memory_analysis()``
bytes. Saved residuals (what reverse mode keeps for the backward pass) come
from ``jax._src.ad_checkpoint.saved_residuals`` -- a private API, used because the
public ``print_saved_residuals`` only prints.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
from jax._src.ad_checkpoint import saved_residuals


def time_call(fn: Callable[..., Any], *args, min_reps: int = 5, max_reps: int = 50, budget_s: float = 1.0) -> dict:
    """Median wall-clock of ``fn(*args)`` after one warm-up call (which also compiles).

    Repeats until ``budget_s`` is spent or ``max_reps`` is reached, never fewer
    than ``min_reps``. Returns seconds: median, min, max, and the rep count.
    """
    jax.block_until_ready(fn(*args))
    times: list[float] = []
    started = time.perf_counter()
    while len(times) < min_reps or (len(times) < max_reps and time.perf_counter() - started < budget_s):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return {"median": statistics.median(times), "min": min(times), "max": max(times), "reps": len(times)}


def compiled_stats(fn: Callable[..., Any], *args) -> dict:
    """FLOPs and memory of ``jax.jit(fn)`` compiled for ``args``, without running it."""
    started = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - started
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list | tuple) else cost
    memory = compiled.memory_analysis()
    return {
        "flops": float(cost.get("flops", float("nan"))),
        "peak_bytes": int(memory.peak_memory_in_bytes),
        "temp_bytes": int(memory.temp_size_in_bytes),
        "argument_bytes": int(memory.argument_size_in_bytes),
        "output_bytes": int(memory.output_size_in_bytes),
        "compile_s": compile_s,
    }


def residual_stats(fn: Callable[..., Any], *args) -> dict:
    """What reverse mode of ``fn`` saves for the backward pass: values computed in
    the forward pass, not ``fn``'s arguments or the constants it closes over
    (both exist whether or not anything is saved)."""
    residuals = [
        (aval, why)
        for aval, why in saved_residuals(fn, *args)
        if "from the argument" not in why and "from a constant" not in why
    ]
    return {
        "count": len(residuals),
        "bytes": int(sum(aval.size * aval.dtype.itemsize for aval, _ in residuals)),
    }


def summarise(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "per_seed": values}
