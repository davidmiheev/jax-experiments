"""N2: the gradient noise scale (McCandlish, Kaplan, Amodei et al. 2018), from per-example gradients.

For a batch of B examples (here: sequences) with per-example gradients g_i and
mean G_B,

    E[ mean_i |g_i|^2 ] = |G|^2 + tr(Sigma)          E[ |G_B|^2 ] = |G|^2 + tr(Sigma) / B,

where G is the true gradient and Sigma the per-example gradient covariance. So
one batch gives unbiased estimates

    tr(Sigma) ~ (mean_i |g_i|^2 - |G_B|^2) * B / (B - 1)
    |G|^2     ~ (B |G_B|^2 - mean_i |g_i|^2) / (B - 1),

and the simple noise scale is B_simple = tr(Sigma) / |G|^2. As in the paper, the
numerator and denominator are averaged over batches before the ratio is taken
(the ratio of noisy estimates is biased; the ratio of averages much less so).

Per-example gradients are ``vmap(grad)`` over the batch axis, in chunks, so only
``chunk`` gradient copies exist at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from jaxexp import train as TR


def make_per_example(apply: Callable, chunk: int = 16) -> Callable:
    """(params, inputs [n, T], targets [n, T]) -> (per-example |g_i|^2 [n], sum_i g_i)."""

    def loss_one(params, x, y):
        return TR.cross_entropy(apply(params, x[None]), y[None]).mean()

    grads = jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0))

    @jax.jit
    def chunk_fn(params, xs, ys):
        g = grads(params, xs, ys)
        sq = sum(jnp.sum(jnp.square(leaf.astype(jnp.float32)).reshape(leaf.shape[0], -1), axis=1)
                 for leaf in jax.tree.leaves(g))
        return sq, jax.tree.map(lambda leaf: leaf.astype(jnp.float32).sum(0), g)

    def per_example(params, inputs, targets):
        sqs, total = [], None
        for start in range(0, len(inputs), chunk):
            sq, s = chunk_fn(params, jnp.asarray(inputs[start : start + chunk]), jnp.asarray(targets[start : start + chunk]))
            sqs.append(np.asarray(sq, np.float64))
            total = s if total is None else jax.tree.map(jnp.add, total, s)
        return np.concatenate(sqs), total

    return per_example


def batch_estimates(sq_norms: np.ndarray, grad_sum) -> tuple[float, float]:
    """Unbiased (tr Sigma, |G|^2) from one batch: per-example squared norms and the gradient sum."""
    b = len(sq_norms)
    mean_sq = float(np.mean(sq_norms))
    g_batch_sq = float(sum(np.sum(np.square(np.asarray(leaf, np.float64))) for leaf in jax.tree.leaves(grad_sum))) / b**2
    return (mean_sq - g_batch_sq) * b / (b - 1), (b * g_batch_sq - mean_sq) / (b - 1)


def noise_scale(params, apply: Callable, batches: Iterable[tuple[np.ndarray, np.ndarray]], chunk: int = 16,
                per_example: Callable | None = None) -> dict:
    """B_simple from several batches: ratio of the averaged unbiased estimates."""
    per_example = per_example or make_per_example(apply, chunk)
    traces, norms = [], []
    for inputs, targets in batches:
        sq, total = per_example(params, inputs, targets)
        t, g = batch_estimates(sq, total)
        traces.append(t)
        norms.append(g)
    trace, norm = float(np.mean(traces)), float(np.mean(norms))
    return {
        "trace_sigma": trace,
        "grad_norm_sq": norm,
        "b_simple": trace / norm if norm > 0 else float("inf"),
        "batches": len(traces),
        "per_batch_b_simple": [t / g if g > 0 else None for t, g in zip(traces, norms, strict=True)],
    }
