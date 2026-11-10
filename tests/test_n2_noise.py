"""N2: per-example gradients and the noise-scale estimator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxexp import noise as N
from jaxexp import recurrent as R
from jaxexp import text
from jaxexp import train as TR

DATA = np.random.default_rng(5).integers(0, 256, size=20_000).astype(np.int32)


def _gru():
    params = R.init_like_torch(jax.random.key(0), "gru", 16, 1)
    return params, lambda p, ids: R.apply(p, ids, "gru")


def test_vmapped_per_example_gradients_equal_a_loop():
    params, apply = _gru()
    inputs, targets = text.Batches(DATA, 24, 6, 0)()
    sq, total = N.make_per_example(apply, chunk=4)(params, inputs, targets)  # chunk does not divide 6
    loop = [jax.grad(lambda p, x=x, y=y: TR.cross_entropy(apply(p, x[None]), y[None]).mean())(params)
            for x, y in zip(jnp.asarray(inputs), jnp.asarray(targets), strict=True)]
    want_sq = [sum(float(jnp.sum(leaf**2)) for leaf in jax.tree.leaves(g)) for g in loop]
    np.testing.assert_allclose(sq, want_sq, rtol=1e-5)
    want_sum = jax.tree.map(lambda *g: sum(g), *loop)
    for a, b in zip(jax.tree.leaves(total), jax.tree.leaves(want_sum), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-7)
    # The batch-mean gradient is the gradient of the batch loss.
    batch_grad = jax.grad(lambda p: TR.cross_entropy(apply(p, jnp.asarray(inputs)), jnp.asarray(targets)).mean())(params)
    for a, b in zip(jax.tree.leaves(total), jax.tree.leaves(batch_grad), strict=True):
        np.testing.assert_allclose(np.asarray(a) / 6, np.asarray(b), rtol=1e-4, atol=1e-7)


def test_batch_estimates_are_unbiased():
    """A population of per-example 'gradients' with known mean and covariance; batches drawn with
    replacement. Averaged over many batches the estimates match tr(Sigma) and |G|^2."""
    rng = np.random.default_rng(0)
    population = rng.normal(0.3, 1.0, size=(4000, 50))
    true_g2 = float(np.sum(population.mean(0) ** 2))
    true_trace = float(np.sum(population.var(0)))
    traces, norms = [], []
    for _ in range(3000):
        batch = population[rng.integers(0, len(population), size=8)]
        t, g = N.batch_estimates(np.sum(batch**2, axis=1), {"w": batch.sum(0)})
        traces.append(t)
        norms.append(g)
    assert abs(np.mean(traces) / true_trace - 1) < 0.02
    assert abs(np.mean(norms) / true_g2 - 1) < 0.05
    assert abs((np.mean(traces) / np.mean(norms)) / (true_trace / true_g2) - 1) < 0.06


def test_noise_scale_of_a_known_problem():
    """Per-example gradients g_i = mu + noise with |mu|^2 = 1 and tr(Sigma) = 10: B_simple = 10."""
    rng = np.random.default_rng(1)
    mu = np.zeros(40)
    mu[0] = 1.0
    sigma = np.sqrt(10 / 40)

    def fake_per_example(params, inputs, targets):
        g = mu + rng.normal(0, sigma, size=(len(inputs), 40))
        return np.sum(g**2, axis=1), {"w": g.sum(0)}

    batches = [(np.zeros((32, 1)), np.zeros((32, 1)))] * 400
    result = N.noise_scale(None, None, batches, per_example=fake_per_example)
    assert abs(result["b_simple"] / 10 - 1) < 0.05


def test_warmup_constant_schedule_and_measurement_hook():
    assert TR.lr_at(10, 100, 1.0, "warmup-constant") == 1.0 and TR.lr_at(99, 100, 1.0, "warmup-constant") == 1.0
    assert TR.lr_at(2, 100, 1.0, "warmup-constant") == TR.lr_at(2, 100, 1.0)  # same warmup as the protocol
    params, apply = _gru()
    seen = []
    result = TR.train(params, apply, text.Batches(DATA, 24, 4, 0), steps=7, peak_lr=1e-3, weight_decay=0.0,
                      log_every=100, measure_every=3, on_log=lambda step, p: seen.append(step) or {"step": step})
    assert seen == [0, 3, 6] and [s for s, _ in result.measurements] == [0, 3, 6]
