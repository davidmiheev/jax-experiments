"""N7: the empirical NTK identity, the analytic kernel, and the kernel-regression path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxexp import ntk as K


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def inputs(n=24, d=7, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, d), jnp.float32)


def test_structured_ntk_equals_jacobian():
    X = inputs()
    params = K.init(jax.random.key(1), X.shape[1], 48, 2)
    fast, slow = K.empirical_ntk(params, X, X), K.empirical_ntk_jacobian(params, X, X)
    np.testing.assert_allclose(fast, slow, rtol=2e-5, atol=1e-5)


def test_empirical_approaches_analytic(x64):
    X = inputs(n=32, d=10)
    analytic = np.asarray(K.analytic_ntk(X.astype(jnp.float64), X.astype(jnp.float64), 2)[0])

    def error(width):
        errs = []
        for s in range(3):
            emp = np.asarray(K.empirical_ntk(K.init(jax.random.key(s), 10, width, 2), X, X), np.float64)
            errs.append(np.linalg.norm(emp - analytic) / np.linalg.norm(analytic))
        return np.median(errs)

    small, large = error(64), error(2048)
    assert large < 0.1
    assert large < small / 3  # about sqrt(32) = 5.7 from width^(-1/2)


def test_analytic_diagonal_in_float64(x64):
    """On the diagonal theta = 0, so step expectations are exactly 1/2: Theta^{l} = Theta^{l-1} s_w2/2 + Sigma^l."""
    X = inputs(n=5, d=4).astype(jnp.float64)
    ntk, nngp = K.analytic_ntk(X, X, 1)
    s1 = K.S_W2 * jnp.sum(X * X, 1) / 4 + K.S_B2
    s2 = K.S_W2 * s1 / 2 + K.S_B2  # E[relu(u)^2] = k/2
    np.testing.assert_allclose(jnp.diag(nngp), s2, rtol=1e-12)
    np.testing.assert_allclose(jnp.diag(ntk), s1 * K.S_W2 / 2 + s2, rtol=1e-12)


def test_kernel_regression_path_limits(x64):
    X = inputs(n=20, d=5).astype(jnp.float64)
    theta, _ = K.analytic_ntk(X, X, 2)
    y = jnp.sign(X[:, 0])
    lr = float(20 / jnp.linalg.eigvalsh(theta)[-1])
    train, _ = K.kernel_regression_path(theta, theta, y, lr, [0, 1, 100_000])
    np.testing.assert_allclose(train[0], 0.0, atol=1e-12)  # starts at g = 0
    np.testing.assert_allclose(train[1], lr * theta @ y / 20, rtol=1e-10)  # one GD step
    np.testing.assert_allclose(train[2], y, atol=1e-6)  # interpolates in the limit
