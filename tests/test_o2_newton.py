"""O2: the solvers reach the same optimum, and the autodiff booster is XGBoost's algorithm."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxexp import boosting as B
from jaxexp import newton as N


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def logistic_problem(n=1500, d=12, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    X = jax.random.normal(k1, (n, d), jnp.float64)
    w = jax.random.normal(k2, (d,), jnp.float64)
    y = (jax.random.uniform(k3, (n,), jnp.float64) < jax.nn.sigmoid(X @ w)).astype(jnp.float64)
    return X, y


def test_autodiff_hessian_is_irls(x64):
    X, y = logistic_problem()
    loss = N.logistic_loss(X, y, 1e-3)
    w = jnp.linspace(-0.5, 0.5, X.shape[1] + 1)
    np.testing.assert_allclose(jax.hessian(loss)(w), N.logistic_hessian(X, w, 1e-3), atol=1e-14)


def test_solvers_agree(x64):
    X, y = logistic_problem()
    loss = N.logistic_loss(X, y, 1e-3)
    w0 = jnp.zeros(X.shape[1] + 1)
    results = {
        "newton": jax.jit(lambda w: N.newton(loss, w))(w0),
        "newton_cg": jax.jit(lambda w: N.newton_cg(loss, w))(w0),
        "lbfgs": jax.jit(lambda w: N.lbfgs(loss, w))(w0),
        "gd": jax.jit(lambda w: N.gradient_descent(loss, w, 2.0))(w0),
    }
    for name, r in results.items():
        assert float(r.grad_norm) <= 1e-8, name
        np.testing.assert_allclose(r.w, results["newton"].w, atol=1e-6, err_msg=name)
    assert int(results["newton"].iterations) < 15  # quadratic convergence
    assert int(results["gd"].iterations) > 5 * int(results["newton"].iterations)


@pytest.mark.parametrize(
    "loss, g, h",
    [
        (B.log_loss, lambda y, f: jax.nn.sigmoid(f) - y, lambda y, f: jax.nn.sigmoid(f) * (1 - jax.nn.sigmoid(f))),
        (B.log_cosh, lambda y, f: jnp.tanh(f - y), lambda y, f: 1 - jnp.tanh(f - y) ** 2),
        (
            B.pseudo_huber(2.0),
            lambda y, f: (f - y) / jnp.sqrt(1 + ((f - y) / 2.0) ** 2),
            lambda y, f: (1 + ((f - y) / 2.0) ** 2) ** -1.5,
        ),
    ],
)
def test_derivatives_match_closed_forms(x64, loss, g, h):
    y = jnp.array([0.0, 1.0, 1.0, 0.0, 3.0])
    f = jnp.array([-2.0, 0.3, 4.0, 1.5, -1.0])
    gd, hd = B.derivatives(loss)(y, f)
    np.testing.assert_allclose(gd, g(y, f), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hd, h(y, f), rtol=1e-12, atol=1e-14)


def test_squared_loss_tree_is_a_regression_tree(x64):
    """With h = 1 and lambda = 0, Newton leaves are mean residuals (ml-ladder A4's identity)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(300, 3))
    y = np.sin(4 * X[:, 0]) + X[:, 1]
    model = B.Booster(lambda y, f: 0.5 * (y - f) ** 2, rounds=1, eta=1.0, max_depth=1, reg_lambda=0.0).fit(X, y)
    tree = model.trees[0]
    goes_left = X[:, tree.feature[0]].astype(np.float32) < np.float32(tree.threshold[0])
    np.testing.assert_allclose(tree.value[tree.left[0]], y[goes_left].mean(), rtol=1e-6)
    np.testing.assert_allclose(tree.value[tree.right[0]], y[~goes_left].mean(), rtol=1e-6)


def test_matches_xgboost_on_log_loss(x64):
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(800, 5))
    y = (np.sin(3 * X[:, 0]) + X[:, 1] - X[:, 2] + 0.3 * rng.normal(size=800) > 0.4).astype(np.float64)
    ours = B.Booster(B.log_loss, rounds=10, eta=0.3, max_depth=3).fit(X, y)
    params = {
        "objective": "binary:logistic",
        "eta": 0.3,
        "max_depth": 3,
        "lambda": 1.0,
        "min_child_weight": 1.0,
        "tree_method": "exact",
        "nthread": 1,
        "base_score": 0.5,
        "verbosity": 0,
    }
    bst = xgb.train(params, xgb.DMatrix(X.astype(np.float32), label=y), num_boost_round=10)
    theirs = bst.predict(xgb.DMatrix(X.astype(np.float32)), output_margin=True)
    np.testing.assert_allclose(ours.margin(X), theirs, atol=1e-5)
