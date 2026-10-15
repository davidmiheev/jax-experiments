"""Neural tangent kernels of ReLU MLPs: infinite-width (analytic) and empirical (finite).

The network, in NTK parameterisation, with standard-normal weights and biases:

    h_1     = s_w W_1 x / sqrt(d) + s_b b_1
    h_l     = s_w W_l relu(h_{l-1}) / sqrt(n) + s_b b_l          l = 2..L
    f(x)    = s_w w_out . relu(h_L) / sqrt(n) + s_b b_out        (scalar output)

**Analytic NTK** (Jacot et al. 2018; arc-cosine kernels of Cho & Saul 2009).
For (u, v) ~ N(0, [[k11, k12], [k12, k22]]) and theta = arccos(k12 / sqrt(k11 k22)):

    E[relu(u) relu(v)] = sqrt(k11 k22) (sin theta + (pi - theta) cos theta) / (2 pi)
    E[step(u) step(v)] = (pi - theta) / (2 pi)

    Sigma^1 = s_w^2 x.x'/d + s_b^2,   Theta^1 = Sigma^1
    Sigma^l = s_w^2 E[relu relu](Sigma^{l-1}) + s_b^2
    dSigma^l = s_w^2 E[step step](Sigma^{l-1})
    Theta^l = Theta^{l-1} dSigma^l + Sigma^l,       l = 2..L+1 (L+1 is the output)

**Empirical NTK** Theta(x, x') = <df/dw(x), df/dw(x')>, computed per layer without
forming the Jacobian. With delta_l = df/dh_l (a vector-Jacobian product) and the
layer's scaled input a_{l-1}, df/dW_l = delta_l a_{l-1}^T is an outer product, so

    Theta = sum_l (Delta_l Delta_l^T) * (s_w^2 A_{l-1} A_{l-1}^T / fan_in + s_b^2)
            + s_w^2 A_L A_L^T / n + s_b^2          (* elementwise)

which costs O(N^2 width) rather than O(N^2 params). ``empirical_ntk_jacobian``
is the brute-force J J^T, used to check it.

**Precision.** Compute ``analytic_ntk`` in float64. On the diagonal cos(theta) = 1,
where arccos has infinite slope: float32 rounding of cos by ~1e-7 becomes ~3e-4
in theta, and the NTK's diagonal comes out 7e-4 off. In float64 this module
matches neural-tangents 0.6.5 to 1.2e-8 (NTK) and 7e-15 (NNGP).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

S_W2, S_B2 = 2.0, 0.1  # He-style weight variance; small bias variance


def init(key, d: int, width: int, depth: int, dtype=jnp.float32) -> dict:
    keys = jax.random.split(key, 2 * depth + 2)
    sizes = [d] + [width] * depth
    return {
        "W": [jax.random.normal(keys[i], (sizes[i + 1], sizes[i]), dtype) for i in range(depth)],
        "b": [jax.random.normal(keys[depth + i], (sizes[i + 1],), dtype) for i in range(depth)],
        "w_out": jax.random.normal(keys[-2], (width,), dtype),
        "b_out": jax.random.normal(keys[-1], (), dtype),
    }


def apply(params: dict, x: jax.Array, eps: list | None = None, *, s_w2=S_W2, s_b2=S_B2, act=jax.nn.relu) -> jax.Array:
    """Scalar output for one input x [d]. ``eps`` (optional) is added to each pre-activation h_l,
    so grad w.r.t. eps gives delta_l = df/dh_l. ``act`` may be any activation for the empirical
    kernel; ``analytic_ntk`` is ReLU only."""
    s_w, s_b = math.sqrt(s_w2), math.sqrt(s_b2)
    a = x
    for i, (W, b) in enumerate(zip(params["W"], params["b"], strict=True)):
        h = s_w * (W @ a) / math.sqrt(a.shape[0]) + s_b * b
        if eps is not None:
            h = h + eps[i]
        a = act(h)
    return s_w * (params["w_out"] @ a) / math.sqrt(a.shape[0]) + s_b * params["b_out"]


def features(params: dict, X: jax.Array, *, s_w2=S_W2, s_b2=S_B2, act=jax.nn.relu):
    """Per-layer inputs A_0..A_L [N, fan] and backprop vectors Delta_1..Delta_L [N, width]."""
    depth = len(params["W"])
    width = params["W"][0].shape[0]
    eps0 = [jnp.zeros(width, params["w_out"].dtype) for _ in range(depth)]

    def one(x):
        deltas = jax.grad(lambda e: apply(params, x, e, s_w2=s_w2, s_b2=s_b2, act=act))(eps0)
        s_w, s_b = math.sqrt(s_w2), math.sqrt(s_b2)
        acts, a = [x], x
        for W, b in zip(params["W"], params["b"], strict=True):
            a = act(s_w * (W @ a) / math.sqrt(a.shape[0]) + s_b * b)
            acts.append(a)
        return acts, deltas

    return jax.vmap(one)(X)


def empirical_ntk(params: dict, X1: jax.Array, X2: jax.Array, *, s_w2=S_W2, s_b2=S_B2, act=jax.nn.relu) -> jax.Array:
    """Theta [N1, N2] by the per-layer outer-product identity (no Jacobian formed)."""
    A1, D1 = features(params, X1, s_w2=s_w2, s_b2=s_b2, act=act)
    A2, D2 = features(params, X2, s_w2=s_w2, s_b2=s_b2, act=act)
    theta = 0.0
    for layer in range(len(params["W"])):
        fan = A1[layer].shape[1]
        inputs = s_w2 * (A1[layer] @ A2[layer].T) / fan + s_b2
        theta = theta + (D1[layer] @ D2[layer].T) * inputs
    width = A1[-1].shape[1]
    return theta + s_w2 * (A1[-1] @ A2[-1].T) / width + s_b2


def empirical_ntk_jacobian(params: dict, X1: jax.Array, X2: jax.Array, *, s_w2=S_W2, s_b2=S_B2) -> jax.Array:
    """Brute force: flatten df/dw for every input and take J1 J2^T. For checking only."""
    flat, unravel = ravel_pytree(params)

    def jac(X):
        return jax.vmap(jax.grad(lambda w, x: apply(unravel(w), x, s_w2=s_w2, s_b2=s_b2)), (None, 0))(flat, X)

    return jac(X1) @ jac(X2).T


def _relu_expectations(k11, k12, k22):
    norm = jnp.sqrt(k11 * k22)
    cos = jnp.clip(k12 / norm, -1.0, 1.0)
    theta = jnp.arccos(cos)
    e_relu = norm * (jnp.sin(theta) + (jnp.pi - theta) * cos) / (2 * jnp.pi)
    e_step = (jnp.pi - theta) / (2 * jnp.pi)
    return e_relu, e_step


def analytic_ntk(X1: jax.Array, X2: jax.Array, depth: int, *, s_w2=S_W2, s_b2=S_B2) -> tuple[jax.Array, jax.Array]:
    """Infinite-width (NTK, NNGP) kernels [N1, N2] for the network above."""
    d = X1.shape[1]
    s12 = s_w2 * (X1 @ X2.T) / d + s_b2
    s11 = s_w2 * jnp.sum(X1 * X1, 1) / d + s_b2
    s22 = s_w2 * jnp.sum(X2 * X2, 1) / d + s_b2
    ntk = s12
    for _ in range(depth):  # hidden layers 2..L, then the output layer
        e12, step12 = _relu_expectations(s11[:, None], s12, s22[None, :])
        e11, _ = _relu_expectations(s11, s11, s11)
        e22, _ = _relu_expectations(s22, s22, s22)
        s12_next = s_w2 * e12 + s_b2
        ntk = ntk * (s_w2 * step12) + s12_next
        s12, s11, s22 = s12_next, s_w2 * e11 + s_b2, s_w2 * e22 + s_b2
    return ntk, s12


def kernel_regression_path(theta_train, theta_test_train, y, lr: float, steps):
    """Function values on train and test of the linearised (kernel) model after t GD steps,
    from g_0 = 0, on loss 0.5 * mean((g - y)^2):  g_t = Theta_*X Theta^{-1} (I - (I - lr Theta / n)^t) y.
    Exact for gradient descent (not gradient flow). Returns arrays [len(steps), N]."""
    n = theta_train.shape[0]
    evals, evecs = jnp.linalg.eigh(theta_train)
    proj = evecs.T @ y
    out_train, out_test = [], []
    for t in steps:
        decay = 1.0 - (1.0 - lr * evals / n) ** t
        coef = evecs @ (decay / evals * proj)  # Theta^{-1} (I - (I - lr Theta/n)^t) y
        out_train.append(theta_train @ coef)
        out_test.append(theta_test_train @ coef)
    return jnp.stack(out_train), jnp.stack(out_test)
