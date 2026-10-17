"""Second-order and first-order solvers for smooth convex losses, each one compiled loop.

Every solver runs inside a single ``lax.while_loop`` to a relative gradient-norm
tolerance and returns the iterate, the number of outer iterations, the work it
did (gradients, Hessians, Hessian-vector products) and the loss at every
iteration (a fixed-size buffer, NaN past the end).

- ``newton``: the full Hessian from ``jax.hessian`` (forward-over-reverse),
  solved by Cholesky, with Armijo backtracking. For logistic regression this is
  IRLS: ``jax.hessian`` of the log-loss is exactly X^T S X / n + lambda I.
- ``newton_cg``: truncated Newton. Conjugate gradient on forward-over-reverse
  Hessian-vector products (O1's cheapest HVP), never forming H, with the
  Eisenstat-Walker forcing term min(0.5, sqrt(|g|)) |g|.
- ``lbfgs``: ``optax.lbfgs`` with its zoom line search.
- ``gradient_descent``: a constant step, which the caller tunes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class Result(NamedTuple):
    w: jax.Array
    iterations: jax.Array
    grad_norm: jax.Array
    gradients: jax.Array  # gradient evaluations
    hessians: jax.Array  # full Hessians formed
    hvps: jax.Array  # Hessian-vector products
    history: jax.Array  # loss at iterations 0..iterations, NaN after


def _history(max_iter: int, dtype) -> jax.Array:
    return jnp.full((max_iter + 1,), jnp.nan, dtype)


def _armijo(loss: Callable, w, f, g, step, *, shrink: float = 0.5, c: float = 1e-4, max_halvings: int = 30):
    """Largest t in {1, 1/2, ...} with f(w + t step) <= f + c t g.step."""
    slope = jnp.vdot(g, step)

    def cond(state):
        t, f_new, k = state
        return (f_new > f + c * t * slope) & (k < max_halvings)

    def body(state):
        t, _, k = state
        t = t * shrink
        return t, loss(w + t * step), k + 1

    t, f_new, k = jax.lax.while_loop(cond, body, (jnp.asarray(1.0, w.dtype), loss(w + step), 0))
    return t, f_new, k


def newton(loss: Callable, w0: jax.Array, *, tol: float = 1e-8, max_iter: int = 100) -> Result:
    grad, hess = jax.grad(loss), jax.hessian(loss)
    g0 = grad(w0)
    g0n = jnp.linalg.norm(g0)

    def cond(s):
        w, k, gn, *_ = s
        return (gn > tol * g0n) & (k < max_iter)

    def body(s):
        w, k, gn, ng, nh, hist = s
        f, g = jax.value_and_grad(loss)(w)
        chol = jax.scipy.linalg.cho_factor(hess(w))
        step = -jax.scipy.linalg.cho_solve(chol, g)
        t, f_new, _ = _armijo(loss, w, f, g, step)
        w = w + t * step
        g_new = grad(w)
        return w, k + 1, jnp.linalg.norm(g_new), ng + 2, nh + 1, hist.at[k + 1].set(f_new)

    hist = _history(max_iter, w0.dtype).at[0].set(loss(w0))
    w, k, gn, ng, nh, hist = jax.lax.while_loop(cond, body, (w0, 0, g0n, 1, 0, hist))
    return Result(w, k, gn / g0n, ng, nh, jnp.asarray(0), hist)


def newton_cg(loss: Callable, w0: jax.Array, *, tol: float = 1e-8, max_iter: int = 100, max_cg: int | None = None):
    grad = jax.grad(loss)
    max_cg = max_cg or w0.size

    def hvp(w, v):
        return jax.jvp(grad, (w,), (v,))[1]

    def cg(w, g):
        """Solve H p = -g approximately; returns p and the number of HVPs."""
        gn = jnp.linalg.norm(g)
        forcing = jnp.minimum(0.5, jnp.sqrt(gn)) * gn
        p, r = jnp.zeros_like(g), -g
        d = r

        def cond(s):
            _, r, _, k = s
            return (jnp.linalg.norm(r) > forcing) & (k < max_cg)

        def body(s):
            p, r, d, k = s
            hd = hvp(w, d)
            curvature = jnp.vdot(d, hd)
            alpha = jnp.vdot(r, r) / curvature
            p, r_new = p + alpha * d, r - alpha * hd
            beta = jnp.vdot(r_new, r_new) / jnp.vdot(r, r)
            return p, r_new, r_new + beta * d, k + 1

        p, _, _, k = jax.lax.while_loop(cond, body, (p, r, d, 0))
        return p, k

    g0n = jnp.linalg.norm(grad(w0))

    def cond(s):
        w, k, gn, *_ = s
        return (gn > tol * g0n) & (k < max_iter)

    def body(s):
        w, k, gn, ng, nv, hist = s
        f, g = jax.value_and_grad(loss)(w)
        step, used = cg(w, g)
        t, f_new, _ = _armijo(loss, w, f, g, step)
        w = w + t * step
        return w, k + 1, jnp.linalg.norm(grad(w)), ng + 2, nv + used, hist.at[k + 1].set(f_new)

    hist = _history(max_iter, w0.dtype).at[0].set(loss(w0))
    w, k, gn, ng, nv, hist = jax.lax.while_loop(cond, body, (w0, 0, g0n, 1, 0, hist))
    return Result(w, k, gn / g0n, ng, jnp.asarray(0), nv, hist)


def lbfgs(loss: Callable, w0: jax.Array, *, tol: float = 1e-8, max_iter: int = 1000, memory: int = 10) -> Result:
    optimizer = optax.lbfgs(memory_size=memory)
    value_and_grad = optax.value_and_grad_from_state(loss)
    g0n = jnp.linalg.norm(jax.grad(loss)(w0))

    def cond(s):
        w, state, k, gn, hist = s
        return (gn > tol * g0n) & (k < max_iter)

    def body(s):
        w, state, k, _, hist = s
        f, g = value_and_grad(w, state=state)
        updates, state = optimizer.update(g, state, w, value=f, grad=g, value_fn=loss)
        w = optax.apply_updates(w, updates)
        f_new = optax.tree.get(state, "value")
        g_new = optax.tree.get(state, "grad")
        return w, state, k + 1, jnp.linalg.norm(g_new), hist.at[k + 1].set(f_new)

    hist = _history(max_iter, w0.dtype).at[0].set(loss(w0))
    w, state, k, gn, hist = jax.lax.while_loop(cond, body, (w0, optimizer.init(w0), 0, g0n, hist))
    # The line search's own evaluations are not counted: optax does not expose them.
    return Result(w, k, gn / g0n, k + 1, jnp.asarray(0), jnp.asarray(0), hist)


def gradient_descent(loss: Callable, w0: jax.Array, lr: float, *, tol: float = 1e-8, max_iter: int = 100_000):
    f0, g0 = jax.value_and_grad(loss)(w0)
    g0n = jnp.linalg.norm(g0)

    def cond(s):
        w, g, k, gn, hist = s
        return (gn > tol * g0n) & (k < max_iter) & jnp.isfinite(gn)

    def body(s):
        w, g, k, _, hist = s
        w = w - lr * g
        f, g = jax.value_and_grad(loss)(w)  # one gradient per step, carried to the next
        return w, g, k + 1, jnp.linalg.norm(g), hist.at[k + 1].set(f)

    hist = _history(max_iter, w0.dtype).at[0].set(f0)
    w, _, k, gn, hist = jax.lax.while_loop(cond, body, (w0, g0, 0, g0n, hist))
    return Result(w, k, gn / g0n, k + 1, jnp.asarray(0), jnp.asarray(0), hist)


def logistic_loss(X: jax.Array, y: jax.Array, l2: float = 0.0) -> Callable[[jax.Array], jax.Array]:
    """Mean log-loss of a linear model on [X, 1] (last weight is the intercept, unpenalised)."""
    Xb = jnp.concatenate([X, jnp.ones((X.shape[0], 1), X.dtype)], axis=1)

    def loss(w):
        z = Xb @ w
        return jnp.mean(jax.nn.softplus(z) - y * z) + 0.5 * l2 * jnp.vdot(w[:-1], w[:-1])

    return loss


def logistic_hessian(X: jax.Array, w: jax.Array, l2: float = 0.0) -> jax.Array:
    """The textbook IRLS Hessian X^T S X / n + lambda I (no penalty on the intercept)."""
    Xb = jnp.concatenate([X, jnp.ones((X.shape[0], 1), X.dtype)], axis=1)
    p = jax.nn.sigmoid(Xb @ w)
    H = (Xb * (p * (1 - p))[:, None]).T @ Xb / X.shape[0]
    return H + l2 * jnp.diag(jnp.concatenate([jnp.ones(X.shape[1], X.dtype), jnp.zeros(1, X.dtype)]))
