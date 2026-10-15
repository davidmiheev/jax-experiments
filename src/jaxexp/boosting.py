"""Gradient-boosted trees with Newton leaves from autodiff, for any per-sample loss.

The loss is a JAX function ``loss(y, F) -> scalar`` of one target and one raw
score. Its gradient and Hessian with respect to F come from ``jax.grad``
twice, vmapped over samples, so the same booster serves log-loss, pseudo-Huber,
or a loss XGBoost has no formula for. Each round fits a depth-limited tree by
exact greedy search on the second-order objective (Chen & Guestrin, 2016):

    gain  = G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)
    leaf  = -G/(H+lambda),  scaled by eta

Tree growth is NumPy: it is discrete and data-dependent, which is not what
``jit`` is for. The conventions follow XGBoost's exact updater so the two can be
compared leaf by leaf:

- gradient pairs are stored in float32 and summed in float64;
- features are float32, and a split sits at the float32 midpoint between
  adjacent distinct values, with ``x < threshold`` going left;
- a split needs both children's Hessian sums >= ``min_child_weight`` and a
  gain above 1e-6; on equal gains the lower feature index wins.

``hessian="ones"`` replaces h by 1 everywhere, which turns the same code into
classic first-order gradient boosting: each tree is fit to the negative
gradient by least squares, and leaves hold the mean negative gradient.
``hessian="leaves"`` grows the first-order tree but fills its leaves with the
Newton value, which separates what h does to the leaves from what it does to
the splits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

RT_EPS = 1e-6  # XGBoost's minimum loss change for a split


def derivatives(loss: Callable[[jax.Array, jax.Array], jax.Array]):
    """(g, h) per sample: first and second derivative of ``loss(y, F)`` in F."""
    d1 = jax.grad(loss, argnums=1)
    d2 = jax.grad(d1, argnums=1)
    return jax.jit(jax.vmap(lambda y, f: (d1(y, f), d2(y, f))))


@dataclass
class Tree:
    feature: list[int] = field(default_factory=list)  # -1 for a leaf
    threshold: list[float] = field(default_factory=list)
    left: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)  # leaf value, eta included
    gain: list[float] = field(default_factory=list)
    cover: list[float] = field(default_factory=list)  # Hessian sum at the node

    def _add(self) -> int:
        for column in (self.feature, self.left, self.right):
            column.append(-1)
        for column in (self.threshold, self.value, self.gain, self.cover):
            column.append(0.0)
        return len(self.feature) - 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        node = np.zeros(len(X), np.int64)
        feature, threshold = np.asarray(self.feature), np.asarray(self.threshold, np.float32)
        left, right = np.asarray(self.left), np.asarray(self.right)
        while True:
            internal = feature[node] >= 0
            if not internal.any():
                return np.asarray(self.value)[node]
            rows = np.nonzero(internal)[0]
            goes_left = X[rows, feature[node[rows]]] < threshold[node[rows]]
            node[rows] = np.where(goes_left, left[node[rows]], right[node[rows]])


def _best_split(X: np.ndarray, g: np.ndarray, h: np.ndarray, reg_lambda: float, min_child_weight: float):
    G, H = g.sum(), h.sum()
    parent = G * G / (H + reg_lambda)
    best = (RT_EPS, -1, 0.0)  # (gain, feature, threshold); must beat RT_EPS
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        x, gs, hs = X[order, j], np.cumsum(g[order]), np.cumsum(h[order])
        # Candidate after position i when x[i] != x[i+1].
        distinct = np.nonzero(x[:-1] != x[1:])[0]
        if len(distinct) == 0:
            continue
        GL, HL = gs[distinct], hs[distinct]
        GR, HR = G - GL, H - HL
        ok = (HL >= min_child_weight) & (HR >= min_child_weight)
        if not ok.any():
            continue
        gains = np.where(ok, GL * GL / (HL + reg_lambda) + GR * GR / (HR + reg_lambda) - parent, -np.inf)
        i = int(np.argmax(gains))  # first maximum: lowest threshold on ties
        if gains[i] > best[0]:
            threshold = np.float32((np.float32(x[distinct[i]]) + np.float32(x[distinct[i] + 1])) * np.float32(0.5))
            best = (float(gains[i]), j, float(threshold))
    return best


def grow_tree(
    X: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    *,
    h_leaf: np.ndarray | None = None,
    max_depth: int,
    eta: float,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
) -> Tree:
    tree = Tree()
    h_leaf = h if h_leaf is None else h_leaf

    def build(rows: np.ndarray, depth: int) -> int:
        node = tree._add()
        G, H = g[rows].sum(), h[rows].sum()
        tree.cover[node] = float(H)
        if depth < max_depth:
            gain, feature, threshold = _best_split(X[rows], g[rows], h[rows], reg_lambda, min_child_weight)
            if feature >= 0:
                goes_left = X[rows, feature] < np.float32(threshold)
                tree.feature[node], tree.threshold[node], tree.gain[node] = feature, threshold, gain
                tree.left[node] = build(rows[goes_left], depth + 1)
                tree.right[node] = build(rows[~goes_left], depth + 1)
                return node
        tree.value[node] = float(eta * (-G / (h_leaf[rows].sum() + reg_lambda)))
        return node

    build(np.arange(len(X)), 0)
    return tree


@dataclass
class Booster:
    loss: Callable[[jax.Array, jax.Array], jax.Array]
    rounds: int = 100
    eta: float = 0.1
    max_depth: int = 3
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    base_margin: float = 0.0
    hessian: str = "exact"  # "ones": first-order boosting; "leaves": first-order trees, Newton leaves
    float32_gradients: bool = True  # XGBoost stores gradient pairs in float32
    float32_margin: bool = True  # and its running predictions (bst_float), which g and h are computed from
    trees: list[Tree] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Booster:
        X = np.asarray(X, np.float32)
        derivs = derivatives(self.loss)
        margin = np.full(len(X), self.base_margin, np.float32 if self.float32_margin else np.float64)
        y64 = jnp.asarray(y, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
        self.trees = []
        for _ in range(self.rounds):
            g, h = (np.asarray(a, np.float64) for a in derivs(y64, jnp.asarray(margin, y64.dtype)))
            if self.float32_gradients:
                g, h = g.astype(np.float32).astype(np.float64), h.astype(np.float32).astype(np.float64)
            h_split = h if self.hessian == "exact" else np.ones_like(h)
            h_leaf = np.ones_like(h) if self.hessian == "ones" else h
            tree = grow_tree(
                X,
                g,
                h_split,
                h_leaf=h_leaf,
                max_depth=self.max_depth,
                eta=self.eta,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
            )
            self.trees.append(tree)
            margin += tree.predict(X).astype(margin.dtype)
        return self

    def margin(self, X: np.ndarray, rounds: int | None = None) -> np.ndarray:
        out = np.full(len(X), self.base_margin, np.float64)
        for tree in self.trees[: rounds or len(self.trees)]:
            out += tree.predict(X)
        return out

    def staged_margins(self, X: np.ndarray):
        out = np.full(len(X), self.base_margin, np.float64)
        for tree in self.trees:
            out = out + tree.predict(X)
            yield out


# Per-sample losses: (target, raw score) -> scalar.


def log_loss(y, f):
    return jax.nn.softplus(f) - y * f


def pseudo_huber(delta: float = 1.0):
    def loss(y, f):
        z = (f - y) / delta
        return delta**2 * (jnp.sqrt(1 + z * z) - 1)

    return loss


def log_cosh(y, f):
    z = jnp.abs(f - y)
    return z + jax.nn.softplus(-2 * z) - jnp.log(2.0)
