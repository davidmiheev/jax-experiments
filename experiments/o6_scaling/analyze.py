"""O6: fit the scaling law (``docs/data/o6/o6-*.json``).

- **LR per size** (study ``lr``): the best peak LR per size on seed 0, flagged if at the grid's
  edge; writes ``docs/data/o6/lr-table.json`` for the IsoFLOP study.
- **IsoFLOP (Chinchilla's approach 2):** per budget C, a parabola in log N through the
  bits/byte of every run (all seeds); its vertex is N*(C), D*(C) = C / (6 N*). The vertex is
  *interior* only if it lies strictly inside the sampled sizes. Then N* ~ C^a and D* ~ C^b
  by a line in log-log space, over the interior contours.
- **The parametric law (approach 3):** L(N, D) = E + A / N^alpha + B / D^beta, by Huber loss
  (delta 1e-3) on log L, minimised from a grid of starting points (Hoffmann et al. 2022).
  It implies N* ~ C^(beta / (alpha + beta)).
- **Uncertainty:** both fits are repeated on 400 bootstrap resamples of the runs; the
  ranges reported are the 5th to 95th percentiles. G1's lesson: a parameter whose range
  is wide is not a finding.

- **Two parameter counts.** ``N`` is non-embedding (Kaplan's convention; the budgets are
  defined by it). The parametric fit is repeated with the 4096-way output layer counted
  ("with head"), which at these sizes is 0.4M-2.1M parameters, more than the blocks for the
  smallest models.
- **The true cost.** A training token costs 6 (N + head) FLOPs in the matmuls plus
  12 layers T d in causal attention (XLA computes the full T x T scores), 1.17x to 3.2x the
  nominal 6 N. The parametric law does not depend on how compute is counted, so the
  compute-optimal N under the true cost is found by minimising L(N, C / cost(N)) over N.

Losses are in bits per byte (a constant multiple of nats per token on this fixed held-out
text, so the exponents are the same).

    uv run python experiments/o6_scaling/analyze.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "o6"
FIG = ROOT / "docs" / "figures"
BOOT = 400


def _load(key: str) -> list[dict]:
    return [r for p in sorted(RUNS.glob("o6-*.json")) for r in json.loads(p.read_text()).get(key, [])]


VOCAB, CONTEXT = 4096, 256


def head_params(width: int) -> int:
    return VOCAB * width + VOCAB


def flops_per_token(n_non_embedding: float, width: float, layers: float) -> float:
    """Training FLOPs per token: 6 x (blocks + output layer) + 12 L T d for attention."""
    return 6.0 * (n_non_embedding + head_params(width)) + 12.0 * layers * CONTEXT * width


def lr_table() -> dict:
    rows = _load("lr")
    out = {}
    for size in sorted({r["size"] for r in rows}):
        mine = sorted((r for r in rows if r["size"] == size), key=lambda r: r["peak_lr"])
        best = min(mine, key=lambda r: r["bits_per_byte"])
        lrs = [r["peak_lr"] for r in mine]
        out[size] = {"best_lr": best["peak_lr"], "at_edge": best["peak_lr"] in (lrs[0], lrs[-1]) and len(lrs) > 1,
                     "curve": {f"{r['peak_lr']:g}": r["bits_per_byte"] for r in mine},
                     "non_embedding": best["non_embedding"], "tokens": best["tokens"]}
    return out


def isoflop_fit(rows: list[dict]) -> dict:
    """Parabola in log N per budget; then the exponents over interior contours."""
    contours = {}
    for c in sorted({r["budget"] for r in rows}):
        mine = [r for r in rows if r["budget"] == c]
        x = np.log([r["non_embedding"] for r in mine])
        y = np.array([r["bits_per_byte"] for r in mine])
        if len(set(x)) < 3:
            continue
        a2, a1, a0 = np.polyfit(x, y, 2)
        if a2 <= 0:
            contours[str(c)] = {"interior": False, "reason": "not convex"}
            continue
        xv = -a1 / (2 * a2)
        interior = x.min() < xv < x.max()
        n_star = math.exp(xv)
        contours[str(c)] = {"n_star": n_star, "d_star": c / (6 * n_star), "loss_at_vertex": float(a0 + a1 * xv + a2 * xv * xv),
                            "interior": bool(interior), "sizes": int(len(set(x)))}
    good = [(float(c), v) for c, v in contours.items() if v.get("interior")]
    out = {"contours": contours}
    if len(good) >= 2:
        lc = np.log([c for c, _ in good])
        out["a"] = float(np.polyfit(lc, np.log([v["n_star"] for _, v in good]), 1)[0])
        out["b"] = float(np.polyfit(lc, np.log([v["d_star"] for _, v in good]), 1)[0])
        out["tokens_per_parameter"] = {str(c): v["d_star"] / v["n_star"] for c, v in good}
    return out


def _huber(r, delta=1e-3):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def parametric_fit(rows: list[dict], count: str = "non_embedding", starts: list | None = None) -> dict:
    """log L = LSE(a - alpha log N, b - beta log D, e), with A = e^a, B = e^b, E = e^e.
    ``count``: "non_embedding" or "with_head" (the output layer counted in N)."""
    n = [r["non_embedding"] + (head_params(r["width"]) if count == "with_head" else 0) for r in rows]
    logn = np.log(n)
    logd = np.log([r["tokens"] for r in rows])
    logl = np.log([r["bits_per_byte"] for r in rows])

    def objective(p):
        a, b, e, alpha, beta = p
        pred = logsumexp(np.stack([a - alpha * logn, b - beta * logd, np.full_like(logn, e)]), axis=0)
        return _huber(pred - logl).sum()

    if starts is None:  # Hoffmann et al.'s grid of initialisations
        starts = [[a0, b0, e0, al, be] for a0 in (0.0, 5.0, 10.0) for b0 in (0.0, 5.0, 10.0)
                  for e0 in (-1.0, -0.5, 0.0) for al in (0.1, 0.3, 0.6) for be in (0.1, 0.3, 0.6)]
    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B",
                       bounds=[(-10, 30), (-10, 30), (-3, 1), (0.0, 2.0), (0.0, 2.0)])
        if best is None or res.fun < best.fun:
            best = res
    a, b, e, alpha, beta = best.x
    return {"count": count, "x": [float(v) for v in best.x], "A": float(math.exp(a)), "B": float(math.exp(b)), "E": float(math.exp(e)),
            "alpha": float(alpha), "beta": float(beta), "a_implied": float(beta / (alpha + beta)),
            "objective": float(best.fun)}


def true_cost_optimum(fit: dict, budgets=(1e14, 1e15, 1e16)) -> dict:
    """The compute-optimal non-embedding N under the true FLOP count, from a non-embedding fit:
    minimise L(N, C / flops_per_token(N)) over N, with width and depth following the sizes'
    shape (layers = width / 64)."""
    def loss(n, c):
        width = (n * 64 / 12) ** (1 / 3)  # 12 L d^2 = n with L = d / 64
        layers = width / 64
        d = c / flops_per_token(n, width, layers)
        return fit["E"] + fit["A"] / n ** fit["alpha"] + fit["B"] / d ** fit["beta"]

    grid = np.geomspace(1e4, 1e10, 4000)
    n_star = {}
    for c in budgets:
        n_star[c] = float(grid[int(np.argmin([loss(n, c) for n in grid]))])
    lc, ln = np.log(list(n_star)), np.log(list(n_star.values()))
    return {"n_star": {f"{c:g}": v for c, v in n_star.items()}, "a_true_cost": float(np.polyfit(lc, ln, 1)[0])}


def _near(x: list, rng) -> list:
    """The full-data optimum and four perturbations of it: the resamples' starting points."""
    x = np.asarray(x)
    return [x.tolist()] + [(x + rng.normal(0, [1.0, 1.0, 0.2, 0.05, 0.05])).tolist() for _ in range(4)]


def bootstrap(rows: list[dict], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    full, full_head = parametric_fit(rows), parametric_fit(rows, "with_head")
    iso, par, head, true_a = [], [], [], []
    for _ in range(BOOT):
        sample = [rows[i] for i in rng.integers(0, len(rows), size=len(rows))]
        f = isoflop_fit(sample)
        if "a" in f:
            iso.append((f["a"], f["b"]))
        p = parametric_fit(sample, starts=_near(full["x"], rng))
        par.append((p["alpha"], p["beta"], p["E"], p["a_implied"]))
        true_a.append(true_cost_optimum(p)["a_true_cost"])
        h = parametric_fit(sample, "with_head", starts=_near(full_head["x"], rng))
        head.append((h["alpha"], h["beta"], h["E"], h["a_implied"]))

    def rng5_95(values):
        v = np.asarray(values)
        return [float(np.percentile(v, 5)), float(np.percentile(v, 95))] if len(v) else None

    iso, par, head = np.asarray(iso), np.asarray(par), np.asarray(head)
    return {
        "isoflop_a": rng5_95(iso[:, 0]) if len(iso) else None,
        "isoflop_b": rng5_95(iso[:, 1]) if len(iso) else None,
        "isoflop_resamples_with_two_interior_contours": int(len(iso)),
        "alpha": rng5_95(par[:, 0]), "beta": rng5_95(par[:, 1]), "E": rng5_95(par[:, 2]),
        "a_implied": rng5_95(par[:, 3]), "a_true_cost": rng5_95(true_a),
        "with_head": {"alpha": rng5_95(head[:, 0]), "beta": rng5_95(head[:, 1]), "E": rng5_95(head[:, 2]),
                      "a_implied": rng5_95(head[:, 3])},
        "resamples": BOOT,
    }


def summarise() -> dict:
    out: dict = {"lr": lr_table()}
    rows = _load("isoflop")
    if rows:
        out["isoflop"] = isoflop_fit(rows)
        out["parametric"] = parametric_fit(rows)
        out["parametric_with_head"] = parametric_fit(rows, "with_head")
        out["true_cost"] = true_cost_optimum(out["parametric"])
        out["flops_per_token_over_6n"] = {r["size"]: flops_per_token(r["non_embedding"], r["width"], r["layers"])
                                          / (6 * r["non_embedding"]) for r in rows}
        out["bootstrap"] = bootstrap(rows)
        out["runs"] = len(rows)
    return out


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load("isoflop")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    budgets = sorted({r["budget"] for r in rows})
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(budgets):
        mine = sorted((r for r in rows if r["budget"] == c), key=lambda r: r["non_embedding"])
        color = cmap(i / max(1, len(budgets) - 1))
        axes[0].plot([r["non_embedding"] for r in mine], [r["bits_per_byte"] for r in mine], "o", color=color, alpha=0.6)
        v = summary["isoflop"]["contours"].get(str(c), {})
        if "n_star" in v:
            ns = np.geomspace(min(r["non_embedding"] for r in mine), max(r["non_embedding"] for r in mine), 50)
            x = np.log([r["non_embedding"] for r in mine])
            coef = np.polyfit(x, [r["bits_per_byte"] for r in mine], 2)
            axes[0].plot(ns, np.polyval(coef, np.log(ns)), "-", color=color, label=f"C={c:.0e}" + ("" if v["interior"] else " (edge)"))
            axes[0].plot([v["n_star"]], [v["loss_at_vertex"]], "*", color=color, ms=12)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("non-embedding parameters N")
    axes[0].set_ylabel("validation bits/byte")
    axes[0].set_title("IsoFLOP curves (C = 6ND)")
    axes[0].legend(fontsize=8)
    good = [(float(c), v) for c, v in summary["isoflop"]["contours"].items() if v.get("interior")]
    if good:
        axes[1].plot([c for c, _ in good], [v["n_star"] for _, v in good], "o-", label="N*")
        axes[1].plot([c for c, _ in good], [v["d_star"] for _, v in good], "s-", label="D* (tokens)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("compute C = 6ND (FLOPs)")
    axes[1].set_title("Compute-optimal N and D")
    axes[1].legend(fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "o6-scaling.png", dpi=130)


def main() -> None:
    s = summarise()
    for size, v in s["lr"].items():
        print(f"lr {size} N={v['non_embedding']:,} best {v['best_lr']:g}{' EDGE' if v['at_edge'] else ''}  "
              + "  ".join(f"{k}:{b:.4f}" for k, b in v["curve"].items()))
    if s["lr"]:
        (RUNS / "lr-table.json").write_text(json.dumps({k: v["best_lr"] for k, v in s["lr"].items()}, indent=1) + "\n")
    if "isoflop" in s:
        print(json.dumps({k: s[k] for k in ("isoflop", "parametric", "parametric_with_head", "true_cost", "bootstrap")}, indent=1))
    plot(s)
    (ROOT / "docs" / "data" / "o6-summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print("wrote docs/data/o6-summary.json")


if __name__ == "__main__":
    main()
