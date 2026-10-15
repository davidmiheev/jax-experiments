"""O2: Newton's method where ml-ladder used first order.

``logistic``  Logistic regression four ways, each one compiled ``while_loop`` to
              the same tolerance (relative gradient norm 1e-8), float64:
              Newton/IRLS (``jax.hessian`` + Cholesky), truncated Newton-CG
              (forward-over-reverse HVPs), L-BFGS (optax), and gradient descent
              with its constant step tuned on seed 0. Data: A1's two blobs, and
              a sweep over dimension d in {10, 100, 1000} and condition number
              kappa in {10, 1000} at n = 10,000.
``boosting``  Boosting with autodiff Newton leaves (``jaxexp.boosting``):
              (1) parity with XGBoost's exact updater on log-loss and
              pseudo-Huber (XGBoost's own objectives) and log-cosh (a custom
              objective fed to XGBoost from the same autodiff); (2) does the
              Newton step pay on log-loss? First-order, first-order trees with
              Newton leaves, and full Newton, each with eta tuned on a
              validation split, on A4's Friedman #1 binarised at its median.
``plot``      Figures into ``docs/figures/``.

    uv sync --extra boost
    uv run python experiments/o2_newton/run.py logistic boosting plot
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxexp import boosting as B  # noqa: E402
from jaxexp import newton as N  # noqa: E402
from jaxexp.bench import compiled_stats, summarise, time_call  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
FIGURES = ROOT / "docs" / "figures"
SEEDS = (0, 1, 2)
TOL = 1e-8


def ladder_datasets(path: str):
    sys.path[:0] = [str(Path(path).resolve() / "src")]
    from mlladder import datasets

    return datasets


def environment() -> dict:
    import optax

    env = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "optax": optax.__version__,
        "device": str(jax.devices()[0]),
        "cpu_count": __import__("os").cpu_count(),
        "x64": True,
    }
    try:
        import xgboost

        env["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return env


def write(name: str, report: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"o2-{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote docs/data/o2-{name}.json", flush=True)


# ---------------------------------------------------------------- logistic


def synthetic(seed: int, n: int, d: int, kappa: float):
    """Gaussian features with condition number kappa (random rotation), labels from a logistic model."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scales = np.logspace(0, -0.5 * np.log10(kappa), d)  # covariance eigenvalues 1 .. 1/kappa
    X = (rng.normal(size=(n, d)) * scales) @ q.T
    w = q @ (rng.normal(size=d) / scales) / np.sqrt(d) * 2.0  # comparable signal in every direction
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ w)))).astype(np.float64)
    return jnp.asarray(X), jnp.asarray(y)


def solve_all(loss, d: int, lr_scale: float | None, L: float, *, max_gd: int):
    """Run the four solvers; returns rows and the chosen GD scale."""
    w0 = jnp.zeros(d + 1)
    solvers = {
        "Newton (IRLS)": lambda w: N.newton(loss, w, tol=TOL),
        "Newton-CG": lambda w: N.newton_cg(loss, w, tol=TOL),
        "L-BFGS": lambda w: N.lbfgs(loss, w, tol=TOL, max_iter=5000),
    }
    if lr_scale is None:  # tune: the scale c in lr = c / L that converges in the fewest steps
        best = None  # (c, iterations); later candidates are capped at the best count so far
        for c in (64, 32, 16, 8, 4, 2, 1):
            cap = best[1] if best else max_gd
            r = jax.jit(lambda w, c=c, cap=cap: N.gradient_descent(loss, w, c / L, tol=TOL, max_iter=cap))(w0)
            ok = bool(r.grad_norm <= TOL)
            print(
                f"[logistic]   GD tune c={c:>2}: {int(r.iterations):>6} its, grad {float(r.grad_norm):.1e}", flush=True
            )
            if ok and (best is None or int(r.iterations) < best[1]):
                best = (c, int(r.iterations))
        lr_scale = best[0] if best else 1.0
    solvers["gradient descent"] = lambda w: N.gradient_descent(loss, w, lr_scale / L, tol=TOL, max_iter=max_gd)
    rows = {}
    for name, fn in solvers.items():
        compiled = jax.jit(fn)
        started = time.perf_counter()
        r = jax.block_until_ready(compiled(w0))
        first = time.perf_counter() - started
        # One timed re-run when the first took over a second; otherwise a median of several.
        reps = 1 if first > 1.0 else 3
        seconds = time_call(compiled, w0, min_reps=reps, max_reps=max(reps, 10), budget_s=2.0)["median"]
        hist = np.asarray(r.history)
        rows[name] = {
            "iterations": int(r.iterations),
            "gradients": int(r.gradients),
            "hessians": int(r.hessians),
            "hvps": int(r.hvps),
            "relative_grad_norm": float(r.grad_norm),
            "converged": bool(r.grad_norm <= TOL),
            "loss": float(loss(r.w)),
            "seconds": seconds,
            "compile_and_first_run_s": first,
            "history": hist[~np.isnan(hist)].tolist(),
        }
    return rows, lr_scale


def run_logistic(ladder: str) -> None:
    datasets = ladder_datasets(ladder)
    report = {"experiment": "O2 logistic regression", "environment": environment(), "tolerance": TOL, "configs": {}}

    configs = [("A1 blobs (2 features, 450 train)", None)] + [
        (f"d={d}, kappa={k:g}", (d, k)) for k in (10.0, 1000.0) for d in (10, 100, 1000)
    ]
    for label, spec in configs:
        per_seed, scale = [], None
        for seed in SEEDS:
            if spec is None:
                rng = np.random.default_rng(seed)
                X, y, _ = datasets.make_blobs(600, centers=2, cluster_std=2.2, center_box=(-3.0, 3.0), rng=rng)
                Xtr, _, ytr, _ = datasets.train_test_split(X, y, rng=rng)
                (Xtr,) = datasets.standardize(Xtr)
                X, y, l2 = jnp.asarray(Xtr), jnp.asarray(ytr, jnp.float64), 0.0  # A1: no penalty
            else:
                X, y = synthetic(seed, 10_000, *spec)
                l2 = 1e-4
            loss = N.logistic_loss(X, y, l2)
            Xb = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
            L = float(jnp.linalg.eigvalsh(Xb.T @ Xb)[-1]) / (4 * X.shape[0]) + l2  # Hessian bound
            rows, scale = solve_all(loss, X.shape[1], scale, L, max_gd=100_000)
            best = min(r["loss"] for r in rows.values())
            for r in rows.values():
                r["suboptimality"] = r["loss"] - best
            # Cost of one Hessian: jax.hessian against the hand-written IRLS X^T S X.
            w = jnp.full(X.shape[1] + 1, 0.01)
            auto = compiled_stats(jax.hessian(loss), w)
            irls = jax.jit(lambda w, X=X, l2=l2: N.logistic_hessian(X, w, l2))
            hand = compiled_stats(irls, w)
            rows["_hessian_cost"] = {
                "autodiff_flops": auto["flops"],
                "irls_flops": hand["flops"],
                "autodiff_s": time_call(jax.jit(jax.hessian(loss)), w, budget_s=1.0)["median"],
                "irls_s": time_call(irls, w, budget_s=1.0)["median"],
                "max_abs_difference": float(jnp.abs(jax.hessian(loss)(w) - irls(w)).max()),
            }
            per_seed.append(rows)
            print(
                f"[logistic] {label:<34} seed {seed}: "
                + "  ".join(
                    f"{k} {v['iterations']} its {v['seconds'] * 1e3:.1f} ms" for k, v in rows.items() if k[0] != "_"
                ),
                flush=True,
            )
        names = [k for k in per_seed[0] if k[0] != "_"]
        report["configs"][label] = {
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "l2": l2,
            "gd_lr_scale_over_L": scale,
            "solvers": {
                name: {
                    "iterations": summarise([r[name]["iterations"] for r in per_seed]),
                    "seconds": summarise([r[name]["seconds"] for r in per_seed]),
                    "gradients": summarise([r[name]["gradients"] for r in per_seed]),
                    "hessians": summarise([r[name]["hessians"] for r in per_seed]),
                    "hvps": summarise([r[name]["hvps"] for r in per_seed]),
                    "converged_all": all(r[name]["converged"] for r in per_seed),
                    "suboptimality": summarise([r[name]["suboptimality"] for r in per_seed]),
                    "compile_and_first_run_s": summarise([r[name]["compile_and_first_run_s"] for r in per_seed]),
                    "history_seed0": per_seed[0][name]["history"],
                }
                for name in names
            },
            "hessian_cost": {
                k: summarise([r["_hessian_cost"][k] for r in per_seed])
                for k in ("autodiff_flops", "irls_flops", "autodiff_s", "irls_s", "max_abs_difference")
            },
        }
    write("logistic", report)


# ---------------------------------------------------------------- boosting


def xgb_trees(booster) -> list[dict]:
    return [json.loads(t) for t in booster.get_dump(dump_format="json", with_stats=True)]


def flatten_xgb(node: dict) -> list[tuple]:
    """Preorder (feature, threshold, leaf) with the 'yes' (x < threshold) child first, as ours."""
    if "leaf" in node:
        return [(-1, 0.0, node["leaf"])]
    children = {c["nodeid"]: c for c in node["children"]}
    feature = int(node["split"].lstrip("f"))
    return (
        [(feature, node["split_condition"], 0.0)]
        + flatten_xgb(children[node["yes"]])
        + flatten_xgb(children[node["no"]])
    )


def flatten_ours(tree: B.Tree) -> list[tuple]:
    def walk(i):
        if tree.feature[i] < 0:
            return [(-1, 0.0, tree.value[i])]
        return [(tree.feature[i], tree.threshold[i], 0.0)] + walk(tree.left[i]) + walk(tree.right[i])

    return walk(0)


def parity(Xtr, ytr, Xte, *, loss, objective: dict, base_margin: float, rounds: int, seed: int) -> dict:
    import xgboost as xgb

    common = {"eta": 0.3, "max_depth": 3, "reg_lambda": 1.0, "min_child_weight": 1.0}
    started = time.perf_counter()
    ours = B.Booster(loss, rounds=rounds, base_margin=base_margin, **common).fit(Xtr, ytr)
    ours_s = time.perf_counter() - started

    params = {
        "eta": 0.3,
        "max_depth": 3,
        "lambda": 1.0,
        "min_child_weight": 1.0,
        "tree_method": "exact",
        "nthread": 1,
        "seed": seed,
        "verbosity": 0,
        **objective.get("params", {}),
    }
    dtrain = xgb.DMatrix(np.asarray(Xtr, np.float32), label=ytr)
    if "obj" in objective:  # custom objective: XGBoost gets the same autodiff g, h
        derivs = B.derivatives(loss)
        params["base_score"] = base_margin

        def obj(margin, dmatrix):
            g, h = derivs(jnp.asarray(dmatrix.get_label(), jnp.float64), jnp.asarray(margin, jnp.float64))
            return np.asarray(g), np.asarray(h)

        started = time.perf_counter()
        bst = xgb.train(params, dtrain, num_boost_round=rounds, obj=obj)
    else:
        started = time.perf_counter()
        bst = xgb.train(params, dtrain, num_boost_round=rounds)
    xgb_s = time.perf_counter() - started

    same_structure, leaf_diffs = 0, []
    for theirs, mine in zip(xgb_trees(bst), ours.trees, strict=True):
        a, b = flatten_xgb(theirs), flatten_ours(mine)
        if len(a) == len(b) and all(
            fa == fb and np.float32(ta) == np.float32(tb) for (fa, ta, _), (fb, tb, _) in zip(a, b, strict=True)
        ):
            same_structure += 1
            leaf_diffs += [abs(la - lb) for (fa, _, la), (_, _, lb) in zip(a, b, strict=True) if fa < 0]
    m_x = bst.predict(xgb.DMatrix(np.asarray(Xte, np.float32)), output_margin=True).astype(np.float64)
    m_o = ours.margin(Xte)
    return {
        "rounds": rounds,
        "trees_with_identical_structure": same_structure,
        "max_abs_leaf_difference": max(leaf_diffs) if leaf_diffs else None,
        "max_abs_test_margin_difference": float(np.abs(m_x - m_o).max()),
        "max_abs_test_margin": float(np.abs(m_x).max()),
        "ours_fit_s": ours_s,
        "xgboost_fit_s": xgb_s,
    }


def log_loss_np(y, margin):
    return float(np.mean(np.logaddexp(0.0, margin) - y * margin))


def run_boosting(ladder: str) -> None:
    datasets = ladder_datasets(ladder)
    report = {"experiment": "O2 autodiff Newton boosting", "environment": environment()}

    # (1) Parity with XGBoost.
    losses = {
        "log-loss (binary:logistic)": (B.log_loss, {"params": {"objective": "binary:logistic", "base_score": 0.5}}),
        "pseudo-Huber, delta 1 (reg:pseudohubererror)": (
            B.pseudo_huber(1.0),
            {"params": {"objective": "reg:pseudohubererror", "huber_slope": 1.0}},
        ),
        "log-cosh (custom objective)": (B.log_cosh, {"obj": True}),
    }
    parity_rows = {}
    for name, (loss, objective) in losses.items():
        rows = []
        for seed in SEEDS:
            X, y = datasets.make_friedman1(2000, n_features=10, noise=1.0, rng=seed)
            Xtr, Xte, ytr, yte = datasets.train_test_split(X, y, rng=seed)
            if "binary" in str(objective):
                cut = np.median(ytr)
                ytr, yte = (ytr > cut).astype(np.float64), (yte > cut).astype(np.float64)
                base = 0.0  # margin of base_score 0.5
            else:
                base = float(np.mean(ytr))
                objective.setdefault("params", {})["base_score"] = base
            rows.append(parity(Xtr, ytr, Xte, loss=loss, objective=objective, base_margin=base, rounds=100, seed=seed))
        parity_rows[name] = {
            "per_seed": rows,
            "trees_with_identical_structure": [r["trees_with_identical_structure"] for r in rows],
            "max_abs_leaf_difference": summarise([r["max_abs_leaf_difference"] for r in rows]),
            "max_abs_test_margin_difference": summarise([r["max_abs_test_margin_difference"] for r in rows]),
            "max_abs_test_margin": summarise([r["max_abs_test_margin"] for r in rows]),
            "ours_fit_s": summarise([r["ours_fit_s"] for r in rows]),
            "xgboost_fit_s": summarise([r["xgboost_fit_s"] for r in rows]),
        }
        p = parity_rows[name]
        print(
            f"[boosting] parity {name:<46} identical trees {p['trees_with_identical_structure']}/100  "
            f"leaf |d| {p['max_abs_leaf_difference']['max']:.1e}  "
            f"margin |d| {p['max_abs_test_margin_difference']['max']:.1e}",
            flush=True,
        )
    report["parity"] = parity_rows

    # (2) Does the Newton step pay? Friedman #1 at its median, train / validation / test.
    etas, rounds = (0.03, 0.1, 0.3, 1.0), 300
    arms = {"first-order": "ones", "first-order trees, Newton leaves": "leaves", "Newton (XGBoost-style)": "exact"}
    results = {arm: [] for arm in arms}
    for seed in SEEDS:
        X, y = datasets.make_friedman1(5000, n_features=10, noise=1.0, rng=seed)
        Xrest, Xte, yrest, yte = datasets.train_test_split(X, y, rng=seed)
        Xtr, Xva, ytr, yva = datasets.train_test_split(Xrest, yrest, rng=seed + 100)
        cut = np.median(ytr)
        ytr, yva, yte = ((a > cut).astype(np.float64) for a in (ytr, yva, yte))
        for arm, hessian in arms.items():
            curves, best = {}, None
            for eta in etas:
                model = B.Booster(B.log_loss, rounds=rounds, eta=eta, hessian=hessian).fit(Xtr, ytr)
                val = [log_loss_np(yva, m) for m in model.staged_margins(Xva)]
                curves[eta] = val
                r = int(np.argmin(val)) + 1
                if best is None or val[r - 1] < best[0]:
                    best = (val[r - 1], eta, r, model)
            _, eta, r, model = best
            margin = model.margin(Xte, rounds=r)
            results[arm].append(
                {
                    "eta": eta,
                    "rounds": r,
                    "val_log_loss": best[0],
                    "test_log_loss": log_loss_np(yte, margin),
                    "test_accuracy": float(np.mean((margin > 0) == (yte > 0.5))),
                    "val_curves": {str(e): c for e, c in curves.items()},
                }
            )
            print(
                f"[boosting] seed {seed} {arm:<34} eta {eta:<5} rounds {r:>3}  "
                f"test log-loss {results[arm][-1]['test_log_loss']:.4f}  acc {results[arm][-1]['test_accuracy']:.4f}",
                flush=True,
            )
    report["newton_pays"] = {
        "data": "Friedman #1 (10 features, noise 1), 5000 samples, label y > median(train); 2812/938/1250 split",
        "etas": etas,
        "max_rounds": rounds,
        "settings": {"max_depth": 3, "lambda": 1.0, "min_child_weight": 1.0},
        "arms": {
            arm: {
                "test_log_loss": summarise([r["test_log_loss"] for r in rows]),
                "test_accuracy": summarise([r["test_accuracy"] for r in rows]),
                "val_log_loss": summarise([r["val_log_loss"] for r in rows]),
                "chosen_eta": [r["eta"] for r in rows],
                "chosen_rounds": [r["rounds"] for r in rows],
                "val_curves_seed0": rows[0]["val_curves"],
            }
            for arm, rows in results.items()
        },
    }
    write("boosting", report)


# ---------------------------------------------------------------- plot


def run_plot() -> None:
    import matplotlib.pyplot as plt

    from jaxexp.plotting import INK_SECONDARY, SERIES, save, style

    style()
    lo = json.loads((DATA / "o2-logistic.json").read_text())
    names = ["Newton (IRLS)", "Newton-CG", "L-BFGS", "gradient descent"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    config = lo["configs"]["d=100, kappa=1000"]
    for name, color in zip(names, SERIES, strict=False):
        hist = np.asarray(config["solvers"][name]["history_seed0"])
        best = min(min(config["solvers"][n]["history_seed0"]) for n in names)
        sub = np.maximum(hist - best, 1e-17)
        left.plot(np.arange(len(sub)), sub, color=color, label=name, marker="o" if len(sub) < 40 else None)
    left.set_xscale("symlog", linthresh=10)
    left.set_yscale("log")
    left.set_ylim(1e-17, None)
    left.set_xlabel("iteration")
    left.set_ylabel("loss − best loss")
    left.set_title("d = 100, κ = 1000, seed 0")
    left.legend(loc="lower left")
    for kappa, style_ in (("1000", "-"), ("10", (0, (3, 2)))):
        dims = [10, 100, 1000]
        for name, color in zip(names, SERIES, strict=False):
            secs = [lo["configs"][f"d={d}, kappa={kappa}"]["solvers"][name]["seconds"]["median"] * 1e3 for d in dims]
            right.plot(
                dims, secs, color=color, linestyle=style_, marker="o", label=f"{name}" if kappa == "1000" else None
            )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("dimension d   (n = 10,000)")
    right.set_ylabel("ms to tolerance (median of seeds)")
    right.set_title("Wall-clock to |∇L| ≤ 1e-8 |∇L₀|;  solid κ=1000, dashed κ=10")
    right.title.set_fontsize(9.5)
    right.legend(loc="upper left", fontsize=7.5)
    save(fig, FIGURES / "o2_logistic.png")

    bo = json.loads((DATA / "o2-boosting.json").read_text())
    arms = bo["newton_pays"]["arms"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    for (arm, row), color in zip(arms.items(), SERIES, strict=False):
        eta = str(row["chosen_eta"][0])
        curve = row["val_curves_seed0"][eta]
        left.plot(np.arange(1, len(curve) + 1), curve, color=color, label=f"{arm} (η={eta})")
    left.set_xlabel("boosting round")
    left.set_ylabel("validation log-loss")
    left.set_title("Seed 0, each arm at its tuned η")
    left.set_ylim(None, 0.5)
    left.legend(loc="upper right", fontsize=7.5)
    ys = np.arange(len(arms))
    med = [row["test_log_loss"]["median"] for row in arms.values()]
    lo_ = [row["test_log_loss"]["median"] - row["test_log_loss"]["min"] for row in arms.values()]
    hi_ = [row["test_log_loss"]["max"] - row["test_log_loss"]["median"] for row in arms.values()]
    right.barh(
        ys,
        med,
        xerr=[lo_, hi_],
        color=SERIES[: len(arms)],
        height=0.55,
        edgecolor="white",
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3},
    )
    for y_, m in zip(ys, med, strict=True):
        right.text(m * 0.02, y_, f"{m:.4f}", va="center", ha="left", color="white", fontsize=8, fontweight="bold")
    right.set_yticks(ys, [a.replace(", ", ",\n") for a in arms], fontsize=8)
    right.invert_yaxis()
    right.set_xlabel("test log-loss (median, min–max over seeds)")
    right.set_title("Held-out log-loss")
    right.grid(axis="y", visible=False)
    save(fig, FIGURES / "o2_boosting.png")
    print("figures written", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="+", choices=["logistic", "boosting", "plot"])
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    args = parser.parse_args()
    for part in args.parts:
        started = time.time()
        if part == "plot":
            run_plot()
        else:
            {"logistic": run_logistic, "boosting": run_boosting}[part](args.ml_ladder)
        print(f"[{part}] {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
