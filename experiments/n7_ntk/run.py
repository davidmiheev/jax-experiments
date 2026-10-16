"""N7: the empirical neural tangent kernel against the infinite-width prediction.

Two-hidden-layer ReLU MLPs in NTK parameterisation (``jaxexp.ntk``), scalar
output, on two tasks: ml-ladder B1's two moons (noise 0.18; 256 train, 1024
test) and MNIST even-vs-odd (B2's normalisation; 1000 train and 1000 test,
100 per digit). Targets are +1 / -1. Seeds 0-2 vary the data draw and the init.

``kernel``  At initialisation: relative Frobenius distance between the empirical
            NTK and the analytic one on 256 inputs, for widths 32..8192.
``train``   Full-batch gradient descent on the centred, alpha-scaled model
            g = alpha (f(w) - f(w0)), loss 0.5 mean((g - y)^2) / alpha^2 (Chizat,
            Oyallon & Bach 2019). Its function-space dynamics at t = 0 are
            g' = -lr Theta (g - y) / n for every alpha, so one learning rate,
            lr = n / lambda_max(Theta_inf), serves every width and alpha. Compared at
            log-spaced steps with two predictions that need no network at all:
            the linearised model (the network's own empirical NTK at init) and
            infinite-width kernel regression (the analytic NTK). Width sweep at
            alpha = 1.
``alpha``   The same at width 512 for alpha in {0.03 .. 10}: small alpha forces the
            weights to move far (feature learning), large alpha keeps them lazy.
``plot``    Figures.

Kernels and kernel regression run in float64; the networks train in float32.

    uv run python experiments/n7_ntk/run.py kernel train alpha plot
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

from jaxexp import data as D  # noqa: E402
from jaxexp import ntk as K  # noqa: E402
from jaxexp.bench import summarise  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
FIGURES = ROOT / "docs" / "figures"
CACHE = ROOT / "runs" / "n7"
SEEDS = (0, 1, 2)
DEPTH = 2


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "cpu_count": __import__("os").cpu_count(),
    }


def load_task(name: str, seed: int, ladder: str):
    """(X_train, y_train, X_test, y_test) as float32 arrays, y in {+1, -1}."""
    rng = np.random.default_rng(seed)
    if name == "moons":
        sys.path[:0] = [str(Path(ladder).resolve() / "src")]
        from mlladder import datasets

        X, y = datasets.make_moons(256 + 1024, noise=0.18, rng=rng)
        Xtr, Xte, ytr, yte = datasets.train_test_split(X, y, test_size=1024 / 1280, rng=rng)
        Xtr, Xte = datasets.standardize(Xtr, Xte)
    else:
        images, labels = D.mnist("train")
        tr = D.balanced_subset(labels, 100, rng)
        timages, tlabels = D.mnist("test")
        te = D.balanced_subset(tlabels, 100, rng)
        Xtr, ytr, Xte, yte = images[tr], labels[tr] % 2 == 0, timages[te], tlabels[te] % 2 == 0
    sign = lambda v: np.where(np.asarray(v) > 0.5, 1.0, -1.0).astype(np.float32)  # noqa: E731
    return np.asarray(Xtr, np.float32), sign(ytr), np.asarray(Xte, np.float32), sign(yte)


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cached(key: str, fn):
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    result = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


# ---------------------------------------------------------------- kernel at init


def run_kernel(ladder: str) -> None:
    widths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    report = {
        "experiment": "N7 kernel at initialisation",
        "environment": environment(),
        "widths": widths,
        "depth": DEPTH,
        "points": 256,
        "tasks": {},
    }
    for task in ("moons", "mnist"):
        per_width = {w: [] for w in widths}
        diag_err = {w: [] for w in widths}
        for seed in SEEDS:
            Xtr, *_ = load_task(task, seed, ladder)
            X = Xtr[:256]
            analytic = np.asarray(K.analytic_ntk(jnp.asarray(X, jnp.float64), jnp.asarray(X, jnp.float64), DEPTH)[0])
            for w in widths:

                def one(w=w, X=X, seed=seed, analytic=analytic):
                    emp = np.asarray(K.empirical_ntk(K.init(jax.random.key(seed), X.shape[1], w, DEPTH), X, X))
                    return {"frobenius": rel(emp, analytic), "diagonal": rel(np.diag(emp), np.diag(analytic))}

                r = cached(f"kernel-{task}-w{w}-s{seed}", one)
                per_width[w].append(r["frobenius"])
                diag_err[w].append(r["diagonal"])
        logw = np.log(widths[3:])
        slope = float(np.polyfit(logw, np.log([np.median(per_width[w]) for w in widths[3:]]), 1)[0])
        report["tasks"][task] = {
            "frobenius_relative_error": {str(w): summarise(per_width[w]) for w in widths},
            "diagonal_relative_error": {str(w): summarise(diag_err[w]) for w in widths},
            "loglog_slope_width_256_up": slope,
        }
        print(
            f"[kernel] {task}: "
            + "  ".join(f"{w}:{np.median(per_width[w]):.3f}" for w in widths)
            + f"  slope {slope:.3f}",
            flush=True,
        )
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "n7-kernel.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote docs/data/n7-kernel.json", flush=True)


# ---------------------------------------------------------------- training


ACTIVATIONS = {"relu": jax.nn.relu, "gelu": lambda h: jax.nn.gelu(h, approximate=False)}


def train_run(
    task: str, width: int, alpha: float, seed: int, steps: list[int], ladder: str, act: str = "relu", dtype=jnp.float32
) -> dict:
    """``act`` other than relu is compared only with its own linearisation (the analytic kernel is ReLU's)."""
    activation = ACTIVATIONS[act]
    Xtr, ytr, Xte, yte = load_task(task, seed, ladder)
    n = len(Xtr)
    X64, Xte64 = jnp.asarray(Xtr, jnp.float64), jnp.asarray(Xte, jnp.float64)
    th_inf, _ = K.analytic_ntk(X64, X64, DEPTH)
    th_inf_te, _ = K.analytic_ntk(Xte64, X64, DEPTH)
    lr = float(n / jnp.linalg.eigvalsh(th_inf)[-1])  # lr * lambda_max / n = 1 for the infinite-width kernel

    params0 = K.init(jax.random.key(seed), Xtr.shape[1], width, DEPTH, dtype)
    th0 = jnp.asarray(K.empirical_ntk(params0, Xtr, Xtr, act=activation), jnp.float64)
    th0_te = jnp.asarray(K.empirical_ntk(params0, Xte, Xtr, act=activation), jnp.float64)
    y = jnp.asarray(ytr, jnp.float64)
    lin_tr, lin_te = K.kernel_regression_path(th0, th0_te, y, lr, steps)
    inf_tr, inf_te = K.kernel_regression_path(th_inf, th_inf_te, y, lr, steps)

    f = jax.vmap(lambda p, x: K.apply(p, x, act=activation), (None, 0))
    f0_tr, f0_te = f(params0, Xtr), f(params0, Xte)

    def g(params, X, f0):
        return alpha * (f(params, X) - f0)

    def loss(params):
        return 0.5 * jnp.mean((g(params, Xtr, f0_tr) - ytr) ** 2) / alpha**2

    @jax.jit
    def run_steps(params, k):
        def body(_, p):
            grads = jax.grad(loss)(p)
            return jax.tree.map(lambda a, b: a - lr * b, p, grads)

        return jax.lax.fori_loop(0, k, body, params)

    out = {"lr": lr, "alpha": alpha, "width": width, "activation": act, "steps": steps, "checkpoints": []}
    params, done = params0, 0
    started = time.perf_counter()
    for i, t in enumerate(steps):
        params = run_steps(params, t - done)
        done = t
        net_tr, net_te = np.asarray(g(params, Xtr, f0_tr)), np.asarray(g(params, Xte, f0_te))
        finite = bool(np.all(np.isfinite(net_te)))
        out["checkpoints"].append(
            {
                "step": t,
                "finite": finite,
                "train_mse": float(np.mean((net_tr - ytr) ** 2)),
                "test_mse": float(np.mean((net_te - yte) ** 2)),
                "test_accuracy": float(np.mean(np.sign(net_te) == yte)),
                "net_vs_linearised_test": rel(net_te, lin_te[i]) if t > 0 and finite else None,
                "net_vs_infinite_test": rel(net_te, inf_te[i]) if t > 0 and finite else None,
                "linearised_vs_infinite_test": rel(lin_te[i], inf_te[i]) if t > 0 else None,
                "infinite_test_mse": float(np.mean((np.asarray(inf_te[i]) - yte) ** 2)),
                "infinite_test_accuracy": float(np.mean(np.sign(np.asarray(inf_te[i])) == yte)),
                "linearised_test_accuracy": float(np.mean(np.sign(np.asarray(lin_te[i])) == yte)),
            }
        )
    out["train_seconds"] = time.perf_counter() - started
    # How far the kernel moved: empirical NTK on the training set, at the end vs at init.
    if np.all(np.isfinite(np.asarray(jax.tree.leaves(params)[0]))):
        thT = jnp.asarray(K.empirical_ntk(params, Xtr, Xtr, act=activation), jnp.float64)
        out["kernel_drift"] = rel(thT, th0)
    else:
        out["kernel_drift"] = None
    weights0 = jnp.concatenate([jnp.ravel(a) for a in jax.tree.leaves(params0)])
    weights = jnp.concatenate([jnp.ravel(a) for a in jax.tree.leaves(params)])
    out["weight_relative_change"] = float(jnp.linalg.norm(weights - weights0) / jnp.linalg.norm(weights0))
    return out


def summarise_runs(runs: list[dict]) -> dict:
    final = [r["checkpoints"][-1] for r in runs]
    keys = (
        "train_mse",
        "test_mse",
        "test_accuracy",
        "net_vs_linearised_test",
        "net_vs_infinite_test",
        "linearised_vs_infinite_test",
        "infinite_test_mse",
        "infinite_test_accuracy",
        "linearised_test_accuracy",
    )
    out = {k: summarise([c[k] for c in final]) for k in keys if all(c[k] is not None for c in final)}
    out["diverged_seeds"] = sum(not c["finite"] for c in final)
    for k in ("kernel_drift", "weight_relative_change"):
        vals = [r[k] for r in runs if r[k] is not None]
        if vals:
            out[k] = summarise(vals)
    out["lr"] = [r["lr"] for r in runs]
    out["train_seconds"] = summarise([r["train_seconds"] for r in runs])
    out["curves"] = [
        {
            "step": c["step"],
            **{
                k: c[k]
                for k in (
                    "net_vs_linearised_test",
                    "net_vs_infinite_test",
                    "test_mse",
                    "train_mse",
                    "infinite_test_mse",
                )
            },
        }
        for c in runs[0]["checkpoints"]
    ]
    return out


STEPS = {"moons": [0, 10, 30, 100, 300, 1000], "mnist": [0, 10, 30, 100, 300, 500]}


def run_train(ladder: str) -> None:
    widths = [64, 256, 1024, 2048]
    report = {
        "experiment": "N7 training vs kernel predictions, width sweep at alpha = 1",
        "environment": environment(),
        "depth": DEPTH,
        "widths": widths,
        "steps": STEPS,
        "tasks": {},
    }
    for task in ("moons", "mnist"):
        report["tasks"][task] = {}
        for w in widths:
            runs = [
                cached(
                    f"train-{task}-w{w}-a1-s{s}",
                    lambda s=s, task=task, w=w: train_run(task, w, 1.0, s, STEPS[task], ladder),
                )
                for s in SEEDS
            ]
            summary = summarise_runs(runs)
            report["tasks"][task][str(w)] = summary
            print(
                f"[train] {task} width {w:>5}: net vs linearised {summary['net_vs_linearised_test']['median']:.3e}  "
                f"vs infinite {summary['net_vs_infinite_test']['median']:.3e}  "
                f"drift {summary['kernel_drift']['median']:.3e}  "
                f"test acc {summary['test_accuracy']['median']:.3f} "
                f"(NTK {summary['infinite_test_accuracy']['median']:.3f})",
                flush=True,
            )
    (DATA / "n7-train.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote docs/data/n7-train.json", flush=True)


def run_alpha(ladder: str, act: str = "relu") -> None:
    """ReLU (the default) and, as the control for ReLU's kinks, exact-erf GELU."""
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    width = 512
    report = {
        "experiment": f"N7 lazy vs feature learning: alpha sweep at width 512, {act}",
        "activation": act,
        "environment": environment(),
        "depth": DEPTH,
        "width": width,
        "alphas": alphas,
        "steps": STEPS,
        "tasks": {},
    }
    for task in ("moons", "mnist"):
        report["tasks"][task] = {}
        for a in alphas:
            runs = [
                cached(
                    f"alpha-{task}-w{width}-a{a}-s{s}" + ("" if act == "relu" else f"-{act}"),
                    lambda s=s, a=a, task=task: train_run(task, width, a, s, STEPS[task], ladder, act),
                )
                for s in SEEDS
            ]
            summary = summarise_runs(runs)
            report["tasks"][task][str(a)] = summary
            lin = summary.get("net_vs_linearised_test", {}).get("median", float("nan"))
            drift = summary.get("kernel_drift", {}).get("median", float("nan"))
            print(
                f"[alpha-{act}] {task} alpha {a:>5}: net vs linearised {lin:.3e}  drift {drift:.3e}  "
                f"test acc {summary['test_accuracy']['median']:.3f}  diverged {summary['diverged_seeds']}",
                flush=True,
            )
    name = "n7-alpha.json" if act == "relu" else f"n7-alpha-{act}.json"
    (DATA / name).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote docs/data/{name}", flush=True)


# ---------------------------------------------------------------- plot


def run_converged(ladder: str) -> None:
    """Control for the alpha sweep's accuracy: infinite-width kernel regression trained further.

    Small-alpha networks reach a much lower training loss in the same number of steps (their
    kernel grows as features move), so their accuracy gain could be "further along" rather
    than "better features". This runs the same kernel model to 10x, 100x steps and to
    convergence (the interpolant), on the same data draws."""
    report = {"experiment": "N7 kernel regression trained to convergence", "environment": environment(), "tasks": {}}
    for task in ("moons", "mnist"):
        rows = []
        for seed in SEEDS:
            Xtr, ytr, Xte, yte = load_task(task, seed, ladder)
            X64, Xte64 = jnp.asarray(Xtr, jnp.float64), jnp.asarray(Xte, jnp.float64)
            th, _ = K.analytic_ntk(X64, X64, DEPTH)
            th_te, _ = K.analytic_ntk(Xte64, X64, DEPTH)
            lr = float(len(Xtr) / jnp.linalg.eigvalsh(th)[-1])
            t0 = STEPS[task][-1]
            steps = [t0, 10 * t0, 100 * t0, 10**12]
            tr, te = K.kernel_regression_path(th, th_te, jnp.asarray(ytr, jnp.float64), lr, steps)
            rows.append(
                {
                    str(t): {
                        "train_mse": float(np.mean((np.asarray(tr[i]) - ytr) ** 2)),
                        "test_mse": float(np.mean((np.asarray(te[i]) - yte) ** 2)),
                        "test_accuracy": float(np.mean(np.sign(np.asarray(te[i])) == yte)),
                    }
                    for i, t in enumerate(steps)
                }
            )
        report["tasks"][task] = {
            t: {k: summarise([r[t][k] for r in rows]) for k in ("train_mse", "test_mse", "test_accuracy")}
            for t in rows[0]
        }
        print(
            f"[converged] {task}: "
            + "  ".join(
                f"t={t}: acc {v['test_accuracy']['median']:.3f} train mse {v['train_mse']['median']:.3g}"
                for t, v in report["tasks"][task].items()
            ),
            flush=True,
        )
    (DATA / "n7-converged.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote docs/data/n7-converged.json", flush=True)


def run_floor(ladder: str) -> None:
    """Is the upturn at large alpha float32? Moons, width 512, seed 0, float32 vs float64 params."""
    report = {"experiment": "N7 precision floor at large alpha", "environment": environment(), "runs": []}
    for act in ("relu", "gelu"):
        for alpha in (30.0, 100.0):
            for name, dtype in (("float32", jnp.float32), ("float64", jnp.float64)):
                r = cached(
                    f"floor-moons-w512-a{alpha}-s0-{act}-{name}",
                    lambda act=act, alpha=alpha, dtype=dtype: train_run(
                        "moons", 512, alpha, 0, STEPS["moons"], ladder, act, dtype
                    ),
                )
                end = r["checkpoints"][-1]
                report["runs"].append(
                    {
                        "activation": act,
                        "alpha": alpha,
                        "dtype": name,
                        "net_vs_linearised_test": end["net_vs_linearised_test"],
                        "kernel_drift": r["kernel_drift"],
                        "weight_relative_change": r["weight_relative_change"],
                    }
                )
                print(
                    f"[floor] {act} alpha {alpha:>5} {name}: net vs linearised "
                    f"{end['net_vs_linearised_test']:.2e}  drift {r['kernel_drift']:.2e}",
                    flush=True,
                )
    (DATA / "n7-floor.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote docs/data/n7-floor.json", flush=True)


def run_plot() -> None:
    import matplotlib.pyplot as plt

    from jaxexp.plotting import INK_SECONDARY, SERIES, save, style

    style()
    kern = json.loads((DATA / "n7-kernel.json").read_text())
    train = json.loads((DATA / "n7-train.json").read_text())
    relu = json.loads((DATA / "n7-alpha.json").read_text())
    gelu = json.loads((DATA / "n7-alpha-gelu.json").read_text())
    converged = json.loads((DATA / "n7-converged.json").read_text())
    tasks = (("moons", SERIES[0]), ("mnist", SERIES[1]))
    guide = {"color": INK_SECONDARY, "linestyle": (0, (3, 3)), "linewidth": 1}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    (k_ax, step_ax, width_ax), (gap_ax, drift_ax, acc_ax) = axes

    widths = kern["widths"]
    for task, color in tasks:
        r = kern["tasks"][task]["frobenius_relative_error"]
        k_ax.plot(
            widths,
            [r[str(w)]["median"] for w in widths],
            marker="o",
            color=color,
            label=f"{task} (slope {kern['tasks'][task]['loglog_slope_width_256_up']:.2f})",
        )
        k_ax.fill_between(
            widths,
            [r[str(w)]["min"] for w in widths],
            [r[str(w)]["max"] for w in widths],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ref = np.asarray(widths, float)
    k_ax.plot(ref, 0.1 * np.sqrt(512 / ref), label="width$^{-1/2}$", **guide)
    k_ax.set(
        xscale="log", yscale="log", xlabel="width", ylabel="‖Θ_emp − Θ_∞‖ / ‖Θ_∞‖", title="Kernel at initialisation"
    )
    k_ax.set_xscale("log", base=2)
    k_ax.legend(fontsize=7.5)

    tw = train["widths"]
    for w, color in zip(tw, SERIES, strict=False):
        curve = [c for c in train["tasks"]["mnist"][str(w)]["curves"] if c["step"] > 0]
        step_ax.plot(
            [c["step"] for c in curve],
            [c["net_vs_linearised_test"] for c in curve],
            marker="o",
            color=color,
            label=f"width {w}",
        )
    step_ax.set(
        xscale="log",
        yscale="log",
        xlabel="GD step (MNIST, α = 1, seed 0)",
        ylabel="‖g_net − g_lin‖ / ‖g_lin‖ (test)",
        title="Linearisation holds for a while",
    )
    step_ax.legend(fontsize=7.5)

    for task, color in tasks:
        t = train["tasks"][task]
        width_ax.plot(
            tw,
            [t[str(w)]["net_vs_linearised_test"]["median"] for w in tw],
            marker="o",
            color=color,
            label=f"{task}: net vs linearised",
        )
        width_ax.plot(
            tw,
            [t[str(w)]["kernel_drift"]["median"] for w in tw],
            marker="s",
            color=color,
            linestyle=(0, (4, 2)),
            label=f"{task}: kernel drift",
        )
    width_ax.set(
        xscale="log",
        yscale="log",
        xlabel="width (α = 1, end of training)",
        ylabel="relative change",
        title="Lazier as width grows",
    )
    width_ax.set_xscale("log", base=2)
    width_ax.legend(fontsize=7)

    def alpha_panel(ax, key, title, ylabel):
        for task, color in tasks:
            for report, ls, name in ((relu, "-", "ReLU"), (gelu, (0, (4, 2)), "GELU")):
                t = report["tasks"][task]
                al = [a for a in report["alphas"] if key in t[str(a)]]
                ax.plot(
                    al,
                    [t[str(a)][key]["median"] for a in al],
                    marker="o" if name == "ReLU" else "s",
                    color=color,
                    linestyle=ls,
                    label=f"{task}, {name}",
                )
        al = np.asarray(relu["alphas"], float)
        anchor = relu["tasks"]["mnist"]["1.0"][key]["median"]
        ax.plot(al, anchor / al, label="α$^{-1}$", **guide)
        ax.plot(al, anchor / np.sqrt(al), label="α$^{-1/2}$", color=INK_SECONDARY, linestyle=(0, (1, 2)), linewidth=1)
        ax.set(xscale="log", yscale="log", xlabel="α (width 512)", ylabel=ylabel, title=title)
        ax.legend(fontsize=6.5, ncol=2)

    alpha_panel(gap_ax, "net_vs_linearised_test", "Distance from the linearised model", "relative (test, end)")
    alpha_panel(drift_ax, "kernel_drift", "Kernel drift", "‖Θ_T − Θ_0‖ / ‖Θ_0‖")

    for task, color in tasks:
        for report, ls, name in ((relu, "-", "ReLU"), (gelu, (0, (4, 2)), "GELU")):
            t = report["tasks"][task]
            al = report["alphas"]
            acc_ax.plot(
                al,
                [t[str(a)]["test_accuracy"]["median"] for a in al],
                marker="o",
                color=color,
                linestyle=ls,
                label=f"{task}, {name}",
            )
        ntk_acc = relu["tasks"][task]["1.0"]["infinite_test_accuracy"]["median"]
        acc_ax.axhline(ntk_acc, color=color, linewidth=1, alpha=0.6, linestyle=(0, (1, 2)))
        conv_acc = converged["tasks"][task]["1000000000000"]["test_accuracy"]["median"]
        acc_ax.axhline(conv_acc, color=color, linewidth=1.2, alpha=0.8, linestyle=(0, (6, 2, 1, 2)))
    acc_ax.set(
        xscale="log",
        xlabel="α (width 512)",
        ylabel="test accuracy (end of training)",
        title="Test accuracy",
    )
    acc_ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, FIGURES / "n7_ntk.png")
    print("figures written", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "parts", nargs="+", choices=["kernel", "train", "alpha", "alpha-gelu", "converged", "floor", "plot"]
    )
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    args = parser.parse_args()
    for part in args.parts:
        started = time.time()
        if part == "plot":
            run_plot()
        else:
            {
                "kernel": run_kernel,
                "train": run_train,
                "alpha": run_alpha,
                "alpha-gelu": lambda ladder: run_alpha(ladder, "gelu"),
                "converged": run_converged,
                "floor": run_floor,
            }[part](args.ml_ladder)
        print(f"[{part}] {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
