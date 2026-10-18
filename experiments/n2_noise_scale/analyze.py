"""N2: does the measured noise scale predict the critical batch size?

Reads ``docs/data/n2/n2-*.json``; writes ``docs/data/n2-summary.json`` and the figure.

- **Steps to a target.** For each batch size B and target training loss L, S(B, L) is the
  first logged step at which the smoothed loss (mean of 8 logs, 200 steps) is <= L, at the
  best of that batch size's learning rates (the fewest steps), with a flag if the best LR is
  an edge of the grid.
- **Fit.** McCandlish et al.: S(B) = S_min (1 + B_crit / B), linear in 1/B, fitted by least
  squares over the batch sizes that reached L. B_crit is where doubling the batch stops
  halving the steps (at B = B_crit it costs 2 S_min steps and 2 E_min examples).
- **Prediction.** B_simple from the noise run, interpolated to the step at which that run's
  smoothed loss crosses L.

    uv run python experiments/n2_noise_scale/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "n2"
FIG = ROOT / "docs" / "figures"
WINDOW = 8
TARGETS = [1.6, 1.5, 1.4, 1.3, 1.25, 1.2]


def smoothed(history):
    steps = np.array([s for s, _ in history], float)
    loss = np.array([l for _, l in history], float)
    return steps[WINDOW - 1 :], np.convolve(loss, np.ones(WINDOW) / WINDOW, mode="valid")


def first_reach(history, level):
    steps, loss = smoothed(history)
    hit = np.nonzero(loss <= level)[0]
    return float(steps[hit[0]]) if hit.size else None


def load(key):
    return [r for p in sorted(RUNS.glob("n2-*.json")) for r in json.loads(p.read_text()).get(key, [])]


def summarise() -> dict:
    runs, noise = load("batch"), load("noise")
    batches = sorted({r["batch"] for r in runs})
    out = {"targets": {}, "batches": batches}
    out["noise_runs"] = [
        {
            "seed": run["seed"],
            "noise_batches": run["noise_batches"],
            "steps": [m["step"] for m in run["measurements"]],
            "b_simple": [m["b_simple"] for m in run["measurements"]],
            "trace_sigma": [m["trace_sigma"] for m in run["measurements"]],
            "grad_norm_sq": [m["grad_norm_sq"] for m in run["measurements"]],
            "measure_seconds": float(np.median([m["seconds"] for m in run["measurements"]])),
        }
        for run in noise
    ]
    for level in TARGETS:
        per_b = {}
        for b in batches:
            cands = [(first_reach(r["train_history"], level), r["peak_lr"]) for r in runs if r["batch"] == b]
            reached = [(s, lr) for s, lr in cands if s is not None]
            lrs = sorted(lr for _, lr in cands)
            if reached:
                s, lr = min(reached)
                per_b[b] = {"steps": s, "lr": lr, "lr_at_grid_edge": lr in (lrs[0], lrs[-1]),
                            "examples": s * b, "all_lrs": {f"{l:g}": st for st, l in cands}}
        entry = {"per_batch": per_b}
        if len(per_b) >= 2:
            bs = np.array(sorted(per_b), float)
            s = np.array([per_b[int(b)]["steps"] for b in bs])
            a, slope = np.linalg.lstsq(np.stack([np.ones_like(bs), 1 / bs], 1), s, rcond=None)[0]
            entry["fit"] = {"s_min": float(a), "b_crit": float(slope / a) if a > 0 else None,
                            "residual_rel": float(np.max(np.abs(a + slope / bs - s) / s))}
        at_level = []
        for run, nr in zip(noise, out["noise_runs"], strict=True):
            steps, loss = smoothed(run["train_history"])
            hit = np.nonzero(loss <= level)[0]
            if hit.size:
                at = float(steps[hit[0]])
                at_level.append({"seed": run["seed"], "noise_batches": run["noise_batches"], "step": at,
                                 "b_simple": float(np.interp(at, nr["steps"], nr["b_simple"]))})
        if at_level:
            entry["b_simple_at_level"] = at_level
            precise = [a["b_simple"] for a in at_level if a["noise_batches"] >= 32] or [a["b_simple"] for a in at_level]
            entry["b_simple_median"] = float(np.median(precise))
        out["targets"][f"{level:g}"] = entry
    return out


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for nr in summary["noise_runs"]:
        style = "o-" if nr["noise_batches"] >= 32 else "x:"
        axes[0].plot(nr["steps"], nr["b_simple"], style, ms=4, lw=1,
                     label=f"seed {nr['seed']}, {nr['noise_batches']} batches per point")
    axes[0].set_xlabel("step (B=64, lr 8e-3, warmup-constant)")
    axes[0].set_ylabel("B_simple (sequences of 256 bytes)")
    axes[0].set_yscale("log")
    axes[0].set_title("Noise scale through training")
    axes[0].legend(fontsize=7)
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    levels = [k for k, v in summary["targets"].items() if "fit" in v]
    for i, key in enumerate(levels):
        v = summary["targets"][key]
        color = cmap(i / max(1, len(levels) - 1))
        bs = sorted(int(b) for b in v["per_batch"])
        ax.plot(bs, [v["per_batch"][b]["steps"] for b in bs], "o", color=color, label=f"loss {key}")
        grid = np.geomspace(min(bs) / 2, max(bs) * 2, 50)
        ax.plot(grid, v["fit"]["s_min"] * (1 + v["fit"]["b_crit"] / grid), "-", color=color, lw=0.8)
        if v.get("b_simple_median"):
            ax.axvline(v["b_simple_median"], color=color, ls=":", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B (sequences)")
    ax.set_ylabel("steps to reach the loss")
    ax.set_title("Steps vs batch: fit S_min(1 + B_crit/B); dotted: measured B_simple")
    ax.legend(fontsize=7)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "n2-noise-scale.png", dpi=130)


def main() -> None:
    summary = summarise()
    for nr in summary["noise_runs"]:
        print(f"B_simple, seed {nr['seed']}, {nr['noise_batches']} batches:",
              [(s, round(b, 1)) for s, b in zip(nr["steps"], nr["b_simple"], strict=True)])
    for key, v in summary["targets"].items():
        line = f"loss {key}: " + ", ".join(
            f"B={b}: {e['steps']:.0f} steps (lr {e['lr']:g}{' EDGE' if e['lr_at_grid_edge'] else ''})" for b, e in v["per_batch"].items())
        if "fit" in v:
            line += f" | S_min {v['fit']['s_min']:.0f}, B_crit {v['fit']['b_crit']:.1f}, max resid {v['fit']['residual_rel']:.2f}"
        if "b_simple_median" in v:
            line += f" | B_simple {v['b_simple_median']:.1f} " + str([(a["seed"], a["noise_batches"], round(a["b_simple"], 1)) for a in v["b_simple_at_level"]])
        print(line)
    plot(summary)
    (ROOT / "docs" / "data" / "n2-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote docs/data/n2-summary.json")


if __name__ == "__main__":
    main()
