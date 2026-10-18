"""N3: summarise the hypergradient runs (``docs/data/n3/n3-*.json``).

- **Landscape:** for each (start, K), the meta-loss after K steps against the LR, the
  K-step-greedy best LR (grid minimum, refined by where the hypergradient changes sign), and
  the tuned schedule's LR at that point. The ratio of the two is the short-horizon bias.
  The hypergradients are also checked against finite differences of the grid's meta-loss.
- **Online:** each run's LR trajectory and final bits/byte, against the tuned reference
  run in the same fp32 setting and ml-ladder's fp16 runs at 8e-3 (``c-gpu-lr2-256.json``).

Writes ``docs/data/n3-summary.json`` and ``docs/figures/n3-hypergrad.png``.

    uv run python experiments/n3_hypergrad/analyze.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "n3"
LADDER = Path(os.environ.get("ML_LADDER", ROOT.parent / "ml-ladder")) / "docs" / "writeups" / "data"
FIG = ROOT / "docs" / "figures"
STEPS, PEAK = 3000, 8e-3


def schedule_lr(step: int) -> float:
    warmup = int(STEPS * 0.05)
    if step < warmup:
        return PEAK * (step + 1) / warmup
    progress = (step - warmup) / (STEPS - warmup)
    return PEAK * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _rows(key: str) -> list[dict]:
    return [r for p in sorted(RUNS.glob("n3-*.json")) for r in json.loads(p.read_text()).get(key, [])]


def greedy_lr(points: list[dict]) -> dict:
    """Grid minimum, and the zero crossing of the hypergradient (in log lr) bracketing it."""
    lrs = np.array([p["lr"] for p in points])
    loss = np.array([p["meta_loss"] for p in points])
    grad = np.array([p["hypergrad"] for p in points])
    i = int(np.argmin(loss))
    crossing = None
    for j in range(len(lrs) - 1):
        if grad[j] < 0 <= grad[j + 1]:  # descending, then ascending: a minimum in between
            x0, x1 = math.log(lrs[j]), math.log(lrs[j + 1])
            crossing = math.exp(x0 + (x1 - x0) * (-grad[j]) / (grad[j + 1] - grad[j]))
            if abs(math.log(crossing) - math.log(lrs[i])) <= abs(x1 - x0) * 1.01:
                break
    # Finite differences of the grid against the measured hypergradients (central, in log lr).
    fd = (loss[2:] - loss[:-2]) / (np.log(lrs[2:]) - np.log(lrs[:-2]))
    return {
        "grid_best_lr": float(lrs[i]),
        "grid_best_at_edge": i in (0, len(lrs) - 1),
        "zero_crossing_lr": crossing,
        "grid_fd_vs_hypergrad": [{"lr": float(lrs[k + 1]), "fd": float(fd[k]), "hypergrad": float(grad[k + 1])}
                                 for k in range(len(fd))],
    }


def summarise() -> dict:
    out: dict = {"landscape": [], "online": [], "reference": []}
    for r in _rows("landscape"):
        g = greedy_lr(r["points"])
        best = g["zero_crossing_lr"] or g["grid_best_lr"]
        per_step = np.asarray(r["schedule_hypergrad_per_step"])
        out["landscape"].append({
            "seed": r["seed"], "start": r["start"], "k": r["k"], **g,
            "schedule_lr_mean": r["schedule_lr_mean"], "greedy_over_schedule": best / r["schedule_lr_mean"],
            "meta_loss_before": r["meta_loss_before"],
            "meta_loss_at_schedule": r["schedule_meta_loss"],
            "meta_loss_at_grid_best": min(p["meta_loss"] for p in r["points"]),
            "per_step_hypergrad_first_half_mean": float(per_step[: len(per_step) // 2].mean()),
            "per_step_hypergrad_second_half_mean": float(per_step[len(per_step) // 2 :].mean()),
            "per_step_hypergrad_negative_fraction": float((per_step < 0).mean()),
        })
    for r in _rows("reference"):
        out["reference"].append({"seed": r["seed"], "bits_per_byte": r["bits_per_byte"], "steps_per_second": r["steps_per_second"]})
    for r in _rows("online"):
        lrs = np.array([t["lr"] for t in r["trajectory"]])
        out["online"].append({
            "seed": r["seed"], "k": r["k"], "init_lr": r["init_lr"], "meta_lr": r["meta_lr"],
            "bits_per_byte": r["bits_per_byte"], "train_seconds": r["train_seconds"],
            "lr_at": {str(s): float(lrs[min(s // r["k"], len(lrs) - 1)]) for s in (0, 150, 500, 1000, 1500, 2000, 2500, 2970)},
            "lr_max": float(lrs.max()), "lr_final": float(lrs[-1]),
        })
    ladder = json.loads((LADDER / "c-gpu-lr2-256.json").read_text())["enwik8"]
    out["ml_ladder_8e-3"] = {r["seed"]: r["bits_per_byte"] for r in ladder if r["model"] == "transformer-4M" and r["peak_lr"] == 8e-3}
    return out


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    land = _rows("landscape")
    starts = sorted({r["start"] for r in land})
    fig, axes = plt.subplots(1, len(starts) + 1, figsize=(4.4 * (len(starts) + 1), 4), constrained_layout=True)
    colors = {10: "tab:blue", 30: "tab:orange", 100: "tab:green"}
    for ax, start in zip(axes, starts):
        for r in sorted((r for r in land if r["start"] == start and r["seed"] == 0), key=lambda r: r["k"]):
            lrs = [p["lr"] for p in r["points"]]
            rel = [p["meta_loss"] - r["meta_loss_before"] for p in r["points"]]
            ax.plot(lrs, rel, "o-", color=colors.get(r["k"], "0.5"), label=f"K={r['k']}")
            ax.axvline(r["schedule_lr_mean"], color=colors.get(r["k"], "0.5"), ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("constant LR over the K steps")
        ax.set_title(f"start: step {start}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("meta-loss after K steps minus before (nats)")
    ax = axes[-1]
    t = np.arange(STEPS)
    ax.plot(t, [schedule_lr(s) for s in t], "k-", lw=2, label="tuned schedule (8e-3)")
    for r in _rows("online"):
        steps = [p["step"] for p in r["trajectory"]]
        ax.plot(steps, [p["lr"] for p in r["trajectory"]], lw=1, label=f"online K={r['k']}, from {r['init_lr']:g}, s{r['seed']}")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("Online hypergradient LR")
    ax.legend(fontsize=7)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "n3-hypergrad.png", dpi=130)


def main() -> None:
    s = summarise()
    for r in s["landscape"]:
        print(f"start {r['start']:5d} K={r['k']:<4} greedy {r['zero_crossing_lr'] or r['grid_best_lr']:.2e} "
              f"(grid {r['grid_best_lr']:.1e}{' EDGE' if r['grid_best_at_edge'] else ''}) schedule {r['schedule_lr_mean']:.2e} "
              f"ratio {r['greedy_over_schedule']:.2f}  loss before {r['meta_loss_before']:.4f} sched {r['meta_loss_at_schedule']:.4f} "
              f"best {r['meta_loss_at_grid_best']:.4f}  per-step g<0 {r['per_step_hypergrad_negative_fraction']:.2f}")
    print("reference", s["reference"], "ml-ladder 8e-3", s["ml_ladder_8e-3"])
    for r in s["online"]:
        print("online", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "lr_at"}, r["lr_at"])
    plot(s)
    (ROOT / "docs" / "data" / "n3-summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print("wrote docs/data/n3-summary.json")


if __name__ == "__main__":
    main()
