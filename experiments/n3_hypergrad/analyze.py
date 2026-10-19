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


EXPLODED = 10.0  # |dL/dlog lr| in nats; over this grid the meta-loss moves by a few nats at most


def exploded(g: float) -> bool:
    return not math.isfinite(g) or abs(g) > EXPLODED


def greedy_lr(points: list[dict]) -> dict:
    """The K-step-greedy LR from the meta-loss curve itself (the vertex of a parabola in log lr
    through the grid minimum and its two neighbours), and the hypergradient checked against the
    curve: central differences in log lr at each interior grid point, and the hypergradient's own
    zero crossing (looked for below the first LR where it has exploded)."""
    lrs = np.array([p["lr"] for p in points])
    loss = np.array([p["meta_loss"] for p in points])
    grad = np.array([p["hypergrad"] for p in points])
    x = np.log(lrs)
    i = int(np.argmin(loss))
    vertex = None
    if 0 < i < len(lrs) - 1:
        a2, a1, _ = np.polyfit(x[i - 1 : i + 2], loss[i - 1 : i + 2], 2)
        vertex = float(math.exp(-a1 / (2 * a2))) if a2 > 0 else None
    first_bad = next((j for j in range(len(lrs)) if exploded(float(grad[j]))), len(lrs))
    crossing = None
    for j in range(min(first_bad, len(lrs)) - 1):
        if grad[j] < 0 <= grad[j + 1]:
            crossing = math.exp(x[j] + (x[j + 1] - x[j]) * (-grad[j]) / (grad[j + 1] - grad[j]))
            break
    fd = (loss[2:] - loss[:-2]) / (x[2:] - x[:-2])
    checks = [{"lr": float(lrs[k + 1]), "fd": float(fd[k]), "hypergrad": float(grad[k + 1])} for k in range(len(fd))]
    # the hypergradient "agrees" with the curve at a point if it is within half the central
    # difference, or 0.02 nats near zero (the difference is itself only as accurate as the grid's
    # factor-of-2 spacing allows)
    agrees = [abs(c["hypergrad"] - c["fd"]) <= max(0.5 * abs(c["fd"]), 0.02) for c in checks]
    agrees_up_to = None
    for c, ok in zip(checks, agrees):
        if not ok:
            break
        agrees_up_to = c["lr"]
    return {
        "grid_best_lr": float(lrs[i]),
        "grid_best_at_edge": i in (0, len(lrs) - 1),
        "greedy_lr": vertex if vertex is not None else float(lrs[i]),
        "zero_crossing_lr": crossing,
        "hypergrad_explodes_from_lr": float(lrs[first_bad]) if first_bad < len(lrs) else None,
        "hypergrad_agrees_with_curve_up_to_lr": agrees_up_to,
        "grid_fd_vs_hypergrad": checks,
    }


def summarise() -> dict:
    out: dict = {"landscape": [], "online": [], "reference": []}
    for r in _rows("landscape"):
        g = greedy_lr(r["points"])
        best = g["greedy_lr"]
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
        out["reference"].append({"seed": r["seed"], "schedule": r.get("schedule", "warmup-cosine"), "peak_lr": r.get("peak_lr", PEAK),
                                 "bits_per_byte": r["bits_per_byte"], "steps_per_second": r["steps_per_second"]})
    for r in _rows("online"):
        lrs = np.array([t["lr"] for t in r["trajectory"]])
        gs = [t["hypergrad"] for t in r["trajectory"]]
        out["online"].append({
            "seed": r["seed"], "k": r["k"], "init_lr": r["init_lr"], "meta_lr": r["meta_lr"],
            "clip_hypergrad": r.get("clip_hypergrad"),
            "guard_halvings": int(sum(t["diverged"] for t in r["trajectory"])),
            "segments_exploded": int(sum(exploded(g) for g in gs)),
            "max_abs_finite_hypergrad": float(max(abs(g) for g in gs if math.isfinite(g))),
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
    fig, axes = plt.subplots(1, len(starts) + 2, figsize=(4.4 * (len(starts) + 2), 4), constrained_layout=True)
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
        if start > 0:  # the large-LR blow-ups would flatten the region that matters
            lo = min(p["meta_loss"] - r["meta_loss_before"] for r in land if r["start"] == start and r["seed"] == 0
                     for p in r["points"])
            ax.set_ylim(lo - 0.02, lo + 0.35)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("meta-loss after K steps minus before (nats)")
    ax = axes[-2]
    k_show = 30
    for start, color in zip(starts, ("tab:red", "tab:purple", "tab:brown")):
        for r in (r for r in summary["landscape"] if r["start"] == start and r["k"] == k_show and r["seed"] == 0):
            c = r["grid_fd_vs_hypergrad"]
            ax.plot([q["lr"] for q in c], [abs(q["hypergrad"]) if math.isfinite(q["hypergrad"]) else np.nan for q in c],
                    "o-", color=color, label=f"hypergradient, from step {start}")
            ax.plot([q["lr"] for q in c], [abs(q["fd"]) for q in c], "x--", color=color, alpha=0.6,
                    label=f"finite difference of the curve, from step {start}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("constant LR over the K steps")
    ax.set_ylabel("|d meta-loss / d log lr| (nats)")
    ax.set_title(f"Hypergradient vs the curve (K={k_show}, seed 0)")
    ax.legend(fontsize=6)
    ax = axes[-1]
    t = np.arange(STEPS)
    ax.plot(t, [schedule_lr(s) for s in t], "k-", lw=2, label="tuned schedule (8e-3)")
    for r in _rows("online"):
        steps = [p["step"] for p in r["trajectory"]]
        ax.plot(steps, [p["lr"] for p in r["trajectory"]], lw=1, label=f"from {r['init_lr']:g}, s{r['seed']}" + (f", clip {r['clip_hypergrad']:g}" if r.get("clip_hypergrad") else ""))
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
        print(f"s{r['seed']} start {r['start']:5d} K={r['k']:<4} greedy {r['greedy_lr']:.2e} crossing {r['zero_crossing_lr'] or float('nan'):.2e} "
              f"(grid {r['grid_best_lr']:.1e}{' EDGE' if r['grid_best_at_edge'] else ''}) schedule {r['schedule_lr_mean']:.2e} "
              f"ratio {r['greedy_over_schedule']:.2f}  loss before {r['meta_loss_before']:.4f} sched {r['meta_loss_at_schedule']:.4f} "
              f"best {r['meta_loss_at_grid_best']:.4f}  per-step g<0 {r['per_step_hypergrad_negative_fraction']:.2f}  "
              f"explodes from {r['hypergrad_explodes_from_lr']}  agrees up to {r['hypergrad_agrees_with_curve_up_to_lr']}")
    print("reference", s["reference"], "ml-ladder 8e-3", s["ml_ladder_8e-3"])
    for r in s["online"]:
        print("online", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "lr_at"}, r["lr_at"])
    plot(s)
    (ROOT / "docs" / "data" / "n3-summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print("wrote docs/data/n3-summary.json")


if __name__ == "__main__":
    main()
