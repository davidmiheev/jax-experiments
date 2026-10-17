"""O3: AdamW, Lion and Muon on R2's 4M transformer at T=256, each at its own tuned LR.

Reads ``docs/data/o3/o3-*.json`` (rows written by ``experiments/n1_vmap_sweep/run.py
--optimizer ...``) and ml-ladder's ``c-gpu-lr2-256.json`` (AdamW at this
setting); writes ``docs/data/o3-summary.json`` and the figure.

- **Tuning:** each optimiser's LR curve on seed 0; best LR; a flag if the best is at
  an edge of the grid.
- **Comparison:** seeds 0-2 at each optimiser's best LR, median (min-max) bits/byte.
- **Steps to a target:** the first logged step at which the smoothed training loss
  (mean of the last 8 logs, 200 steps) reaches AdamW's final smoothed loss at its best
  LR (median over seeds).

    uv run python experiments/o3_optimizers/analyze.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "o3"
LADDER = Path(os.environ.get("ML_LADDER", ROOT.parent / "ml-ladder")) / "docs" / "writeups" / "data"
FIG = ROOT / "docs" / "figures"
OPTIMIZERS = ("adamw", "lion", "muon")
WINDOW = 8


def _spread(values) -> dict:
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def rows() -> list[dict]:
    return [r for p in sorted(RUNS.glob("o3-*.json")) for r in json.loads(p.read_text()).get("enwik8", [])]


def smoothed(history) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array([s for s, _ in history], float)
    loss = np.array([l for _, l in history], float)
    kernel = np.ones(WINDOW) / WINDOW
    return steps[WINDOW - 1 :], np.convolve(loss, kernel, mode="valid")


def summarise() -> dict:
    data = rows()
    out = {"optimizers": {}}
    for opt in OPTIMIZERS:
        mine = [r for r in data if r.get("optimizer") == opt]
        if not mine:
            continue
        seed0 = sorted((r for r in mine if r["seed"] == 0), key=lambda r: r["peak_lr"])
        curve = {f"{r['peak_lr']:g}": r["bits_per_byte"] for r in seed0}
        best = min(seed0, key=lambda r: r["bits_per_byte"])["peak_lr"]
        lrs = [r["peak_lr"] for r in seed0]
        at_best = sorted((r for r in mine if r["peak_lr"] == best), key=lambda r: r["seed"])
        out["optimizers"][opt] = {
            "weight_decay": seed0[0].get("weight_decay"),
            "adam_peak": seed0[0].get("adam_peak"),
            "seed0_curve": curve,
            "best_lr": best,
            "best_at_grid_edge": best in (min(lrs), max(lrs)),
            "best_lr_seeds": [r["seed"] for r in at_best],
            "best_lr_bits": _spread([r["bits_per_byte"] for r in at_best]),
            "best_lr_bits_by_seed": {r["seed"]: r["bits_per_byte"] for r in at_best},
            "skipped_steps": {f"{r['peak_lr']:g}/s{r['seed']}": r["skipped_steps"] for r in mine if r["skipped_steps"]},
            "group_steady_tokens_per_second": _spread([r["group_steady_tokens_per_second"] for r in mine]),
        }
    if "adamw" in out["optimizers"]:
        adam = out["optimizers"]["adamw"]
        finals = [smoothed(r["train_history"])[1][-1] for r in data if r.get("optimizer") == "adamw" and r["peak_lr"] == adam["best_lr"]]
        target = float(np.median(finals))
        out["target_train_loss"] = target
        for opt, entry in out["optimizers"].items():
            reached = {}
            for r in data:
                if r.get("optimizer") == opt and r["peak_lr"] == entry["best_lr"]:
                    steps, loss = smoothed(r["train_history"])
                    hit = np.nonzero(loss <= target)[0]
                    reached[r["seed"]] = int(steps[hit[0]]) if hit.size else None
            entry["steps_to_target"] = reached
        ref = {r["peak_lr"]: r for r in json.loads((LADDER / "c-gpu-lr2-256.json").read_text())["enwik8"]
               if r["model"] == "transformer-4M" and r["seed"] == 0}
        ours = {r["peak_lr"]: r for r in data if r.get("optimizer") == "adamw" and r["seed"] == 0}
        out["adamw_vs_ml_ladder_seed0"] = {
            f"{lr:g}": {"ours": ours[lr]["bits_per_byte"], "ml_ladder": ref[lr]["bits_per_byte"]}
            for lr in sorted(set(ref) & set(ours))
        }
    return out


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = rows()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = {"adamw": "tab:blue", "lion": "tab:orange", "muon": "tab:green"}
    for opt, entry in summary["optimizers"].items():
        lrs = [float(k) for k in entry["seed0_curve"]]
        axes[0].plot(lrs, list(entry["seed0_curve"].values()), "o-", color=colors[opt], label=opt)
        b = entry["best_lr_bits"]
        axes[0].errorbar([entry["best_lr"]], [b["median"]], yerr=[[b["median"] - b["min"]], [b["max"] - b["median"]]],
                         fmt="s", color=colors[opt], ms=8, capsize=4)
        for r in data:
            if r.get("optimizer") == opt and r["peak_lr"] == entry["best_lr"]:
                steps, loss = smoothed(r["train_history"])
                axes[1].plot(steps, loss, color=colors[opt], lw=1, alpha=0.8, label=opt if r["seed"] == 0 else None)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("peak learning rate")
    axes[0].set_ylabel("validation bits/byte")
    axes[0].set_title("LR sweep, seed 0 (squares: seeds 0-2 at the best LR)")
    axes[0].legend(fontsize=8)
    if "target_train_loss" in summary:
        axes[1].axhline(summary["target_train_loss"], color="0.4", ls="--", lw=0.8)
    axes[1].set_ylim(top=2.0)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(f"training loss (nats, mean of {WINDOW} logs)")
    axes[1].set_title("Each optimiser at its best LR, seeds 0-2")
    axes[1].legend(fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "o3-optimizers.png", dpi=130)


def main() -> None:
    summary = summarise()
    for opt, e in summary["optimizers"].items():
        print(opt, "curve", {k: round(v, 4) for k, v in e["seed0_curve"].items()}, "best", e["best_lr"],
              "EDGE" if e["best_at_grid_edge"] else "", "seeds", e["best_lr_bits"], "steps->target", e.get("steps_to_target"))
    if "adamw_vs_ml_ladder_seed0" in summary:
        print("adamw vs ml-ladder (seed 0):", summary["adamw_vs_ml_ladder_seed0"], "target", summary["target_train_loss"])
    plot(summary)
    (ROOT / "docs" / "data" / "o3-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote docs/data/o3-summary.json")


if __name__ == "__main__":
    main()
