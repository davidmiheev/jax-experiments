"""N8: summarise the diffusion runs (``docs/data/n8/n8-s*.json``).

- **E3 reproduction** (T=400, eps, DDPM): median (min-max) fd over seeds per schedule and step
  count, against ml-ladder's ``e-metrics.json`` (rung E3, the same statistic).
- **T=1000 arms:** fd by schedule x parameterisation x sampler x steps (median and range over
  seeds); the "knee" (fewest steps within 10% of the arm's own full-step fd, E3's definition).
- **Cost:** sampling wall time against step count (one ``lax.scan`` per run), as ms per step
  for 500 images, and compile time.

Writes ``docs/data/n8-summary.json`` and ``docs/figures/n8-diffusion.png``.

    uv run python experiments/n8_diffusion/analyze.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "n8"
LADDER = Path(os.environ.get("ML_LADDER", ROOT.parent / "ml-ladder")) / "docs" / "writeups" / "data"
FIG = ROOT / "docs" / "figures"


def _spread(values) -> dict:
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def rows() -> list[dict]:
    return [r for p in sorted(RUNS.glob("n8-s*.json")) for r in json.loads(p.read_text())["rows"]]


def floors() -> dict:
    return {json.loads(p.read_text())["settings"]["seed"]: json.loads(p.read_text())["real_vs_real_fd"]
            for p in sorted(RUNS.glob("n8-s*.json"))}


def summarise() -> dict:
    data = rows()
    out: dict = {"real_vs_real_fd": _spread(list(floors().values())) if floors() else None, "e3": {}, "t1000": {}, "cost": {}}
    e3 = json.loads((LADDER / "e-metrics.json").read_text())["rungs"]["E3"]["metrics"]
    for r in data:
        key = f"T{r['T']}/{r['schedule']}/{r['parameterisation']}/{r['sampler']}" + ("/torch" if r.get("framework") == "torch" else "")
        bucket = out["e3" if r["T"] == 400 else "t1000"].setdefault(key, {})
        bucket.setdefault(str(r["steps"]), []).append(r["fd"])
    for section in ("e3", "t1000"):
        for key, by_steps in out[section].items():
            spread = {s: _spread(v) for s, v in by_steps.items()}
            full = spread[max(spread, key=int)]["median"]
            knee = min((int(s) for s, v in spread.items() if v["median"] <= 1.1 * full), default=None)
            out[section][key] = {"fd": spread, "knee_steps": knee}
            if section == "e3" and not key.endswith("/torch"):
                schedule = key.split("/")[1]
                out[section][key]["ml_ladder"] = {s: e3.get(f"{schedule}/steps/{s}/fd") for s in by_steps}
    for r in data:
        key = f"T{r['T']}/{r['sampler']}" + ("/torch" if r.get("framework") == "torch" else "")
        out["cost"].setdefault(key, {}).setdefault(str(r["steps"]), {"seconds": [], "compile": []})
        out["cost"][key][str(r["steps"])]["seconds"].append(r["sample_seconds"])
        if "compile_seconds" in r:  # the PyTorch control is not compiled
            out["cost"][key][str(r["steps"])]["compile"].append(r["compile_seconds"])
    for key, by_steps in out["cost"].items():
        for s, v in by_steps.items():
            by_steps[s] = {"seconds": _spread(v["seconds"]), "ms_per_step": 1e3 * float(np.median(v["seconds"])) / int(s),
                           "compile_seconds": _spread(v["compile"]) if v["compile"] else None}
    out["train_mse"] = {}
    for r in data:
        key = f"T{r['T']}/{r['schedule']}/{r['parameterisation']}" + ("/torch" if r.get("framework") == "torch" else "")
        out["train_mse"].setdefault(key, {})[r["seed"]] = r["train_mse"]
    return out


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, grid = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = [grid[0, 0], grid[0, 1], grid[1, 0]]
    legend_ax = grid[1, 1]
    # E3 reproduction.
    ax = axes[0]
    for key, entry in summary["e3"].items():
        schedule = key.split("/")[1]
        steps = sorted(entry["fd"], key=int)
        color = "tab:blue" if schedule == "linear" else "tab:orange"
        med = [entry["fd"][s]["median"] for s in steps]
        if key.endswith("/torch"):
            ax.plot([int(s) for s in steps], med, "s:", color=color, label=f"E3's PyTorch, same scorer, {schedule}")
            continue
        ax.plot([int(s) for s in steps], med, "o-", color=color, label=f"JAX {schedule}")
        ref = [entry["ml_ladder"][s]["median"] if entry["ml_ladder"].get(s) else np.nan for s in steps]
        ax.plot([int(s) for s in steps], ref, "x--", color=color, label=f"E3 as recorded {schedule}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sampling steps (DDPM, T=400)")
    ax.set_ylabel("fd, median of seeds 0-2")
    ax.set_title("E3 reproduction")
    ax.legend(fontsize=8)
    # T=1000: one panel per schedule.
    colors = {"eps": "tab:blue", "x0": "tab:green", "v": "tab:purple"}
    for ax, schedule in zip(axes[1:], ("linear", "cosine")):
        for key, entry in summary["t1000"].items():
            _, sched, p, sampler = key.split("/")
            if sched != schedule:
                continue
            steps = sorted(entry["fd"], key=int)
            style = {"ddpm": "s--", "ddim": "o-", "ddpm-clip": "s:", "ddim-clip": "o:"}[sampler]
            ax.plot([int(s) for s in steps], [entry["fd"][s]["median"] for s in steps],
                    style, color=colors[p], label=f"{p} {sampler}")
        if summary["real_vs_real_fd"]:
            ax.axhline(summary["real_vs_real_fd"]["median"], color="0.5", lw=0.8, ls=":", label="real vs real")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sampling steps (T=1000)")
        ax.set_title(f"T=1000, {schedule} schedule")
        ax.set_ylabel("fd, median of seeds 0-2")
    # one legend for both T=1000 panels
    handles, labels = axes[1].get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=2, fontsize=10,
                     title="T=1000 panels: parameterisation, sampler")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "n8-diffusion.png", dpi=130)


def main() -> None:
    s = summarise()
    print("real vs real fd:", s["real_vs_real_fd"])
    for section in ("e3", "t1000"):
        for key, entry in s[section].items():
            line = "  ".join(f"{st}:{v['median']:.2f}" for st, v in sorted(entry["fd"].items(), key=lambda x: int(x[0])))
            extra = ""
            if "ml_ladder" in entry:
                extra = "  E3: " + "  ".join(f"{st}:{v['median']:.2f}" for st, v in entry["ml_ladder"].items() if v)
            print(f"{key:<26} knee {entry['knee_steps']}  {line}{extra}")
    for key, by_steps in s["cost"].items():
        print(key, {st: round(v["ms_per_step"], 2) for st, v in sorted(by_steps.items(), key=lambda x: int(x[0]))})
    plot(s)
    (ROOT / "docs" / "data" / "n8-summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print("wrote docs/data/n8-summary.json")


if __name__ == "__main__":
    main()
