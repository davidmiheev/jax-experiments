"""N4: linear recurrences (LRU, Mamba-2) against ml-ladder's GRU and 4M transformer.

Reads ``docs/data/n4/`` (sweeps from ``experiments/n1_vmap_sweep/run.py --model lru|mamba2``,
copy runs from ``experiments/r2_enwik8/run.py --studies copy``, throughput and memory from
``experiments/n4_scans/run.py`` and ``n1 --studies memory``) and ml-ladder's JSON for the
references (3,000 steps, 16,384 tokens per step, as here):

- GRU at its tuned 2e-3 (``c-gpu.json``; 1e-3 / 4e-3 / 8e-3 in ``c-gpu-lr3-gru-*.json`` are worse);
- 4M transformer at its tuned LR: 2e-3 at T=1024 (``c-gpu-lr.json``), 8e-3 at T=256
  (``c-gpu-lr2-256.json``);
- copy task: ml-ladder's LSTM and transformer (``c-gpu.json``) and R3's JAX runs.

Writes ``docs/data/n4-summary.json`` and ``docs/figures/n4-*.png``.

    uv run python experiments/n4_scans/analyze.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "n4"
R2 = ROOT / "docs" / "data" / "r2"
LADDER = Path(os.environ.get("ML_LADDER", ROOT.parent / "ml-ladder")) / "docs" / "writeups" / "data"
FIG = ROOT / "docs" / "figures"
OURS = ("lru", "mamba2")
CONTEXTS = (256, 1024)
# Position bins for the loss-by-position table: [lo, hi) in 0-based positions.
BINS = ((0, 8), (8, 64), (64, 256), (256, 1024))


def _spread(values) -> dict:
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _load(path: Path, key: str) -> list[dict]:
    return json.loads(path.read_text()).get(key, []) if path.exists() else []


def our_rows(repeats: bool = False) -> list[dict]:
    """enwik8 rows; ``repeats=True`` returns only the deliberate repeats of a run (``*-repeat.json``),
    which are kept out of the seed statistics."""
    return [r for p in sorted(RUNS.glob("n4-*.json")) if ("-repeat" in p.name) == repeats for r in _load(p, "enwik8")]


def reference_rows() -> dict[tuple[str, int], list[dict]]:
    """ml-ladder's runs at each baseline's tuned LR, seeds 0-2."""
    base = _load(LADDER / "c-gpu.json", "enwik8")
    out = {("gru", t): [r for r in base if r["model"] == "gru" and r["context"] == t] for t in CONTEXTS}
    lr1024 = _load(LADDER / "c-gpu-lr.json", "enwik8")
    lr256 = _load(LADDER / "c-gpu-lr2-256.json", "enwik8")
    out[("transformer-4M", 1024)] = [r for r in lr1024 if r["peak_lr"] == 2e-3]
    out[("transformer-4M", 256)] = [r for r in lr256 if r["peak_lr"] == 8e-3]
    return out


def by_position(runs: list[dict], context: int) -> dict:
    """Median over seeds of the mean bits in each position bin (bins inside the context)."""
    table = {}
    for lo, hi in BINS:
        if lo >= context:
            continue
        hi = min(hi, context)
        table[f"{lo}-{hi}"] = float(np.median([np.mean(r["bits_by_position"][lo:hi]) for r in runs]))
    return table


def context_gain(runs: list[dict], context: int) -> dict:
    """Bits at positions [T/8, T/4) minus bits in the last quarter: what the second half of
    the context buys. A model that stops using context after ~T/8 bytes scores ~0."""
    gains = [np.mean(r["bits_by_position"][context // 8 : context // 4]) - np.mean(r["bits_by_position"][3 * context // 4 :])
             for r in runs]
    return _spread(gains)


def gap_to_gru(runs: list[dict], gru: list[dict], context: int) -> dict:
    """Median curve minus the GRU's median curve, binned. Both are evaluated on the same validation
    windows, so the text's own difficulty at each position (which moves both curves by ~0.03
    bits from bin to bin) cancels. ``trend`` is the gap in the last quarter minus the gap at
    positions [T/8, T/4): negative means the model gains on the GRU as the context grows."""
    diff = np.median([r["bits_by_position"] for r in runs], 0) - np.median([g["bits_by_position"] for g in gru], 0)
    bins = {}
    for lo, hi in BINS:
        if lo < context:
            bins[f"{lo}-{min(hi, context)}"] = float(diff[lo : min(hi, context)].mean())
    return {"bins": bins, "trend": float(diff[3 * context // 4 :].mean() - diff[context // 8 : context // 4].mean())}


def summarise() -> dict:
    data = our_rows()
    out: dict = {"enwik8": {}, "references": {}}
    for model in OURS:
        for t in CONTEXTS:
            mine = [r for r in data if r["model"] == model and r["context"] == t]
            if not mine:
                continue
            seed0 = sorted((r for r in mine if r["seed"] == 0), key=lambda r: r["peak_lr"])
            best = min(seed0, key=lambda r: r["bits_per_byte"])["peak_lr"]
            lrs = [r["peak_lr"] for r in seed0]
            at_best = sorted((r for r in mine if r["peak_lr"] == best), key=lambda r: r["seed"])
            out["enwik8"][f"{model}-{t}"] = {
                "model": model,
                "context": t,
                "parameters": mine[0]["parameters"],
                "seed0_curve": {f"{r['peak_lr']:g}": r["bits_per_byte"] for r in seed0},
                "best_lr": best,
                "best_at_grid_edge": best in (min(lrs), max(lrs)),
                "best_lr_bits": _spread([r["bits_per_byte"] for r in at_best]),
                "best_lr_bits_by_seed": {r["seed"]: r["bits_per_byte"] for r in at_best},
                "bits_by_position_bins": by_position(at_best, t),
                "context_gain": context_gain(at_best, t),
                "gap_to_gru": gap_to_gru(at_best, reference_rows()[("gru", t)], t),
                "skipped_steps": {f"{r['peak_lr']:g}/s{r['seed']}": r["skipped_steps"] for r in mine if r.get("skipped_steps")},
                "group_steady_tokens_per_second": _spread([r["group_steady_tokens_per_second"] for r in mine]),
            }
    for (model, t), runs in reference_rows().items():
        if not runs:
            continue
        out["references"][f"{model}-{t}"] = {
            "model": model,
            "context": t,
            "source": "ml-ladder (PyTorch, T4)",
            "peak_lr": runs[0].get("peak_lr", 2e-3),
            "parameters": runs[0]["parameters"],
            "bits": _spread([r["bits_per_byte"] for r in runs]),
            "bits_by_position_bins": by_position(runs, t),
            "context_gain": context_gain(runs, t),
            "gap_to_gru": gap_to_gru(runs, reference_rows()[("gru", t)], t),
            "tokens_per_second_ml_ladder": _spread([r["tokens_per_second"] for r in runs]),
        }
    out["repeats"] = {
        f"{r['model']}-{r['context']}/s{r['seed']}/lr{r['peak_lr']:g}": {
            "repeat": r["bits_per_byte"],
            "original": next((o["bits_per_byte"] for o in data if (o["model"], o["context"], o["seed"], o["peak_lr"])
                              == (r["model"], r["context"], r["seed"], r["peak_lr"])), None),
        }
        for r in our_rows(repeats=True)
    }
    out["copy"] = copy_summary()
    out["throughput"] = throughput_summary()
    return out


def copy_summary() -> dict:
    """Second-half accuracy by (source, model, n, LR). Runs without a recorded LR used their
    model's default, 2e-3 (R3's and n4-copy's protocol)."""
    ours = [r for p in sorted(RUNS.glob("n4-copy*.json")) for r in _load(p, "copy")]
    r3 = [r for p in (R2 / "r3-lstm.json", R2 / "r3-transformer-fused-colab.json") for r in _load(p, "copy")]
    ladder = _load(LADDER / "c-gpu.json", "copy")
    out = {}
    for source, rows in (("jax", ours + r3), ("ml-ladder", ladder)):
        groups: dict = {}
        for r in rows:
            key = f"{source}/{r['model']}/{r['n']}" + (f"/lr{r['peak_lr']:g}" if r.get("peak_lr") not in (None, 2e-3) else "")
            groups.setdefault(key, []).append(r)
        for key in sorted(groups):
            runs = sorted(groups[key], key=lambda r: r["seed"])
            out[key] = {
                "by_seed": {r["seed"]: r["second_half_accuracy"] for r in runs},
                "second_half_accuracy": _spread([r["second_half_accuracy"] for r in runs]),
            }
    return out


def throughput_summary() -> dict:
    rows = [r for p in sorted(RUNS.glob("n4-throughput*.json")) for r in _load(p, "rows")]
    out = {}
    for r in rows:
        key = f"{r['model']}/{r['scan']}/{r['remat']}"
        out.setdefault(key, {})[str(r["context"])] = (
            {"tokens_per_second": r["tokens_per_second"], "temp_gib": r["temp_gib"]}
            if "tokens_per_second" in r
            else {"error": r["error"][:60], "temp_gib": r.get("temp_gib")}
        )
    memory = {}
    for p in sorted(RUNS.glob("n4-memory-*.json")):
        for r in _load(p, "memory"):
            memory.setdefault(f"{r['model']}/{r['remat']}", {})[str(r["k"])] = r["temp_bytes"] / 2**30
    return {"by_arm": out, "vmapped_temp_gib_T1024": memory}


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = our_rows()
    refs = reference_rows()
    colors = {"gru": "tab:red", "transformer-4M": "tab:blue", "lru": "tab:green", "mamba2": "tab:purple"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    # Loss by position at T=1024, each model at its tuned LR, median over seeds (smoothed for display).
    kernel = np.ones(16) / 16
    for model in ("gru", "transformer-4M", *OURS):
        if model in OURS:
            entry = summary["enwik8"].get(f"{model}-1024")
            if not entry:
                continue
            runs = [r for r in data if r["model"] == model and r["context"] == 1024 and r["peak_lr"] == entry["best_lr"]]
        else:
            runs = refs[(model, 1024)]
        curve = np.median([r["bits_by_position"] for r in runs], axis=0)
        smooth = np.convolve(curve, kernel, mode="valid")
        axes[0].plot(np.arange(len(smooth)) + 8, smooth, color=colors[model], label=model)
    axes[0].set_xscale("log")
    axes[0].set_ylim(1.4, 2.6)
    axes[0].set_xlabel("position in the window")
    axes[0].set_ylabel("validation bits/byte (16-position mean)")
    axes[0].set_title("Loss by position, T=1024, tuned LR, median of seeds")
    axes[0].legend(fontsize=8)
    # LR curves (seed 0).
    for key, entry in summary["enwik8"].items():
        lrs = [float(k) for k in entry["seed0_curve"]]
        axes[1].plot(lrs, list(entry["seed0_curve"].values()), "o-" if entry["context"] == 1024 else "s--",
                     color=colors[entry["model"]], label=key)
    for key, entry in summary["references"].items():
        axes[1].plot([entry["peak_lr"]], [entry["bits"]["median"]], "*" if entry["context"] == 1024 else "x",
                     color=colors[entry["model"]], ms=10, label=f"{key} (ml-ladder)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("peak learning rate")
    axes[1].set_ylabel("validation bits/byte")
    axes[1].set_title("LR sweep, seed 0")
    axes[1].legend(fontsize=7)
    # Throughput against T.
    styles = {"associative": "-", "chunked": "-", "sequential": ":", "-": "-"}
    for key, by_t in summary["throughput"]["by_arm"].items():
        model, scan, _ = key.split("/")
        ts = sorted((int(t) for t, v in by_t.items() if "tokens_per_second" in v))
        if ts:
            axes[2].plot(ts, [by_t[str(t)]["tokens_per_second"] / 1e3 for t in ts], "o" + styles[scan],
                         color=colors.get(model, "0.5"), label=model if scan == "-" else f"{model} {scan}")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("sequence length T (16,384 tokens per step)")
    axes[2].set_ylabel("training tokens/s (thousands), one T4, fp16")
    axes[2].set_title("Throughput against T")
    axes[2].legend(fontsize=7)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "n4-scans.png", dpi=130)


def main() -> None:
    summary = summarise()
    for key, e in {**summary["references"], **summary["enwik8"]}.items():
        print(f"{key:<22}", "curve" if "seed0_curve" in e else "ref  ",
              {k: round(v, 4) for k, v in e.get("seed0_curve", {}).items()} or e["peak_lr"],
              "EDGE" if e.get("best_at_grid_edge") else "",
              "bits", {k: round(v, 4) for k, v in e.get("best_lr_bits", e.get("bits")).items()},
              "bins", {k: round(v, 3) for k, v in e["bits_by_position_bins"].items()},
              "gain", round(e["context_gain"]["median"], 3),
              "vs GRU", {k: round(v, 3) for k, v in e["gap_to_gru"]["bins"].items()}, "trend", round(e["gap_to_gru"]["trend"], 3))
    print("repeats:", summary["repeats"])
    for key, e in summary["copy"].items():
        print(f"copy {key:<28}", {k: round(v, 3) for k, v in e["by_seed"].items()})
    plot(summary)
    (ROOT / "docs" / "data" / "n4-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote docs/data/n4-summary.json")


if __name__ == "__main__":
    main()
