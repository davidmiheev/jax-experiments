"""R2/R3 against ml-ladder's recorded runs: bits/byte, per-position loss, throughput, copy accuracy.

Reads this repo's Kaggle outputs from ``docs/data/r2/`` (``r2-*-s*.json``, ``r3-*.json``)
and ml-ladder's raw JSON (``$ML_LADDER/docs/writeups/data``), writes
``docs/data/r2-summary.json`` and the figures.

    uv run python experiments/r2_enwik8/analyze.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "r2"
LADDER = Path(os.environ.get("ML_LADDER", ROOT.parent / "ml-ladder")) / "docs" / "writeups" / "data"
FIG = ROOT / "docs" / "figures"

# arm: (model, peak_lr, ml-ladder file holding the 12,000-step T=1024 reference rows)
ARMS = {
    "transformer-25M-rope": ("transformer-25M-rope", 6e-4, "c-gpu-rope-1024.json"),
    "gru": ("gru", 2e-3, "c-gpu-long.json"),
    "transformer-4M (lr 2e-3)": ("transformer-4M", 2e-3, "c-gpu-4m-long.json"),
}


def _spread(values) -> dict:
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _reference_rows(model, lr, fname):
    rows = json.loads((LADDER / fname).read_text())["enwik8"]
    return sorted(
        (
            r
            for r in rows
            if r["model"] == model
            and r["context"] == 1024
            and r["steps"] == 12000
            and abs(r.get("peak_lr", lr) - lr) < 1e-12
        ),
        key=lambda r: r["seed"],
    )


def _our_rows(model, lr):
    rows = []
    for path in sorted(RUNS.glob("r2-*-s*.json")):
        for r in json.loads(path.read_text()).get("enwik8", []):
            if r["model"] == model and abs(r["peak_lr"] - lr) < 1e-12 and r["steps"] == 12000:
                rows.append(r)
    return sorted(rows, key=lambda r: r["seed"])


def enwik8_summary() -> dict:
    out = {}
    for arm, (model, lr, fname) in ARMS.items():
        ref, ours = _reference_rows(model, lr, fname), _our_rows(model, lr)
        entry = {
            "reference_file": fname,
            "reference_seeds": [r["seed"] for r in ref],
            "seeds": [r["seed"] for r in ours],
            "reference_bits_per_byte": _spread([r["bits_per_byte"] for r in ref]),
            "reference_tokens_per_second": _spread([r["tokens_per_second"] for r in ref]),
        }
        if ours:
            entry["bits_per_byte"] = _spread([r["bits_per_byte"] for r in ours])
            entry["bits_last_quarter"] = _spread([r["bits_last_quarter"] for r in ours])
            entry["reference_bits_last_quarter"] = _spread([r["bits_last_quarter"] for r in ref])
            entry["delta_median_bits"] = entry["bits_per_byte"]["median"] - entry["reference_bits_per_byte"]["median"]
            entry["steady_tokens_per_second"] = _spread([r["steady_tokens_per_second"] for r in ours])
            entry["compile_seconds"] = _spread([r["compile_seconds"] for r in ours])
            entry["train_seconds"] = _spread([r["train_seconds"] for r in ours])
            entry["peak_memory_mb"] = _spread([r["peak_memory_mb"] for r in ours])
            entry["reference_peak_memory_mb"] = _spread([r["peak_memory_mb"] for r in ref if "peak_memory_mb" in r] or [np.nan])
            entry["skipped_steps"] = [r["skipped_steps"] for r in ours]
            entry["final_loss_scale"] = [r["final_loss_scale"] for r in ours]
            ours_pos = np.median([r["bits_by_position"] for r in ours], axis=0)
            ref_pos = np.median([r["bits_by_position"] for r in ref], axis=0)
            bands = {"0": slice(0, 1), "1-15": slice(1, 16), "16-255": slice(16, 256), "256-1023": slice(256, 1024)}
            entry["by_position_median_bits"] = {
                band: {"ours": float(ours_pos[s].mean()), "reference": float(ref_pos[s].mean())} for band, s in bands.items()
            }
            entry["by_position_max_abs_gap"] = float(np.abs(ours_pos - ref_pos).max())
            entry["_curves"] = (ours_pos.tolist(), ref_pos.tolist())
        out[arm] = entry
    return out


def copy_summary() -> dict:
    ref = json.loads((LADDER / "c-gpu.json").read_text())["copy"]
    ours = [r for p in sorted(RUNS.glob("r3-*.json")) for r in json.loads(p.read_text()).get("copy", [])]
    out = {}
    for model in ("transformer-4M", "lstm"):
        for n in (32, 128, 512):
            key = f"{model} n={n}"
            entry = {"reference": _spread([r["second_half_accuracy"] for r in ref if r["model"] == model and r["n"] == n])}
            mine = [r["second_half_accuracy"] for r in ours if r["model"] == model and r["n"] == n]
            if mine:
                entry["ours"] = _spread(mine)
            out[key] = entry
    return out


def plot(enwik8: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a, e in enwik8.items() if "_curves" in e]
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 3.6), constrained_layout=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        ours, ref = (np.asarray(c) for c in enwik8[arm]["_curves"])
        pos = np.arange(1, len(ours) + 1)
        ax.plot(pos, ref, color="0.4", lw=1.6, label="ml-ladder (PyTorch, cuDNN)")
        ax.plot(pos, ours, color="tab:blue", lw=1.0, label="jaxexp (JAX, XLA)")
        ax.set_xscale("log")
        ax.set_xlabel("position in context")
        ax.set_ylabel("bits/byte (median over seeds)")
        e = enwik8[arm]
        ax.set_title(f"{arm}\n{e['bits_per_byte']['median']:.3f} vs {e['reference_bits_per_byte']['median']:.3f}", fontsize=10)
        ax.legend(fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "r2-per-position.png", dpi=130)


def main() -> None:
    enwik8 = enwik8_summary()
    copy = copy_summary()
    plot(enwik8)
    for arm, e in enwik8.items():
        r = e["reference_bits_per_byte"]
        line = f"{arm:26s} ref {r['median']:.4f} [{r['min']:.4f}, {r['max']:.4f}]"
        if "bits_per_byte" in e:
            b = e["bits_per_byte"]
            line += (
                f"  ours {b['median']:.4f} [{b['min']:.4f}, {b['max']:.4f}] (n={b['n']})  delta {e['delta_median_bits']:+.4f}"
                f"  {e['steady_tokens_per_second']['median']:,.0f} tok/s vs {e['reference_tokens_per_second']['median']:,.0f}"
            )
        print(line)
    for k, e in copy.items():
        print(f"{k:22s} ref {e['reference']['median']:.3f} [{e['reference']['min']:.3f}, {e['reference']['max']:.3f}]",
              f" ours {e['ours']['median']:.3f} [{e['ours']['min']:.3f}, {e['ours']['max']:.3f}]" if "ours" in e else "")
    for e in enwik8.values():
        e.pop("_curves", None)
    summary = {
        "what": "R2/R3 (jaxexp on a Kaggle T4, fp16 compute with dynamic loss scaling) against ml-ladder's recorded "
        "PyTorch runs on a Kaggle T4 (fp16 autocast), same protocol: 12,000 steps, 16 x 1024 tokens, seeds 0-2",
        "enwik8": enwik8,
        "copy": copy,
    }
    path = ROOT / "docs" / "data" / "r2-summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
