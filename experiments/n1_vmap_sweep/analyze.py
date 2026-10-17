"""N1 against ml-ladder's §7 sweep, plus the vmap throughput study.

Reads ``docs/data/n1/`` (this repo's Kaggle outputs) and ml-ladder's
``c-gpu-lr.json`` / ``c-gpu-lr2-1024.json``; writes ``docs/data/n1-summary.json``
and the figures.

    uv run python experiments/n1_vmap_sweep/analyze.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "data" / "n1"
LADDER = Path(os.environ.get("ML_LADDER", ROOT.parent / "ml-ladder")) / "docs" / "writeups" / "data"
FIG = ROOT / "docs" / "figures"


def _spread(values) -> dict:
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def reference() -> dict:
    rows = []
    for name in ("c-gpu-lr.json", "c-gpu-lr2-1024.json"):
        rows += [r for r in json.loads((LADDER / name).read_text())["enwik8"]
                 if r["model"] == "transformer-4M" and r["context"] == 1024 and r["steps"] == 3000]
    return {(r["peak_lr"], r["seed"]): r for r in rows}


def ours() -> dict:
    rows = [r for p in sorted(RUNS.glob("n1-sweep-*.json")) for r in json.loads(p.read_text()).get("enwik8", [])]
    return {(r["peak_lr"], r["seed"]): r for r in rows}


def sweep_summary() -> dict:
    ref, mine = reference(), ours()
    lrs = sorted({lr for lr, _ in ref})
    out = {"by_lr": {}, "pairs": {}}
    for lr in lrs:
        entry = {"reference": _spread([ref[(lr, s)]["bits_per_byte"] for s in (0, 1, 2) if (lr, s) in ref])}
        got = [mine[(lr, s)]["bits_per_byte"] for s in (0, 1, 2) if (lr, s) in mine]
        if got:
            entry["ours"] = _spread(got)
        out["by_lr"][f"{lr:g}"] = entry
    deltas = []
    for key in sorted(mine):
        if key in ref:
            d = mine[key]["bits_per_byte"] - ref[key]["bits_per_byte"]
            out["pairs"][f"lr={key[0]:g} s{key[1]}"] = {
                "ours": mine[key]["bits_per_byte"], "reference": ref[key]["bits_per_byte"], "delta": d,
                "skipped_steps": mine[key]["skipped_steps"], "vmapped_copies": mine[key]["vmapped_copies"],
            }
            deltas.append(d)
    if deltas:
        out["abs_delta"] = _spread(np.abs(deltas))
        best = lambda table: min(table, key=lambda lr: table[lr][1])  # noqa: E731
        med = {k: (float(k), v["reference"]["median"]) for k, v in out["by_lr"].items()}
        out["best_lr_reference"] = best(med)
        med_ours = {k: (float(k), v["ours"]["median"]) for k, v in out["by_lr"].items() if "ours" in v}
        out["best_lr_ours"] = best(med_ours)
        groups = {tuple(map(tuple, r["group"])): r for r in mine.values()}
        out["programs"] = [
            {"copies": r["vmapped_copies"], "train_seconds": r["group_train_seconds"],
             "compile_seconds": r["group_compile_seconds"], "steady_tokens_per_second": r["group_steady_tokens_per_second"]}
            for r in groups.values()
        ]
        out["reference_seconds_per_run"] = _spread([r["train_seconds"] for r in ref.values()])
    return out


def throughput_summary() -> dict:
    rows = [r for p in sorted(RUNS.glob("n1-throughput-*.json")) for r in json.loads(p.read_text()).get("throughput", [])]
    out = {}
    for r in rows:
        key = f"{r.get('model', 'transformer-4M')} remat={r['remat']}"
        out.setdefault(key, {})[r["k"]] = r.get("tokens_per_second", None)
    for key, table in out.items():
        base = table.get(1)
        out[key] = {
            str(k): {"tokens_per_second": v, "speedup_vs_k1": (v / base if (v and base) else None)} for k, v in sorted(table.items())
        }
    memory = [r for p in sorted(RUNS.glob("n1-memory*.json")) for r in json.loads(p.read_text()).get("memory", [])]
    return {
        "throughput": out,
        "memory_temp_gib": {f"K={r['k']} remat={r['remat']}": r["temp_bytes"] / 2**30 for r in memory},
    }


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax = axes[0]
    by_lr = summary["sweep"]["by_lr"]
    lrs = np.array([float(k) for k in by_lr])
    for key, color, label, shift in (("reference", "0.35", "ml-ladder: 18 sequential PyTorch runs", 0.97),
                                     ("ours", "tab:blue", "jaxexp: vmapped, 3 copies per program", 1.03)):
        if all(key in v for v in by_lr.values()):
            med = np.array([v[key]["median"] for v in by_lr.values()])
            lo = med - np.array([v[key]["min"] for v in by_lr.values()])
            hi = np.array([v[key]["max"] for v in by_lr.values()]) - med
            ax.errorbar(lrs * shift, med, yerr=[lo, hi], fmt="o-", color=color, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("validation bits/byte (median, min-max over seeds)")
    ax.set_title("4M transformer, T=1024, 3,000 steps")
    ax.legend(fontsize=8)
    ax = axes[1]
    for key, table in summary["throughput"]["throughput"].items():
        ks = [int(k) for k, v in table.items() if v["speedup_vs_k1"]]
        sp = [table[str(k)]["speedup_vs_k1"] for k in ks]
        if ks:
            ax.plot(ks, sp, "o-", label=key)
    ax.axhline(1, color="0.5", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("copies K in one vmapped program")
    ax.set_ylabel("total throughput / K=1 throughput")
    ax.set_title("vmap speedup over K sequential runs (T4, fp16)")
    ax.legend(fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "n1-sweep-and-throughput.png", dpi=130)


def main() -> None:
    summary = {"sweep": sweep_summary(), "throughput": throughput_summary()}
    for lr, v in summary["sweep"]["by_lr"].items():
        r = v["reference"]
        line = f"lr={lr:>7}  ref {r['median']:.4f} [{r['min']:.4f}, {r['max']:.4f}]"
        if "ours" in v:
            o = v["ours"]
            line += f"  ours {o['median']:.4f} [{o['min']:.4f}, {o['max']:.4f}]"
        print(line)
    if "abs_delta" in summary["sweep"]:
        print("per-pair |delta|", summary["sweep"]["abs_delta"], "best lr ref/ours",
              summary["sweep"]["best_lr_reference"], summary["sweep"]["best_lr_ours"])
    for key, table in summary["throughput"]["throughput"].items():
        print(key, {k: round(v["speedup_vs_k1"], 2) if v["speedup_vs_k1"] else None for k, v in table.items()})
    plot(summary)
    (ROOT / "docs" / "data" / "n1-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote docs/data/n1-summary.json")


if __name__ == "__main__":
    main()
