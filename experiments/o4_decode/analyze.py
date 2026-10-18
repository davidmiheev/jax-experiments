"""O4: summarise the decode benchmark (``docs/data/o4/o4-*.json``).

For each (model, batch): ms per generated token against the context length C for the three
modes, the bandwidth floor (bytes a cached step must read / 320 GB/s), the achieved bandwidth,
and the overhead above the floor per kernel launch (from ``o4-kernels.json``). Also the
command-buffer (CUDA graph) runs, if present. Writes ``docs/data/o4-summary.json`` and
``docs/figures/o4-decode.png``.

    uv run python experiments/o4_decode/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data" / "o4"
FIG = ROOT / "docs" / "figures"
MODES = ("recompute", "cached", "loop")


def rows(pattern: str = "o4-*.json") -> list[dict]:
    out = []
    for p in sorted(DATA.glob(pattern)):
        if p.name in ("o4-kernels.json",) or p.name.startswith("o4-cmdbuf"):
            continue
        out += [r for r in json.loads(p.read_text())["rows"] if "error" not in r]
    return out


def summarise() -> dict:
    data = rows()
    kernels = json.loads((DATA / "o4-kernels.json").read_text())
    table = {}
    for r in data:
        key = f"{r['model']}/B{r['batch']}"
        entry = table.setdefault(key, {})
        c = entry.setdefault(str(r["context"]), {"floor_ms": r["step_ms_at_bandwidth"], "step_bytes_model": r["step_bytes_model"],
                                                 "cache_bytes": r["cache_bytes"], "weight_bytes": r["weight_bytes"],
                                                 "prefill_ms": r.get("prefill_ms")})
        c[r["mode"]] = {
            "ms_per_token_step": r["ms_per_step"],
            "achieved_GBps": r["achieved_GBps"],
            "compile_seconds": r["compile_seconds"],
            "temp_MiB": r["temp_bytes"] / 2**20,
            "xla_bytes_accessed_MB": r["xla_bytes_accessed"] / 1e6,
        }
        if r["mode"] == "loop" and r["batch"] == 1 and r["model"] in kernels:
            n = kernels[r["model"]]["launches"]
            c["loop"]["us_above_floor_per_kernel"] = 1e3 * (r["ms_per_step"] - r["step_ms_at_bandwidth"]) / n
        if r["mode"] == "cached":
            c["cached"]["xla_bytes_vs_model"] = r["xla_bytes_accessed"] / r["step_bytes_model"]
        agree = r.get("token_agreement", {})
        c["token_agreement"] = agree
    summary = {"by_model_batch": table, "kernels": {k: v for k, v in kernels.items() if isinstance(v, dict)}}
    # Speedups at each point.
    for key, entry in table.items():
        for c, e in entry.items():
            if "recompute" in e and "cached" in e:
                e["recompute_over_cached"] = e["recompute"]["ms_per_token_step"] / e["cached"]["ms_per_token_step"]
            if "loop" in e and "cached" in e:
                e["cached_over_loop"] = e["cached"]["ms_per_token_step"] / e["loop"]["ms_per_token_step"]
    cmd = {}
    for p in sorted(DATA.glob("o4-cmdbuf-*.json")):
        variant = p.stem.split("-")[2]
        for r in json.loads(p.read_text())["rows"]:
            if "error" in r:
                cmd.setdefault(variant, {})[f"{r['model']}/C{r['context']}"] = {"error": r["error"][:120]}
                continue
            cmd.setdefault(variant, {}).setdefault(f"{r['model']}/C{r['context']}", {})[r["mode"]] = r["ms_per_step"]
    if cmd:
        summary["command_buffers"] = cmd
    return summary


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [k for k in ("transformer-25M-rope/B1", "transformer-25M-rope/B8", "transformer-4M/B1", "transformer-4M/B8")
              if k in summary["by_model_batch"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), constrained_layout=True, sharey=True)
    colors = {"recompute": "tab:red", "cached": "tab:blue", "loop": "tab:green"}
    for ax, key in zip(axes, panels):
        entry = summary["by_model_batch"][key]
        cs = sorted(int(c) for c in entry)
        for mode in MODES:
            pts = [(c, entry[str(c)][mode]["ms_per_token_step"]) for c in cs if mode in entry[str(c)]]
            if pts:
                ax.plot(*zip(*pts), "o-", color=colors[mode], label=mode)
        ax.plot(cs, [entry[str(c)]["floor_ms"] for c in cs], "k--", lw=1, label="bandwidth floor")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(key.replace("/", ", "))
        ax.set_xlabel("context length C (static cache)")
    axes[0].set_ylabel("ms per decode step (one token per sequence)")
    axes[0].legend(fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "o4-decode.png", dpi=130)


def main() -> None:
    summary = summarise()
    for key, entry in summary["by_model_batch"].items():
        for c in sorted(entry, key=int):
            e = entry[c]
            print(f"{key:<26} C={c:<5} floor {e['floor_ms']:.3f}  " + "  ".join(
                f"{m} {e[m]['ms_per_token_step']:.3f}" for m in MODES if m in e),
                f" rec/cached {e.get('recompute_over_cached', float('nan')):.1f}x  cached/loop {e.get('cached_over_loop', float('nan')):.2f}x",
                f" us/kernel {e.get('loop', {}).get('us_above_floor_per_kernel', float('nan')):.2f}")
    if "command_buffers" in summary:
        print("command buffers:", json.dumps(summary["command_buffers"], indent=1))
    plot(summary)
    (ROOT / "docs" / "data" / "o4-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote docs/data/o4-summary.json")


if __name__ == "__main__":
    main()
