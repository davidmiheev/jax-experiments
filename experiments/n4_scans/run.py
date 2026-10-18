"""N4: speed and memory of parallel scans against sequential scans and attention.

Four ~4M byte-level models (R2's GRU and 4M transformer, ``jaxexp.ssm``'s LRU and
selective SSM), fp16 compute, a fixed 16,384 tokens per step, T = 256 / 1024 / 4096.
For each (model, T, scan mode): steady training tokens/s over ``--steps`` steps,
compile time, and XLA's compiled temp memory of the training step. Random bytes
stand in for data (speed and memory do not depend on the values).

The enwik8 and copy-task numbers come from the existing runners:
``experiments/n1_vmap_sweep/run.py --model lru|mamba`` (LR sweeps) and
``experiments/r2_enwik8/run.py --studies copy --copy-models lru mamba``.

    python experiments/n4_scans/run.py --contexts 256 1024 4096
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxexp import train as TR
from jaxexp.bench import compiled_stats

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("r2_run", ROOT / "experiments" / "r2_enwik8" / "run.py")
r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2)

# (model, scan mode, remat): the GRU is sequential by nature; Mamba-1's [B, T, 512, 16] states need remat;
# Mamba-2 (SSD) runs its scan as chunked matrix products.
ARMS = [
    ("transformer-4M", "-", "none"),
    ("gru", "-", "none"),
    ("lru", "associative", "none"),
    ("lru", "sequential", "none"),
    ("mamba", "associative", "dots"),
    ("mamba", "sequential", "dots"),
    ("mamba2", "chunked", "none"),
    ("mamba2", "sequential", "none"),
]


def _save(args, report):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "runs" / "n4-throughput.json"))
    parser.add_argument("--contexts", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--tokens-per-step", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--arms", nargs="+", default=None, help="model:scan:remat; default: all")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    # accepted for the Kaggle runner's common arguments
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--ml-ladder", default=None)
    args = parser.parse_args()
    arms = [tuple(a.split(":")) for a in args.arms] if args.arms else ARMS
    report = {"device": str(jax.devices()[0]), "environment": {"python": platform.python_version(), "jax": jax.__version__},
              "settings": vars(args), "rows": []}
    rng = np.random.default_rng(0)
    for context in args.contexts:
        batch = args.tokens_per_step // context
        for name, scan, remat in arms:
            ns = argparse.Namespace(amp=args.amp, attention="fused", remat=remat,
                                    scan=scan if scan != "-" else "associative")
            params, apply, count, _ = r2.build(name, context, 0, ns, None)
            row = {"model": name, "scan": scan, "remat": remat, "context": context, "batch": batch, "parameters": count}
            ids = jnp.zeros((batch, context), jnp.int32)
            try:
                stats = compiled_stats(jax.grad(lambda p: TR.cross_entropy(apply(p, ids), ids).mean()), params)
                row.update(temp_gib=stats["temp_bytes"] / 2**30, compile_only_seconds=stats["compile_s"])

                def sample(batch=batch, context=context):
                    x = rng.integers(0, 256, size=(batch, context + 1)).astype(np.int32)
                    return x[:, :-1], x[:, 1:]

                result = TR.train(params, apply, sample, steps=args.steps, peak_lr=1e-3, weight_decay=0.0,
                                  amp=args.amp, log_every=10**9)
                steady = result.train_seconds - result.compile_seconds
                row.update(compile_seconds=result.compile_seconds,
                           tokens_per_second=(args.steps - 1) * batch * context / steady,
                           ms_per_step=1000 * steady / (args.steps - 1))
            except Exception as error:  # noqa: BLE001  (out of memory etc. is a result)
                row["error"] = str(error)[:300]
            report["rows"].append(row)
            tps = f"{row['tokens_per_second']:>10,.0f} tok/s" if "tokens_per_second" in row else f"ERROR {row['error'][:60]}"
            print(f"[n4] T={context:<5} {name:<15} {scan:<12} remat={remat:<5} {tps}  temp "
                  f"{row.get('temp_gib', float('nan')):6.2f} GiB  compile {row.get('compile_seconds', float('nan')):5.1f}s",
                  flush=True)
            _save(args, report)


if __name__ == "__main__":
    main()
