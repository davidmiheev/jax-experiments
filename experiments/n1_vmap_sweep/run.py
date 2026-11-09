"""N1: ml-ladder's learning-rate sweep (Part C §7) as vmapped programs.

The sweep is the 4M transformer at T=1024, 3,000 steps of 16 x 1024 bytes,
peak LRs {3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3} x seeds 0-2: 18 runs that
ml-ladder trained one by one (``c-gpu-lr.json``, ``c-gpu-lr2-1024.json``).
Here K of them train as one program, ``jaxexp.sweep.train_many``. Every copy
starts from ml-ladder's initial weights for its seed and sees that seed's
batches, so copy (lr, seed) is the same experiment as ml-ladder's run.

``memory``      XLA temp memory of the vmapped step for K copies, remat none / dots
                (compile only): where memory stops K.
``throughput``  Steady tokens/s of K copies trained together for a few hundred steps,
                against K = 1, which is what K sequential runs achieve.
``sweep``       The 18 runs, ``--group`` copies per program, with per-position
                evaluation as R2's: JSON rows in ml-ladder's format.

    python experiments/n1_vmap_sweep/run.py --studies memory throughput --ks 1 2 4 8
    python experiments/n1_vmap_sweep/run.py --studies sweep --seeds 0 1 2 --group 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxexp import sweep as S
from jaxexp import text
from jaxexp import train as TR
from jaxexp.bench import compiled_stats

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("r2_run", ROOT / "experiments" / "r2_enwik8" / "run.py")
r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2)

LRS = [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3]  # ml-ladder Part C §7, T=1024
MODEL = "transformer-4M"


def _save(args, report) -> None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


def _builder_args(args, remat: str):
    return argparse.Namespace(amp=args.amp, attention=args.attention, remat=remat)


def study_memory(args, report) -> None:
    rows = report.setdefault("memory", [])
    batch = args.tokens_per_step // args.context
    for remat in args.remats:
        for k in args.ks:
            params, apply, _, _ = r2.build(args.model, args.context, 0, _builder_args(args, remat), None)
            opt = TR.optimizer(0.1)
            stacked = S.stack([params] * k)
            opt_state = jax.vmap(opt.init)(stacked)
            scale = (jnp.ones((k,), jnp.float32), jnp.zeros((k,), jnp.int32))
            ids = jnp.zeros((k, batch, args.context), jnp.int32)
            lr = jnp.zeros((k,), jnp.float32)
            step = jax.vmap(TR.make_step(apply, opt, args.amp))
            stats = compiled_stats(step, stacked, opt_state, scale, ids, ids, lr)
            rows.append({"model": args.model, "k": k, "remat": remat, "context": args.context, "batch": batch, **stats})
            print(
                f"[memory] K={k:<3} remat={remat:<5} temp {stats['temp_bytes'] / 2**30:6.2f} GiB "
                f"({stats['temp_bytes'] / 2**30 / k:5.2f} per copy)  args {stats['argument_bytes'] / 2**30:5.2f} GiB  "
                f"compile {stats['compile_s']:.1f}s",
                flush=True,
            )
            _save(args, report)


def study_throughput(args, report) -> None:
    rows = report.setdefault("throughput", [])
    train_ids, _ = text.split(text.enwik8(Path(args.data_dir) if args.data_dir else None))
    batch = args.tokens_per_step // args.context
    for remat in args.remats:
        for k in args.ks:
            params, apply, _, _ = r2.build(args.model, args.context, 0, _builder_args(args, remat), None)
            sampler = text.Batches(train_ids, args.context, batch, 0)
            try:
                _, timing = S.train_many(
                    [params] * k,
                    apply,
                    [sampler] * k,
                    steps=args.throughput_steps,
                    peak_lrs=[2e-3] * k,
                    weight_decay=0.1,
                    amp=args.amp,
                    log_every=10**9,
                )
            except Exception as error:  # noqa: BLE001  (out of memory ends the K sweep)
                rows.append({"model": args.model, "k": k, "remat": remat, "error": str(error)[:300]})
                print(f"[throughput] K={k} remat={remat}: {str(error)[:120]}", flush=True)
                _save(args, report)
                break
            steady = timing["train_seconds"] - timing["compile_seconds"]
            tokens = (args.throughput_steps - 1) * batch * args.context * k
            rows.append(
                {
                    "model": args.model,
                    "parameters": int(sum(np.prod(v.shape) for v in jax.tree.leaves(params))),
                    "k": k,
                    "remat": remat,
                    "steps": args.throughput_steps,
                    "compile_seconds": timing["compile_seconds"],
                    "steady_seconds": steady,
                    "tokens_per_second": tokens / steady,
                    "peak_memory_mb": (r2.device_peak_bytes() or 0) / 2**20,
                }
            )
            base = next(
                (r for r in rows if r.get("k") == 1 and r["remat"] == remat and r.get("model") == args.model
                 and "tokens_per_second" in r),
                None,
            )
            speedup = rows[-1]["tokens_per_second"] / base["tokens_per_second"] if base else float("nan")
            print(
                f"[throughput] {args.model} K={k:<3} remat={remat:<5} {rows[-1]['tokens_per_second']:>10,.0f} tok/s total "
                f"({speedup:.2f}x K=1, i.e. {speedup:.2f}x {k} sequential runs' rate)  compile {timing['compile_seconds']:.1f}s",
                flush=True,
            )
            _save(args, report)


def study_sweep(args, report) -> None:
    ladder = r2.ladder_module(args.ml_ladder)
    train_ids, val_ids = text.split(text.enwik8(Path(args.data_dir) if args.data_dir else None))
    batch = args.tokens_per_step // args.context
    windows = text.validation_windows(val_ids, args.context, args.eval_tokens)
    rows = report.setdefault("enwik8", [])
    grid = [(seed, lr) for seed in args.seeds for lr in args.lrs]
    done = {(r["seed"], r["peak_lr"]) for r in rows}
    grid = [g for g in grid if g not in done]
    builder = _builder_args(args, args.remat)
    for start in range(0, len(grid), args.group):
        group = grid[start : start + args.group]
        built = {seed: r2.build(args.model, args.context, seed, builder, ladder) for seed in {s for s, _ in group}}
        samplers = {seed: text.Batches(train_ids, args.context, batch, seed) for seed in built}
        apply = next(iter(built.values()))[1]
        label = "K=%d %s" % (len(group), " ".join(f"s{s}/{lr:g}" for s, lr in group))
        results, timing = S.train_many(
            [built[s][0] for s, _ in group],
            apply,
            [samplers[s] for s, _ in group],
            steps=args.steps,
            peak_lrs=[lr for _, lr in group],
            weight_decay=r2.MODELS[args.model][3],
            amp=args.amp,
            log_every=args.log_every,
            label=label,
        )
        steady = timing["train_seconds"] - timing["compile_seconds"]
        for (seed, lr), result in zip(group, results, strict=True):
            curve = TR.per_position_loss(result.params, apply, windows, batch_size=max(1, 65536 // args.context))
            bits = curve / math.log(2)
            rows.append(
                {
                    "seed": seed,
                    "model": args.model,
                    "context": args.context,
                    "batch": batch,
                    "peak_lr": lr,
                    "parameters": built[seed][2],
                    "steps": args.steps,
                    "tokens_seen": args.steps * batch * args.context,
                    "bits_per_byte": float(bits.mean()),
                    "bits_last_quarter": float(bits[-max(1, args.context // 4) :].mean()),
                    "bits_by_position": [round(float(b), 4) for b in bits],
                    "train_history": result.history,
                    "skipped_steps": result.skipped_steps,
                    "final_loss_scale": result.final_scale,
                    "vmapped_copies": len(group),
                    "group": [list(g) for g in group],
                    "group_train_seconds": timing["train_seconds"],
                    "group_compile_seconds": timing["compile_seconds"],
                    "group_steady_tokens_per_second": (args.steps - 1) * batch * args.context * len(group) / steady,
                    "peak_memory_mb": (r2.device_peak_bytes() or 0) / 2**20,
                    "framework": "jax",
                    "init": built[seed][3],
                    "amp": args.amp,
                    "remat": args.remat,
                    "attention": args.attention,
                }
            )
            print(f"[sweep] s{seed} lr={lr:g}  {rows[-1]['bits_per_byte']:.4f} bits/byte  ({label})", flush=True)
        print(f"[sweep] group of {len(group)}: {timing['train_seconds']:.0f}s, "
              f"{rows[-1]['group_steady_tokens_per_second']:,.0f} tok/s total", flush=True)
        _save(args, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", nargs="+", choices=["memory", "throughput", "sweep"], required=True)
    parser.add_argument("--out", default=str(ROOT / "runs" / "n1.json"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    parser.add_argument("--model", default=MODEL, help="a model of R2's run.py, or a --size")
    parser.add_argument("--size", default=None, help="width,depth,heads: a learned-position transformer, lr 2e-3, wd 0.1")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--tokens-per-step", type=int, default=16384)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--remats", nargs="+", default=["none", "dots"])
    parser.add_argument("--throughput-steps", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lrs", nargs="+", type=float, default=LRS)
    parser.add_argument("--group", type=int, default=6, help="copies per vmapped program")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-tokens", type=int, default=262144)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--remat", choices=["none", "nothing", "dots"], default="none")
    parser.add_argument("--attention", choices=["explicit", "fused"], default="fused")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args()
    if args.size:  # smaller transformers, to find where one copy stops filling the GPU
        width, depth, heads = (int(x) for x in args.size.split(","))
        args.model = f"transformer-w{width}d{depth}"
        r2.MODELS[args.model] = ("learned", (width, depth, heads), 2e-3, 0.1)
    report = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    report.update(
        {
            "device": str(jax.devices()[0]),
            "environment": {"python": platform.python_version(), "jax": jax.__version__},
            "settings": {k: v for k, v in vars(args).items()},
        }
    )
    started = time.time()
    for study in args.studies:
        {"memory": study_memory, "throughput": study_throughput, "sweep": study_sweep}[study](args, report)
    report["wall_seconds"] = time.time() - started
    _save(args, report)


if __name__ == "__main__":
    main()
