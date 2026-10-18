"""N2: the gradient noise scale through training, and whether it predicts the critical batch size.

R2's 4M transformer (learned positions) at T=256, AdamW as the protocol, from
ml-ladder's initial weights. "Batch size" counts sequences of 256 bytes.

``noise``  One run at B=64 (the protocol's 16,384 tokens per step), peak LR 8e-3 (O3's
           tuned AdamW), warmup-constant schedule. Every ``--measure-every`` steps,
           B_simple = tr(Sigma) / |G|^2 from per-example gradients (``jaxexp.noise``,
           fp32) on ``--noise-batches`` batches of 64 from a separate sampler.
``batch``  Steps to reach fixed training-loss levels at B = 16, 64, 256, each with its own
           3-point LR grid (vmapped), warmup-constant schedule so that the steps to a
           target do not depend on the run's length. The analysis fits
           S(B) = S_min (1 + B_crit / B) and compares B_crit with B_simple at that loss.

    python experiments/n2_noise_scale/run.py --studies noise
    python experiments/n2_noise_scale/run.py --studies batch --batches 16 64
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
import numpy as np

from jaxexp import noise as N
from jaxexp import sweep as S
from jaxexp import text
from jaxexp import train as TR

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("r2_run", ROOT / "experiments" / "r2_enwik8" / "run.py")
r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2)

MODEL = "transformer-4M"
# Per batch size: (steps, peak LRs). The grid is centred on sqrt scaling from O3's 8e-3 at B=64.
PLAN = {16: (12000, [2e-3, 4e-3, 8e-3]), 64: (6000, [4e-3, 8e-3, 1.6e-2]), 256: (3000, [8e-3, 1.6e-2, 3.2e-2])}


def _save(args, report) -> None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


def _builder(amp: bool):
    return argparse.Namespace(amp=amp, attention="fused", remat="none")


def study_noise(args, report) -> None:
    ladder = r2.ladder_module(args.ml_ladder)
    train_ids, _ = text.split(text.enwik8(Path(args.data_dir) if args.data_dir else None))
    params, apply, count, init = r2.build(MODEL, args.context, args.seed, _builder(args.amp), ladder)
    # The measurement uses an fp32 forward (no fp16 underflow in per-example gradients).
    _, apply32, _, _ = r2.build(MODEL, args.context, args.seed, _builder(False), None)
    per_example = N.make_per_example(apply32, chunk=args.chunk)
    probe = text.Batches(train_ids, args.context, 64, 10_000 + args.seed)
    rows = report.setdefault("noise", [])

    def measure(step, p):
        t0 = time.time()
        p32 = jax.tree.map(lambda x: x.astype(np.float32) if hasattr(x, "astype") else x, p)
        out = N.noise_scale(p32, apply32, [probe() for _ in range(args.noise_batches)], per_example=per_example)
        out["seconds"] = time.time() - t0
        print(f"  [noise] step {step:5d}  B_simple {out['b_simple']:8.1f} sequences "
              f"(tr {out['trace_sigma']:.3g}, |G|^2 {out['grad_norm_sq']:.3g})  {out['seconds']:.0f}s", flush=True)
        return out

    result = TR.train(
        params, apply, text.Batches(train_ids, args.context, 64, args.seed), steps=args.noise_steps,
        peak_lr=args.noise_lr, weight_decay=0.1, amp=args.amp, schedule="warmup-constant",
        log_every=25, measure_every=args.measure_every, on_log=measure, label=f"noise B=64 lr={args.noise_lr:g}",
    )
    rows.append({
        "seed": args.seed, "model": MODEL, "context": args.context, "batch": 64, "peak_lr": args.noise_lr,
        "schedule": "warmup-constant", "steps": args.noise_steps, "parameters": count, "init": init,
        "noise_batches": args.noise_batches, "measurements": [{"step": s, **m} for s, m in result.measurements],
        "train_history": result.history, "train_seconds": result.train_seconds, "skipped_steps": result.skipped_steps,
    })
    _save(args, report)


def study_batch(args, report) -> None:
    ladder = r2.ladder_module(args.ml_ladder)
    train_ids, _ = text.split(text.enwik8(Path(args.data_dir) if args.data_dir else None))
    rows = report.setdefault("batch", [])
    for batch in args.batches:
        steps, lrs = PLAN[batch]
        lrs = args.lrs or lrs
        group = args.group if batch < 256 else 1
        for start in range(0, len(lrs), group):
            chunk = lrs[start : start + group]
            params, apply, count, init = r2.build(MODEL, args.context, args.seed, _builder(args.amp), ladder)
            sampler = text.Batches(train_ids, args.context, batch, args.seed)
            results, timing = S.train_many(
                [params] * len(chunk), apply, [sampler] * len(chunk), steps=steps, peak_lrs=chunk,
                weight_decay=0.1, amp=args.amp, schedule="warmup-constant", log_every=25,
                label=f"B={batch} " + " ".join(f"{lr:g}" for lr in chunk),
            )
            for lr, res in zip(chunk, results, strict=True):
                rows.append({
                    "seed": args.seed, "model": MODEL, "context": args.context, "batch": batch, "peak_lr": lr,
                    "schedule": "warmup-constant", "steps": steps, "tokens_per_step": batch * args.context,
                    "parameters": count, "init": init, "train_history": res.history,
                    "skipped_steps": res.skipped_steps, "final_loss_scale": res.final_scale,
                    "vmapped_copies": len(chunk), "group_train_seconds": timing["train_seconds"],
                })
                last = np.mean([l for _, l in res.history[-8:]])
                print(f"[batch] B={batch} lr={lr:g}  final train loss {last:.4f} (mean of last 8 logs)", flush=True)
            _save(args, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", nargs="+", choices=["noise", "batch"], required=True)
    parser.add_argument("--out", default=str(ROOT / "runs" / "n2.json"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--noise-steps", type=int, default=6000)
    parser.add_argument("--noise-lr", type=float, default=8e-3)
    parser.add_argument("--measure-every", type=int, default=250)
    parser.add_argument("--noise-batches", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 64, 256], choices=list(PLAN))
    parser.add_argument("--lrs", nargs="+", type=float, default=None, help="override the per-batch LR grid")
    parser.add_argument("--group", type=int, default=3)
    parser.add_argument("--steps-override", type=int, default=None, help="smoke tests: cap every run's steps")
    args = parser.parse_args()
    if args.steps_override:
        for b in PLAN:
            PLAN[b] = (args.steps_override, PLAN[b][1])
    report = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    report.update({"device": str(jax.devices()[0]),
                   "environment": {"python": platform.python_version(), "jax": jax.__version__},
                   "settings": vars(args)})
    started = time.time()
    for study in args.studies:
        {"noise": study_noise, "batch": study_batch}[study](args, report)
    report["wall_seconds"] = time.time() - started
    _save(args, report)


if __name__ == "__main__":
    main()
