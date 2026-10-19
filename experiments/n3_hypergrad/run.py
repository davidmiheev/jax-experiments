"""N3: learning the learning rate by differentiating through training.

R2's 4M transformer at T=256 (ml-ladder's initial weights and batch stream for the seed;
AdamW 0.9/0.95, wd 0.1, clip 1.0; 64 x 256 bytes per step), in fp32 with explicit attention,
since the hypergradient differentiates through the optimiser (``jaxexp.hypergrad``).

The meta-objective is the loss on held-out bytes: windows drawn from the validation split
*after* its first 1,000,000 bytes, so that they never overlap the 262,144 bytes on which
bits/byte is reported.

Studies:

- ``reference``: 3,000 steps of the tuned protocol (peak 8e-3, 5% warmup, cosine to 10%; the
  best LR at T=256 in ml-ladder's ``c-gpu-lr2-256.json`` and O3), in this fp32 setting. Saves
  checkpoints at ``--starts`` and reports bits/byte. The baseline for ``online``.
- ``landscape``: from each start (step 0, and the reference's checkpoints), for each K in
  ``--ks``: the meta-loss after K steps at a constant LR on a log grid, and its hypergradient
  d L / d log lr. The same K training batches and the same 64 meta-validation windows at every
  LR, so the curve is a deterministic function of the LR. Also the per-step hypergradient
  (one LR per step) at the reference schedule's LR, to show which steps want more or less.
- ``online``: 3,000 steps whose LR is set by hypergradient descent: every K steps, the
  hypergradient of the meta-loss after that segment (fresh meta-validation windows each
  time) updates log lr by Adam (step ``--meta-lr``). No schedule; the LR is what it learns.
  A segment whose loss or hypergradient is not finite is undone and the LR halved.

    python experiments/n3_hypergrad/run.py --studies reference landscape --out runs/n3.json
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
import optax

from jaxexp import hypergrad as H
from jaxexp import text
from jaxexp import train as TR

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("r2_run", ROOT / "experiments" / "r2_enwik8" / "run.py")
r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2)

CONTEXT, BATCH, STEPS, PEAK = 256, 64, 3000, 8e-3  # STEPS: the default of --steps
META_OFFSET = 1_000_000  # meta-validation windows come from val[META_OFFSET:]


def _save(args, report):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


class Setup:
    def __init__(self, args):
        ladder = r2.ladder_module(args.ml_ladder) if args.init == "torch" else None
        ns = argparse.Namespace(amp=False, attention="explicit", remat="none", init=args.init)
        self.params0, self.apply, self.count, self.init = r2.build("transformer-4M", CONTEXT, args.seed, ns, ladder)
        train, val = text.split(text.enwik8(Path(args.data_dir) if args.data_dir else None))
        self.train, self.val = train, val
        self.batches = text.Batches(train, CONTEXT, BATCH, args.seed)
        self.eval_windows = text.validation_windows(val, CONTEXT)
        self.meta_rng = np.random.default_rng(10_000 + args.seed)
        self.unroll, self.loss_fn = H.make_unroll(self.apply)
        self.meta = H.make_meta_loss(self.apply)
        self.unroll_j = jax.jit(self.unroll)
        self.meta_vg = jax.jit(jax.value_and_grad(self.meta, has_aux=True))

    def next_windows(self, k: int) -> np.ndarray:
        out = []
        for _ in range(k):
            x, y = self.batches()
            out.append(np.concatenate([x, y[:, -1:]], axis=1))
        return np.stack(out)

    def meta_windows(self, n: int = BATCH, rng: np.random.Generator | None = None) -> np.ndarray:
        region = self.val[META_OFFSET:]
        starts = (rng or self.meta_rng).integers(0, len(region) - CONTEXT - 1, size=n)
        return np.stack([region[s : s + CONTEXT + 1] for s in starts])

    def keyed_windows(self, seed: int, start: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The landscape's K training batches and meta-validation windows for (seed, start, K),
        drawn from their own generators, so that a rerun with another LR grid sees the same bytes."""
        key = 1_000_003 * (seed + 1) + 1009 * start + k
        batches = text.Batches(self.train, CONTEXT, BATCH, key)
        out = []
        for _ in range(k):
            x, y = batches()
            out.append(np.concatenate([x, y[:, -1:]], axis=1))
        return np.stack(out), self.meta_windows(rng=np.random.default_rng(key + 1))

    def bits(self, params) -> dict:
        nats = TR.per_position_loss(params, self.apply, self.eval_windows, 64)
        return {"bits_per_byte": float(nats.mean() / math.log(2)), "bits_by_position": (nats / math.log(2)).round(4).tolist()}


def study_reference(args, report, setup: Setup, checkpoints: dict):
    params, state = setup.params0, H.adamw_init(setup.params0)
    chunk, history, started = 50, [], time.time()
    if 0 in args.starts:
        checkpoints[0] = (params, state)
    for t0 in range(0, args.steps, chunk):
        windows = setup.next_windows(chunk)
        lrs = np.array([TR.lr_at(t, args.steps, PEAK) for t in range(t0, t0 + chunk)], np.float32)
        params, state, losses = setup.unroll_j(params, state, jnp.asarray(windows), jnp.asarray(lrs))
        history.append([t0, float(np.mean(losses))])
        if t0 + chunk in args.starts:
            checkpoints[t0 + chunk] = (params, state)
        if (t0 // chunk) % 10 == 0:
            print(f"[reference] step {t0:5d} loss {history[-1][1]:.4f} ({time.time() - started:.0f}s)", flush=True)
    seconds = time.time() - started
    row = {"seed": args.seed, "peak_lr": PEAK, "steps": args.steps, "train_history": history, "train_seconds": seconds,
           "steps_per_second": args.steps / seconds, **setup.bits(params)}
    report.setdefault("reference", []).append(row)
    print(f"[reference] s{args.seed} {row['bits_per_byte']:.4f} bits/byte, {row['steps_per_second']:.2f} steps/s", flush=True)
    _save(args, report)


def study_landscape(args, report, setup: Setup, checkpoints: dict):
    rows = report.setdefault("landscape", [])
    grid = [float(x) for x in args.grid]
    for start in args.starts:
        params, state = checkpoints[start]
        for k in args.ks:
            windows, val = (jnp.asarray(a) for a in setup.keyed_windows(args.seed, start, k))
            before = float(setup.loss_fn(params, val))
            points, started = [], time.time()
            for lr in grid:
                (loss, (_, _, train_losses)), g = setup.meta_vg(jnp.log(jnp.float32(lr)), params, state, windows, val)
                points.append({"lr": lr, "meta_loss": float(loss), "hypergrad": float(g),
                               "train_loss_mean": float(jnp.mean(train_losses))})
            schedule_lrs = np.array([TR.lr_at(start + i, args.steps, PEAK) for i in range(k)], np.float32)
            (loss_s, _), g_per_step = setup.meta_vg(jnp.log(jnp.asarray(schedule_lrs)), params, state, windows, val)
            row = {"seed": args.seed, "start": start, "k": k, "meta_loss_before": before, "points": points,
                   "schedule_lr_mean": float(schedule_lrs.mean()), "schedule_meta_loss": float(loss_s),
                   "schedule_hypergrad_per_step": np.asarray(g_per_step).tolist(), "seconds": time.time() - started}
            best = min(points, key=lambda p: p["meta_loss"])
            rows.append(row)
            print(f"[landscape] start {start:5d} K={k:<4} best lr {best['lr']:.1e} (loss {best['meta_loss']:.4f}; "
                  f"before {before:.4f}); schedule lr {row['schedule_lr_mean']:.1e}; {row['seconds']:.0f}s", flush=True)
            _save(args, report)


def study_online(args, report, setup: Setup):
    k, steps = args.online_k, args.steps
    params, state = setup.params0, H.adamw_init(setup.params0)
    log_lr = jnp.log(jnp.float32(args.online_init))
    meta_opt = optax.adam(args.meta_lr)
    meta_state = meta_opt.init(log_lr)
    trajectory, started = [], time.time()
    for seg in range(steps // k):
        windows = jnp.asarray(setup.next_windows(k))
        val = jnp.asarray(setup.meta_windows())
        (loss, (new_params, new_state, train_losses)), g = setup.meta_vg(log_lr, params, state, windows, val)
        diverged = not (np.isfinite(float(loss)) and np.isfinite(float(g)))
        trajectory.append({"step": seg * k, "lr": float(jnp.exp(log_lr)), "meta_loss": float(loss),
                           "hypergrad": float(g), "train_loss": float(jnp.mean(train_losses)), "diverged": diverged})
        if diverged:
            # Undo the segment and halve the LR (the meta-optimiser's state is kept): without this
            # guard, a run started above the stable LR is NaN before the hypergradient can act.
            log_lr = log_lr - jnp.log(2.0)
            continue
        params, state = new_params, new_state
        upd, meta_state = meta_opt.update(g, meta_state, log_lr)
        log_lr = optax.apply_updates(log_lr, upd)
        if seg % 10 == 0:
            print(f"[online] step {seg * k:5d} lr {trajectory[-1]['lr']:.2e} meta {float(loss):.4f} "
                  f"({time.time() - started:.0f}s)", flush=True)
    row = {"seed": args.seed, "k": k, "init_lr": args.online_init, "meta_lr": args.meta_lr, "steps": steps,
           "trajectory": trajectory, "train_seconds": time.time() - started, **setup.bits(params)}
    report.setdefault("online", []).append(row)
    print(f"[online] s{args.seed} K={k} init {args.online_init:g}: {row['bits_per_byte']:.4f} bits/byte", flush=True)
    _save(args, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", nargs="+", choices=["reference", "landscape", "online"], required=True)
    parser.add_argument("--out", default=str(ROOT / "runs" / "n3.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=["torch", "jax"], default="torch")
    parser.add_argument("--starts", nargs="+", type=int, default=[0, 500, 1500])
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 30, 100])
    parser.add_argument("--grid", nargs="+",
                        default=["2.5e-4", "5e-4", "1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2", "3.2e-2", "6.4e-2"])
    parser.add_argument("--steps", type=int, default=STEPS, help="training length (3,000 = ml-ladder's protocol)")
    parser.add_argument("--online-k", type=int, default=30)
    parser.add_argument("--online-init", type=float, default=1e-3)
    parser.add_argument("--meta-lr", type=float, default=0.05, help="Adam step on log lr")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    args = parser.parse_args()
    report = {"device": str(jax.devices()[0]), "environment": {"python": platform.python_version(), "jax": jax.__version__},
              "settings": vars(args)}
    setup = Setup(args)
    report["parameters"] = setup.count
    checkpoints: dict = {}
    if "landscape" in args.studies and set(args.starts) - {0} and "reference" not in args.studies:
        raise SystemExit("landscape from later starts needs the reference study in the same run")
    if "landscape" in args.studies and "reference" not in args.studies:
        checkpoints[0] = (setup.params0, H.adamw_init(setup.params0))
    for study in args.studies:
        if study == "reference":
            study_reference(args, report, setup, checkpoints)
        elif study == "landscape":
            study_landscape(args, report, setup, checkpoints)
        else:
            study_online(args, report, setup)
    _save(args, report)


if __name__ == "__main__":
    main()
