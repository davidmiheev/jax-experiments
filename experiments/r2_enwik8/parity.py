"""Training-step parity: ml-ladder's own ``train()`` (torch) against ``jaxexp.train`` (JAX).

The last check before R2 spends GPU hours (R1's open thread). For each small
configuration of ml-ladder's models:
- build the model with ml-ladder's own builder after ``torch.manual_seed(seed)``,
  exactly as ``study_enwik8`` does, and port the initial weights to JAX;
- draw batches from enwik8 with ml-ladder's sampling code (an inline copy of the
  ``sample_batch`` closure in ``study_enwik8``) and with ``jaxexp.text.Batches``,
  from the same seed, and check they are byte-identical;
- train both for N steps with ml-ladder's protocol (AdamW (0.9, 0.95),
  clipping at 1.0, warmup + cosine). On CPU, fp32 only: ml-ladder disables
  autocast off CUDA, so the fp16 path is checked on the T4;
- compare the loss at every step, the final parameters, and the per-position
  validation loss.

    uv sync --extra torch && uv run python experiments/r2_enwik8/parity.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from jaxexp import recurrent as R
from jaxexp import text
from jaxexp import train as TR
from jaxexp import transformer as T
from jaxexp.torch_port import load_torch_state, numpy_state

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = [
    # name, kind, size, context, batch, peak_lr, weight_decay (ml-ladder's values for each family)
    ("gru 2x64", "gru", 64, 64, 8, 2e-3, 0.0),
    ("lstm 2x64", "lstm", 64, 64, 8, 2e-3, 0.0),
    ("transformer 64x2, learned", "learned", 64, 64, 8, 1e-3, 0.1),
    ("transformer 64x2, rope", "rope", 64, 64, 8, 1e-3, 0.1),
]


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(ROOT / "docs" / "data" / "r2-parity.json"))
    args = parser.parse_args()
    sys.path[:0] = [str(Path(args.ml_ladder).resolve() / "src"), str(Path(args.ml_ladder).resolve())]
    import torch
    from tools.gpu import c_context_scaling as ladder

    torch.set_num_threads(4)
    train_ids, val_ids = text.split(text.enwik8())
    report = {
        "experiment": "R2 training-step parity, ml-ladder train() vs jaxexp.train, CPU fp32",
        "environment": {"python": platform.python_version(), "jax": jax.__version__, "torch": torch.__version__},
        "steps": args.steps,
        "runs": [],
    }
    for name, kind, size, context, batch, peak_lr, wd in CONFIGS:
        for seed in args.seeds:
            torch.manual_seed(seed)
            if kind in ("gru", "lstm"):
                tmodel = ladder.recurrent_lm(kind, size, 2)
                params = R.from_torch_state(numpy_state(tmodel))
                apply = lambda p, ids, kind=kind: R.apply(p, ids, kind)  # noqa: E731
            else:
                tmodel = ladder.transformer_lm(size, 2, 4, context, positional=kind)
                jmodel = T.CharTransformer(256, size, 4, 2, context, positional=kind, rngs=nnx.Rngs(0))
                load_torch_state(jmodel, numpy_state(tmodel))
                params, apply = TR.nnx_model(jmodel)

            # ml-ladder's sample_batch closure (study_enwik8), copied: rng, then integers, then stack.
            rng = np.random.default_rng(seed)

            def sample_batch(rng=rng, context=context, batch=batch):
                starts = rng.integers(0, len(train_ids) - context - 1, size=batch)
                chunk = np.stack([train_ids[s : s + context + 1] for s in starts]).astype(np.int64)
                tensor = torch.from_numpy(chunk)
                return tensor[:, :-1], tensor[:, 1:]

            ours = text.Batches(train_ids, context, batch, seed)
            probe_a, probe_b = np.random.default_rng(seed), text.Batches(train_ids, context, batch, seed)
            starts = probe_a.integers(0, len(train_ids) - context - 1, size=batch)
            same_batches = bool(
                np.array_equal(np.stack([train_ids[s : s + context + 1] for s in starts])[:, :-1], probe_b()[0])
            )

            info = ladder.train(
                tmodel,
                sample_batch,
                steps=args.steps,
                peak_lr=peak_lr,
                schedule="warmup-cosine",
                device=torch.device("cpu"),
                weight_decay=wd,
                log_every=1,
                label="",
            )
            result = TR.train(params, apply, ours, steps=args.steps, peak_lr=peak_lr, weight_decay=wd, log_every=1)
            torch_losses = np.array([loss for _, loss in info["train_history"]])
            jax_losses = np.array([loss for _, loss in result.history])

            if kind in ("gru", "lstm"):
                param_rel = rel(
                    np.concatenate([np.ravel(result.params[k]) for k in sorted(result.params)]),
                    np.concatenate([np.ravel(v) for _, v in sorted(numpy_state(tmodel).items())]),
                )
            else:
                twin = T.CharTransformer(256, size, 4, 2, context, positional=kind, rngs=nnx.Rngs(0))
                load_torch_state(twin, numpy_state(tmodel))
                trained = nnx.merge(nnx.graphdef(jmodel), result.params)
                flat = lambda m: np.concatenate(  # noqa: E731
                    [np.ravel(v.get_value()) for _, v in nnx.to_flat_state(nnx.state(m, nnx.Param))]
                )
                param_rel = rel(flat(trained), flat(twin))

            windows = text.validation_windows(val_ids, context, eval_tokens=64 * context)
            torch_curve = ladder.per_position_loss(
                tmodel, windows.astype(np.int64), device=torch.device("cpu"), batch_size=64
            )
            jax_curve = TR.per_position_loss(result.params, apply, windows, batch_size=64)
            row = {
                "config": name,
                "seed": seed,
                "context": context,
                "batch": batch,
                "peak_lr": peak_lr,
                "weight_decay": wd,
                "byte_identical_batches": same_batches,
                "loss_step0": [float(torch_losses[0]), float(jax_losses[0])],
                "loss_final": [float(torch_losses[-1]), float(jax_losses[-1])],
                "max_relative_loss_difference": float(np.max(np.abs(jax_losses - torch_losses) / torch_losses)),
                "relative_loss_difference_by_step": (np.abs(jax_losses - torch_losses) / torch_losses).tolist(),
                "final_parameter_relative_difference": param_rel,
                "validation_bits_per_byte": [
                    float(torch_curve.mean() / np.log(2)),
                    float(jax_curve.mean() / np.log(2)),
                ],
                "validation_curve_relative_difference": rel(jax_curve, torch_curve),
            }
            report["runs"].append(row)
            print(
                f"[parity] {name:<26} seed {seed}: batches identical {same_batches}  loss {torch_losses[0]:.4f} -> "
                f"{torch_losses[-1]:.4f}  max rel loss diff {row['max_relative_loss_difference']:.2e}  "
                f"params {param_rel:.2e}  val curve {row['validation_curve_relative_difference']:.2e}",
                flush=True,
            )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
