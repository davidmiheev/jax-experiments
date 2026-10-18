"""R2/R3: ml-ladder's Part C enwik8 study and copy task, in JAX.

Mirrors ``ml-ladder/tools/gpu/c_context_scaling.py``: the same command-line
arguments, the same models and learning rates, the same data, schedule,
evaluation and JSON fields, so each row compares with ml-ladder's. What may
differ is stated per row: framework, ``init`` ("torch" = ml-ladder's own
initial weights, built with ``torch.manual_seed(seed)`` and ported, the default
when torch and ml-ladder are importable), ``amp`` (fp16 matmuls with dynamic
loss scaling, as autocast + GradScaler), ``remat`` and ``attention``.

Studies:
- ``memory``: compile-only memory and FLOPs of R2's training steps (no
  training). Run it first on the T4: O1 found the 25M RoPE step needs 19.7 GiB
  in fp32 on CPU.
- ``enwik8``: bits/byte and per-position loss, as ml-ladder's ``study_enwik8``.
- ``copy``: R3, as ml-ladder's ``study_copy``.

    uv run python experiments/r2_enwik8/run.py --studies enwik8 --models gru --contexts 64 \
        --enwik8-steps 50 --seeds 0 --out runs/r2-smoke.json          # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jaxexp import recurrent as R
from jaxexp import ssm
from jaxexp import text
from jaxexp import train as TR
from jaxexp import transformer as T
from jaxexp.bench import compiled_stats
from jaxexp.torch_port import load_torch_state, numpy_state

ROOT = Path(__file__).resolve().parents[2]

# name: (family, size spec, peak_lr, weight_decay): ml-ladder's ENWIK8_MODELS.
MODELS = {
    "rnn": ("rnn", (1024, 2), 1e-3, 0.0),
    "gru": ("gru", (592, 2), 2e-3, 0.0),
    "lstm": ("lstm", (512, 2), 2e-3, 0.0),
    "transformer-4M": ("learned", (256, 5, 4), 1e-3, 0.1),
    "transformer-25M": ("learned", (512, 8, 8), 6e-4, 0.1),
    "transformer-25M-rope": ("rope", (512, 8, 8), 6e-4, 0.1),
    # N4: linear recurrences (jaxexp.ssm), ~4.08M parameters each; LRs are N4's own sweeps.
    "lru": ("lru", None, 2e-3, 0.1),
    "mamba": ("mamba", None, 2e-3, 0.1),
}


def ladder_module(path: str | None):
    if not path:
        return None
    try:
        sys.path[:0] = [str(Path(path).resolve() / "src"), str(Path(path).resolve())]
        import torch  # noqa: F401
        from tools.gpu import c_context_scaling

        return c_context_scaling
    except Exception as error:  # noqa: BLE001
        print(f"[r2] ml-ladder/torch not importable ({error}); falling back to JAX-side init", flush=True)
        return None


def build(name: str, context: int, seed: int, args, ladder, vocab: int = 256):
    """(params, apply, parameter count, init label)."""
    family, size, _, _ = MODELS[name]
    dtype = jnp.float16 if args.amp else None
    if family in ssm.KINDS:  # N4: new models, so JAX-side init (no ml-ladder twin)
        params = ssm.KINDS[family][1](jax.random.key(seed), vocab=vocab)
        scan, remat = getattr(args, "scan", "associative"), args.remat != "none"

        def apply(p, ids):
            return ssm.apply(p, ids, family, scan=scan, compute_dtype=dtype, remat=remat)

        return params, apply, ssm.parameter_count(params), "jax"
    if family in ("rnn", "gru", "lstm"):
        hidden, layers = size
        if ladder is not None:
            import torch

            torch.manual_seed(seed)
            params = R.from_torch_state(numpy_state(ladder.recurrent_lm(family, hidden, layers, vocab=vocab)))
            init = "torch"
        else:
            params, init = R.init_like_torch(jax.random.key(seed), family, hidden, layers, vocab), "jax"

        def apply(p, ids):
            return R.apply(p, ids, family, compute_dtype=dtype)

        return params, apply, R.parameter_count(params), init

    width, depth, heads = size
    numerics = T.Numerics(attention=args.attention, compute_dtype=dtype)
    model = T.CharTransformer(
        vocab, width, heads, depth, context, positional=family, numerics=numerics, rngs=nnx.Rngs(seed)
    )
    init = "jax"
    if ladder is not None:
        import torch

        torch.manual_seed(seed)
        load_torch_state(
            model, numpy_state(ladder.transformer_lm(width, depth, heads, context, vocab=vocab, positional=family))
        )
        init = "torch"
    params, nnx_apply = TR.nnx_model(model)
    policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable if args.remat == "dots" else None

    def apply(p, ids):
        return nnx_apply(p, ids, remat=args.remat != "none", policy=policy)

    count = int(sum(np.prod(v.shape) for v in jax.tree.leaves(params)))
    return params, apply, count, init


def device_peak_bytes() -> int | None:
    stats = jax.devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def study_memory(args, report) -> None:
    rows = report.setdefault("memory", [])
    for name in args.models:
        for context in args.contexts:
            batch = args.tokens_per_step // context
            params, apply, count, _ = build(name, context, 0, args, None)
            ids = jnp.zeros((batch, context), jnp.int32)

            def loss(p, apply=apply, ids=ids):
                return TR.cross_entropy(apply(p, ids), ids).mean()

            stats = compiled_stats(jax.grad(loss), params)
            rows.append(
                {
                    "model": name,
                    "context": context,
                    "batch": batch,
                    "parameters": count,
                    "amp": args.amp,
                    "remat": args.remat,
                    "attention": args.attention,
                    **stats,
                }
            )
            print(
                f"[memory] {name:<22} T={context:<5} amp={args.amp} remat={args.remat:<5} "
                f"temp {stats['temp_bytes'] / 2**30:6.2f} GiB  args {stats['argument_bytes'] / 2**30:5.2f} GiB  "
                f"compile {stats['compile_s']:.1f}s",
                flush=True,
            )
            save(args, report)


def study_enwik8(args, report) -> None:
    ladder = ladder_module(args.ml_ladder) if args.init == "torch" else None
    train_ids, val_ids = text.split(text.enwik8(Path(args.data_dir) if args.data_dir else None))
    rows = report.setdefault("enwik8", [])
    done = {(r["seed"], r["model"], r["context"], r.get("peak_lr")) for r in rows}
    for seed in args.seeds:
        for context in args.contexts:
            batch = args.tokens_per_step // context
            windows = text.validation_windows(val_ids, context, args.eval_tokens)
            for name in args.models:
                _, _, default_lr, weight_decay = MODELS[name]
                peak_lr = args.peak_lr or default_lr
                if (seed, name, context, peak_lr) in done:
                    continue
                params, apply, count, init = build(name, context, seed, args, ladder)
                result = TR.train(
                    params,
                    apply,
                    text.Batches(train_ids, context, batch, seed),
                    steps=args.enwik8_steps,
                    peak_lr=peak_lr,
                    weight_decay=weight_decay,
                    amp=args.amp,
                    log_every=args.log_every,
                    label=f"enwik8 {name} T={context} s{seed}",
                )
                curve = TR.per_position_loss(result.params, apply, windows, batch_size=max(1, 65536 // context))
                bits = curve / math.log(2)
                rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "context": context,
                        "batch": batch,
                        "peak_lr": peak_lr,
                        "parameters": count,
                        "steps": args.enwik8_steps,
                        "tokens_seen": args.enwik8_steps * batch * context,
                        "bits_per_byte": float(bits.mean()),
                        "bits_last_quarter": float(bits[-max(1, context // 4) :].mean()),
                        "bits_by_position": [round(float(b), 4) for b in bits],
                        "train_seconds": result.train_seconds,
                        "compile_seconds": result.compile_seconds,
                        "tokens_per_second": args.enwik8_steps * batch * context / result.train_seconds,
                        "steady_tokens_per_second": (args.enwik8_steps - 1)
                        * batch
                        * context
                        / max(result.train_seconds - result.compile_seconds, 1e-9),
                        "peak_memory_mb": (device_peak_bytes() or 0) / 2**20,
                        "train_history": result.history,
                        "skipped_steps": result.skipped_steps,
                        "final_loss_scale": result.final_scale,
                        "framework": "jax",
                        "init": init,
                        "amp": args.amp,
                        "remat": args.remat,
                        "attention": args.attention,
                    }
                )
                r = rows[-1]
                print(
                    f"[enwik8] s{seed} T={context:<5} {name:<22} {r['bits_per_byte']:.4f} bits/byte "
                    f"(last quarter {r['bits_last_quarter']:.4f}) params={count:,} {r['train_seconds']:.0f}s "
                    f"{r['steady_tokens_per_second']:,.0f} tok/s init={init}",
                    flush=True,
                )
                save(args, report)


def study_copy(args, report) -> None:
    ladder = ladder_module(args.ml_ladder) if args.init == "torch" else None
    rows = report.setdefault("copy", [])
    for seed in args.seeds:
        for n in args.copy_lengths:
            length = 2 * n
            batch = max(8, args.tokens_per_step // length)
            for name in args.copy_models:
                _, _, peak_lr, weight_decay = MODELS[name]
                params, apply, count, init = build(name, length - 1, seed, args, ladder, vocab=text.COPY_VOCAB)
                batches = text.CopyBatches(n, batch, seed)
                result = TR.train(
                    params,
                    apply,
                    batches,
                    steps=args.copy_steps,
                    peak_lr=peak_lr,
                    weight_decay=weight_decay,
                    amp=args.amp,
                    log_every=max(args.copy_steps // 4, 1),
                    label=f"copy {name} n={n} s{seed}",
                )
                # ml-ladder's 512 evaluation sequences, scored 32 at a time: all 512 at once
                # needs 8.6 GB of attention scores at n=512 and runs out of memory on a T4.
                inputs, targets = batches(512)
                predict = jax.jit(lambda p, x: jnp.argmax(apply(p, x), -1))
                preds = np.concatenate(
                    [np.asarray(predict(result.params, jnp.asarray(inputs[i : i + 32]))) for i in range(0, len(inputs), 32)]
                )
                correct = preds == targets
                rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "n": n,
                        "parameters": count,
                        "second_half_accuracy": float(correct[:, n - 1 :].mean()),
                        "first_half_accuracy": float(correct[:, : n - 1].mean()),
                        "chance": 1 / text.COPY_VOCAB,
                        "train_seconds": result.train_seconds,
                        "framework": "jax",
                        "init": init,
                        "amp": args.amp,
                    }
                )
                print(
                    f"[copy] s{seed} n={n:<4} {name:<15} second-half acc {rows[-1]['second_half_accuracy']:.3f}",
                    flush=True,
                )
                save(args, report)


def save(args, report) -> None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="r2.json")
    parser.add_argument("--data-dir", default=None, help="directory holding enwik8 (default: data/text)")
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    parser.add_argument("--init", choices=["torch", "jax"], default="torch")
    parser.add_argument("--studies", nargs="+", default=["enwik8"], choices=["memory", "enwik8", "copy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--contexts", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--models", nargs="+", default=["gru", "transformer-25M-rope"], choices=list(MODELS))
    parser.add_argument("--copy-models", nargs="+", default=["lstm", "transformer-4M"], choices=list(MODELS))
    parser.add_argument("--tokens-per-step", type=int, default=16384)
    parser.add_argument("--enwik8-steps", type=int, default=12000)
    parser.add_argument("--eval-tokens", type=int, default=262144)
    parser.add_argument("--copy-lengths", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--copy-steps", type=int, default=1500)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--amp", action="store_true", help="fp16 matmuls + dynamic loss scaling (as autocast)")
    parser.add_argument("--remat", choices=["none", "nothing", "dots"], default="none")
    # ml-ladder routes attention through SDPA (fused, fp32 score accumulation).
    parser.add_argument("--attention", choices=["explicit", "fused"], default="fused")
    parser.add_argument("--scan", choices=["associative", "sequential"], default="associative", help="lru / mamba")
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    report = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    report.setdefault("device", str(jax.devices()[0]))
    report.setdefault("environment", {"python": platform.python_version(), "jax": jax.__version__})
    report["settings"] = vars(args)
    started = time.time()
    for study in args.studies:
        {"memory": study_memory, "enwik8": study_enwik8, "copy": study_copy}[study](args, report)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["wall_seconds"] = time.time() - started
    save(args, report)


if __name__ == "__main__":
    main()
