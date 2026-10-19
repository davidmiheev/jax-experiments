"""O6: a scaling law on BPE tokens, with a corpus the optimum can move in.

Transformers with RoPE (``jaxexp.transformer``, fused attention, fp16 matmuls with loss
scaling) on ml-ladder's BPE-4096 tokens (``jaxexp.bpe``), 256-token contexts, 64 sequences per
step (16,384 tokens, ml-ladder's BPE protocol), AdamW (0.9, 0.95), weight decay 0.1, clipping
at 1.0, 5% warmup then cosine to 10% *over each run's own length* (G1's first rule). Training
tokens come from enwik9 without enwik8's validation and test bytes (``--corpus enwik9``,
~340M tokens), so no run here repeats data (G1's second rule). Scored in bits per byte on
ml-ladder's held-out bytes: total NLL of the target tokens over the bytes they spell, on
windows covering at least 262,144 bytes of enwik8's validation split.

``N`` is the non-embedding parameter count, 12 * layers * width^2 plus biases and
LayerNorms (G1's third rule); the embedding and the 4096-way output layer are reported
separately, as are the FLOPs XLA counts for one training step.

Studies:

- ``tokenize``: build the caches (once per machine) and check them against ml-ladder's
  recorded tokenizer (31,095,000 enwik8 training tokens; 2.8865 bytes per validation token).
- ``lr``: per model size, peak LRs on a log grid at a fixed token budget (seed 0).
- ``isoflop``: for each compute budget C = 6 N D, each size runs D = C / (6 N) tokens, at its
  tuned LR (``--lr-table``), for ``--seeds``.

    python experiments/o6_scaling/run.py --studies tokenize
    python experiments/o6_scaling/run.py --studies lr --sizes s1 s2 --lrs 1e-3 2e-3 4e-3 --tokens 2e7
    python experiments/o6_scaling/run.py --studies isoflop --budgets 1e14 3e14 --lr-table lr.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jaxexp import bpe
from jaxexp import train as TR
from jaxexp import transformer as T
from jaxexp.bench import compiled_stats

ROOT = Path(__file__).resolve().parents[2]
CONTEXT, BATCH = 256, 64
TOKENS_PER_STEP = CONTEXT * BATCH
ATTENTION = ["fused"]  # --attention; "explicit" for CPU smoke tests (XLA:CPU has no fused fp16 backward)
# (width, layers, heads): width / layers ~ 64, head width 64 (32 for the two smallest)
SIZES = {
    "s0": (96, 2, 3),
    "s1": (128, 2, 4),
    "s2": (160, 3, 5),
    "s3": (192, 4, 3),
    "s4": (256, 4, 4),
    "s5": (320, 5, 5),
    "s6": (384, 6, 6),
    "s7": (448, 7, 7),
}


def counts(params) -> dict:
    """Non-embedding parameters (blocks), and the token embedding + output layer + final norm."""
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    total = embed = 0
    for path, leaf in flat:
        size = int(np.prod(leaf.shape))
        total += size
        name = jax.tree_util.keystr(path)
        if "blocks" not in name:
            embed += size
    return {"total": total, "non_embedding": total - embed, "embedding_and_head": embed}


def build(size: str, seed: int, vocab: int):
    width, layers, heads = SIZES[size]
    numerics = T.Numerics(attention=ATTENTION[0], compute_dtype=jnp.float16)
    model = T.CharTransformer(vocab, width, heads, layers, CONTEXT, positional="rope", numerics=numerics,
                              rngs=nnx.Rngs(seed))
    params, apply = TR.nnx_model(model)
    return params, apply


def eval_windows(val: np.ndarray, lengths: np.ndarray, min_bytes: int = 262_144) -> tuple[np.ndarray, np.ndarray]:
    """ml-ladder's BPE evaluation: consecutive windows of CONTEXT + 1 tokens until their target
    tokens spell at least ``min_bytes`` bytes. Returns (windows, target bytes per position)."""
    windows, covered, start = [], 0, 0
    while covered < min_bytes:
        window = np.asarray(val[start : start + CONTEXT + 1], np.int32)
        windows.append(window)
        covered += int(lengths[window[1:]].sum())
        start += CONTEXT + 1
    windows = np.stack(windows)
    return windows, lengths[windows[:, 1:]]


def run_one(data: dict, size: str, *, tokens: float, peak_lr: float, seed: int) -> dict:
    vocab = int(data["meta"]["vocab"])
    params, apply = build(size, seed, vocab)
    n = counts(params)
    steps = max(20, int(round(tokens / TOKENS_PER_STEP)))
    train = data["train"]
    rng = np.random.default_rng(seed)

    def sample():
        starts = rng.integers(0, len(train) - CONTEXT - 1, size=BATCH)
        chunk = np.stack([np.asarray(train[s : s + CONTEXT + 1], np.int32) for s in starts])
        return chunk[:, :-1], chunk[:, 1:]

    ids = jnp.zeros((BATCH, CONTEXT), jnp.int32)
    stats = compiled_stats(jax.grad(lambda p: TR.cross_entropy(apply(p, ids), ids).mean()), params)
    result = TR.train(params, apply, sample, steps=steps, peak_lr=peak_lr, weight_decay=0.1, amp=True,
                      log_every=max(steps // 10, 1), label=f"o6 {size} lr={peak_lr:g} s{seed}")
    windows, target_bytes = data["eval"]
    curve = TR.per_position_loss(result.params, apply, windows, 64)  # mean nats per position
    nats = float((curve * len(windows)).sum())
    d = steps * TOKENS_PER_STEP
    return {
        "size": size, "width": SIZES[size][0], "layers": SIZES[size][1], "heads": SIZES[size][2],
        "seed": seed, "peak_lr": peak_lr, "steps": steps, "tokens": d,
        "parameters": n["total"], "non_embedding": n["non_embedding"], "embedding_and_head": n["embedding_and_head"],
        "compute_6ND": 6.0 * n["non_embedding"] * d,
        "xla_flops_per_step_fwd_bwd": stats["flops"],
        "bits_per_byte": nats / math.log(2) / float(target_bytes.sum()),
        "nats_per_token": nats / float(windows[:, 1:].size),
        "train_history": [[int(a), float(b)] for a, b in result.history],
        "skipped_steps": result.skipped_steps,
        "train_seconds": result.train_seconds,
        "tokens_per_second": d / max(result.train_seconds - result.compile_seconds, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", nargs="+", choices=["tokenize", "lr", "isoflop"], required=True)
    parser.add_argument("--out", default=str(ROOT / "runs" / "o6.json"))
    parser.add_argument("--corpus", choices=["enwik8", "enwik9"], default="enwik9")
    parser.add_argument("--data-dir", default=None, help="directory holding enwik8 (and enwik9)")
    parser.add_argument("--cache-dir", default=None, help="token caches (default: <data-dir>/bpe)")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 2e-3, 4e-3, 8e-3])
    parser.add_argument("--tokens", type=float, default=2e7, help="lr study: tokens per run")
    parser.add_argument("--budgets", nargs="+", type=float, default=[1e14, 3e14, 1e15, 3e15])
    parser.add_argument("--min-tokens", type=float, default=2e6, help="isoflop: skip runs shorter than this")
    parser.add_argument("--lr-table", default=None, help='isoflop: JSON {"s1": 4e-3, ...}')
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--attention", choices=["fused", "explicit"], default="fused")
    parser.add_argument("--ml-ladder", default=None)  # accepted for the Kaggle runner's common arguments
    args = parser.parse_args()
    ATTENTION[0] = args.attention
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / "text"
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "bpe"
    report = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    report.update({"device": str(jax.devices()[0]),
                   "environment": {"python": platform.python_version(), "jax": jax.__version__},
                   "settings": vars(args)})

    def save():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))

    started = time.time()
    if "tokenize" in args.studies:
        check = bpe.corpus(data_dir, cache_dir, "enwik8")
        report["tokenizer_check"] = {**check["meta"], "ml_ladder_train_tokens": 31_095_000,
                                     "ml_ladder_bytes_per_token_val": 2.8865277092227446}
        print(f"[o6] enwik8 check: {check['meta']}", flush=True)
        assert check["meta"]["train_tokens"] == 31_095_000, "not ml-ladder's tokenization"
        save()
    data = bpe.corpus(data_dir, cache_dir, args.corpus)
    data["eval"] = eval_windows(data["val"], data["token_bytes"])
    report["corpus"] = {**data["meta"], "seconds": time.time() - started,
                        "eval_windows": int(len(data["eval"][0])), "eval_bytes": int(data["eval"][1].sum())}
    print(f"[o6] corpus {report['corpus']}", flush=True)
    save()

    if "lr" in args.studies:
        rows = report.setdefault("lr", [])
        done = {(r["size"], r["peak_lr"], r["seed"]) for r in rows}
        for size in args.sizes:
            for lr in args.lrs:
                if (size, lr, 0) in done:
                    continue
                row = run_one(data, size, tokens=args.tokens, peak_lr=lr, seed=0)
                rows.append(row)
                print(f"[o6] lr {size} N={row['non_embedding']:>9,} lr={lr:g}: {row['bits_per_byte']:.4f} bits/byte "
                      f"({row['tokens_per_second']:,.0f} tok/s)", flush=True)
                save()
    if "isoflop" in args.studies:
        table = json.loads(Path(args.lr_table).read_text()) if args.lr_table else {}
        rows = report.setdefault("isoflop", [])
        done = {(r["size"], r["budget"], r["seed"]) for r in rows}
        for budget in args.budgets:
            for size in args.sizes:
                width, layers, _ = SIZES[size]
                params, _ = build(size, 0, int(data["meta"]["vocab"]))
                n = counts(params)["non_embedding"]
                tokens = budget / (6 * n)
                if tokens < args.min_tokens or tokens > len(data["train"]):
                    print(f"[o6] skip {size} at C={budget:.0e}: {tokens:.3g} tokens", flush=True)
                    continue
                for seed in args.seeds:
                    if (size, budget, seed) in done:
                        continue
                    row = run_one(data, size, tokens=tokens, peak_lr=float(table[size]), seed=seed)
                    row["budget"] = budget
                    rows.append(row)
                    print(f"[o6] C={budget:.0e} {size} N={row['non_embedding']:>9,} D={row['tokens']:.3g} "
                          f"s{seed}: {row['bits_per_byte']:.4f} bits/byte", flush=True)
                    save()
    report["wall_seconds"] = time.time() - started
    save()


if __name__ == "__main__":
    main()
