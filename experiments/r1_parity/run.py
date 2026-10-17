"""R1: parity harness — C3's PyTorch transformer against the Flax NNX port.

The reference is ml-ladder's own code, imported from a checkout (``--ml-ladder``,
default ``../ml-ladder``), not a copy: C3's ``make_transformer`` for the block,
and ``tools/gpu/c_context_scaling.transformer_lm`` for the SDPA/RoPE models R2
trains. Weights travel as NumPy; both sides run on the same input; the verdict
is relative error ||jax - torch|| / ||torch|| and max |diff|.

Arms (block at G3's settings: width 64, 4 heads, length 16, seeds 0-2, batch 1 and 4):

- matched: PyTorch numerics, explicit softmax attention;
- fused attention: ``jax.nn.dot_product_attention`` instead of the explicit form;
- one silent Flax/JAX default at a time: tanh GELU, LayerNorm epsilon 1e-6,
  fast variance; then all three together (what a naive port computes);
- fast variance again with a +100 offset on the input (catastrophic cancellation);
- the square output projection loaded untransposed (the silent orientation bug);
- ``jax.default_matmul_precision("highest")`` against the default.

Then whole models, R2's configurations: 4M learned and RoPE at T=256, 25M RoPE at
T=1024 (logits), plus loss and gradient parity on the 4M model.

    uv run python experiments/r1_parity/run.py --out docs/data/r1-parity.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jaxexp import transformer as T
from jaxexp.torch_port import load_torch_state, numpy_state

SEEDS = (0, 1, 2)
BATCHES = (1, 4)


def disagreement(reference: np.ndarray, ported: np.ndarray) -> dict[str, float]:
    diff = np.asarray(ported, np.float64) - np.asarray(reference, np.float64)
    return {
        "relative": float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-30)),
        "max_abs": float(np.abs(diff).max()),
    }


def summarise(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def block_arms():
    """(name, Numerics, input offset, transpose_linear, matmul precision)."""
    ln_eps = T.Numerics(ln_epsilon=1e-6)
    fast = T.Numerics(ln_fast_variance=True)
    return [
        ("matched", T.TORCH, 0.0, True, None),
        ("fused attention", T.Numerics(attention="fused"), 0.0, True, None),
        ("tanh GELU (jax.nn.gelu default)", T.Numerics(gelu_approximate=True), 0.0, True, None),
        ("LayerNorm eps 1e-6 (Flax default)", ln_eps, 0.0, True, None),
        ("fast variance (Flax default)", fast, 0.0, True, None),
        ("all three Flax/JAX defaults", T.FLAX_DEFAULTS, 0.0, True, None),
        ("matched, input +100", T.TORCH, 100.0, True, None),
        ("fast variance, input +100", fast, 100.0, True, None),
        ("projection untransposed", T.TORCH, 0.0, False, None),
        ("matmul precision highest", T.TORCH, 0.0, True, "highest"),
    ]


def run_block(torch, make_transformer, width=64, heads=4, length=16) -> dict:
    CharTransformer, _ = make_transformer()
    rows: dict[str, dict] = {}
    sdpa_floor = []
    for name, numerics, offset, transpose, precision in block_arms():
        per = {f"batch{b}": {"relative": [], "max_abs": []} for b in BATCHES}
        for seed in SEEDS:
            torch.manual_seed(seed)
            block = CharTransformer(65, width=width, heads=heads, depth=1, block_size=length).blocks[0].eval()
            ported = T.Block(width, heads, numerics=numerics, rngs=nnx.Rngs(seed))
            load_torch_state(ported, numpy_state(block), transpose_linear=transpose)
            for batch in BATCHES:
                generator = torch.Generator().manual_seed(seed + batch)  # G3's input convention
                x = torch.randn(batch, length, width, generator=generator) + offset
                with torch.no_grad():
                    reference = block(x)[0].numpy()
                if precision:
                    with jax.default_matmul_precision(precision):
                        out = np.asarray(ported(jnp.asarray(x.numpy())))
                else:
                    out = np.asarray(ported(jnp.asarray(x.numpy())))
                d = disagreement(reference, out)
                per[f"batch{batch}"]["relative"].append(d["relative"])
                per[f"batch{batch}"]["max_abs"].append(d["max_abs"])
                if name == "matched":
                    # PyTorch against itself: explicit softmax vs SDPA, the noise floor.
                    attention = block.attention
                    with torch.no_grad():
                        h = block.norm1(x)
                        explicit = attention(h)[0]
                        q, k, v = attention.qkv(h).chunk(3, dim=-1)
                        q, k, v = (
                            t.view(batch, length, heads, width // heads).transpose(1, 2) for t in (q, k, v)
                        )
                        fused = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                        fused = attention.projection(fused.transpose(1, 2).reshape(batch, length, width))
                    sdpa_floor.append(disagreement(explicit.numpy(), fused.numpy())["relative"])
        rows[name] = {
            b: {metric: {"per_seed": vals, **summarise(vals)} for metric, vals in m.items()} for b, m in per.items()
        }
        print(
            f"[block] {name:<36} rel b1 {rows[name]['batch1']['relative']['median']:.2e}  "
            f"b4 {rows[name]['batch4']['relative']['median']:.2e}  "
            f"max|d| b4 {rows[name]['batch4']['max_abs']['median']:.2e}",
            flush=True,
        )
    return {
        "settings": {"width": width, "heads": heads, "length": length, "seeds": SEEDS, "batches": BATCHES},
        "arms": rows,
        "torch_explicit_vs_sdpa_attention_relative": summarise(sdpa_floor),
    }


def run_models(torch, transformer_lm) -> dict:
    """Logits parity for R2's configurations, against the SDPA forward R2 used."""
    configs = [
        ("transformer-4M learned, T=256", (256, 5, 4), 256, "learned"),
        ("transformer-4M rope, T=256", (256, 5, 4), 256, "rope"),
        ("transformer-25M rope, T=1024", (512, 8, 8), 1024, "rope"),
    ]
    rows = {}
    for name, (width, depth, heads), length, positional in configs:
        per = {(attn, b): [] for attn in ("explicit", "fused") for b in BATCHES}
        params = None
        for seed in SEEDS:
            torch.manual_seed(seed)
            reference_model = transformer_lm(width, depth, heads, length, positional=positional).eval()
            params = sum(p.numel() for p in reference_model.parameters())
            state = numpy_state(reference_model)
            for attn in ("explicit", "fused"):
                ported = T.CharTransformer(
                    256, width, heads, depth, length, positional=positional,
                    numerics=T.Numerics(attention=attn), rngs=nnx.Rngs(seed),
                )
                load_torch_state(ported, state)
                forward = nnx.jit(lambda m, ids: m(ids))
                for batch in BATCHES:
                    generator = torch.Generator().manual_seed(seed + batch)
                    ids = torch.randint(0, 256, (batch, length), generator=generator)
                    with torch.no_grad():
                        reference = reference_model(ids)[0].numpy()
                    out = np.asarray(forward(ported, jnp.asarray(ids.numpy())))
                    per[(attn, batch)].append(disagreement(reference, out)["relative"])
        rows[name] = {
            "parameters": params,
            **{f"{attn}/batch{b}": {"per_seed": v, **summarise(v)} for (attn, b), v in per.items()},
        }
        print(
            f"[model] {name:<30} {params:>10,} params  explicit b4 {rows[name]['explicit/batch4']['median']:.2e}  "
            f"fused b4 {rows[name]['fused/batch4']['median']:.2e}",
            flush=True,
        )
    return rows


def run_gradients(torch, transformer_lm, width=256, depth=5, heads=4, length=256, batch=4) -> dict:
    """Next-byte cross-entropy and its gradient, 4M learned model, both frameworks."""
    loss_rel, grad_rel, worst_tensor = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        reference_model = transformer_lm(width, depth, heads, length).train()
        ported = T.CharTransformer(256, width, heads, depth, length, rngs=nnx.Rngs(seed))
        load_torch_state(ported, numpy_state(reference_model))
        generator = torch.Generator().manual_seed(seed + 100)
        ids = torch.randint(0, 256, (batch, length + 1), generator=generator)
        x, y = ids[:, :-1], ids[:, 1:]

        logits = reference_model(x)[0]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        torch_grads = {k: p.grad.numpy() for k, p in reference_model.named_parameters()}

        def loss_fn(model, x, y):
            return optax.softmax_cross_entropy_with_integer_labels(model(x), y).mean()

        jax_loss, grads = nnx.value_and_grad(loss_fn)(ported, jnp.asarray(x.numpy()), jnp.asarray(y.numpy()))
        # Map the gradient back through the same loader: load grads into a twin, read them as torch names.
        twin = T.CharTransformer(256, width, heads, depth, length, rngs=nnx.Rngs(0))
        load_torch_state(twin, torch_grads)
        twin_state = nnx.to_flat_state(nnx.state(twin, nnx.Param))
        grad_state = nnx.to_flat_state(grads)
        mine = {p: np.asarray(v.get_value()) for p, v in grad_state}
        ref = {p: np.asarray(v.get_value()) for p, v in twin_state}
        num = sum(float(np.sum((mine[p] - ref[p]) ** 2)) for p in ref)
        den = sum(float(np.sum(ref[p] ** 2)) for p in ref)
        per_tensor = {".".join(map(str, p)): disagreement(ref[p], mine[p])["relative"] for p in ref}
        name, value = max(per_tensor.items(), key=lambda kv: kv[1])
        loss_rel.append(abs(float(jax_loss) - loss.item()) / loss.item())
        grad_rel.append((num / den) ** 0.5)
        worst_tensor.append({"tensor": name, "relative": value})
    print(
        f"[grad]  loss rel {statistics.median(loss_rel):.2e}  grad rel (global) {statistics.median(grad_rel):.2e}  "
        f"worst tensor {worst_tensor[0]['tensor']} {worst_tensor[0]['relative']:.2e}",
        flush=True,
    )
    return {
        "settings": {"width": width, "depth": depth, "heads": heads, "length": length, "batch": batch},
        "loss_relative": {"per_seed": loss_rel, **summarise(loss_rel)},
        "grad_relative_global": {"per_seed": grad_rel, **summarise(grad_rel)},
        "worst_tensor_per_seed": worst_tensor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ml-ladder", default=str(Path(__file__).resolve().parents[3] / "ml-ladder"))
    parser.add_argument("--out", default="docs/data/r1-parity.json")
    args = parser.parse_args()

    root = Path(args.ml_ladder).resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    import subprocess

    import torch
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer
    from tools.gpu.c_context_scaling import transformer_lm

    ladder_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    started = time.time()
    report = {
        "experiment": "R1 parity harness",
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "flax": __import__("flax").__version__,
            "torch": torch.__version__,
            "device": str(jax.devices()[0]),
            "cpu": platform.processor() or platform.machine(),
            "ml_ladder_commit": ladder_commit,
        },
        "block": run_block(torch, make_transformer),
        "models": run_models(torch, transformer_lm),
        "gradients": run_gradients(torch, transformer_lm),
    }
    report["wall_clock_s"] = round(time.time() - started, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} in {report['wall_clock_s']} s")


if __name__ == "__main__":
    main()
