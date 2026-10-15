"""O1: autodiff internals — forward vs reverse mode, Hessian-vector products, rematerialisation.

Three experiments, each a subcommand that writes ``docs/data/o1-<name>.json``:

``jacobian``  Full Jacobians of one function family f: R^n -> R^m (a tanh MLP,
              hidden width 512, 4 hidden layers) with n*m fixed at 2^16 and n/m
              swept from 2^-8 to 2^8. ``jacfwd`` pushes n tangents forward,
              ``jacrev`` pulls m cotangents back; find where they cross, in
              wall-clock and in compiled FLOPs. Plus single jvp / vjp / forward
              costs at n = m, unbatched and at batch 256.
``hvp``       Hessian-vector products four ways on an MLP classifier loss:
              forward-over-reverse, reverse-over-reverse, reverse-over-forward
              (all exact in exact arithmetic), and finite differences of the
              gradient (central and one-sided) over a step-size sweep. Accuracy
              against a float64 forward-over-reverse reference; cost relative to
              one gradient, on the MLP and on R2's 4M transformer.
``remat``     ``jax.checkpoint`` on a 12-layer MLP and on the R2 transformer:
              what reverse mode saves (``saved_residuals``), the compiled
              program's peak and temp memory and FLOPs (``memory_analysis`` /
              ``cost_analysis``), and wall-clock. The 25M RoPE model at R2's
              real step (16 x 1024 tokens) is compiled, not run.
``plot``      Figures from the three JSON files into ``docs/figures/``.

Seeds 0-2 throughout; medians with min-max.

    uv run python experiments/o1_autodiff/run.py jacobian hvp remat plot
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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
FIGURES = ROOT / "docs" / "figures"
SEEDS = (0, 1, 2)


def environment() -> dict:
    import flax

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "flax": flax.__version__,
        "device": str(jax.devices()[0]),
        "cpu_count": __import__("os").cpu_count(),
        "x64": bool(jax.config.jax_enable_x64),
    }


def write(name: str, report: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"o1-{name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def mlp_params(key, n_in: int, n_out: int, hidden: int, depth: int, dtype=jnp.float32) -> list[jax.Array]:
    sizes = [n_in] + [hidden] * depth + [n_out]
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(k, (b, a), dtype) / math.sqrt(a)).astype(dtype)
        for k, a, b in zip(keys, sizes[:-1], sizes[1:], strict=True)
    ]


def mlp_apply(ws: list[jax.Array], x: jax.Array) -> jax.Array:
    """x: [..., n_in] -> [..., n_out]; tanh between layers, linear output.

    ``einsum`` rather than ``x @ w.T``: the explicit transpose would be traced as
    an op of its own and counted by ``saved_residuals`` as a saved weight copy,
    though XLA folds it into the matmul.
    """
    for w in ws[:-1]:
        x = jnp.tanh(jnp.einsum("...i,oi->...o", x, w))
    return jnp.einsum("...i,oi->...o", x, ws[-1])


# ---------------------------------------------------------------- jacobian


def crossover(log_ratio: list[float], log_cost_ratio: list[float]) -> float | None:
    """log2(n/m) where log(cost_fwd/cost_rev) crosses zero, by linear interpolation."""
    for (x0, y0), (x1, y1) in zip(
        zip(log_ratio, log_cost_ratio, strict=True), zip(log_ratio[1:], log_cost_ratio[1:], strict=True), strict=False
    ):
        if (y0 <= 0 <= y1) or (y1 <= 0 <= y0):
            return x0 + (0 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    return None


def run_jacobian(hidden: int = 512, depth: int = 4, log_total: int = 16) -> None:
    from jaxexp.bench import compiled_stats, residual_stats, summarise, time_call

    ks = list(range(4, log_total - 3))  # n = 2^k from 16 to 4096
    rows = []
    for k in ks:
        n, m = 2**k, 2 ** (log_total - k)
        per = {"fwd_s": [], "rev_s": [], "agree": []}
        stats = {}
        for seed in SEEDS:
            ws = mlp_params(jax.random.key(seed), n, m, hidden, depth)
            x = jax.random.normal(jax.random.key(seed + 1000), (n,))
            fwd = jax.jit(jax.jacfwd(mlp_apply, argnums=1))
            rev = jax.jit(jax.jacrev(mlp_apply, argnums=1))
            jf, jr = np.asarray(fwd(ws, x)), np.asarray(rev(ws, x))
            per["agree"].append(float(np.linalg.norm(jf - jr) / np.linalg.norm(jr)))
            per["fwd_s"].append(time_call(fwd, ws, x, budget_s=0.6)["median"])
            per["rev_s"].append(time_call(rev, ws, x, budget_s=0.6)["median"])
            if seed == SEEDS[0]:
                stats = {
                    "fwd": compiled_stats(jax.jacfwd(mlp_apply, argnums=1), ws, x),
                    "rev": compiled_stats(jax.jacrev(mlp_apply, argnums=1), ws, x),
                }
        row = {
            "n": n,
            "m": m,
            "log2_ratio": k - (log_total - k),
            "jacfwd_s": summarise(per["fwd_s"]),
            "jacrev_s": summarise(per["rev_s"]),
            "fwd_over_rev_time": summarise([f / r for f, r in zip(per["fwd_s"], per["rev_s"], strict=True)]),
            "jacfwd_flops": stats["fwd"]["flops"],
            "jacrev_flops": stats["rev"]["flops"],
            "jacfwd_temp_bytes": stats["fwd"]["temp_bytes"],
            "jacrev_temp_bytes": stats["rev"]["temp_bytes"],
            "fwd_rev_relative_disagreement": summarise(per["agree"]),
        }
        rows.append(row)
        print(
            f"[jac] n={n:5d} m={m:5d}  jacfwd {row['jacfwd_s']['median'] * 1e3:8.2f} ms  "
            f"jacrev {row['jacrev_s']['median'] * 1e3:8.2f} ms  fwd/rev {row['fwd_over_rev_time']['median']:.3f}  "
            f"flops fwd/rev {row['jacfwd_flops'] / row['jacrev_flops']:.3f}",
            flush=True,
        )

    x_axis = [r["log2_ratio"] for r in rows]
    cross_time = [
        crossover(x_axis, [math.log2(r["jacfwd_s"]["per_seed"][i] / r["jacrev_s"]["per_seed"][i]) for r in rows])
        for i in range(len(SEEDS))
    ]
    cross_flops = crossover(x_axis, [math.log2(r["jacfwd_flops"] / r["jacrev_flops"]) for r in rows])

    # Single products at n = m: one forward, one jvp, one vjp (forward + pullback), per batch size.
    products = {}
    width = 2 ** (log_total // 2)
    for batch in (1, 256):
        per = {name: [] for name in ("forward", "jvp", "vjp", "linearize_apply", "vjp_weights_and_input")}
        stats = {}
        residuals = None
        for seed in SEEDS:
            ws = mlp_params(jax.random.key(seed), width, width, hidden, depth)
            shape = (width,) if batch == 1 else (batch, width)
            x = jax.random.normal(jax.random.key(seed + 1000), shape)
            t = jax.random.normal(jax.random.key(seed + 2000), shape)

            def f(x, ws=ws):
                return mlp_apply(ws, x)

            def jvp(x, t, f=f):
                return jax.jvp(f, (x,), (t,))

            def vjp(x, t, f=f):
                y, pullback = jax.vjp(f, x)
                return y, pullback(t)

            def vjp_weights(ws, x, t):
                y, pullback = jax.vjp(mlp_apply, ws, x)
                return y, pullback(t)

            fns = {
                "forward": (f, (x,)),
                "jvp": (jvp, (x, t)),
                "vjp": (vjp, (x, t)),
                "vjp_weights_and_input": (vjp_weights, (ws, x, t)),
            }
            _, f_lin = jax.linearize(f, x)
            fns["linearize_apply"] = (f_lin, (t,))
            for name, (fn, args) in fns.items():
                per[name].append(time_call(jax.jit(fn), *args, budget_s=0.4)["median"])
                if seed == SEEDS[0]:
                    stats[name] = compiled_stats(fn, *args)
            if seed == SEEDS[0]:
                residuals = residual_stats(f, x)
        products[f"batch{batch}"] = {
            "n": width,
            "m": width,
            **{
                name: {
                    "seconds": summarise(per[name]),
                    "time_vs_forward": summarise([a / b for a, b in zip(per[name], per["forward"], strict=True)]),
                    "flops": stats[name]["flops"],
                    "flops_vs_forward": stats[name]["flops"] / stats["forward"]["flops"],
                }
                for name in per
            },
            "vjp_residuals": residuals,
        }
        p = products[f"batch{batch}"]
        print(
            f"[jac] batch {batch:3d}, n=m={width}: jvp {p['jvp']['time_vs_forward']['median']:.2f}x forward "
            f"({p['jvp']['flops_vs_forward']:.2f}x flops), vjp {p['vjp']['time_vs_forward']['median']:.2f}x "
            f"({p['vjp']['flops_vs_forward']:.2f}x flops), vjp keeps {residuals['bytes'] / 1024:.0f} KiB; "
            f"weights+input vjp {p['vjp_weights_and_input']['time_vs_forward']['median']:.2f}x "
            f"({p['vjp_weights_and_input']['flops_vs_forward']:.2f}x flops)",
            flush=True,
        )

    write(
        "jacobian",
        {
            "experiment": "O1 jvp vs vjp",
            "environment": environment(),
            "settings": {"hidden": hidden, "depth": depth, "n_times_m": 2**log_total, "seeds": SEEDS},
            "sweep": rows,
            "crossover_log2_ratio_time": summarise([c for c in cross_time if c is not None]),
            "crossover_log2_ratio_flops": cross_flops,
            "products": products,
        },
    )


# ---------------------------------------------------------------- hvp


def run_hvp() -> None:
    jax.config.update("jax_enable_x64", True)
    import optax
    from flax import nnx
    from jax.flatten_util import ravel_pytree

    from jaxexp import transformer as T
    from jaxexp.bench import compiled_stats, summarise, time_call

    n_in, hidden, classes, batch = 32, 128, 10, 512
    epsilons = [10.0**e for e in np.arange(-8, 0.25, 0.5)]

    def make(seed, dtype):
        # Draw in float64 and cast: the same key gives different numbers per dtype,
        # and the fp32 problem must be the fp64 reference's problem, rounded.
        ws = [w.astype(dtype) for w in mlp_params(jax.random.key(seed), n_in, classes, hidden, 1, jnp.float64)]
        x = jax.random.normal(jax.random.key(seed + 1000), (batch, n_in), jnp.float64).astype(dtype)
        y = jax.random.randint(jax.random.key(seed + 2000), (batch,), 0, classes)
        flat, unravel = ravel_pytree(ws)

        def loss(p):
            logits = mlp_apply(unravel(p), x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        v = jax.random.normal(jax.random.key(seed + 3000), flat.shape, jnp.float64)
        return loss, flat, (v / jnp.linalg.norm(v)).astype(dtype)

    def methods(loss):
        grad = jax.grad(loss)
        return {
            "forward-over-reverse": lambda p, v: jax.jvp(grad, (p,), (v,))[1],
            "reverse-over-reverse": lambda p, v: jax.grad(lambda q: jnp.vdot(grad(q), v))(p),
            "reverse-over-forward": lambda p, v: jax.grad(lambda q: jax.jvp(loss, (q,), (v,))[1])(p),
        }

    accuracy = {name: [] for name in ("forward-over-reverse", "reverse-over-reverse", "reverse-over-forward")}
    reference_check = []
    fd = {f"{kind} {bits}": [] for kind in ("central", "one-sided") for bits in ("fp32", "fp64")}
    dimension = None
    for seed in SEEDS:
        loss64, p64, v64 = make(seed, jnp.float64)
        exact = methods(loss64)["forward-over-reverse"](p64, v64)
        reference_check.append(
            float(jnp.linalg.norm(methods(loss64)["reverse-over-reverse"](p64, v64) - exact) / jnp.linalg.norm(exact))
        )
        dimension = int(p64.size)
        loss32, p32, v32 = make(seed, jnp.float32)
        for name, fn in methods(loss32).items():
            hv = jax.jit(fn)(p32, v32).astype(jnp.float64)
            accuracy[name].append(float(jnp.linalg.norm(hv - exact) / jnp.linalg.norm(exact)))
        for bits, (loss, p, v) in (("fp32", (loss32, p32, v32)), ("fp64", (loss64, p64, v64))):
            grad = jax.jit(jax.grad(loss))
            g0 = grad(p)
            central, one_sided = [], []
            for eps in epsilons:
                e = jnp.asarray(eps, p.dtype)
                gp, gm = grad(p + e * v), grad(p - e * v)
                c = ((gp - gm) / (2 * e)).astype(jnp.float64)
                o = ((gp - g0) / e).astype(jnp.float64)
                central.append(float(jnp.linalg.norm(c - exact) / jnp.linalg.norm(exact)))
                one_sided.append(float(jnp.linalg.norm(o - exact) / jnp.linalg.norm(exact)))
            fd[f"central {bits}"].append(central)
            fd[f"one-sided {bits}"].append(one_sided)

    fd_summary = {}
    for name, curves in fd.items():
        med = np.median(np.array(curves), axis=0)
        best = int(np.argmin(med))
        fd_summary[name] = {
            "epsilons": epsilons,
            "relative_error_median": med.tolist(),
            "relative_error_per_seed": curves,
            "best_epsilon": epsilons[best],
            "best_relative_error": float(med[best]),
        }
        print(f"[hvp] FD {name:<16} best eps {epsilons[best]:.0e}  rel err {med[best]:.2e}", flush=True)
    for name, vals in accuracy.items():
        print(f"[hvp] {name:<22} rel err (fp32) {np.median(vals):.2e}", flush=True)

    # Cost relative to one gradient: the MLP above (fp32), and R2's 4M transformer.
    def cost_rows(loss, p, v, label):
        grad = jax.grad(loss)
        fns = {"gradient": (grad, (p,))}
        fns.update({name: (fn, (p, v)) for name, fn in methods(loss).items()})
        fns["finite difference (central)"] = (lambda p, v: (grad(p + 1e-2 * v) - grad(p - 1e-2 * v)) / 2e-2, (p, v))
        out = {}
        for name, (fn, args) in fns.items():
            seconds = time_call(jax.jit(fn), *args, min_reps=10, budget_s=3.0)["median"]
            out[name] = {"seconds": seconds, **compiled_stats(fn, *args)}
        g = out["gradient"]
        for name, row in out.items():
            row["time_vs_gradient"] = row["seconds"] / g["seconds"]
            row["flops_vs_gradient"] = row["flops"] / g["flops"]
            row["temp_bytes_vs_gradient"] = row["temp_bytes"] / max(g["temp_bytes"], 1)
            print(
                f"[hvp] cost {label:<28} {name:<28} {row['time_vs_gradient']:.2f}x time  "
                f"{row['flops_vs_gradient']:.2f}x flops",
                flush=True,
            )
        return out

    cost = {"mlp": [], "transformer-4M": []}
    for seed in SEEDS:
        loss32, p32, v32 = make(seed, jnp.float32)
        cost["mlp"].append(cost_rows(loss32, p32, v32, f"MLP ({dimension:,} params)"))
        model = T.CharTransformer(256, 256, 4, 5, 128, rngs=nnx.Rngs(seed))
        graphdef, state = nnx.split(model)
        flat, unravel = ravel_pytree(state)
        ids = jax.random.randint(jax.random.key(seed), (4, 129), 0, 256).astype(jnp.int32)

        def tloss(p, unravel=unravel, graphdef=graphdef, ids=ids):
            m = nnx.merge(graphdef, unravel(p))
            return optax.softmax_cross_entropy_with_integer_labels(m(ids[:, :-1]), ids[:, 1:]).mean()

        v = jax.random.normal(jax.random.key(seed + 3000), flat.shape, jnp.float32)
        cost["transformer-4M"].append(cost_rows(tloss, flat, v / jnp.linalg.norm(v), "transformer-4M, T=128, b=4"))

    cost_summary = {
        model: {
            name: {
                "time_vs_gradient": summarise([r[name]["time_vs_gradient"] for r in rows]),
                "flops_vs_gradient": rows[0][name]["flops_vs_gradient"],
                "temp_bytes_vs_gradient": rows[0][name]["temp_bytes_vs_gradient"],
                "gradient_seconds": summarise([r["gradient"]["seconds"] for r in rows]),
            }
            for name in rows[0]
        }
        for model, rows in cost.items()
    }
    write(
        "hvp",
        {
            "experiment": "O1 Hessian-vector products",
            "environment": environment(),
            "settings": {
                "mlp": {"n_in": n_in, "hidden": hidden, "classes": classes, "batch": batch, "params": dimension},
                "transformer": {"width": 256, "depth": 5, "heads": 4, "length": 128, "batch": 4},
                "reference": "forward-over-reverse in float64",
                "seeds": SEEDS,
            },
            "reference_fwd_rev_vs_rev_rev_fp64": summarise(reference_check),
            "ad_relative_error_fp32": {name: summarise(vals) for name, vals in accuracy.items()},
            "finite_differences": fd_summary,
            "cost": cost_summary,
        },
    )


# ---------------------------------------------------------------- remat


def run_remat(width: int = 512, depth: int = 12, batch: int = 4096) -> None:
    import optax
    from flax import nnx

    from jaxexp import transformer as T
    from jaxexp.bench import compiled_stats, residual_stats, summarise, time_call

    policies = jax.checkpoint_policies

    def layer(w, x):
        return jnp.tanh(jnp.einsum("bi,oi->bo", x, w))

    def forward_plain(ws, x):
        for w in ws:
            x = layer(w, x)
        return x

    def forward_per_layer(policy=None):
        step = jax.checkpoint(layer, policy=policy)

        def forward(ws, x):
            for w in ws:
                x = step(w, x)
            return x

        return forward

    def forward_segments(size: int):
        @jax.checkpoint
        def segment(ws, x):
            for w in ws:
                x = layer(w, x)
            return x

        def forward(ws, x):
            for i in range(0, len(ws), size):
                x = segment(ws[i : i + size], x)
            return x

        return forward

    arms = {
        "no remat": forward_plain,
        "per layer, save nothing": forward_per_layer(),
        "per layer, save matmuls": forward_per_layer(policies.dots_saveable),
        "4 segments of 3 layers": forward_segments(3),
        "whole network": jax.checkpoint(forward_plain),
    }
    mlp_rows = {}
    reference_grads = {}
    for name, forward in arms.items():

        def loss(ws, x, forward=forward):
            return jnp.mean(forward(ws, x) ** 2)

        times, agree = [], []
        stats = residuals = None
        for seed in SEEDS:
            ws = [
                jax.random.normal(k, (width, width)) / math.sqrt(width)
                for k in jax.random.split(jax.random.key(seed), depth)
            ]
            x = jax.random.normal(jax.random.key(seed + 1000), (batch, width))
            grad = jax.jit(jax.grad(loss))
            g = grad(ws, x)
            if name == "no remat":
                reference_grads[seed] = g
            num = sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(g, reference_grads[seed], strict=True))
            den = sum(float(jnp.sum(b**2)) for b in reference_grads[seed])
            agree.append((num / den) ** 0.5)
            times.append(time_call(grad, ws, x, budget_s=1.5)["median"])
            if seed == SEEDS[0]:
                stats = compiled_stats(jax.grad(loss), ws, x)
                residuals = residual_stats(loss, ws, x)
        mlp_rows[name] = {
            "seconds": summarise(times),
            "grad_relative_vs_no_remat": summarise(agree),
            "residuals": residuals,
            **{k: stats[k] for k in ("flops", "peak_bytes", "temp_bytes", "argument_bytes", "output_bytes")},
            "total_bytes": stats["temp_bytes"] + stats["argument_bytes"] + stats["output_bytes"],
        }
    base = mlp_rows["no remat"]
    for name, row in mlp_rows.items():
        row["flops_vs_no_remat"] = row["flops"] / base["flops"]
        row["time_vs_no_remat"] = row["seconds"]["median"] / base["seconds"]["median"]
        print(
            f"[remat] MLP {name:<26} residuals {row['residuals']['bytes'] / 2**20:7.1f} MiB  "
            f"temp {row['temp_bytes'] / 2**20:7.1f} MiB  total {row['total_bytes'] / 2**20:7.1f} MiB  "
            f"flops {row['flops_vs_no_remat']:.3f}x  time {row['time_vs_no_remat']:.3f}x "
            f"({row['seconds']['min'] / base['seconds']['median']:.3f}-"
            f"{row['seconds']['max'] / base['seconds']['median']:.3f})",
            flush=True,
        )

    # The transformer: per-block remat via the model's own option.
    transformer_arms = {
        "no remat": dict(remat=False),
        "per block, save nothing": dict(remat=True, policy=None),
        "per block, save matmuls without batch dims": dict(
            remat=True, policy=policies.dots_with_no_batch_dims_saveable
        ),
    }

    def transformer_loss_factory(graphdef, kwargs):
        def loss(state, ids):
            model = nnx.merge(graphdef, state)
            logits = model(ids[:, :-1], **kwargs)
            return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]).mean()

        return loss

    transformer_rows = {}
    configs = [
        ("transformer-4M learned, T=256, batch 16", (256, 5, 4), 256, "learned", 16, True),
        ("transformer-25M rope, T=1024, batch 16 (R2's step; compiled only)", (512, 8, 8), 1024, "rope", 16, False),
    ]
    for label, (w, d, h), length, positional, b, timed in configs:
        transformer_rows[label] = {}
        for name, kwargs in transformer_arms.items():
            times = []
            stats = residuals = None
            for seed in SEEDS if timed else SEEDS[:1]:
                model = T.CharTransformer(256, w, h, d, length, positional=positional, rngs=nnx.Rngs(seed))
                graphdef, state = nnx.split(model)
                ids = jax.random.randint(jax.random.key(seed), (b, length + 1), 0, 256).astype(jnp.int32)
                loss = transformer_loss_factory(graphdef, kwargs)
                if timed:
                    times.append(time_call(jax.jit(jax.grad(loss)), state, ids, min_reps=3, budget_s=3.0)["median"])
                if seed == SEEDS[0]:
                    stats = compiled_stats(jax.grad(loss), state, ids)
                    residuals = residual_stats(loss, state, ids)
            row = {
                "residuals": residuals,
                **{
                    k: stats[k]
                    for k in ("flops", "peak_bytes", "temp_bytes", "argument_bytes", "output_bytes", "compile_s")
                },
                "total_bytes": stats["temp_bytes"] + stats["argument_bytes"] + stats["output_bytes"],
            }
            if timed:
                row["seconds"] = summarise(times)
            transformer_rows[label][name] = row
        base = transformer_rows[label]["no remat"]
        for name, row in transformer_rows[label].items():
            row["flops_vs_no_remat"] = row["flops"] / base["flops"]
            if timed:
                row["time_vs_no_remat"] = row["seconds"]["median"] / base["seconds"]["median"]
            print(
                f"[remat] {label[:24]:<24} {name:<42} residuals {row['residuals']['bytes'] / 2**20:8.1f} MiB  "
                f"temp {row['temp_bytes'] / 2**20:8.1f} MiB  total {row['total_bytes'] / 2**20:8.1f} MiB  "
                f"flops {row['flops_vs_no_remat']:.3f}x"
                + (
                    f"  time {row['time_vs_no_remat']:.3f}x "
                    f"({row['seconds']['min'] / base['seconds']['median']:.3f}-"
                    f"{row['seconds']['max'] / base['seconds']['median']:.3f})"
                    if timed
                    else ""
                ),
                flush=True,
            )

    # Where did the activations go? Same 12-layer MLP at 1024 wide, batch 512 (weights
    # outweigh activations): compile the gradient w.r.t. the weights (48 MiB of outputs)
    # and w.r.t. the input only (2 MiB). If temp grows by the activations when the
    # weight-gradient buffers are gone, XLA was keeping activations in them.
    reuse = {}
    ws = [jax.random.normal(k, (1024, 1024)) / math.sqrt(1024) for k in jax.random.split(jax.random.key(0), depth)]
    x = jax.random.normal(jax.random.key(1000), (512, 1024))

    def plain_loss(ws, x):
        return jnp.mean(forward_plain(ws, x) ** 2)

    for name, argnums in (("gradient w.r.t. weights", 0), ("gradient w.r.t. input only", 1)):
        st = compiled_stats(jax.grad(plain_loss, argnums=argnums), ws, x)
        reuse[name] = {k: st[k] for k in ("temp_bytes", "argument_bytes", "output_bytes", "peak_bytes")}
        print(
            f"[remat] buffer reuse, {name:<28} temp {st['temp_bytes'] / 2**20:5.1f} MiB  "
            f"outputs {st['output_bytes'] / 2**20:5.1f} MiB",
            flush=True,
        )
    reuse["saved_residuals_bytes"] = residual_stats(plain_loss, ws, x)["bytes"]

    write(
        "remat",
        {
            "experiment": "O1 rematerialisation",
            "environment": environment(),
            "settings": {"mlp": {"width": width, "depth": depth, "batch": batch}, "seeds": SEEDS},
            "mlp": mlp_rows,
            "transformer": transformer_rows,
            "xla_buffer_reuse_check": {"width": 1024, "depth": depth, "batch": 512, **reuse},
        },
    )


# ---------------------------------------------------------------- plot


def run_plot() -> None:
    import matplotlib.pyplot as plt

    from jaxexp.plotting import INK_SECONDARY, SERIES, save, style

    style()
    jac = json.loads((DATA / "o1-jacobian.json").read_text())
    rows = jac["sweep"]
    x = [r["log2_ratio"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, label, color in (
        ("jacfwd_s", "jacfwd (n tangents)", SERIES[0]),
        ("jacrev_s", "jacrev (m cotangents)", SERIES[1]),
    ):
        med = [r[key]["median"] * 1e3 for r in rows]
        lo = [r[key]["min"] * 1e3 for r in rows]
        hi = [r[key]["max"] * 1e3 for r in rows]
        left.plot(x, med, marker="o", color=color, label=label)
        left.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    left.set_yscale("log")
    left.set_xticks(x, [f"$2^{{{v}}}$" for v in x])
    left.set_xlabel("input / output dimension, n / m   (n · m = 65,536)")
    left.set_ylabel("ms per full Jacobian (median of seeds)")
    left.set_title("Full Jacobian: wall-clock")
    left.legend(loc="upper center")
    ratio_t = [r["fwd_over_rev_time"]["median"] for r in rows]
    ratio_f = [r["jacfwd_flops"] / r["jacrev_flops"] for r in rows]
    right.plot(x, ratio_t, marker="o", color=SERIES[0], label="wall-clock")
    right.plot(x, ratio_f, marker="s", color=SERIES[1], label="compiled FLOPs")
    right.axhline(1.0, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
    right.set_yscale("log", base=2)
    right.set_xticks(x, [f"$2^{{{v}}}$" for v in x])
    right.set_xlabel("input / output dimension, n / m")
    right.set_ylabel("jacfwd cost / jacrev cost")
    ct = jac["crossover_log2_ratio_time"]["median"]
    right.set_title(f"Crossover at n/m ≈ $2^{{{ct:.2f}}}$ (time)")
    right.legend(loc="upper left")
    save(fig, FIGURES / "o1_jvp_vjp.png")

    hvp = json.loads((DATA / "o1-hvp.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for name, color, style_ in (
        ("central fp32", SERIES[0], "-"),
        ("one-sided fp32", SERIES[1], "-"),
        ("central fp64", SERIES[0], (0, (4, 2))),
        ("one-sided fp64", SERIES[1], (0, (4, 2))),
    ):
        fdr = hvp["finite_differences"][name]
        left.plot(fdr["epsilons"], fdr["relative_error_median"], color=color, linestyle=style_, label=f"FD {name}")
    ad = hvp["ad_relative_error_fp32"]
    ad_median = max(v["median"] for v in ad.values())
    left.axhline(ad_median, color=SERIES[2], linewidth=1.6, label="AD, all three orders (fp32)")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("finite-difference step ε")
    left.set_ylabel("relative error vs float64 HVP")
    left.set_title("HVP accuracy")
    left.legend(fontsize=7.5, loc="lower right")
    names = list(hvp["cost"]["mlp"].keys())[1:]
    ys = np.arange(len(names))
    for j, (model, color) in enumerate((("mlp", SERIES[0]), ("transformer-4M", SERIES[1]))):
        vals = [hvp["cost"][model][n]["time_vs_gradient"]["median"] for n in names]
        bars = right.barh(
            ys + (j - 0.5) * 0.36, vals, height=0.34, color=color, label=model, edgecolor="white", linewidth=1
        )
        right.bar_label(bars, fmt="%.2f×", fontsize=7.5, padding=2, color=INK_SECONDARY)
    right.set_yticks(
        ys, [n.replace(" (central)", "\n(central)").replace("-over-", "-over-\n") for n in names], fontsize=8
    )
    right.invert_yaxis()
    right.set_xlabel("wall-clock per HVP / per gradient")
    right.set_title("HVP cost")
    right.set_xlim(0, 3.4)
    right.legend(loc="lower right")
    right.grid(axis="y", visible=False)
    save(fig, FIGURES / "o1_hvp.png")

    rem = json.loads((DATA / "o1-remat.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    mlp = rem["mlp"]
    names = list(mlp)
    ys = np.arange(len(names))
    temp = [mlp[n]["temp_bytes"] / 2**20 for n in names]
    bars = left.barh(ys, temp, height=0.6, color=SERIES[0], edgecolor="white")
    left.bar_label(
        bars,
        labels=[f"{t:.0f} MiB · {mlp[n]['flops_vs_no_remat']:.2f}× FLOPs" for t, n in zip(temp, names, strict=True)],
        fontsize=7.5,
        padding=3,
        color=INK_SECONDARY,
    )
    left.set_yticks(ys, [n.replace(", ", ",\n") for n in names], fontsize=8)
    left.invert_yaxis()
    left.set_xlim(0, max(temp) * 1.6)
    left.set_xlabel("compiled temp memory (MiB)")
    left.set_title("12-layer MLP (512 wide, batch 4096)")
    left.grid(axis="y", visible=False)
    label = next(k for k in rem["transformer"] if "25M" in k)
    tr = rem["transformer"][label]
    names = list(tr)
    resid = [tr[n]["residuals"]["bytes"] / 2**30 for n in names]
    temp = [tr[n]["temp_bytes"] / 2**30 for n in names]
    ys = np.arange(len(names))
    b1 = right.barh(ys - 0.18, resid, height=0.34, color=SERIES[0], label="saved residuals (jaxpr)", edgecolor="white")
    b2 = right.barh(ys + 0.18, temp, height=0.34, color=SERIES[1], label="compiled temp (XLA)", edgecolor="white")
    for bars in (b1, b2):
        right.bar_label(bars, fmt="%.2f GiB", fontsize=7.5, padding=2, color=INK_SECONDARY)
    short = {"per block, save matmuls without batch dims": "per block,\nsave matmuls"}
    right.set_yticks(ys, [short.get(n, n.replace(", ", ",\n")) for n in names], fontsize=8)
    right.invert_yaxis()
    right.set_xlim(0, max(resid + temp) * 1.18)
    right.set_xlabel("GiB")
    right.set_title("25M RoPE, T=1024, 16 sequences")
    right.legend(loc="lower right")
    right.grid(axis="y", visible=False)
    save(fig, FIGURES / "o1_remat.png")
    print("figures written to docs/figures/", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="+", choices=["jacobian", "hvp", "remat", "plot"])
    args = parser.parse_args()
    for part in args.parts:
        started = time.time()
        {"jacobian": run_jacobian, "hvp": run_hvp, "remat": run_remat, "plot": run_plot}[part]()
        print(f"[{part}] {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
