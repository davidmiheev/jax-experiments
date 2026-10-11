"""Pallas on a Kaggle T4 (sm_75): the default GPU lowering refuses pre-Hopper cards;
does the Triton backend (compiler_params=pallas.triton.CompilerParams) work?"""
import json, time, traceback
from pathlib import Path
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

out = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}
tri = pltriton.CompilerParams(num_warps=4, num_stages=1)

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def mm_kernel(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

for name, fn in (
    ("add_default", lambda x: pl.pallas_call(add_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x, x)),
    ("add_triton", lambda x: pl.pallas_call(add_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), compiler_params=tri)(x, x)),
):
    try:
        x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
        out[name] = bool(jnp.allclose(fn(x), 2 * x))
    except Exception as e:
        out[name + "_error"] = repr(e)[:600]

try:
    a = jax.random.normal(jax.random.key(0), (128, 128), jnp.float32)
    r = pl.pallas_call(mm_kernel, out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32), compiler_params=tri)(a, a)
    out["matmul_triton_max_abs_err"] = float(jnp.max(jnp.abs(r - a @ a)))
except Exception as e:
    out["matmul_triton_error"] = repr(e)[:600]

# A blocked fp16 matmul over a grid, timed against XLA: the shape N5 needs.
try:
    n, bm = 2048, 64
    def blocked(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)
    a = jax.random.normal(jax.random.key(1), (n, n), jnp.float16)
    b = jax.random.normal(jax.random.key(2), (n, n), jnp.float16)
    f = jax.jit(lambda a, b: pl.pallas_call(
        blocked, out_shape=jax.ShapeDtypeStruct((n, n), jnp.float16), grid=(n // bm, n // bm),
        in_specs=[pl.BlockSpec((bm, n), lambda i, j: (i, 0)), pl.BlockSpec((n, bm), lambda i, j: (0, j))],
        out_specs=pl.BlockSpec((bm, bm), lambda i, j: (i, j)), compiler_params=tri)(a, b))
    ref = jax.jit(lambda a, b: a @ b)
    def timed(g):
        g(a, b).block_until_ready(); t = time.perf_counter()
        for _ in range(10): r = g(a, b)
        r.block_until_ready(); return (time.perf_counter() - t) / 10
    out["blocked_fp16_rel_err"] = float(jnp.linalg.norm((f(a, b) - ref(a, b)).astype(jnp.float32)) / jnp.linalg.norm(ref(a, b).astype(jnp.float32)))
    out["blocked_fp16_pallas_tflops"] = round(2 * n**3 / timed(f) / 1e12, 2)
    out["blocked_fp16_xla_tflops"] = round(2 * n**3 / timed(ref) / 1e12, 2)
except Exception as e:
    out["blocked_error"] = repr(e)[:800]

Path("/kaggle/working/jax-probe-pallas-gpu.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
