"""Probe whether this Kaggle image can run the jax-experiments plan, and record everything as JSON.

Each step runs in a fresh subprocess, so a reinstall is seen by the next import, and
a failure in one check is recorded instead of ending the probe.
"""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

WORK = Path("/kaggle/working")
report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "target": TARGET}


def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return {"rc": r.returncode, "out": (r.stdout + r.stderr)[-4000:]}


def py(code, timeout=1800):
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip().splitlines()
    try:
        return json.loads(out[-1]) | {"rc": r.returncode}
    except Exception:
        return {"rc": r.returncode, "error": (r.stdout + r.stderr)[-4000:]}


def save():
    (WORK / f"jax-probe-{TARGET}.json").write_text(json.dumps(report, indent=2))


PROBE = """
import json, platform
out = {"python": platform.python_version()}
try:
    import jax
    out.update(jax=jax.__version__, backend=jax.default_backend(),
               devices=[str(d) for d in jax.devices()], device_count=jax.device_count())
except Exception as e:
    out["jax_error"] = repr(e)[:800]
for m in ("jaxlib", "flax", "optax", "libtpu", "torch"):
    try:
        mod = __import__(m); out[m] = getattr(mod, "__version__", "present")
    except Exception as e:
        out[m] = None
print(json.dumps(out))
"""

BENCH = """
import json, time
import jax, jax.numpy as jnp
out = {"backend": jax.default_backend(), "device_count": jax.device_count(), "device_kind": jax.devices()[0].device_kind}

def timed(f, *args, reps=10):
    f(*args).block_until_ready()  # compile + warm
    t = time.perf_counter()
    for _ in range(reps):
        r = f(*args)
    r.block_until_ready()
    return (time.perf_counter() - t) / reps

# 1. matmul throughput by dtype (one device)
n = 4096
mm = jax.jit(lambda a, b: a @ b)
for name, dt in (("fp32", jnp.float32), ("bf16", jnp.bfloat16), ("fp16", jnp.float16)):
    try:
        a = jax.random.normal(jax.random.key(0), (n, n), dt); b = jax.random.normal(jax.random.key(1), (n, n), dt)
        s = timed(mm, a, b)
        out[f"matmul_{name}_tflops"] = round(2 * n**3 / s / 1e12, 2)
    except Exception as e:
        out[f"matmul_{name}_error"] = repr(e)[:300]
for prec in ("default", "highest"):
    try:
        with jax.default_matmul_precision(prec):
            a = jax.random.normal(jax.random.key(0), (n, n), jnp.float32)
            s = timed(jax.jit(lambda a: a @ a), a)
        out[f"matmul_fp32_precision_{prec}_tflops"] = round(2 * n**3 / s / 1e12, 2)
    except Exception as e:
        out[f"matmul_fp32_precision_{prec}_error"] = repr(e)[:300]

# 2. scan (the GRU path) and associative_scan (the SSM path)
try:
    def step(h, x):
        h = jnp.tanh(h @ W + x); return h, h
    W = jax.random.normal(jax.random.key(2), (512, 512)) * 0.05
    xs = jax.random.normal(jax.random.key(3), (1024, 64, 512))
    f = jax.jit(lambda xs: jax.lax.scan(step, jnp.zeros((64, 512)), xs)[1])
    t = time.perf_counter(); f(xs).block_until_ready(); out["scan_compile_s"] = round(time.perf_counter() - t, 2)
    out["scan_1024x64x512_ms"] = round(timed(f, xs, reps=5) * 1e3, 2)
    a = jax.random.uniform(jax.random.key(4), (1024, 64, 512)); b = xs
    g = jax.jit(lambda a, b: jax.lax.associative_scan(lambda l, r: (l[0] * r[0], r[0] * l[1] + r[1]), (a, b))[1])
    out["associative_scan_1024x64x512_ms"] = round(timed(g, a, b, reps=5) * 1e3, 2)
except Exception as e:
    out["scan_error"] = repr(e)[:500]

# 3. an NNX + Optax train step, jitted
try:
    import optax
    from flax import nnx
    class MLP(nnx.Module):
        def __init__(self, rngs):
            self.l1 = nnx.Linear(512, 2048, rngs=rngs); self.l2 = nnx.Linear(2048, 512, rngs=rngs)
        def __call__(self, x):
            return self.l2(nnx.gelu(self.l1(x)))
    model = MLP(nnx.Rngs(0)); opt = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
    x = jax.random.normal(jax.random.key(5), (256, 512))
    @nnx.jit
    def train_step(model, opt, x):
        loss, grads = nnx.value_and_grad(lambda m: jnp.mean((m(x) - x) ** 2))(model)
        opt.update(model, grads); return loss
    l0 = float(train_step(model, opt, x))
    for _ in range(50): l = train_step(model, opt, x)
    out["nnx_optax_loss_first_last"] = [round(l0, 4), round(float(l), 4)]
except Exception as e:
    out["nnx_error"] = repr(e)[:500]

# 4. vmap over K model copies (the N1 sweep idea): per-copy step time vs K
try:
    def loss(w, x): return jnp.mean(jnp.tanh(x @ w) ** 2)
    upd = jax.jit(jax.vmap(lambda w, x, lr: w - lr * jax.grad(loss)(w, x), in_axes=(0, None, 0)))
    x = jax.random.normal(jax.random.key(6), (1024, 1024))
    for k in (1, 4, 8):
        ws = jax.random.normal(jax.random.key(7), (k, 1024, 1024)) * 0.03
        out[f"vmap_K{k}_ms"] = round(timed(upd, ws, x, jnp.linspace(1e-3, 1e-2, k)) * 1e3, 3)
except Exception as e:
    out["vmap_error"] = repr(e)[:500]

# 5. Pallas: an elementwise kernel and a small matmul kernel
try:
    from jax.experimental import pallas as pl
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    r = pl.pallas_call(add_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x, x)
    out["pallas_add_ok"] = bool(jnp.allclose(r, 2 * x))
except Exception as e:
    out["pallas_add_error"] = repr(e)[:800]
try:
    from jax.experimental import pallas as pl
    def mm_kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)
    a = jax.random.normal(jax.random.key(8), (128, 128), jnp.float32)
    r = pl.pallas_call(mm_kernel, out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32))(a, a)
    out["pallas_matmul_max_abs_err"] = float(jnp.max(jnp.abs(r - a @ a)))
except Exception as e:
    out["pallas_matmul_error"] = repr(e)[:800]

# 6. sharding across every device: data-parallel matmul with a NamedSharding mesh
try:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    devs = jax.devices(); mesh = Mesh(devs, ("data",))
    x = jax.device_put(jnp.ones((len(devs) * 1024, 1024)), NamedSharding(mesh, P("data")))
    w = jax.device_put(jnp.ones((1024, 1024)) * 0.001, NamedSharding(mesh, P()))
    f = jax.jit(lambda x, w: jnp.sum(jnp.tanh(x @ w)))
    out["sharded_sum"] = float(f(x, w)); out["sharded_ms"] = round(timed(f, x, w) * 1e3, 3)
    out["sharding_devices"] = len(devs)
except Exception as e:
    out["sharding_error"] = repr(e)[:800]
print(json.dumps(out))
"""

report["image"] = {
    "accelerator": sh("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader; ls /dev | grep -i -E 'accel|vfio' | head"),
    "pip": sh("pip list 2>/dev/null | grep -i -E '^(jax|jaxlib|jax-cuda|flax|optax|libtpu|torch |tensorflow )'"),
    "os": sh("cat /etc/os-release | head -2; nproc; free -g | head -2"),
}
report["preinstalled"] = py(PROBE)
save()
print("preinstalled:", report["preinstalled"], flush=True)

if report["preinstalled"].get("backend") != ACCEL_BACKEND:
    report["install"] = sh(f'pip install -q -U "{JAX_EXTRA}" flax optax 2>&1 | tail -15')
    report["after_install"] = py(PROBE)
    save()
    print("after install:", report["after_install"], flush=True)

report["bench"] = py(BENCH, timeout=3000)
report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
print(json.dumps(report["bench"], indent=1), flush=True)
