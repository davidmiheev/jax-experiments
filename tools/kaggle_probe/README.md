# Kaggle accelerator probe

Checks whether a Kaggle image can run this repo's plan, and records the answer
as JSON:
- which Python and JAX/Flax/Optax versions the image ships, and what JAX sees as
  devices;
- whether a reinstall is needed (`jax[cuda12]` or `jax[tpu]`), done only if the
  preinstalled JAX cannot see the accelerator;
- matmul throughput in fp32/bf16/fp16, and fp32 at default vs highest matmul
  precision;
- `lax.scan` vs `lax.associative_scan`, an NNX + Optax train step, and `vmap`
  over K model copies;
- Pallas: an elementwise kernel and a small matmul kernel;
- `jax.sharding` across every device.

`probe_body.py` expects three names to be defined before it:

```
TARGET = 'gpu'            # or 'tpu'; names the output file jax-probe-<TARGET>.json
ACCEL_BACKEND = 'gpu'     # the jax.default_backend() that counts as "accelerator found"
JAX_EXTRA = 'jax[cuda12]' # what to pip install if it is not found ('jax[tpu]' for TPU)
```

Build a kernel file by prepending those lines to `probe_body.py`. Push it with a
`kernel-metadata.json` whose `machine_shape` is `NvidiaTeslaT4` (with
`"enable_gpu": true`) or `TpuV5E8` (with `"enable_tpu": true`). See
`CLAUDE.md`, "Compute", for why `TpuV5E8` and not the SDK's documented
`Tpu1VmV38`.
