# jax-experiments

My experiments with JAX and its derivatives (Flax, Optax, Pallas).

This repo follows up [ml-ladder](https://github.com/davidmiheev/ml-ladder), a
from-scratch tour of machine learning in which every rung is a measured
experiment. The plan has three tracks:

- **Reproduce.** Re-run ml-ladder's headline results in a second framework,
  for example enwik8 at 1.325 bits/byte with a 25M RoPE transformer.
- **Open threads.** Answer ml-ladder's open threads that JAX makes cheap:
  forward-mode autodiff and HVPs, Newton methods, KV-cached decoding,
  bf16 on TPUs, a scaling law on BPE tokens, and mel front-end parity.
- **New.** JAX-native studies:
  - vmapped hyperparameter sweeps;
  - gradient noise scale and the critical batch size;
  - hypergradients;
  - parallel-scan SSMs against the GRU and the transformer;
  - Pallas kernels;
  - sharding;
  - the empirical NTK;
  - diffusion samplers;
  - LoRA and QLoRA built from scratch (NF4 included).

- [`PLAN.md`](PLAN.md): the experiments, the reference numbers, and the order
  of work.
- [`CLAUDE.md`](CLAUDE.md): the working brief (environment, compute,
  standards).

```
uv sync          # jax 0.11, flax 0.12 (nnx), optax 0.2; Python >= 3.12
uv run pytest    # smoke tests: jit, grad, jvp, vmap, scan, an NNX + Optax step
```
