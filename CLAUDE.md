# Working in jax-experiments

Read `PLAN.md` first. It is the work order: three tracks of experiments, a
milestone table, and the reference numbers from ml-ladder that Track R must
reproduce. This file is the practical brief for picking it up.

## Where things stand (handed over 2026-10-11)

- **Code.** Only the scaffold exists: `pyproject.toml`, the `jaxexp` package
  and a smoke test. No experiment has been run yet. Start at milestone M0.
- **Prior JAX use.** None. JAX was not installed in any environment of the
  originating session, and no repo of this account had imported it before.
- **Verified versions.** On 2026-10-11 these installed from PyPI and passed
  the smoke test on CPU (Python 3.12): `jax[cpu]` 0.11.2, `flax` 0.12.10 (the
  `flax.nnx` API is available) and `optax` 0.2.8.
- **Setup.** `uv sync` installs them; `uv run pytest` runs the smoke test
  (5 tests, passing at handover). JAX 0.11 needs **Python ≥ 3.12**, which
  `pyproject.toml` and `.python-version` pin. Kaggle's GPU image runs 3.13,
  which satisfies it (see Compute).
  There is no GPU in the cloud container; GPU work goes to Kaggle (below).

## The repo this builds on

[davidmiheev/ml-ladder](https://github.com/davidmiheev/ml-ladder) (private) is
the source of every reference number and of the reference implementations.
Attach it to the session to read it. The files that matter most:

- `tools/gpu/c_context_scaling.py` — the Part C GPU study:
  - the models (C3's transformer with learned positions or RoPE, and cuDNN
    recurrences);
  - the training loop: AdamW with β=(0.9, 0.95), clipping at 1.0, 5% warmup
    then cosine to 10%, fp16 autocast;
  - per-position evaluation;
  - the enwik8 loader and split (90M train, next 5M validation);
  - the byte-exact BPE tokenizer.

  R2 must match this protocol to make its comparison meaningful.
- `parts/c_sequence_attention/c3_transformer_from_scratch.py` — the block R1
  ports.
- `docs/writeups/c-sequence-attention.md` sections 1–8 under "Part C at
  scale", `g-scaling-systems.md` (G1–G3) and `i-speech.md` — the narratives
  behind the numbers in `PLAN.md`.
- `docs/writeups/data/*.json` — raw results, including per-position losses.
  Compare against these, not against rounded numbers in prose.

## Compute

Checked from this account on 2026-10-11 with Kaggle CLI 2.2.4.

- **CPU, here.** Unit tests, R1, O1, O2, O7, N7 and small prototypes.
- **Weekly quota** (`kaggle quota`): **30 h GPU and 20 h TPU**, resetting each
  week (the 2026-10-17 00:00 UTC refresh was the next one at handover). Other
  sessions of this account share it, and use GPU heavily.
- **Session limits** (from the server's push errors): at most **2 GPU** and
  **1 TPU** batch sessions at a time, per account. A queued kernel already
  holds its slot.
- **Machine shapes.** Set `"machine_shape"` in `kernel-metadata.json`, or pass
  `kaggle kernels push --accelerator`:
  - `NvidiaTeslaT4` gives T4 ×2 (16 GB each, sm_75), the GPU ml-ladder used.
  - `TpuV5E8` gives a TPU v5e-8, with `"enable_tpu": true`. This is the shape
    every recently run public TPU notebook records. The SDK's docstring still
    lists `Tpu1VmV38` (TPU v3-8), which is stale.
  - **Wrong shape names are not rejected.** The server stores any string and
    the kernel sits in `QUEUED` forever: this happened with `Tpu1VmV38` and
    with a deliberately invalid name. If a kernel stays queued much longer
    than expected, check the shape against a recent public notebook
    (`kaggle kernels pull <ref> -p <dir> -m`) before assuming a hardware
    queue.
- **Kernels.** Push with `kaggle kernels push -p DIR`, check with
  `kaggle kernels status`, and download with `kaggle kernels output`.
  `kaggle kernels logs` is available too. `kaggle kernels delete -y <ref>`
  removes a kernel (CLI 2.2.4); earlier CLI versions had no delete.
- **Credentials** live outside any repo, in `~/.kaggle/` (`access_token`,
  `kaggle.json`) with mode 600. Never commit them. If the session has none,
  ask the user.
- **Getting the code in.** The repos are private, so ml-ladder delivered its
  source as a private Kaggle *dataset*: a `git archive` tarball,
  `kaggle datasets version`, and `dataset_sources` in the kernel metadata.
- **Where files go.**
  - Copy the source to `/kaggle/temp` and keep scratch files there.
  - Write only result JSON and logs to `/kaggle/working`, which gets
    published as kernel output.
  - Fetch enwik8 once, before starting parallel jobs.
- **Use both GPUs.** On T4 ×2, run two independent jobs at once with
  `CUDA_VISIBLE_DEVICES=0` and `1`. GPU sessions last up to 12 h.
- **JAX on Kaggle.** `tools/kaggle_probe/` checks an image end to end:
  versions, devices, dtypes, scan, NNX/Optax, `vmap`, Pallas and sharding.
  **Results on the T4 image** (`results/jax-probe-gpu.json` and
  `results/jax-probe-pallas-gpu.json`, run on 2026-10-11):
  - **JAX is preinstalled and CUDA-enabled.** The image has Python **3.13**,
    `jax` / `jaxlib` 0.11.1 with `jax-cuda12-plugin`, `flax` 0.12.10, `optax`
    0.2.8, `torch` 2.11 (cu128) and TensorFlow 2.20. `jax.devices()` shows
    `cuda:0` and `cuda:1`. Nothing needs installing, but note the version
    skew: locally the lock pins 0.11.2 on Python 3.12.
  - **Matmul, 4096², one T4.** fp32 4.5 TFLOP/s, **fp16 24.9** (5.5×), and
    **bf16 4.5**, the same as fp32: no tensor-core path. PyTorch's bf16 on this
    card ran at 0.77× fp32 (ml-ladder G2), so XLA's bf16 is no slower than
    fp32 here, and no faster either. For speed on a T4 use fp16 with loss
    scaling.
  - **Scans.** `lax.associative_scan` 5.3 ms against sequential `lax.scan`
    23.8 ms on a 1024 × 64 × 512 recurrence (N4's premise holds).
  - **`vmap` over model copies.** K = 8 costs 4.4× K = 1 (N1's premise holds).
  - **Working:** NNX + Optax training steps, and `jax.sharding` across both
    GPUs.
  - **Pallas does not run on a T4 at all.** The default GPU lowering refuses
    pre-Hopper cards ("emit_pipeline on pre-Hopper GPUs only supports
    input-only pipelines"), and the Triton backend
    (`compiler_params=pallas.triton.CompilerParams(...)`) needs compute
    capability 8.0+, while the T4 is 7.5. Kaggle's other GPU, the P100, is
    older still. So N5 runs on the TPU, or with `interpret=True` for
    correctness only.
**TPU v5e-8: quota confirmed, hardware not yet obtained.** A *script*
  kernel with `TpuV5E8` on the default image sat in `QUEUED` for 4.5 h
  (16:23–20:48 UTC, 2026-10-11) without consuming any TPU quota. Every public
  TPU notebook checked is a *notebook* kernel, and some pin the TPU-VM image
  `gcr.io/kaggle-private-byod/python-tpuvm@sha256:a2111cb9…`. So the probe was
  re-pushed in that form, as `davidmiheev/jax-probe-tpu-nb` (notebook,
  `TpuV5E8`, that image; full digest in `tools/kaggle_probe/tpu_notebook/`).
  It was still queued at 01:14 UTC on 2026-10-12, 4.5 h later. Together with
  the script version, that is **about 9 h with no TPU session started and
  0.00 h of TPU quota used**. The cause is unknown: there may be no v5e
  capacity for batch kernels, or this account may not be eligible for batch
  (non-interactive) TPU sessions. Things to try:
  - start a TPU notebook interactively in the Kaggle web UI, which tells the
    two cases apart;
  - use Colab's TPU as a fallback for O5, N5 and N6.

  Until a TPU session has actually run, treat M6 as unscheduled.
  - **Collect it.** Check `kaggle kernels status davidmiheev/jax-probe-tpu-nb`.
    When it reports COMPLETE, run
    `kaggle kernels output davidmiheev/jax-probe-tpu-nb -p tools/kaggle_probe/results/`.
  - **It holds the slot.** While queued it holds the account's single TPU
    slot, so delete it before pushing a different TPU kernel.

## Standards (from ml-ladder)

- **Seeds and reporting.** Use seeds 0–2 and report medians with min–max.
  Every run writes JSON with its settings, and the JSON is committed beside
  the writeup that cites it.
- **Writeup shape.** One writeup per milestone: Question, The idea, The math,
  What we measure and why, What I built, Results, What surprised me, Open
  threads.
- **Verified vs not.** Say explicitly what was run and what was not.
  Numbers quoted from someone else's commit or paper are labelled as such.
- **Arithmetic.** Check every number in prose against the JSON before
  committing. ml-ladder lost time to claims that were plausible and wrong.
- **Fair comparisons.** Tune every arm. ml-ladder's "GRU wins at every
  context" was half an artefact of the transformer's untuned learning rate.
- **Hold the setup still.** When an experiment changes one thing, keep
  everything else at the reference protocol.
- **Attribution.** Trace provenance (`git log`, upstream history) before
  saying whose code something is.
- **Hygiene.** Keep credentials and the private ignite-catalog source out of
  this repo. Do not put model identifiers in commits or PR text.

## Layout

```
PLAN.md            the experiment plan (work order)
CLAUDE.md          this brief
pyproject.toml     uv project: jax, flax, optax; optional [cuda] extra
src/jaxexp/        shared code (models, training loop, data) as it is written
tests/             pytest; test_smoke.py checks jit/grad/vmap and the Flax/Optax APIs
```

As milestones land, add `experiments/<id>_<name>/` for each experiment's
scripts and `docs/` for writeups, with data under `docs/data/` and figures
under `docs/figures/`.
