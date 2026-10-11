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
  `pyproject.toml` and `.python-version` pin; Kaggle's image was on 3.12.
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

- **CPU, here.** Unit tests, R1, O1, O2, O7, N7 and small prototypes.
- **Kaggle GPU (T4 ×2, 16 GB each, sm_75).** How ml-ladder ran its GPU work:
  - **Credentials** live outside any repo, in `~/.kaggle/` (`access_token`,
    `kaggle.json`) with mode 600. Never commit them. If the session has none,
    ask the user.
  - **Kernels.** Push with `kaggle kernels push -p DIR`, where `DIR` holds
    `kernel-metadata.json` (`"enable_gpu": true`, `"is_private": true`).
    Check progress with `kaggle kernels status` and download results with
    `kaggle kernels output`.
  - **Kernels are permanent** and the CLI cannot delete them, so pick names
    deliberately.
  - **Getting the code in.** The repos are private, so ml-ladder delivered its
    source as a private Kaggle *dataset*: a `git archive` tarball,
    `kaggle datasets version`, and `dataset_sources` in the kernel metadata.
  - **Where files go.**
    - Copy the source to `/kaggle/temp` and keep scratch files there.
    - Write only result JSON and logs to `/kaggle/working`, which gets
      published as kernel output.
    - Fetch enwik8 once, before starting parallel jobs.
  - **Use both GPUs.** Run two independent jobs at once, one per GPU, with
    `CUDA_VISIBLE_DEVICES=0` and `1`.
  - **Limits.** Two GPU kernels can run concurrently, and sessions last up to
    12 h.
  - **JAX on Kaggle has not been tested yet.** Do it first:
    - check whether the image ships a CUDA-enabled `jax`;
    - otherwise `pip install -U "jax[cuda12]"`;
    - then run the smoke test there.
- **TPU.** O5, N5 and N6 want one. Kaggle and Colab have offered TPU VMs, but
  this was not verified from this account. Find out how (or whether) a TPU
  kernel can be requested before scheduling M6.

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
