# JAX experiments: the plan

This repo picks up where [ml-ladder](https://github.com/davidmiheev/ml-ladder)
(private) left off. ml-ladder built classical ML through generative models
from scratch in NumPy and PyTorch, ran each rung as a measured experiment, and
ended with a list of open threads. Many of those threads are exactly the things
JAX makes cheap or makes visible:
- forward-mode autodiff, Hessian-vector products and rematerialisation;
- per-example gradients and `vmap` over seeds or hyperparameters;
- whole training loops compiled into one program;
- parallel scans;
- custom kernels (Pallas);
- TPUs and sharding.

The plan has three tracks:

- **Track R — reproduce.** Re-run a handful of ml-ladder's headline results in
  JAX. Two reasons: a second implementation is the strongest check that a
  number is real (ml-ladder's PaddleOCR-VL agreed between candle and
  transformers to 16 digits; its whisper did not, and that disagreement found
  three bugs); and every later experiment needs a trusted JAX baseline.
- **Track O — open threads.** ml-ladder threads that JAX answers better than
  PyTorch did.
- **Track N — new.** Experiments that only make sense in JAX, or that the
  ladder never reached.

Every experiment states its question, what it reuses, what it measures and
what counts as done. Order of work is at the end. Compute: CPU for anything
marked *CPU*; Kaggle T4×2 or a TPU for the rest (see `CLAUDE.md`).

---

## Reference numbers from ml-ladder

These are the targets for Track R. All are medians over seeds 0–2, from
`ml-ladder/docs/writeups/` and the JSON beside each writeup.

| result | value | source |
| --- | --- | --- |
| enwik8 bits/byte, 25M transformer, learned positions, 12,000 steps × 16,384 tokens, T=256 / 1024 | 1.448 / 1.423 | `c-sequence-attention.md` §4, `data/c-gpu-long.json` |
| same with RoPE | **1.399 / 1.325** | §5, `data/c-gpu-rope-*.json` |
| GRU (592 hidden × 2 layers, ~4.5M), same budget, T=256 / 1024 | 1.627 / 1.598 | §4 |
| 4M transformer (256 wide, 5 layers, 4 heads), 12,000 steps, T=1024, peak LR 2e-3 | 1.544 | §6, `data/c-gpu-4m-long.json` |
| LR sweep at 3000 steps: 4M transformer best LR, T=256 / T=1024 | ≥8e-3 (1.631) / 2e-3 (1.825) | §7, `data/c-gpu-lr*.json` |
| LR sweep: GRU best LR, both contexts | 2e-3 (1.701 / 1.687) | §7 |
| BPE (4096 tokens, 2.89 bytes/token), 3000 steps, 25M transformer, 256 tokens | 1.424 bits/byte | §8, `data/c-gpu-bpe.json` |
| copy task: transformer 4M exact recall at n = 32/128/512; LSTM 4.5M | 1.000/1.000/0.999; 0.903 then chance | §2 |
| throughput on a T4 at T=1024 (PyTorch, fp16 autocast): 4M transformer / cuDNN recurrences | 254k / 182–202k tokens/s | §3 |
| RoPE throughput cost of ml-ladder's unfused fp32 rotation, 25M at T=1024 | 44k vs 60k tokens/s (−27%) | §5 |
| AMP on a T4 (G2): fp16 vs fp32, 384-wide transformer | 2.54× faster; bf16 **0.77×** (slower) | `g-scaling-systems.md` G2 |
| fp16 speedup vs width 64 → 512 | 1.25× → 2.99× | G2 |
| PyTorch→Candle port of one transformer block (G3): exact GELU / tanh GELU relative error | 8×10⁻⁸ / 4×10⁻⁵ | G3 |

The protocol behind those numbers:
- **Data:** enwik8, the first 90M bytes for training and the next 5M for
  validation.
- **Batching:** 16,384 tokens per step.
- **Optimiser:** AdamW with β=(0.9, 0.95) and weight decay 0.1 for transformers
  (0 for recurrences), gradient clipping at 1.0, 5% linear warmup, then cosine
  decay to 10% of peak.
- **Precision:** fp16 autocast.
- **Evaluation:** 256 KiB of held-out bytes in non-overlapping windows; loss is
  also recorded at every position in the window.

ml-ladder's `tools/gpu/c_context_scaling.py` is the reference implementation:
models, schedule, evaluation, the BPE tokenizer with its byte-exactness check,
and RoPE.

---

## Track R — reproduce

### R1. Parity harness: one transformer block, PyTorch vs JAX *(CPU)*

- **Question.** Does a Flax block compute the same function as C3's PyTorch
  block, and which silent mismatches would the comparison catch?
- **Reuses.** ml-ladder G3's method: export weights to `.npz`, run both
  implementations on the same input, and report relative error and max
  |diff| at batch 1 *and* batch 4. Batch 4 catches layout bugs that batch 1
  hides.
- **Measures.** Relative error with the same GELU variant, and with the other
  variant on purpose (G3: 8×10⁻⁸ vs 4×10⁻⁵). LayerNorm epsilon conventions.
  The SDPA path against the explicit softmax. Also compare
  `jax.default_matmul_precision`: on GPU, JAX's default may use TF32, which
  bites on Ampere+ but not on a T4.
- **Done when.** A pytest asserts parity below 1e-6 with matched settings, and
  a table records the size of each mismatch.

### R2. enwik8 at scale in JAX *(GPU)*

- **Question.** Do the Part C numbers reproduce in a second framework, and how
  do XLA's fusion and `lax.scan` change the cost picture?
- **Arms.**
  - The 25M transformer with RoPE at T=1024, 12,000 steps; the target is 1.325.
  - The GRU at T=1024, written with `lax.scan`. The target is 1.598. Compare
    its throughput with cuDNN's 182k tokens/s.
  - The 4M transformer at LR 2e-3; the target is 1.544.
- **Measures.**
  - Bits/byte and loss by position against the reference table.
  - Tokens per second, compile time, and steady state measured separately.
  - Peak memory.
- **Done when.** Each arm lands within its seed range (about ±0.01) of the
  reference, or the gap is explained.
- **Watch for.**
  - RNG: per-step data sampling has to be reproducible, so key-split
    discipline matters.
  - Mixed precision: Flax has no autocast. Decide explicitly which ops run in
    fp16 and keep the loss and softmax in fp32.

### R3. The copy task *(GPU, cheap)*

- **Question.** Does R2's transformer still copy 512 symbols back perfectly,
  and does a `lax.scan` LSTM still fail at 128?
- **Why.** It is a correctness test for attention masking and for state
  handling in the scan that costs minutes. Keep it as a regression test for
  every later sequence model (N4 needs it).

---

## Track O — ml-ladder's open threads, answered in JAX

### O1. Autodiff internals: forward mode, HVPs, rematerialisation *(CPU)*

- **ml-ladder thread (B1).** Checkpointing (recompute instead of store),
  forward-mode for Jacobian-vector products, Hessian-vector products for
  second-order methods.
- **Experiments.**
  - `jvp` vs `vjp` cost as a function of the input/output dimension ratio, on
    one function family. Find the crossover.
  - HVPs three ways: forward-over-reverse, reverse-over-reverse, and finite
    differences. Measure accuracy and cost.
  - `jax.checkpoint` (remat) on a 12-layer MLP and on the R2 transformer:
    memory saved against extra compute, read from `jax.make_jaxpr` and the
    compiled HLO, not only from a profiler.
- **Done when.** There is a plot for each experiment, and a sentence saying
  where the textbook claim is and isn't true on real hardware.

### O2. Newton's method where ml-ladder used first order *(CPU)*

- **ml-ladder threads (A1, A4).** IRLS / Newton for logistic regression, "to
  watch second-order information pay off", and boosting with a real Newton
  step on log-loss, where $h_i = p_i(1-p_i)$, which A4 never implemented.
- **Experiments.**
  - Logistic regression by `jax.hessian` + solve against gradient descent:
    iterations and wall-clock to a fixed tolerance.
  - Gradient-boosted trees where the leaf values come from autodiff gradients
    and Hessians of an arbitrary loss. Check it against XGBoost's leaf values
    on log-loss.

### O3. Optimisers beyond Adam *(CPU → GPU)*

- **ml-ladder thread (B5).** Lion and Shampoo, and whether second-moment
  scaling is the right primitive at all.
- **Experiment.** Using `optax` (adamw, lion, and a Shampoo-family optimiser
  such as `optax.contrib` or `distributed_shampoo` if available), run R2's 4M
  transformer at T=256 for 3000 steps, with each optimiser's learning rate
  swept by N1's vmap. Compare bits/byte at the tuned LR, and steps to a target
  loss.
- **Rule.** Every optimiser gets its own LR sweep. ml-ladder §7 showed that an
  untuned baseline produces a fake winner.

### O4. KV-cached decoding, compiled *(GPU)*

- **ml-ladder thread (C3).** KV caching, the difference between O(T) and
  O(T²) per sampled token.
- **Context.** The candle fork's static-decode work, preallocated KV buffers
  replayed in a CUDA graph, is the same idea as a jitted decode step with
  static shapes (davidmiheev/candle#3 and davidmiheev/candle#4).
- **Experiment.** Decode from R2's transformer three ways: no cache
  (recompute), a cache via `lax.dynamic_update_slice` in a jitted step, and
  the whole generation loop inside `lax.fori_loop`. Measure ms/token against
  context length, compile time, and cache memory. Check against C5's
  bandwidth arithmetic, the bytes moved per token.

### O5. Mixed precision where bf16 is native *(TPU, plus the T4 for contrast)*

- **ml-ladder thread (G2).** On a T4, bf16 was **slower** than fp32 (0.77×),
  and "the bf16 result would invert on an A100".
- **Experiment.** G2's five arms re-run in JAX on a T4 and on a TPU:
  - fp32;
  - fp16 with loss scaling (`jmp`, or hand-written);
  - bf16;
  - fp16 without scaling;
  - an accumulation-count sweep.

  On the T4, does XLA's bf16 behave like PyTorch's emulated path? On the TPU,
  bf16 is the native format. Repeat G2's speedup-vs-width curve.
  Already measured by the probe: on a T4, XLA's bf16 matmul runs at fp32
  speed (4.5 TFLOP/s, against fp16's 24.9), not slower as PyTorch's was. So
  the T4 arm's open question is end-to-end training, not the matmul.

### O6. A scaling law on tokens *(GPU)*

- **ml-ladder thread (G1).** "A tokenized corpus large enough that the optimum
  data budget is free to move, at which point the Chinchilla exponent becomes
  measurable."
- **Context.** ml-ladder §8 now has a byte-exact BPE tokenizer on enwik8. G1
  had about one decade of data and ran out.
- **Experiment.** IsoFLOP curves for transformers from about 1M to 30M
  parameters on BPE enwik8, with RoPE. Fit $L = E + A/N^\alpha + B/D^\beta$
  with uncertainty ranges, and check that the minimum is interior. N1 gives
  the per-size LR tuning that G1 found essential.
- **Done when.** The exponent comes with an uncertainty range, or there is an
  honest statement that the data still doesn't pin it down.

### O7. Front-end parity: mel spectrograms, four ways *(CPU)*

- **ml-ladder thread (I).** candle's whisper decoders disagree with
  transformers on about 20 of 61 clips. The shared suspect is the mel front
  end.
- **Experiment.** A pure `jnp` log-mel. Diff it stage by stage against:
  - torchaudio;
  - transformers' `WhisperFeatureExtractor`;
  - candle's `pcm_to_mel`, through the candle fork's `examples/whisper` or a
    tiny Rust bin.

  Stages: framing, window, FFT, power, mel, log, clamp. Locate the first stage
  where candle departs.
- **Done when.** The disagreeing stage is named with its size. If it's small,
  the next suspect is ruled in or out.

---

## Track N — new, JAX-native

### N1. Hyperparameter sweeps as one program: `vmap` over learning rates and seeds *(GPU)*

- **Question.** ml-ladder §7's sweep took about 45 separate runs. Can one
  vmapped program train all of them at once, and how much faster is it?
- **How.** Stack the parameters and optimiser states for K copies, then
  `vmap` the train step over the stack, with the LR as a vmapped argument
  (optax `inject_hyperparams`). Data can be shared or vmapped. Reproduce §7's
  curves (4M transformer, T=1024, LRs 3e-4…8e-3) in one run.
- **Measures.** The same bits/byte as the sequential runs, within seed noise.
  Throughput for K = 1, 2, 4, 8 against K sequential runs. Find where memory
  stops it.
- **Becomes.** The sweep tool for O3, O6 and N2.

### N2. Gradient noise scale and the critical batch size *(GPU)*

- **ml-ladder thread (B5).** "The learning-rate/batch-size scaling rule, a
  scaling-law question."
- **How.** Per-example gradients via `vmap(grad)`, which costs one line in
  JAX. Estimate the simple noise scale
  $B_\text{simple} = \operatorname{tr}\Sigma / |G|^2$ (McCandlish et al. 2018)
  through training on R2's models.
- **Question.** Does the measured critical batch size predict where doubling
  the batch stops halving the steps? Test it with 2–3 batch sizes.

### N3. Learning the learning rate: hypergradients through training *(GPU, small)*

- **Question.** N1 finds a good LR by search. Can `jax.grad` through an
  unrolled training segment find it by gradient instead?
- **How.** Differentiate validation loss with respect to the (log) LR schedule
  through K steps of training, for K = 10…100, on the 4M transformer at T=256.
  Compare with the sweep's optimum (≥8e-3 at T=256 per §7). Watch for
  short-horizon bias, the known failure where hypergradients favour greedy,
  small LRs.

### N4. Parallel scans: linear RNNs and SSMs vs the GRU and the transformer *(GPU)*

- **Question.** The GRU lost long context because its state is fixed-size and
  it must walk the sequence step by step. A diagonal linear recurrence (an
  LRU, or Mamba-style selective SSM) runs in parallel via
  `lax.associative_scan`. Does it get the transformer's speed with the RNN's
  memory profile? And does it still fail the copy task, as theory predicts
  for fixed-state models?
- **Arms.** An LRU and a selective SSM, about 4M parameters, against R2's GRU
  and 4M transformer: enwik8 at T=256/1024, and the R3 copy task at
  n = 32/128/512.
- **Measures.**
  - Bits/byte;
  - loss by position: does the curve stay flat like the GRU's, or keep
    falling like the transformer's?
  - copy accuracy;
  - tokens/s against sequence length: associative scan, sequential scan, and
    attention.

### N5. A fused RoPE / attention kernel in Pallas *(TPU; Kaggle's GPUs cannot run Pallas)*

- **ml-ladder thread (§5).** RoPE cost 27% throughput because the rotation was
  recomputed unfused in fp32 in every layer. Is that inherent?
- **How.** A Pallas kernel that applies the rotation inside the attention
  kernel, or that just fuses rotate-q/k. Compare with plain `jnp` under XLA,
  which may already fuse most of it.
- **Hardware (checked).** Pallas does not run on a T4. The default GPU
  lowering refuses pre-Hopper cards, and the Triton backend needs sm_80+,
  while the T4 is sm_75 (`tools/kaggle_probe/results/`). Develop with
  `interpret=True` on CPU for correctness, and time the kernel on the TPU
  (Mosaic). The XLA baseline can still be measured on the T4.
- **Done when.** A RoPE throughput cost is measured with the fused kernel, and
  the plain-XLA result says whether the kernel was needed at all.

### N6. Data and model parallelism with `jax.sharding` *(TPU)*

- **Question.** How close to linear is scaling R2's 25M transformer across
  8 TPU cores with pure data parallelism, versus 2-way model × 4-way data
  (tensor-parallel attention and MLP)?
- **Measures.** Step time and tokens/s against device count, the
  communication share from the profiler, and the largest model that fits per
  layout.
- **Depends on.** A TPU being available (see `CLAUDE.md`, "Compute").

### N7. The empirical neural tangent kernel *(CPU → GPU, small)*

- **Question.** For small MLPs on a small task, for example ml-ladder's B-part
  data or MNIST-scale, how well does the infinite-width NTK prediction track
  finite networks as width grows? When does feature learning break the
  agreement?
- **How.** `jax.jacfwd`/`jacrev` for the empirical NTK, or the
  `neural-tangents` library if it supports the installed JAX. Use the kernel's
  closed-form predictions, the empirical NTK at init and after training, and
  real training curves.

### N8. Diffusion samplers *(GPU)*

- **ml-ladder threads (E3).** DDIM's deterministic sampler, $T = 1000$,
  v-prediction.
- **Experiment.** E3's DDPM in Flax. Arms: DDPM sampling vs DDIM at
  10/20/50/100 steps; ε-, x₀- and v-prediction; linear vs cosine schedule at
  T=1000. Score with the Fréchet distance E3 used, so the numbers line up.
- **JAX angle.** The sampler loop goes inside `lax.scan`, so the whole
  sampling run compiles to one program. Measure ms per sample against step
  count.

---

## Order of work

Each milestone is small enough to finish and write up on its own.

| milestone | experiments | where | rough cost |
| --- | --- | --- | --- |
| M0 | scaffold, smoke tests, R1 parity harness | CPU | half a day |
| M1 | O1 autodiff internals, O2 Newton/IRLS | CPU | 1 day |
| M2 | R2 enwik8 reproduction (GRU via scan, 4M, 25M+RoPE), R3 copy task | Kaggle T4 | ~6 GPU-hours (the 25M at 12,000 steps took 55–75 min per seed in PyTorch) |
| M3 | N1 vmapped sweeps, then O3 optimisers, N2 noise scale | Kaggle T4 | ~4 GPU-hours |
| M4 | N4 scans vs GRU vs transformer, O4 KV-cached decode | Kaggle T4 | ~4 GPU-hours |
| M5 | O7 mel parity (CPU), N7 NTK (CPU) | CPU | 1 day |
| M6 | O5 precision, N5 Pallas, N6 sharding | TPU (+T4 contrast) | depends on TPU access |
| M7 | O6 scaling law on BPE, N3 hypergradients, N8 diffusion | Kaggle T4 | ~8 GPU-hours |

Start with M0 → M1 → M2. M2's GRU and transformer are the baselines nearly
everything after them compares against. If M2 cannot reproduce the reference
numbers, stop and find out why before going further.

## How results are reported

The same standard as ml-ladder; see `CLAUDE.md` for details.
- **Repeat runs.** Use seeds 0–2 and report medians with min–max ranges.
- **Keep the raw data.** Every run writes JSON, and it is committed beside the
  writeup that uses it.
- **Name what was measured.** Say what was run and what was not: hardware,
  steps, tokens, and wall-clock.
- **Compare tuned against tuned.** A comparison between a tuned and an untuned
  arm is not a result (ml-ladder §7).
- **Say what's open.** A writeup per milestone ends with "What surprised me"
  and "Open threads".
