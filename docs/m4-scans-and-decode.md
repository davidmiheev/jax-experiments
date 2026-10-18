# M4 — Parallel scans and compiled decoding (N4, O4)

> **Status:** N4 done; O4 next ·
> **Hardware:** N4: Colab T4 (calibration, LRU sweeps) and Kaggle T4 ×2 (Mamba-2 sweeps, copy task), JAX 0.11.1; CPU (4 cores) for the equivalence tests ·
> **Data:** N4: [`data/n4-summary.json`](data/n4-summary.json) and [`data/n4/`](data/n4/) (one JSON and log per job); ml-ladder's references from its `docs/writeups/data/c-gpu.json`, `c-gpu-lr.json`, `c-gpu-lr2-256.json` and `c-gpu-lr3-gru-*.json` ·
> **Reproduce:** `uv run python tools/kaggle/push_r2.py <preset> --push` for `n4-calibrate`, `n4-lru`, `n4-mamba2` and `n4-copy`, then `uv run python experiments/n4_scans/analyze.py`. CPU tests: `uv run pytest tests/test_n4_ssm.py`

## N4 — Parallel scans: linear recurrences against the GRU and the transformer

### Question

In ml-ladder's Part C the GRU was the best 4M model at T=256 and T=1024, but
it walks the sequence one step at a time: cuDNN's GRU ran at 86k tokens/s at
T=1024 against the 4M transformer's 136k (both measured here, in JAX, on one T4).
Its state is fixed (two layers of 592 numbers), and fixed-state recurrences
cannot copy a long sequence: R3's LSTM (two layers of 512) fails the copy task
from n=128.

A *linear* recurrence keeps the fixed-size state but drops the nonlinearity
between steps, and that lets it train in parallel. Three questions:

- **Speed and memory.** Does a parallel scan give transformer-like (or better)
  training speed with the RNN's flat memory profile?
- **Quality.** At about 4M parameters and ml-ladder's protocol, where does it
  land between the GRU and the transformer in bits/byte, and does its loss
  keep falling with position like the transformer's, or flatten like the
  GRU's?
- **Copying.** Does it fail the copy task, as a fixed-state model should?

### The idea

**A linear recurrence is a prefix product.** The GRU's update
$h_t = f(h_{t-1}, x_t)$ is nonlinear in $h_{t-1}$, so step $t$ needs step
$t-1$ first. A diagonal linear recurrence

$$h_t = a_t \odot h_{t-1} + b_t,$$

with $a_t$ and $b_t$ computed from the input alone, is different. Write each
step as the pair $(a_t, b_t)$, a map $h \mapsto a_t h + b_t$. Composing two
such maps gives another one:

$$(a_1, b_1) \bullet (a_2, b_2) = (a_1 a_2,\; a_2 b_1 + b_2).$$

This operator is associative, so all $T$ prefixes can be computed by a
parallel scan in $O(\log T)$ depth and $O(T)$ work (`lax.associative_scan`).
The nonlinearity moves out of the recurrence, into the layers between
recurrences.

**Three models**, each about 4M parameters, sharing the byte-level shell of
ml-ladder's language models (embedding, residual blocks, LayerNorm, linear head):

- **LRU** (Orvieto et al. 2023). A complex diagonal recurrence with a fixed
  transition $\lambda = \exp(-e^{\nu} + i e^{\theta})$, $|\lambda|$ initialised
  in $[0.9, 0.999]$. The input is normalised by $\sqrt{1 - |\lambda|^2}$.
  Each block is an LRU mixer and a GELU MLP. The transition does not depend
  on the input, so the LRU decides *how long* to remember per channel, but
  not *what*.
- **Mamba-1 (S6)** (Gu & Dao 2023). The transition depends on the input:
  $a_t = \exp(\Delta_t A)$ and $b_t = \Delta_t B_t x_t$, with $\Delta_t$,
  $B_t$, $C_t$ computed from $x_t$ ("selection"). The state is
  [channels × 16] per position. A causal depthwise convolution precedes it
  and a SiLU gate follows.
- **Mamba-2 (SSD)** (Dao & Gu 2024). The same selection, but with a *scalar*
  decay per head. That turns the scan into matrix products in chunks of 64
  positions: an attention-like masked product inside each chunk, and a
  short recurrence across chunks.

**Why Mamba-2 is here.** Mamba-1 without its fused CUDA kernel materialises a
[batch, T, 512, 16] state for the scan. The parallel scan then stores
$O(\log T)$ copies of it for the backward pass. It ran, but at 13.7k tokens/s
and 2 GiB with remat, and 12.4 GiB of temp memory without. That is a
statement about the missing kernel, not about selective SSMs. Mamba-2's chunked
form uses the matmul units that a T4 has instead, so it is the fair
selective-SSM arm for the quality and copy comparisons.

### The math

**Cost per training step**, for $T$ positions and state width $N$ per channel:

| | work | sequential depth | activations stored |
| --- | --- | --- | --- |
| GRU (`lax.scan` / cuDNN) | $O(T)$ | $T$ | $O(T)$ |
| linear scan, sequential | $O(T)$ | $T$ | $O(T)$ |
| linear scan, associative | $O(T)$ | $O(\log T)$ | $O(T)$ (a few times more) |
| SSD, chunk $Q$ | $O(TQ + T N)$ | $T / Q$ (across chunks) | $O(T Q)$ per head |
| attention | $O(T^2)$ | $1$ | $O(T^2)$ without a fused kernel |

At a fixed 16,384 tokens per step (ml-ladder's protocol), the batch shrinks
as $T$ grows. A recurrence's cost per token is then independent of $T$; a
sequential one's depth grows with $T$; attention's cost per token grows
linearly in $T$.

**Why a fixed state cannot copy.** In the copy task, the model reads $n$
random symbols from a 32-symbol vocabulary and must reproduce them. At the
separator it must carry $n \log_2 32 = 5n$ bits in its state. The LRU's state
is 256 complex numbers per layer, but each is a leaky average of the past, and
storing a sequence in it needs distinct, slowly decaying channels per
position. Nothing stops a linear model from learning a delay line for small
$n$; the question is where the state runs out.

### What we measure and why

- **Equivalence (CPU).** The associative scan equals a Python loop, for real
  and complex inputs; scan modes agree in value and gradient; chunked SSD
  equals the step-by-step recurrence (with padding, one chunk, and ragged
  lengths); all models are causal.
- **Speed and memory (T4).** Training tokens/s and XLA's compiled temp memory at
  T = 256, 1024 and 4096, fp16 matmuls, 16,384 tokens per step. Arms: the 4M
  transformer (fused attention), the GRU (`lax.scan`), LRU (associative and
  sequential), Mamba-1 (associative and sequential, remat) and Mamba-2
  (chunked and sequential).
- **Quality (T4).** enwik8 bits/byte on ml-ladder's protocol (3,000 steps of
  16,384 bytes, AdamW β=(0.9, 0.95), clipping at 1.0, 5% warmup then cosine
  to 10%; 262,144 validation bytes), at T=256 and T=1024. Weight decay is 0.1
  for the new arms and the transformer, and 0 for the GRU (ml-ladder's
  settings).
  Each new arm gets its own LR sweep on seed 0 over {1e-3, 2e-3, 4e-3, 8e-3},
  then seeds 0–2 at its best LR. The baselines are ml-ladder's own runs at
  their tuned LRs: GRU 2e-3 at both lengths; transformer 2e-3 at T=1024 and
  8e-3 at T=256.
- **Loss by position.** Bits at each position in the window, median over
  seeds. Its summary is the *context gain*: the mean bits at positions
  $[T/8, T/4)$ minus those in the last quarter. It is what the second half of
  the window buys.
- **Copying.** R3's task: n = 32, 128, 512, 1,500 steps, second-half accuracy
  on 512 held-out sequences, seeds 0–2, each model at 2e-3 as in R3.

### What I built

- **`jaxexp.ssm`** — the three models as pure functions over dicts of arrays.
  - `linear_scan(a, b, mode)` runs the recurrence as `associative`
    (`lax.associative_scan`) or `sequential` (`lax.scan`).
  - `ssd(x, dta, b, c, chunk)` is Mamba-2's chunked form; `ssd_sequential` is
    the reference recurrence. `_segsum` builds the in-chunk decay matrix as a
    masked cumulative sum (a difference of two cumulative sums loses
    precision), with $-\infty$ above the diagonal so `exp` gives exact zeros.
    A test checks that no NaN from those entries reaches the gradients.
  - `apply(params, ids, kind, scan=..., compute_dtype=..., remat=...)`.
    `compute_dtype` applies to the dense projections only; recurrences,
    gates, LayerNorms and the residual stream stay in fp32.
- **Runners.** `experiments/r2_enwik8/run.py` and
  `experiments/n1_vmap_sweep/run.py` take `--model lru|mamba|mamba2` and
  `--scan`, so the LR sweeps run as vmapped groups (N1) on ml-ladder's
  batches for each seed, and the copy task runs on R3's code unchanged.
  `experiments/n4_scans/run.py` measures throughput and memory against $T$;
  `analyze.py` builds the tables and figure.
- **Tests.** `tests/test_n4_ssm.py`, 13 tests (above).

### Results

RESULTS

### What surprised me

SURPRISES

### Open threads

THREADS
