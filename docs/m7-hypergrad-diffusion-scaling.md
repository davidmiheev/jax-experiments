# M7 — Hypergradients, diffusion samplers, a scaling law (N3, N8, O6)

> **Status:** N3 in progress; N8 and O6 to follow ·
> **Hardware:** Kaggle T4 ×2 (one job per GPU), JAX 0.11.1; CPU (4 cores) for the tests ·
> **Data:** N3: [`data/n3-summary.json`](data/n3-summary.json) and [`data/n3/`](data/n3/) (one JSON and log per job; the first run, whose landscape batches were not yet keyed, in `data/n3/first-run/`) ·
> **Reproduce:** N3: `uv run python tools/kaggle/push_r2.py <preset> --push` for `n3b` and `n3c`, then `uv run python experiments/n3_hypergrad/analyze.py`. CPU tests: `uv run pytest tests/test_n3_hypergrad.py`

## N3 — Learning the learning rate: hypergradients through training

### Question

N1 found the 4M transformer's best learning rate by search: 18 runs, one per
(LR, seed). Training is a differentiable function of its learning rate, so
`jax.grad` can, in principle, find that rate by gradient instead. Two
questions:

- **Does the hypergradient point at the tuned LR?** Starting from a point in
  training and looking K steps ahead, which LR does the gradient prefer, and
  how does it compare with the LR the tuned schedule uses at that point?
- **Can it set the LR online?** Train 3,000 steps with the LR updated by its
  own hypergradient every K steps, and compare the final bits/byte with the
  tuned schedule.

The literature predicts a specific failure: **short-horizon bias**.
Hypergradients over short horizons favour small, greedy learning rates,
because a small step reduces the loss soonest even when a larger one would
reach a lower loss later (Wu et al. 2018).

### The idea

**Training is a function, so it has a derivative.** K steps of AdamW map
$(\theta_0, \eta_1, \dots, \eta_K)$ to $\theta_K$. The validation loss after
them, $L_{\text{val}}(\theta_K)$, is a scalar function of the learning rates,
and reverse mode differentiates it through every step: through the forward
and backward passes of each step (a second derivative of the model), through
Adam's moments and bias correction, through the clipping and the weight decay.

**The meta-loss.** Its gradient with respect to $\log \eta$ (log, so that a
step changes the LR by a factor) is the hypergradient. Its zero crossing is
the LR that minimises the loss K steps later: the *K-step-greedy* LR.

**What it costs.** Reverse mode through K steps stores what each step's
backward pass needs. With `jax.checkpoint` around every step, it stores only
the carry ($\theta$, $m$, $v$: three copies of the parameters) per step and
recomputes the rest, so memory grows with K times 50 MB for the 4M model
rather than K times the activations.

### The math

One AdamW step, as ml-ladder's protocol runs it (clipping at global norm 1,
$\beta = (0.9, 0.95)$, $\epsilon = 10^{-8}$, decoupled weight decay
$\lambda = 0.1$):

$$g \leftarrow g \min\!\left(1, \tfrac{1}{\lVert g \rVert}\right), \quad
m \leftarrow \beta_1 m + (1-\beta_1) g, \quad
v \leftarrow \beta_2 v + (1-\beta_2) g^2,$$

$$\theta \leftarrow \theta - \eta \left(\frac{m / (1-\beta_1^t)}{\sqrt{v / (1-\beta_2^t)} + \epsilon} + \lambda \theta\right).$$

The hypergradient for the segment's shared LR is

$$\frac{\partial L_{\text{val}}(\theta_K)}{\partial \log \eta}
= \sum_{k=1}^{K} \eta_k \frac{\partial L_{\text{val}}(\theta_K)}{\partial \eta_k},$$

the sum of the per-step hypergradients; the code computes both. Each
$\partial \theta_K / \partial \eta_k$ propagates through the later steps'
Jacobians $\partial \theta_{j+1} / \partial \theta_j$, which contain the
Hessian of the training loss. That is where the second derivative enters.

**Why a short horizon prefers a small LR.** On a quadratic valley with
curvature $h$, gradient descent at rate $\eta$ shrinks the error along it by
$(1 - \eta h)$ per step. Over a short horizon the loss is dominated by the
highest-curvature directions, which a large $\eta$ overshoots (or destabilises),
while the low-curvature directions, where a large $\eta$ pays, contribute only
over many steps. The greedy optimum therefore sits below the long-run optimum,
and the shorter the horizon, the further below.

### What we measure and why

- **Equivalence (CPU).** The differentiable AdamW equals the training
  optimiser (`jaxexp.train.optimizer("adamw")`) step for step. The
  hypergradient equals a central finite difference (float64), and remat
  changes nothing.
- **The reference.** The tuned protocol in this setting: 3,000 steps of
  64 × 256 bytes, peak 8e-3 (the best LR at T=256 in ml-ladder's
  `c-gpu-lr2-256.json` and in O3), 5% warmup, cosine to 10%, fp32. It saves
  checkpoints at steps 500 and 1,500.
- **The landscape.** From step 0 and the two checkpoints, for K = 10, 30 and
  100: the meta-loss after K steps at nine constant LRs (2.5e-4 to 6.4e-2,
  factors of 2) and its hypergradient. The same K training batches and the
  same 64 held-out windows are used at every LR, so the curve is a
  deterministic function of the LR; they are keyed by (seed, start, K).
  The K-step-greedy LR is the hypergradient's zero crossing; it is compared
  with the schedule's mean LR over the same K steps.
- **Online.** 3,000 steps with no schedule. Every K = 30 steps, the
  hypergradient of the meta-loss after that segment (fresh held-out windows
  each time) updates $\log \eta$ by Adam (step 0.05). It starts from 1e-3
  (too small), 8e-3 (the tuned peak) and 3.2e-2 (too large). Final bits/byte
  is on the same 262,144 validation bytes as every other run.
- **Held-out windows.** The meta-objective's windows come from the validation
  split after its first 1,000,000 bytes, so they never overlap the bytes on
  which bits/byte is reported.

### What I built

- **`jaxexp.hypergrad`:**
  - `adamw_init` and `adamw_step`: ml-ladder's AdamW written out so the LR is a
    traced input. Its square root has a zero derivative at 0 (see below).
  - `make_unroll`: K steps in `lax.scan`, with `jax.checkpoint` on each step.
  - `make_meta_loss`: the validation loss after K steps, as a function of
    $\log \eta$ (one value, or one per step).
- **`experiments/n3_hypergrad/run.py`** (studies `reference`, `landscape`,
  `online`), `analyze.py`, and the Kaggle presets `n3b` and `n3c`.
- **Tests:** `tests/test_n3_hypergrad.py`, 4 tests.
  - **A NaN the tests found.** The attention key bias has a gradient of
    exactly zero, because softmax ignores a shift shared by all keys. Its
    Adam second moment is then exactly 0, the derivative of $\sqrt{v}$ there
    is infinite, and reverse mode multiplies it by zero: NaN in every
    hypergradient. The fix is a square root whose derivative at 0 is 0; the
    forward value is unchanged.
  - **Finite differences on the transformer agree only to 0.13%.** Even in
    float64, the transformer's meta-loss is not smooth below about $10^{-9}$,
    so the exact check uses a small attention-free model (agreement
    $3.5 \times 10^{-9}$). On the transformer, the test checks that reverse
    and forward mode agree.

### Results

N3RESULTS

### What surprised me

N3SURPRISES

### Open threads

N3THREADS

## N8 — Diffusion samplers: DDIM, x0 and v, and the cosine schedule's first step

### Question

E3 trained a 245k-parameter DDPM on MNIST at T = 400 and sampled it with
DDPM's ancestral sampler. Its open threads were DDIM's deterministic sampler,
T = 1000 and v-prediction. Three questions:

- **Does the JAX port reproduce E3?** Same U-Net, data subset, optimiser,
  schedules and sampler; scored by E3's own Fréchet distance.
- **How few steps can each sampler take?** DDPM against DDIM at 10 to 1,000
  steps, for ε-, $x_0$- and v-prediction, linear and cosine schedules, T = 1000.
- **What does a sampler cost when the whole loop is one compiled program?**

### The idea

**One network, three targets.** The forward process is
$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$. The
network can be trained to output ε (DDPM), $x_0$, or
$v = \sqrt{\bar\alpha_t}\, \epsilon - \sqrt{1-\bar\alpha_t}\, x_0$ (Salimans
and Ho 2022). Any one of them, with $x_t$, determines the other two, so every
sampler runs on every parameterisation.

**Two samplers on the same network.** DDPM's ancestral step adds fresh noise
at every step. DDIM with $\eta = 0$ is deterministic: it predicts $x_0$ and
re-noises it to the next retained timestep along the predicted ε. Both can
skip timesteps; E3 strides DDPM by recomputing the effective $\alpha$ between
retained steps.

### The math

From an ε-prediction, the clean image is

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}},$$

and DDIM's update to the previous retained timestep $s < t$ is

$$x_s = \sqrt{\bar\alpha_s}\, \hat x_0 + \sqrt{1-\bar\alpha_s}\, \hat\epsilon.$$

An error $\delta$ in $\hat\epsilon$ moves $x_s$ by

$$\left(\sqrt{\bar\alpha_s}\,\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}} - \sqrt{1-\bar\alpha_s}\right)\delta .$$

At the first step, $t = T - 1$. **The cosine schedule ends at
$\bar\alpha_T = 2.4 \times 10^{-9}$** (T = 1000; linear: $4.0 \times 10^{-5}$).
With 10 steps, the first jump goes to $s = 888$, where $\bar\alpha_s = 0.030$.
That multiplies $\delta$ by about 3,500 under the cosine schedule and by 1.9
under the linear one. At 1,000 steps the factor is 31 (cosine) against 0.01
(linear). ε-prediction is therefore fragile exactly where few-step sampling
needs it. $x_0$- and v-prediction never divide by $\sqrt{\bar\alpha_t}$.
Clipping $\hat x_0$ to the data range $[-1, 1]$, then recomputing
$\hat\epsilon$ from it, bounds the damage.

### What we measure and why

- **E3's arm** (T = 400, ε, linear and cosine; DDPM at 10, 25, 50, 100 and
  400 steps), seeds 0–2. E3's protocol throughout:
  - the same 12,000 training and 2,000 test images per seed (`rng.choice`
    with `default_rng(seed)`);
  - AdamW at 2e-3, weight decay $10^{-5}$, batch 128, 16 epochs;
  - 500 samples per score.
- **An in-job PyTorch control.** E3's own `UNet`, schedules, training loop
  and sampler, run in the same job on the same GPU and scored by the same
  scorer. ml-ladder's E3 numbers were recorded on a CPU, so this control
  separates "JAX differs" from "the hardware differs".
- **T = 1000.** Every combination of {ε, $x_0$, v} × {linear, cosine} is
  trained under E3's protocol. Each is sampled by DDPM and DDIM, each with and
  without $x_0$ clipping, at 10, 20, 50, 100 and 1,000 steps. The LR is E3's
  for every arm; it was not re-tuned per parameterisation.
- **The score.** ml-ladder's `generative` fd: a Fréchet distance in the
  feature space of a small classifier, trained per seed on that seed's
  training images (CPU, 2 epochs), between 500 samples and 500 test images.
  The floor is fd between two disjoint sets of 500 test images.
- **Cost.** Wall time of one compiled sampling run of 500 images, after a
  warm-up call.

### What I built

- **`jaxexp.diffusion`:**
  - E3's schedules (verbatim) and its U-Net, written as pure functions with
    PyTorch's layer semantics and initialisation;
  - `from_torch`, which loads E3's weights;
  - the loss for each parameterisation;
  - `sample`, which runs DDPM or DDIM (optionally clipped) as one
    `lax.scan` over E3's timestep subsequence.
- **`experiments/n8_diffusion/run.py`:**
  - arms `e3`, `e3torch` and `t1000`, scored by ml-ladder's scorer;
  - `analyze.py`;
  - `diag_seed2.py`, a CPU diagnosis of one outlying seed.
- **Tests:** `tests/test_n8_diffusion.py`, 46 tests.
  - The JAX U-Net matches E3's PyTorch U-Net on loaded weights.
  - Each parameterisation's output converts back to (ε, $x_0$).
  - If the data is a single image, the exact denoiser is known in closed
    form. With it, every sampler, parameterisation and step count (10, 50,
    1,000) ends at that image, to $10^{-4}$.
- **A race fixed on the way.** n8b's two GPU lanes downloaded MNIST into the
  same directory at once, and one read the other's partial file (SHA
  mismatch). `jaxexp.data` now downloads to a per-process temporary file and
  renames it into place.

### Results

N8RESULTS

### What surprised me

N8SURPRISES

### Open threads

N8THREADS

## O6 — A scaling law on tokens

### Question

G1 tried to measure the Chinchilla exponent on enwik8's characters and could
not. Its compute-optimal $N^*$ grew as $C^{1.24}$, because the corpus ran
out: the largest IsoFLOP point repeated the training split 39.7 times. With
ml-ladder's byte-exact BPE tokenizer and a corpus 11× larger, does the
optimum become free to move? Does its exponent come with a range narrow
enough to mean something?

### The idea

**Chinchilla's two approaches.** At a fixed compute budget $C \approx 6ND$,
train models of several sizes $N$, each on $D = C / (6N)$ tokens. The loss
against $\log N$ is a valley (an IsoFLOP curve), and its bottom is the
compute-optimal size $N^*(C)$. Repeating this at several budgets gives
$N^* \propto C^a$ (approach 2). Alternatively, fit one surface
$L(N, D) = E + A/N^\alpha + B/D^\beta$ through every run; it implies
$a = \beta / (\alpha + \beta)$ (approach 3).

**A corpus that does not run out.** enwik8 is the first $10^8$ bytes of
enwik9. Training on enwik9 minus the bytes enwik8 holds out (90M–100M) gives
about 990M bytes, which is 353M BPE tokens. Validation is ml-ladder's own
held-out text, scored in bits per byte.

### The math

**Approach 2.** Per budget, fit $L = c_0 + c_1 \log N + c_2 (\log N)^2$ through
all runs; the vertex $\log N^* = -c_1 / 2c_2$ counts only if it lies strictly
inside the sizes sampled. Then $\log N^* = a \log C + k$ over the interior
contours, and $D^* = C / 6N^*$ gives $b = 1 - a$.

**Approach 3.** Hoffmann et al.'s fit: minimise the Huber loss
($\delta = 10^{-3}$) of $\log L$ against
$\operatorname{LSE}(a - \alpha \log N,\; b - \beta \log D,\; e)$ from a grid of
243 starting points, by L-BFGS-B.

**What C counts.** $6ND$ with non-embedding $N$ is Kaplan's convention, and
the budgets here are defined by it. A training token really costs

$$6\,(N + N_{\text{head}}) + 12\, n_{\text{layer}}\, T\, d$$

FLOPs: the 4096-way output layer ($N_{\text{head}} = 4096\,d$, 0.4M to 2.1M
here) and causal attention (XLA computes all $T \times T$ scores). That is
3.2× the nominal cost for the smallest model and 1.17× for the largest. The
parametric law $L(N, D)$ does not depend on how compute is counted, so the
compute-optimal $N$ under the true cost comes from minimising
$L(N, C / \text{cost}(N))$ over $N$.

### What we measure and why

- **The tokenizer is ml-ladder's.** It must give 31,095,000 tokens on
  enwik8's training split and 2.8865 bytes per validation token.
- **LR per size.** Peak LR on a factor-2 grid at $2 \times 10^7$ tokens,
  seed 0, extended until every optimum is interior. G1 found per-size tuning
  essential.
- **IsoFLOP contours** at $C = 3 \times 10^{13}$, $10^{14}$,
  $3 \times 10^{14}$, $10^{15}$, $3 \times 10^{15}$ and $10^{16}$ (nominal
  6ND), seeds 0–2. Each of nine sizes runs where
  $2 \times 10^6 \le D \le$ the corpus.
- **Fixed for every run:** T = 256, 64 sequences per step, AdamW (0.9, 0.95),
  weight decay 0.1, clipping at 1, 5% warmup then cosine to 10% over each
  run's own length, fp16 matmuls with loss scaling, fused attention.
- **Uncertainty:** 5th–95th percentiles over 400 bootstrap resamples of the
  runs, for both approaches.
- **The parameter count:** approach 3 is repeated with the output layer
  counted in $N$, because that choice has been shown to move the exponent
  at small scale (Pearce and Song 2024).

### What I built

- **`jaxexp.bpe`:**
  - ml-ladder's `train_bpe` and `bpe_encode`;
  - `enwik9`, checked against enwik8 by SHA (enwik8 must be its first
    $10^8$ bytes);
  - `corpus`, which encodes in 100 MB chunks, checks that the tokens spell
    every byte exactly once, and caches the result as uint16 under one file
    lock.
- **`experiments/o6_scaling/run.py`:**
  - nine RoPE transformers, s0–s8: widths 96–512, 2–8 layers, 0.22M–25.2M
    non-embedding parameters;
  - the studies `tokenize`, `lr` and `isoflop`.
- **`analyze.py`:** both approaches, both parameter counts, the true-cost
  optimum and the bootstrap. `synthetic_check.py` runs it on data drawn from
  a known law at this design. It recovers $a = 0.483$ as 0.468 (IsoFLOP) and
  0.460 (parametric, bootstrap 0.43–0.51).
- **`tests/test_o6_bpe.py`:** the tokens spell every byte once, and the
  vocabulary and encoding equal ml-ladder's.
- **The token caches** are built once on CPU and shipped as a private Kaggle
  dataset. One kernel that downloaded and tokenized enwik9 itself ran 3 h
  against 45 min expected, with no output, and was deleted. The runner now
  bounds every job's wall time.

### Results

O6RESULTS

### What surprised me

O6SURPRISES

### Open threads

O6THREADS
