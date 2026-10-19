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
