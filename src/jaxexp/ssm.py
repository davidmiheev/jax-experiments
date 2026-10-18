"""N4: linear recurrences that train in parallel. An LRU, a selective SSM (Mamba-style S6) and
Mamba-2 (SSD: a scalar decay per head, computed as chunked matrix products).

All three share the byte-level shell of ml-ladder's other language models:
embedding -> L residual blocks -> LayerNorm -> linear head. What differs is the
sequence mixer, a *linear* recurrence

    h_t = a_t * h_{t-1} + b_t        (elementwise; a_t, b_t computed from the input)

whose composition (a1, b1) . (a2, b2) = (a1 a2, a2 b1 + b2) is associative, so
``lax.associative_scan`` evaluates all T states in O(log T) parallel depth.
``scan="sequential"`` runs the same recurrence with ``lax.scan`` (for speed
comparisons and as the reference in tests).

- **LRU** (Orvieto et al. 2023, "Resurrecting Recurrent Neural Networks"): a complex
  diagonal recurrence with a fixed (input-independent) transition
  lambda = exp(-exp(nu) + i exp(theta)), |lambda| initialised in [r_min, r_max], an
  input projection B normalised by gamma = sqrt(1 - |lambda|^2), output
  y = Re(C h) + D x. Each block: x + LRU(LN(x)), then x + MLP(LN(x)) (GELU, 4x).
- **Selective SSM** (Gu & Dao 2023, "Mamba"; the S6 layer without the hardware kernel):
  per channel and state, a_t = exp(Delta_t A), b_t = Delta_t B_t x_t, with Delta_t, B_t,
  C_t computed from the input (selection), A = -exp(A_log) initialised to -(1..N), a
  causal depthwise conv (k = 4) before it, and a SiLU gate after:
  y = (C_t . h_t + D x) * silu(z). Each block: x + Mamba(LN(x)) (no separate MLP).
- **Mamba-2** (Dao & Gu 2024): the same selection with a_t = exp(Delta_t A_h), one scalar per
  head, so the recurrence over [heads, head_dim, state] can be evaluated chunk by chunk as
  masked matrix products (``ssd``) instead of a scan over [B, T, channels, state].

Parameters are dicts of arrays. ``compute_dtype`` (e.g. float16) applies to the dense
projections only; the recurrences, gates, LayerNorms and residual stream run in float32.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------------- scans


def linear_scan(a, b, mode: str = "associative"):
    """All states of h_t = a_t * h_{t-1} + b_t, h_{-1} = 0, along axis 1 ([batch, time, ...])."""
    if mode == "associative":

        def combine(left, right):
            a1, b1 = left
            a2, b2 = right
            return a1 * a2, a2 * b1 + b2

        return jax.lax.associative_scan(combine, (a, b), axis=1)[1]
    if mode == "sequential":

        def step(h, ab):
            h = ab[0] * h + ab[1]
            return h, h

        _, hs = jax.lax.scan(step, jnp.zeros_like(b[:, 0]), (jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1)))
        return jnp.swapaxes(hs, 0, 1)
    raise ValueError(f"unknown scan mode {mode!r}")


# ---------------------------------------------------------------------------------- pieces


def _dense(key, fan_in, fan_out, scale=1.0):
    bound = scale / math.sqrt(fan_in)
    return {"w": jax.random.uniform(key, (fan_in, fan_out), jnp.float32, -bound, bound), "b": jnp.zeros((fan_out,))}


def _apply_dense(p, x, dtype):
    if dtype is None:
        return x @ p["w"] + p["b"]
    return (x.astype(dtype) @ p["w"].astype(dtype)).astype(jnp.float32) + p["b"]


def _ln_init(d):
    return {"scale": jnp.ones((d,)), "bias": jnp.zeros((d,))}


def _ln(p, x, eps=1e-5):  # PyTorch's LayerNorm (exact variance, eps 1e-5), as jaxexp.transformer.TORCH
    mean = x.mean(-1, keepdims=True)
    var = jnp.square(x - mean).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * p["scale"] + p["bias"]


def _shell_init(key, vocab, d):
    k1, k2 = jax.random.split(key)
    return {"embedding": jax.random.normal(k1, (vocab, d)) * 0.02, "norm": _ln_init(d), "head": _dense(k2, d, vocab)}


def _shell_out(params, x, dtype):
    return _apply_dense(params["head"], _ln(params["norm"], x), dtype)


# ---------------------------------------------------------------------------------- LRU


def init_lru(key, vocab: int = 256, d: int = 256, layers: int = 5, state: int = 256,
             r_min: float = 0.9, r_max: float = 0.999, max_phase: float = math.pi) -> dict:
    keys = jax.random.split(key, layers + 1)
    params = _shell_init(keys[0], vocab, d)
    blocks = []
    for k in keys[1:]:
        ks = jax.random.split(k, 7)
        u1, u2 = jax.random.uniform(ks[0], (state,)), jax.random.uniform(ks[1], (state,))
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))  # |lambda| uniform on the ring
        theta_log = jnp.log(max_phase * u2)
        modulus = jnp.exp(-jnp.exp(nu_log))
        blocks.append({
            "norm1": _ln_init(d), "norm2": _ln_init(d),
            "nu_log": nu_log, "theta_log": theta_log,
            "gamma_log": jnp.log(jnp.sqrt(1 - modulus**2)),
            "B_re": jax.random.normal(ks[2], (d, state)) / math.sqrt(2 * d),
            "B_im": jax.random.normal(ks[3], (d, state)) / math.sqrt(2 * d),
            "C_re": jax.random.normal(ks[4], (state, d)) / math.sqrt(state),
            "C_im": jax.random.normal(ks[5], (state, d)) / math.sqrt(state),
            "D": jax.random.normal(ks[6], (d,)),
            "up": _dense(jax.random.fold_in(k, 1), d, 4 * d), "down": _dense(jax.random.fold_in(k, 2), 4 * d, d),
        })
    params["blocks"] = blocks
    return params


def lru_lambda(block):
    return jnp.exp(-jnp.exp(block["nu_log"]) + 1j * jnp.exp(block["theta_log"]))


def _lru_mixer(blk, x, dtype, scan):
    lam = lru_lambda(blk)  # [N] complex64
    gamma = jnp.exp(blk["gamma_log"])
    cast = (lambda t: t.astype(dtype)) if dtype is not None else (lambda t: t)
    bu_re = (cast(x) @ cast(blk["B_re"])).astype(jnp.float32)
    bu_im = (cast(x) @ cast(blk["B_im"])).astype(jnp.float32)
    b = (bu_re + 1j * bu_im) * gamma  # [batch, T, N]
    a = jnp.broadcast_to(lam, b.shape)
    h = linear_scan(a, b, scan)
    y = (cast(h.real) @ cast(blk["C_re"])).astype(jnp.float32) - (cast(h.imag) @ cast(blk["C_im"])).astype(jnp.float32)
    return y + blk["D"] * x


def _lru_block(blk, x, dtype, scan):
    x = x + _lru_mixer(blk, _ln(blk["norm1"], x), dtype, scan)
    hidden = jax.nn.gelu(_apply_dense(blk["up"], _ln(blk["norm2"], x), dtype), approximate=False)
    return x + _apply_dense(blk["down"], hidden, dtype)


# ---------------------------------------------------------------------------------- selective SSM


def init_mamba(key, vocab: int = 256, d: int = 256, layers: int = 9, expand: int = 2, state: int = 16,
               conv: int = 4, dt_min: float = 1e-3, dt_max: float = 1e-1) -> dict:
    inner, dt_rank = expand * d, math.ceil(d / 16)
    keys = jax.random.split(key, layers + 1)
    params = _shell_init(keys[0], vocab, d)
    blocks = []
    for k in keys[1:]:
        ks = jax.random.split(k, 6)
        dt = jnp.exp(jax.random.uniform(ks[4], (inner,)) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        blocks.append({
            "norm": _ln_init(d),
            "in_proj": _dense(ks[0], d, 2 * inner),
            "conv_w": jax.random.uniform(ks[1], (conv, inner), minval=-1 / math.sqrt(conv), maxval=1 / math.sqrt(conv)),
            "conv_b": jnp.zeros((inner,)),
            "x_proj": _dense(ks[2], inner, dt_rank + 2 * state),
            "dt_proj": {"w": jax.random.uniform(ks[3], (dt_rank, inner), minval=-dt_rank**-0.5, maxval=dt_rank**-0.5),
                        "b": dt + jnp.log(-jnp.expm1(-dt))},  # softplus^-1(dt): Delta starts in [dt_min, dt_max]
            "A_log": jnp.log(jnp.broadcast_to(jnp.arange(1, state + 1, dtype=jnp.float32), (inner, state))),
            "D": jnp.ones((inner,)),
            "out_proj": _dense(ks[5], inner, d),
        })
    params["blocks"] = blocks
    return params


def _mamba_mixer(blk, x, dtype, scan):
    inner = blk["D"].shape[0]
    state = blk["A_log"].shape[1]
    dt_rank = blk["dt_proj"]["w"].shape[0]
    xz = _apply_dense(blk["in_proj"], x, dtype)
    u, z = xz[..., :inner], xz[..., inner:]
    k = blk["conv_w"].shape[0]  # causal depthwise conv: sum_j w_j * u_{t-k+1+j}
    padded = jnp.pad(u, ((0, 0), (k - 1, 0), (0, 0)))
    u = sum(blk["conv_w"][j] * padded[:, j : j + u.shape[1]] for j in range(k)) + blk["conv_b"]
    u = jax.nn.silu(u)
    proj = _apply_dense(blk["x_proj"], u, dtype)
    delta = jax.nn.softplus(proj[..., :dt_rank] @ blk["dt_proj"]["w"] + blk["dt_proj"]["b"])  # [B, T, inner]
    b_t, c_t = proj[..., dt_rank : dt_rank + state], proj[..., dt_rank + state :]  # [B, T, N]
    A = -jnp.exp(blk["A_log"])  # [inner, N]
    a = jnp.exp(delta[..., None] * A)  # [B, T, inner, N]
    b = (delta * u)[..., None] * b_t[:, :, None, :]
    h = linear_scan(a, b, scan)
    y = jnp.einsum("btin,btn->bti", h, c_t) + blk["D"] * u
    return _apply_dense(blk["out_proj"], y * jax.nn.silu(z), dtype)


def _mamba_block(blk, x, dtype, scan):
    return x + _mamba_mixer(blk, _ln(blk["norm"], x), dtype, scan)


# ---------------------------------------------------------------------------------- Mamba-2 (SSD)


def init_mamba2(key, vocab: int = 256, d: int = 256, layers: int = 9, expand: int = 2, head_dim: int = 64,
                state: int = 64, conv: int = 4, dt_min: float = 1e-3, dt_max: float = 1e-1) -> dict:
    """Mamba-2 (Dao & Gu 2024): the selective SSM with a *scalar* decay per head, which lets the
    scan be computed as chunked matrix products (SSD) instead of a [B, T, channels, state] scan.
    One group of B, C shared by all heads; output gate y * silu(z) as in ``mamba`` (no extra norm)."""
    inner = expand * d
    heads = inner // head_dim
    keys = jax.random.split(key, layers + 1)
    params = _shell_init(keys[0], vocab, d)
    blocks = []
    for k in keys[1:]:
        ks = jax.random.split(k, 5)
        dt = jnp.exp(jax.random.uniform(ks[2], (heads,)) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        conv_dim = inner + 2 * state
        blocks.append({
            "norm": _ln_init(d),
            "in_proj": _dense(ks[0], d, 2 * inner + 2 * state + heads),
            "conv_w": jax.random.uniform(ks[1], (conv, conv_dim), minval=-1 / math.sqrt(conv), maxval=1 / math.sqrt(conv)),
            "conv_b": jnp.zeros((conv_dim,)),
            "dt_bias": dt + jnp.log(-jnp.expm1(-dt)),
            "A_log": jnp.log(jax.random.uniform(ks[3], (heads,), minval=1.0, maxval=16.0)),
            "D": jnp.ones((heads,)),
            "out_proj": _dense(ks[4], inner, d),
        })
    params["blocks"] = blocks
    return params


def _segsum(x):
    """[..., T] -> [..., T, T]: out[i, j] = sum_{k=j+1..i} x_k for i >= j, -inf above the diagonal.
    Summed directly (masked cumsum), not as a difference of two cumulative sums, which loses
    precision when the sums are large."""
    t = x.shape[-1]
    xr = jnp.broadcast_to(x[..., :, None], x.shape + (t,))  # [..., k, j] = x_k
    xr = jnp.where(jnp.tril(jnp.ones((t, t), bool), -1), xr, 0.0)  # keep k > j
    out = jnp.cumsum(xr, axis=-2)  # [..., i, j] = sum_{j < k <= i} x_k
    return jnp.where(jnp.tril(jnp.ones((t, t), bool)), out, -jnp.inf)


def ssd(x, dta, b, c, chunk: int = 64):
    """Chunked SSD: y_t = sum_{s<=t} (C_t . B_s) exp(sum_{r=s+1..t} dta_r) x_s, per head.
    x [batch, T, H, P] (already multiplied by dt), dta [batch, T, H] (dt * A <= 0),
    b, c [batch, T, N] (shared by heads). Returns y [batch, T, H, P]."""
    bsz, t, h, p = x.shape
    n = b.shape[-1]
    q = min(chunk, t)
    pad = (-t) % q
    if pad:
        x, dta = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))), jnp.pad(dta, ((0, 0), (0, pad), (0, 0)))
        b, c = jnp.pad(b, ((0, 0), (0, pad), (0, 0))), jnp.pad(c, ((0, 0), (0, pad), (0, 0)))
    nc = (t + pad) // q
    x = x.reshape(bsz, nc, q, h, p)
    b, c = b.reshape(bsz, nc, q, n), c.reshape(bsz, nc, q, n)
    a = jnp.moveaxis(dta.reshape(bsz, nc, q, h), -1, 1)  # [b, h, c, l]
    a_cum = jnp.cumsum(a, axis=-1)
    # 1. within each chunk: an attention-like masked product
    decay = jnp.exp(_segsum(a))  # [b, h, c, l, s]
    y_diag = jnp.einsum("bcln,bcsn,bhcls,bcshp->bclhp", c, b, decay, x)
    # 2. each chunk's final state, from its own inputs
    to_end = jnp.exp(a_cum[..., -1:] - a_cum)  # [b, h, c, l]
    states = jnp.einsum("bcln,bhcl,bclhp->bchpn", b, to_end, x)
    # 3. carry states across chunks (a short recurrence over nc chunks)
    states = jnp.concatenate([jnp.zeros_like(states[:, :1]), states], axis=1)
    chunk_decay = jnp.exp(_segsum(jnp.pad(a_cum[..., -1], ((0, 0), (0, 0), (1, 0)))))  # [b, h, c+1, c+1]
    states = jnp.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)[:, :-1]  # state entering each chunk
    # 4. what the carried state contributes inside each chunk
    y_off = jnp.einsum("bcln,bchpn,bhcl->bclhp", c, states, jnp.exp(a_cum))
    y = (y_diag + y_off).reshape(bsz, nc * q, h, p)
    return y[:, :t]


def ssd_sequential(x, dta, b, c):
    """The same recurrence step by step: H_t = exp(dta_t) H_{t-1} + x_t B_t^T, y_t = H_t C_t."""

    def step(state, inp):
        xt, at, bt, ct = inp
        state = jnp.exp(at)[..., None, None] * state + xt[..., None] * bt[:, None, None, :]
        return state, jnp.einsum("bhpn,bn->bhp", state, ct)

    bsz, t, h, p = x.shape
    init = jnp.zeros((bsz, h, p, b.shape[-1]), x.dtype)
    _, ys = jax.lax.scan(step, init, tuple(jnp.swapaxes(v, 0, 1) for v in (x, dta, b, c)))
    return jnp.swapaxes(ys, 0, 1)


def _mamba2_mixer(blk, x, dtype, scan):
    heads = blk["A_log"].shape[0]
    inner = blk["out_proj"]["w"].shape[0]
    state = (blk["conv_w"].shape[1] - inner) // 2
    proj = _apply_dense(blk["in_proj"], x, dtype)
    z, xbc, dt_raw = proj[..., :inner], proj[..., inner : 2 * inner + 2 * state], proj[..., 2 * inner + 2 * state :]
    k = blk["conv_w"].shape[0]
    padded = jnp.pad(xbc, ((0, 0), (k - 1, 0), (0, 0)))
    xbc = jax.nn.silu(sum(blk["conv_w"][j] * padded[:, j : j + xbc.shape[1]] for j in range(k)) + blk["conv_b"])
    u, b_t, c_t = xbc[..., :inner], xbc[..., inner : inner + state], xbc[..., inner + state :]
    dt = jax.nn.softplus(dt_raw + blk["dt_bias"])  # [B, T, H]
    A = -jnp.exp(blk["A_log"])  # [H]
    bsz, t, _ = u.shape
    uh = u.reshape(bsz, t, heads, inner // heads)
    xdt = uh * dt[..., None]
    fn = ssd if scan in ("associative", "chunked") else ssd_sequential
    y = fn(xdt, dt * A, b_t, c_t) + blk["D"][:, None] * uh
    return _apply_dense(blk["out_proj"], y.reshape(bsz, t, inner) * jax.nn.silu(z), dtype)


def _mamba2_block(blk, x, dtype, scan):
    return x + _mamba2_mixer(blk, _ln(blk["norm"], x), dtype, scan)


# ---------------------------------------------------------------------------------- models

KINDS = {"lru": (_lru_block, init_lru), "mamba": (_mamba_block, init_mamba), "mamba2": (_mamba2_block, init_mamba2)}


def apply(params: dict, ids, kind: str, *, scan: str = "associative", compute_dtype=None, remat: bool = False):
    """Logits [batch, T, vocab] for byte ids [batch, T]."""
    block_fn = KINDS[kind][0]
    if remat:
        block_fn = jax.checkpoint(block_fn, static_argnums=(2, 3))
    x = params["embedding"][ids]
    for blk in params["blocks"]:
        x = block_fn(blk, x, compute_dtype, scan)
    return _shell_out(params, x, compute_dtype)


def parameter_count(params) -> int:
    return int(sum(np.prod(v.shape) for v in jax.tree.leaves(params)))
