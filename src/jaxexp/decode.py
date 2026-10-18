"""O4: autoregressive decoding from ``jaxexp.transformer.CharTransformer``, three ways.

- **recompute** — no cache: each new token runs the whole model over the sequence so far,
  padded to a fixed length so that one compiled program serves every step.
- **cached** — a key/value cache preallocated for ``length`` positions; a jitted step embeds
  one token, writes its keys and values at ``pos`` with ``lax.dynamic_update_slice`` and
  attends over the cache under a mask. Python drives the loop, one dispatch per token.
- **loop** — the same step inside ``lax.fori_loop``: the whole generation is one program.
  This is the JAX analogue of replaying a captured CUDA graph over preallocated buffers.

All three decode greedily (argmax), so they must emit the same tokens; the tests check that,
and that the cached step's logits equal the model's own forward pass.

The model is read into plain arrays (``weights``) so the decode functions are pure. Matmul
weights are stored in ``compute_dtype`` once, up front: a decode step then reads each weight
once, in that dtype, which is what the bandwidth arithmetic in the writeup counts.
Numerics follow ``jaxexp.transformer.TORCH``: exact GELU, LayerNorm with eps 1e-5 and two-pass
variance, attention scores and softmax in fp32, residual stream in fp32.

Shapes: a cache is ``{"k", "v"}`` of [layers, batch, length, heads, head_width].
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

from jaxexp import transformer as T


def weights(model: T.CharTransformer, compute_dtype=jnp.float16) -> dict:
    """The model's parameters as a pytree of arrays (matmul weights in ``compute_dtype``)."""
    cast = (lambda a: a.astype(compute_dtype)) if compute_dtype is not None else (lambda a: a)

    def dense(layer):
        out = {"kernel": cast(layer.kernel[...])}
        if layer.bias is not None:
            out["bias"] = cast(layer.bias[...])
        return out

    def norm(layer):
        return {"scale": layer.scale[...], "bias": layer.bias[...], "eps": layer.epsilon}

    att0 = model.blocks[0].attention
    w = {
        "token": model.token_embedding.embedding[...],
        "blocks": [
            {
                "norm1": norm(b.norm1),
                "qkv": dense(b.attention.qkv),
                "projection": dense(b.attention.projection),
                "norm2": norm(b.norm2),
                "up": dense(b.mlp.up),
                "down": dense(b.mlp.down),
            }
            for b in model.blocks
        ],
        "norm": norm(model.norm),
        "head": dense(model.head),
    }
    if model.positional == "learned":
        w["position"] = model.position_embedding.embedding[...]
    elif model.positional == "sinusoidal":
        w["position"] = T.sinusoidal_encoding(model.block_size, w["token"].shape[1])
    static = {
        "heads": att0.heads,
        "head_width": att0.head_width,
        "rope": model.positional == "rope",
        "gelu_approximate": model.blocks[0].mlp.gelu_approximate,
        "compute_dtype": compute_dtype,
    }
    return {"w": w, "static": static}


def _norm(p, x):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + p["eps"]) * p["scale"] + p["bias"]


def _dense(p, x, dtype):
    x = x.astype(dtype) if dtype is not None else x
    y = x @ p["kernel"]
    return y + p["bias"] if "bias" in p else y


def _rope_at(t, positions, base: float = 10000.0):
    """``transformer.rotary`` for explicit positions: t [..., dim]; ``positions`` broadcasts against t[..., 0]."""
    dim = t.shape[-1]
    half = dim // 2
    inv_freq = base ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angles = jnp.asarray(positions, jnp.float32)[..., None] * inv_freq
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    first, second = t[..., :half].astype(jnp.float32), t[..., half:].astype(jnp.float32)
    return jnp.concatenate([first * cos - second * sin, first * sin + second * cos], -1).astype(t.dtype)


def _embed(w, ids, positions):
    x = w["token"][ids]
    if "position" in w:
        x = x + w["position"][positions]
    return x


def _mlp(blk, x, s):
    dt = s["compute_dtype"]
    h = jax.nn.gelu(_dense(blk["up"], _norm(blk["norm2"], x), dt), approximate=s["gelu_approximate"])
    return x + _dense(blk["down"], h, dt)


def forward(params: dict, ids):
    """Logits [batch, T, vocab] and per-layer keys and values [layers, batch, T, heads, head_width]
    for ``ids`` [batch, T]: the model's forward pass, written over ``weights``."""
    w, s = params["w"], params["static"]
    batch, length = ids.shape
    heads, hw, dt = s["heads"], s["head_width"], s["compute_dtype"]
    positions = jnp.arange(length)
    x = _embed(w, ids, positions)
    ks, vs = [], []
    mask = jnp.tril(jnp.ones((length, length), bool))
    for blk in w["blocks"]:
        q, k, v = jnp.split(_dense(blk["qkv"], _norm(blk["norm1"], x), dt), 3, -1)
        q, k, v = (t.reshape(batch, length, heads, hw) for t in (q, k, v))
        if s["rope"]:
            q, k = _rope_at(q, positions[:, None]), _rope_at(k, positions[:, None])
        ks.append(k)
        vs.append(v)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / math.sqrt(hw)
        att = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), -1).astype(v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(batch, length, heads * hw)
        x = x + _dense(blk["projection"], out, dt)
        x = _mlp(blk, x, s)
    logits = _dense(w["head"], _norm(w["norm"], x), dt)
    return logits, jnp.stack(ks), jnp.stack(vs)


def empty_cache(params: dict, batch: int, length: int) -> dict:
    s = params["static"]
    layers = len(params["w"]["blocks"])
    dtype = s["compute_dtype"] or jnp.float32
    shape = (layers, batch, length, s["heads"], s["head_width"])
    return {"k": jnp.zeros(shape, dtype), "v": jnp.zeros(shape, dtype)}


def cache_bytes(params: dict, batch: int, length: int) -> int:
    s = params["static"]
    layers = len(params["w"]["blocks"])
    itemsize = jnp.dtype(s["compute_dtype"] or jnp.float32).itemsize
    return 2 * layers * batch * length * s["heads"] * s["head_width"] * itemsize


def weight_bytes(params: dict) -> int:
    return int(sum(a.size * a.dtype.itemsize for a in jax.tree.leaves(params["w"]) if hasattr(a, "dtype")))


def prefill(params: dict, ids, length: int):
    """Run the prompt ``ids`` [batch, P] in one parallel pass; return its last logits and a
    cache of ``length`` positions holding the prompt's keys and values."""
    logits, k, v = forward(params, ids)
    cache = empty_cache(params, ids.shape[0], length)
    cache = {
        "k": jax.lax.dynamic_update_slice(cache["k"], k.astype(cache["k"].dtype), (0, 0, 0, 0, 0)),
        "v": jax.lax.dynamic_update_slice(cache["v"], v.astype(cache["v"].dtype), (0, 0, 0, 0, 0)),
    }
    return logits[:, -1], cache


def step(params: dict, cache: dict, tokens, pos):
    """One token per sequence: ``tokens`` [batch] at position ``pos`` (a traced scalar).
    Writes their keys and values into the cache and returns (logits [batch, vocab], cache).
    Attention reads the whole preallocated cache and masks positions after ``pos``, so the
    step has one static shape for every position."""
    w, s = params["w"], params["static"]
    batch = tokens.shape[0]
    heads, hw, dt = s["heads"], s["head_width"], s["compute_dtype"]
    length = cache["k"].shape[2]
    x = _embed(w, tokens, pos)
    valid = jnp.arange(length) <= pos
    new_k, new_v = cache["k"], cache["v"]
    for i, blk in enumerate(w["blocks"]):
        q, k, v = jnp.split(_dense(blk["qkv"], _norm(blk["norm1"], x), dt), 3, -1)
        q, k, v = (t.reshape(batch, heads, hw) for t in (q, k, v))
        if s["rope"]:
            q, k = _rope_at(q, pos), _rope_at(k, pos)
        new_k = jax.lax.dynamic_update_slice(new_k, k[None, :, None].astype(new_k.dtype), (i, 0, pos, 0, 0))
        new_v = jax.lax.dynamic_update_slice(new_v, v[None, :, None].astype(new_v.dtype), (i, 0, pos, 0, 0))
        scores = jnp.einsum("bhd,bkhd->bhk", q, new_k[i], preferred_element_type=jnp.float32) / math.sqrt(hw)
        att = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), -1).astype(new_v.dtype)
        out = jnp.einsum("bhk,bkhd->bhd", att, new_v[i]).reshape(batch, heads * hw)
        x = x + _dense(blk["projection"], out, dt)
        x = _mlp(blk, x, s)
    logits = _dense(w["head"], _norm(w["norm"], x), dt)
    return logits.astype(jnp.float32), {"k": new_k, "v": new_v}


# ------------------------------------------------------------------------------ generators
# Each returns the generated tokens [batch, steps]. ``prompt`` is [batch, P]; the sequence
# grows to P + steps <= length. ``params["static"]`` is a dict of Python values, so it is
# closed over (static) rather than traced: the jitted functions take only the arrays.


def _split(params):
    return params["w"], params["static"]


def make_recompute(params: dict, length: int):
    """No cache: one compiled forward pass over a [batch, length] buffer per token."""
    w0, s = _split(params)

    @jax.jit
    def next_token(w, buf, pos):
        logits, _, _ = forward({"w": w, "static": s}, buf)
        return jnp.argmax(logits[:, pos - 1].astype(jnp.float32), -1).astype(buf.dtype)

    def generate(prompt, steps: int):
        batch, p = prompt.shape
        buf = jnp.zeros((batch, length), prompt.dtype).at[:, :p].set(prompt)
        out = []
        for pos in range(p, p + steps):
            tok = next_token(w0, buf, pos)
            buf = buf.at[:, pos].set(tok)
            out.append(tok)
        return jnp.stack(out, 1)

    return generate, next_token


def make_cached(params: dict, length: int):
    """Prefill once, then one jitted step per token, driven from Python."""
    w0, s = _split(params)
    prefill_j = jax.jit(lambda w, ids: prefill({"w": w, "static": s}, ids, length))

    @partial(jax.jit, donate_argnums=(1,))
    def next_token(w, cache, tok, pos):
        logits, cache = step({"w": w, "static": s}, cache, tok, pos)
        return jnp.argmax(logits, -1).astype(tok.dtype), cache

    def generate(prompt, steps: int):
        logits, cache = prefill_j(w0, prompt)
        tok = jnp.argmax(logits.astype(jnp.float32), -1).astype(prompt.dtype)
        out = [tok]
        for pos in range(prompt.shape[1], prompt.shape[1] + steps - 1):
            tok, cache = next_token(w0, cache, tok, jnp.int32(pos))
            out.append(tok)
        return jnp.stack(out, 1)

    return generate, (prefill_j, next_token)


def make_loop(params: dict, length: int, steps: int):
    """Prefill and the whole generation as one jitted program (``lax.fori_loop``)."""
    w0, s = _split(params)

    @jax.jit
    def run(w, prompt):
        p = {"w": w, "static": s}
        logits, cache = prefill(p, prompt, length)
        tok = jnp.argmax(logits.astype(jnp.float32), -1).astype(prompt.dtype)
        out = jnp.zeros((prompt.shape[0], steps), prompt.dtype).at[:, 0].set(tok)

        def body(i, carry):
            tok, cache, out = carry
            logits, cache = step(p, cache, tok, prompt.shape[1] + i - 1)
            tok = jnp.argmax(logits, -1).astype(prompt.dtype)
            return tok, cache, out.at[:, i].set(tok)

        _, _, out = jax.lax.fori_loop(1, steps, body, (tok, cache, out))
        return out

    return (lambda prompt, steps=steps: run(w0, prompt)), run
