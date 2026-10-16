"""C3's decoder-only transformer, ported to Flax NNX.

The reference is ml-ladder's ``parts/c_sequence_attention/c3_transformer_from_scratch.py``
(the block and ``CharTransformer``) and, for SDPA attention and RoPE,
``tools/gpu/c_context_scaling.py``. Same structure, same parameter names where
Flax allows, and the same function — R1 measures how close.

Three Flax/JAX defaults differ silently from PyTorch's and are exposed as
options so their cost can be measured rather than assumed:

- ``jax.nn.gelu`` is the tanh approximation by default; ``torch.nn.GELU()`` is
  the exact $x\\Phi(x)$.
- ``nnx.LayerNorm`` uses epsilon 1e-6; ``torch.nn.LayerNorm`` uses 1e-5.
- ``nnx.LayerNorm`` computes the variance as $E[x^2] - E[x]^2$ ("fast
  variance"); PyTorch subtracts the mean first.

The defaults here are PyTorch's, so a model built with no options computes the
reference function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class Numerics:
    """The knobs on which PyTorch and Flax disagree by default."""

    gelu_approximate: bool = False  # torch nn.GELU(): exact
    ln_epsilon: float = 1e-5  # torch nn.LayerNorm
    ln_fast_variance: bool = False  # torch: two-pass variance
    attention: str = "explicit"  # "explicit" softmax(QK^T/sqrt(d) + M)V, or "fused" (jax.nn.dot_product_attention)
    # Matmul dtype, as PyTorch's fp16 autocast: Linear layers and attention run in it;
    # LayerNorm, softmax, RoPE, the residual stream and the loss stay float32. None = float32.
    compute_dtype: object = None


TORCH = Numerics()
FLAX_DEFAULTS = Numerics(gelu_approximate=True, ln_epsilon=1e-6, ln_fast_variance=True)


def sinusoidal_encoding(length: int, width: int) -> jax.Array:
    """C3's fixed table: PE[t, 2i] = sin(t / 10000^(2i/d)), PE[t, 2i+1] = cos(.)."""
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    index = jnp.arange(0, width, 2, dtype=jnp.float32)
    divisor = jnp.exp(-index * math.log(10000.0) / width)
    table = jnp.zeros((length, width), jnp.float32)
    table = table.at[:, 0::2].set(jnp.sin(position * divisor))
    return table.at[:, 1::2].set(jnp.cos(position * divisor))


def rotary(t: jax.Array, base: float = 10000.0) -> jax.Array:
    """Half-split RoPE on [batch, length, heads, dim], as ``c_context_scaling.rotary``.

    Channel pair (i, i + dim/2) at position p is rotated by p * base^(-2i/dim).
    The rotation is done in fp32 whatever the input dtype (angles reach ~1000 rad).
    """
    length, dim = t.shape[-3], t.shape[-1]
    half = dim // 2
    inv_freq = base ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angles = jnp.arange(length, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    cos, sin = jnp.cos(angles)[:, None, :], jnp.sin(angles)[:, None, :]  # broadcast over heads
    first, second = t[..., :half].astype(jnp.float32), t[..., half:].astype(jnp.float32)
    return jnp.concatenate([first * cos - second * sin, first * sin + second * cos], axis=-1).astype(t.dtype)


def layer_norm(width: int, numerics: Numerics, rngs: nnx.Rngs) -> nnx.LayerNorm:
    return nnx.LayerNorm(
        width, epsilon=numerics.ln_epsilon, use_fast_variance=numerics.ln_fast_variance, rngs=rngs
    )


class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, width: int, heads: int, *, rope: bool, numerics: Numerics, rngs: nnx.Rngs) -> None:
        if width % heads:
            raise ValueError(f"width {width} must divide into {heads} heads")
        self.heads, self.head_width = heads, width // heads
        self.rope, self.numerics = rope, numerics
        self.qkv = nnx.Linear(width, 3 * width, dtype=numerics.compute_dtype, rngs=rngs)
        self.projection = nnx.Linear(width, width, dtype=numerics.compute_dtype, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, length, width = x.shape
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        # [batch, length, heads, head_width]: JAX's attention layout, no transpose needed.
        q, k, v = (t.reshape(batch, length, self.heads, self.head_width) for t in (q, k, v))
        if self.rope:
            q, k = rotary(q), rotary(k)
        if self.numerics.attention == "fused":
            out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        elif self.numerics.attention == "explicit":
            scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_width)
            mask = jnp.tril(jnp.ones((length, length), dtype=bool))
            scores = jnp.where(mask, scores, -jnp.inf)
            weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)  # softmax in fp32
            out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
        else:
            raise ValueError(f"unknown attention {self.numerics.attention!r}")
        return self.projection(out.reshape(batch, length, width))


class MLP(nnx.Module):
    def __init__(self, width: int, numerics: Numerics, rngs: nnx.Rngs) -> None:
        self.up = nnx.Linear(width, 4 * width, dtype=numerics.compute_dtype, rngs=rngs)
        self.down = nnx.Linear(4 * width, width, dtype=numerics.compute_dtype, rngs=rngs)
        self.gelu_approximate = numerics.gelu_approximate

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.gelu(self.up(x), approximate=self.gelu_approximate))


class Block(nnx.Module):
    """Pre-LN block: x + attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, width: int, heads: int, *, rope: bool = False, numerics: Numerics = TORCH, rngs: nnx.Rngs):
        self.norm1 = layer_norm(width, numerics, rngs)
        self.attention = MultiHeadSelfAttention(width, heads, rope=rope, numerics=numerics, rngs=rngs)
        self.norm2 = layer_norm(width, numerics, rngs)
        self.mlp = MLP(width, numerics, rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CharTransformer(nnx.Module):
    """Decoder-only transformer over bytes/characters, as C3's ``CharTransformer``.

    ``positional`` is "learned", "sinusoidal", "none", or "rope" (no position
    table; queries and keys rotated in every layer, as ``transformer-25M-rope``).
    """

    def __init__(
        self,
        vocab_size: int,
        width: int = 192,
        heads: int = 4,
        depth: int = 4,
        block_size: int = 128,
        *,
        positional: str = "learned",
        numerics: Numerics = TORCH,
        rngs: nnx.Rngs,
    ) -> None:
        if positional not in ("learned", "sinusoidal", "none", "rope"):
            raise ValueError(f"unknown positional encoding {positional!r}")
        self.block_size, self.positional = block_size, positional
        self.token_embedding = nnx.Embed(vocab_size, width, rngs=rngs)
        if positional == "learned":
            self.position_embedding = nnx.Embed(block_size, width, rngs=rngs)
        self.blocks = nnx.List(
            [Block(width, heads, rope=positional == "rope", numerics=numerics, rngs=rngs) for _ in range(depth)]
        )
        self.norm = layer_norm(width, numerics, rngs)
        self.head = nnx.Linear(width, vocab_size, use_bias=False, dtype=numerics.compute_dtype, rngs=rngs)

    def __call__(self, ids: jax.Array, *, remat: bool = False, policy=None) -> jax.Array:
        """Logits for ``ids``. ``remat=True`` checkpoints each block (``nnx.remat``):
        the backward pass recomputes what ``policy`` (a ``jax.checkpoint_policies``
        entry; None saves nothing) does not keep."""
        length = ids.shape[-1]
        x = self.token_embedding(ids)
        if self.positional == "learned":
            x = x + self.position_embedding(jnp.arange(length))
        elif self.positional == "sinusoidal":
            x = x + sinusoidal_encoding(self.block_size, x.shape[-1])[:length]

        def apply(block: Block, x: jax.Array) -> jax.Array:
            return block(x)

        if remat:
            apply = nnx.remat(apply, policy=policy)
        for block in self.blocks:
            x = apply(block, x)
        return self.head(self.norm(x))
