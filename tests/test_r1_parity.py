"""R1: the Flax port computes C3's function.

The JAX-only tests always run. The parity tests need torch (``uv sync --extra
torch``) and an ml-ladder checkout (``$ML_LADDER``, default ``../ml-ladder``),
and skip without them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxexp import transformer as T
from jaxexp.torch_port import load_torch_state, numpy_state

ML_LADDER = Path(os.environ.get("ML_LADDER", Path(__file__).resolve().parents[2] / "ml-ladder"))


def relative(reference, other) -> float:
    reference, other = np.asarray(reference, np.float64), np.asarray(other, np.float64)
    return float(np.linalg.norm(other - reference) / np.linalg.norm(reference))


# ---------------------------------------------------------------- JAX only


@pytest.mark.parametrize("positional", ["learned", "sinusoidal", "rope"])
@pytest.mark.parametrize("attention", ["explicit", "fused"])
def test_causal_mask_does_not_leak(positional, attention):
    """Changing token t must leave the logits at positions < t untouched."""
    model = T.CharTransformer(
        256, 64, 4, 2, 32, positional=positional, numerics=T.Numerics(attention=attention), rngs=nnx.Rngs(0)
    )
    ids = jax.random.randint(jax.random.key(1), (2, 32), 0, 256)
    t = 20
    changed = ids.at[:, t].set((ids[:, t] + 1) % 256)
    before, after = model(ids), model(changed)
    assert jnp.array_equal(before[:, :t], after[:, :t])
    assert not jnp.allclose(before[:, t], after[:, t])


def test_rope_score_depends_only_on_offset():
    """Same q and k at every position: score(p, r) must be a function of p - r."""
    dim, length = 32, 64
    q = jnp.broadcast_to(jax.random.normal(jax.random.key(0), (dim,)), (1, length, 1, dim))
    k = jnp.broadcast_to(jax.random.normal(jax.random.key(1), (dim,)), (1, length, 1, dim))
    scores = np.asarray(jnp.einsum("bqhd,bkhd->qk", T.rotary(q), T.rotary(k)))
    worst = max(np.ptp(np.diagonal(scores, offset)) for offset in range(-length + 1, length))
    assert worst < 1e-4


def test_loader_is_strict():
    model = T.Block(16, 2, rngs=nnx.Rngs(0))
    with pytest.raises(KeyError, match="unassigned"):
        load_torch_state(model, {"norm1.weight": np.ones(16, np.float32)})
    with pytest.raises(ValueError):
        load_torch_state(model, {"norm1.weight": np.ones(17, np.float32)})


# ------------------------------------------------------- against PyTorch


@pytest.fixture(scope="module")
def ladder():
    torch = pytest.importorskip("torch")
    if not (ML_LADDER / "tools" / "gpu" / "c_context_scaling.py").exists():
        pytest.skip(f"no ml-ladder checkout at {ML_LADDER}")
    sys.path[:0] = [str(ML_LADDER / "src"), str(ML_LADDER)]
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer
    from tools.gpu.c_context_scaling import transformer_lm

    return torch, make_transformer()[0], transformer_lm


def block_pair(ladder, numerics=T.TORCH, transpose=True, seed=0):
    torch, CharTransformer, _ = ladder
    torch.manual_seed(seed)
    block = CharTransformer(65, width=64, heads=4, depth=1, block_size=16).blocks[0].eval()
    ported = T.Block(64, 4, numerics=numerics, rngs=nnx.Rngs(seed))
    load_torch_state(ported, numpy_state(block), transpose_linear=transpose)
    return block, ported


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("attention", ["explicit", "fused"])
def test_block_parity(ladder, batch, attention):
    torch = ladder[0]
    block, ported = block_pair(ladder, T.Numerics(attention=attention))
    x = torch.randn(batch, 16, 64, generator=torch.Generator().manual_seed(batch))
    with torch.no_grad():
        reference = block(x)[0].numpy()
    assert relative(reference, ported(jnp.asarray(x.numpy()))) < 1e-6


@pytest.mark.parametrize(
    "numerics, transpose, floor",
    [
        (T.Numerics(gelu_approximate=True), True, 1e-5),  # jax.nn.gelu's default
        (T.Numerics(ln_epsilon=1e-6), True, 1e-6),  # nnx.LayerNorm's default
        (T.TORCH, False, 1e-2),  # square projection loaded untransposed
    ],
)
def test_harness_catches_silent_mismatches(ladder, numerics, transpose, floor):
    """Each known mismatch must show up well above round-off, or the parity test proves nothing."""
    torch = ladder[0]
    block, ported = block_pair(ladder, numerics, transpose)
    x = torch.randn(4, 16, 64, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        reference = block(x)[0].numpy()
    assert relative(reference, ported(jnp.asarray(x.numpy()))) > floor


@pytest.mark.parametrize("positional", ["learned", "rope"])
def test_model_parity_4m(ladder, positional):
    """R2's 4M configuration (256 wide, 5 layers, 4 heads) against its SDPA forward, T=256, batch 4."""
    torch, _, transformer_lm = ladder
    torch.manual_seed(0)
    reference_model = transformer_lm(256, 5, 4, 256, positional=positional).eval()
    ported = T.CharTransformer(256, 256, 4, 5, 256, positional=positional, rngs=nnx.Rngs(0))
    load_torch_state(ported, numpy_state(reference_model))
    ids = torch.randint(0, 256, (4, 256), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        reference = reference_model(ids)[0].numpy()
    assert relative(reference, ported(jnp.asarray(ids.numpy()))) < 1e-6


@pytest.mark.parametrize("policy", [None, jax.checkpoint_policies.dots_with_no_batch_dims_saveable])
def test_remat_does_not_change_the_gradient(policy):
    """Per-block remat (O1, needed by R2) recomputes; it must not change the function or its gradient."""
    import optax

    model = T.CharTransformer(256, 64, 4, 2, 32, positional="rope", rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    ids = jax.random.randint(jax.random.key(0), (2, 33), 0, 256)

    def loss(state, remat):
        logits = nnx.merge(graphdef, state)(ids[:, :-1], remat=remat, policy=policy)
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]).mean()

    plain, remat = jax.grad(loss)(state, False), jax.grad(loss)(state, True)
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(remat), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
