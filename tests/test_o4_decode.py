"""O4: the three decoders compute the model's own function and emit the same tokens."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxexp import decode as D
from jaxexp import transformer as T


def _model(positional: str, length: int = 48):
    return T.CharTransformer(256, 64, 4, 2, length, positional=positional, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("positional", ["learned", "rope", "sinusoidal"])
def test_functional_forward_is_the_model(positional):
    model = _model(positional)
    ids = jax.random.randint(jax.random.key(1), (2, 40), 0, 256)
    logits, k, v = D.forward(D.weights(model, compute_dtype=None), ids)
    np.testing.assert_allclose(logits, model(ids), rtol=1e-5, atol=1e-5)
    assert k.shape == v.shape == (2, 2, 40, 4, 16)


@pytest.mark.parametrize("positional", ["learned", "rope"])
def test_cached_steps_match_the_full_forward(positional):
    """Prefill 10 tokens, then feed the rest one by one: each step's logits equal the full pass's."""
    model = _model(positional)
    params = D.weights(model, compute_dtype=None)
    ids = jax.random.randint(jax.random.key(2), (3, 30), 0, 256)
    full, _, _ = D.forward(params, ids)
    last, cache = D.prefill(params, ids[:, :10], length=48)
    np.testing.assert_allclose(last, full[:, 9], rtol=1e-5, atol=1e-5)
    step = jax.jit(lambda c, t, p: D.step(params, c, t, p))
    for pos in range(10, 30):
        logits, cache = step(cache, ids[:, pos], jnp.int32(pos))
        np.testing.assert_allclose(logits, full[:, pos], rtol=1e-5, atol=1e-5, err_msg=f"position {pos}")


@pytest.mark.parametrize("positional", ["learned", "rope"])
def test_three_decoders_emit_the_same_tokens(positional):
    model = _model(positional)
    params = D.weights(model, compute_dtype=None)
    prompt = jax.random.randint(jax.random.key(3), (2, 12), 0, 256)
    steps = 20
    recompute, _ = D.make_recompute(params, 48)
    cached, _ = D.make_cached(params, 48)
    loop, _ = D.make_loop(params, 48, steps)
    a, b, c = (np.asarray(f(prompt, steps)) for f in (recompute, cached, loop))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(b, c)
    assert a.shape == (2, steps)


def test_fp16_decoding_stays_close_and_sizes_add_up():
    model = _model("rope")
    p16 = D.weights(model, compute_dtype=jnp.float16)
    p32 = D.weights(model, compute_dtype=None)
    ids = jax.random.randint(jax.random.key(4), (2, 24), 0, 256)
    a, _, _ = D.forward(p16, ids)
    b, _, _ = D.forward(p32, ids)
    np.testing.assert_allclose(np.asarray(a, np.float32), b, atol=2e-2)
    assert D.cache_bytes(p16, batch=2, length=48) == 2 * 2 * 2 * 48 * 64 * 2  # k and v, layers, batch, T, width, fp16
    count = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    assert D.weight_bytes(p16) < 4 * count  # most of it stored in fp16
