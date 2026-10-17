"""ml-ladder's recurrent language models (embedding -> RNN -> linear head), as ``lax.scan``.

The cells follow PyTorch's equations exactly, so cuDNN-trained numbers are
comparable. Gate order and weight layout are ``torch.nn.GRU`` / ``nn.LSTM``'s.

    GRU   r = s(W_ir x + b_ir + W_hr h + b_hr)      z = s(W_iz x + b_iz + W_hz h + b_hz)
          n = tanh(W_in x + b_in + r * (W_hn h + b_hn))      h' = (1 - z) * n + z * h
    LSTM  i, f, g, o = s, s, tanh, s of (W_i x + b_i + W_h h + b_h)  (rows in that order)
          c' = f * c + i * g      h' = o * tanh(c')

Parameters are a dict keyed by PyTorch's ``state_dict`` names
(``embedding.weight``, ``rnn.weight_ih_l0``, ..., ``head.bias``), so weights
ported from ml-ladder's own initialisation load with no renaming. The input
projection of each layer is one matmul over the whole sequence; only the
recurrent matmul is inside the scan.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

GATES = {"gru": 3, "lstm": 4, "rnn": 1}


def init_like_torch(key, kind: str, hidden: int, layers: int, vocab: int = 256) -> dict:
    """PyTorch's default initialisation (different random numbers, same distributions):
    RNN weights and biases U(+-1/sqrt(hidden)), embedding N(0, 1), head U(+-1/sqrt(hidden)),
    and C1's LSTM forget-gate bias of 1."""
    g = GATES[kind]
    keys = iter(jax.random.split(key, 4 * layers + 3))
    bound = 1 / math.sqrt(hidden)

    def uniform(shape):
        return jax.random.uniform(next(keys), shape, jnp.float32, -bound, bound)

    params = {"embedding.weight": jax.random.normal(next(keys), (vocab, hidden), jnp.float32)}
    for layer in range(layers):
        params[f"rnn.weight_ih_l{layer}"] = uniform((g * hidden, hidden))
        params[f"rnn.weight_hh_l{layer}"] = uniform((g * hidden, hidden))
        params[f"rnn.bias_ih_l{layer}"] = uniform((g * hidden,))
        params[f"rnn.bias_hh_l{layer}"] = uniform((g * hidden,))
        if kind == "lstm":
            params[f"rnn.bias_ih_l{layer}"] = params[f"rnn.bias_ih_l{layer}"].at[hidden : 2 * hidden].set(1.0)
            params[f"rnn.bias_hh_l{layer}"] = params[f"rnn.bias_hh_l{layer}"].at[hidden : 2 * hidden].set(0.0)
    params["head.weight"] = uniform((vocab, hidden))
    params["head.bias"] = uniform((vocab,))
    return params


def from_torch_state(state: dict[str, np.ndarray]) -> dict:
    return {k: jnp.asarray(np.asarray(v, np.float32)) for k, v in state.items()}


def _layer(kind: str, xs, W_ih, W_hh, b_ih, b_hh, dtype):
    """One layer over a [batch, time, in] sequence; returns [batch, time, hidden]."""
    hidden = W_hh.shape[1]
    cast = (lambda a: a.astype(dtype)) if dtype is not None else (lambda a: a)
    gx = (cast(xs) @ cast(W_ih).T).astype(jnp.float32) + b_ih  # all time steps at once
    W_hh_t = cast(W_hh).T
    batch = xs.shape[0]
    h0 = jnp.zeros((batch, hidden), jnp.float32)

    if kind == "gru":

        def step(h, gx_t):
            gh = (cast(h) @ W_hh_t).astype(jnp.float32) + b_hh
            r = jax.nn.sigmoid(gx_t[:, :hidden] + gh[:, :hidden])
            z = jax.nn.sigmoid(gx_t[:, hidden : 2 * hidden] + gh[:, hidden : 2 * hidden])
            n = jnp.tanh(gx_t[:, 2 * hidden :] + r * gh[:, 2 * hidden :])
            h = (1 - z) * n + z * h
            return h, h

        carry = h0
    elif kind == "lstm":

        def step(state, gx_t):
            h, c = state
            gates = gx_t + (cast(h) @ W_hh_t).astype(jnp.float32) + b_hh
            i, f, g, o = (gates[:, k * hidden : (k + 1) * hidden] for k in range(4))
            c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            return (h, c), h

        carry = (h0, h0)
    else:  # Elman RNN, tanh

        def step(h, gx_t):
            h = jnp.tanh(gx_t + (cast(h) @ W_hh_t).astype(jnp.float32) + b_hh)
            return h, h

        carry = h0
    _, hs = jax.lax.scan(step, carry, jnp.swapaxes(gx, 0, 1))
    return jnp.swapaxes(hs, 0, 1)


def apply(params: dict, ids: jax.Array, kind: str, *, compute_dtype=None) -> jax.Array:
    """Logits [batch, time, vocab] for ids [batch, time]. ``compute_dtype`` (e.g. float16) is
    used for matmul inputs only; gates, states and the head's output stay float32."""
    layers = sum(1 for k in params if k.startswith("rnn.weight_ih_l"))
    x = params["embedding.weight"][ids]
    for layer in range(layers):
        x = _layer(
            kind,
            x,
            params[f"rnn.weight_ih_l{layer}"],
            params[f"rnn.weight_hh_l{layer}"],
            params[f"rnn.bias_ih_l{layer}"],
            params[f"rnn.bias_hh_l{layer}"],
            compute_dtype,
        )
    W = params["head.weight"]
    if compute_dtype is not None:
        return (x.astype(compute_dtype) @ W.astype(compute_dtype).T).astype(jnp.float32) + params["head.bias"]
    return x @ W.T + params["head.bias"]


def parameter_count(params: dict) -> int:
    return int(sum(np.prod(v.shape) for v in params.values()))
