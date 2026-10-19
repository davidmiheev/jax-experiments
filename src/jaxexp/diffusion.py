"""N8: ml-ladder's E3 DDPM in JAX, with DDIM and three parameterisations.

- ``make_schedule``: E3's linear (range scaled by 1000 / T) and cosine schedules, verbatim.
- ``init_unet`` / ``unet``: E3's 245k-parameter U-Net (two resolutions, a middle block, skip
  concatenations, a sinusoidal timestep embedding added in every block), as pure functions on
  NHWC arrays with PyTorch's layer semantics (Conv2d, GroupNorm(8, eps 1e-5), AvgPool2d(2),
  nearest upsampling) and PyTorch's default initialisation. ``from_torch`` loads E3's weights;
  the tests check both nets compute the same function.
- ``loss``: the regression for an output parameterisation:
  ``eps`` (DDPM: predict the noise), ``x0`` (predict the clean image) or ``v``
  (v = sqrt(abar) eps - sqrt(1 - abar) x0, Salimans & Ho 2022). ``predict_eps_x0`` converts any
  of them back to (eps, x0) for the samplers.
- ``sample``: ``ddpm`` (E3's strided ancestral sampler: each step uses the effective alpha
  between retained timesteps, and adds noise except at the last one) or ``ddim`` (the
  deterministic eta = 0 update x_prev = sqrt(abar_prev) x0 + sqrt(1 - abar_prev) eps); with
  ``-clip``, the x0 estimate is clipped to [-1, 1] and eps recomputed from it before the update
  (E3's sampler does not clip). The
  timestep subsequence is E3's: ``linspace(T - 1, 0, steps)`` truncated to integers. The loop is
  a ``lax.scan``, so a whole sampling run is one compiled program.

Images are in [-1, 1], NHWC, [batch, 28, 28, 1] for MNIST.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

PARAMETERISATIONS = ("eps", "x0", "v")
SAMPLERS = ("ddpm", "ddim", "ddpm-clip", "ddim-clip")


# ------------------------------------------------------------------------------- schedules


def make_schedule(kind: str, steps: int) -> np.ndarray:
    """E3's betas (float64). Linear: DDPM's 1e-4..0.02 scaled by 1000 / T so that the forward
    process ends at N(0, I) for any T. Cosine: Nichol & Dhariwal, offset 0.008, betas <= 0.999."""
    if kind == "linear":
        scale = 1000 / steps
        return np.linspace(1e-4 * scale, 0.02 * scale, steps, dtype=np.float64)
    if kind == "cosine":
        t = np.linspace(0, steps, steps + 1, dtype=np.float64) / steps
        alpha_bar = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        return np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)
    raise ValueError(f"unknown schedule {kind!r}")


def alpha_bar(betas: np.ndarray) -> jnp.ndarray:
    # float32 like E3 (torch.cumprod of a float32 tensor)
    return jnp.cumprod(1.0 - jnp.asarray(betas, jnp.float32))


# ------------------------------------------------------------------------------- the U-Net


def _uniform(key, shape, bound):
    return jax.random.uniform(key, shape, jnp.float32, -bound, bound)


def _conv_init(key, k, cin, cout):
    """torch.nn.Conv2d default: kaiming_uniform(a=sqrt(5)) = U(+-1/sqrt(fan_in)), bias the same."""
    bound = 1 / math.sqrt(cin * k * k)
    kw, kb = jax.random.split(key)
    return {"w": _uniform(kw, (k, k, cin, cout), bound), "b": _uniform(kb, (cout,), bound)}


def _linear_init(key, cin, cout):
    bound = 1 / math.sqrt(cin)
    kw, kb = jax.random.split(key)
    return {"w": _uniform(kw, (cin, cout), bound), "b": _uniform(kb, (cout,), bound)}


def _block_init(key, cin, cout, time_dim):
    ks = jax.random.split(key, 4)
    block = {
        "conv1": _conv_init(ks[0], 3, cin, cout),
        "conv2": _conv_init(ks[1], 3, cout, cout),
        "norm1": {"scale": jnp.ones(cout), "bias": jnp.zeros(cout)},
        "norm2": {"scale": jnp.ones(cout), "bias": jnp.zeros(cout)},
        "time": _linear_init(ks[2], time_dim, cout),
    }
    if cin != cout:
        block["skip"] = _conv_init(ks[3], 1, cin, cout)
    return block


def init_unet(key, channels: int = 32, image_channels: int = 1, time_dim: int = 64) -> dict:
    ks = jax.random.split(key, 8)
    c = channels
    return {
        "time1": _linear_init(ks[0], time_dim, time_dim),
        "time2": _linear_init(ks[1], time_dim, time_dim),
        "down1": _block_init(ks[2], image_channels, c, time_dim),
        "down2": _block_init(ks[3], c, 2 * c, time_dim),
        "middle": _block_init(ks[4], 2 * c, 2 * c, time_dim),
        "up2": _block_init(ks[5], 4 * c, c, time_dim),
        "up1": _block_init(ks[6], 2 * c, c, time_dim),
        "out": _conv_init(ks[7], 1, c, image_channels),
    }


def _conv(p, x):
    k = p["w"].shape[0]
    y = jax.lax.conv_general_dilated(x, p["w"], (1, 1), [(k // 2, k // 2)] * 2,
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + p["b"]


def _group_norm(p, x, groups: int = 8, eps: float = 1e-5):
    b, h, w, c = x.shape
    g = x.reshape(b, h, w, groups, c // groups)
    mean = g.mean(axis=(1, 2, 4), keepdims=True)
    var = ((g - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
    g = (g - mean) / jnp.sqrt(var + eps)
    return g.reshape(b, h, w, c) * p["scale"] + p["bias"]


def _block(p, x, emb):
    h = jax.nn.silu(_group_norm(p["norm1"], _conv(p["conv1"], x)))
    h = h + (emb @ p["time"]["w"] + p["time"]["b"])[:, None, None, :]
    h = jax.nn.silu(_group_norm(p["norm2"], _conv(p["conv2"], h)))
    return h + (_conv(p["skip"], x) if "skip" in p else x)


def time_embedding(params, t, time_dim: int = 64):
    half = time_dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
    angles = t.astype(jnp.float32)[:, None] * freqs[None]
    e = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=1)
    e = jax.nn.silu(e @ params["time1"]["w"] + params["time1"]["b"])
    return e @ params["time2"]["w"] + params["time2"]["b"]


def _pool(x):
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))


def _upsample(x):
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)


def unet(params, x, t):
    """The network's output for images ``x`` [B, H, W, C] at integer timesteps ``t`` [B]."""
    emb = time_embedding(params, t)
    first = _block(params["down1"], x, emb)
    second = _block(params["down2"], _pool(first), emb)
    middle = _block(params["middle"], second, emb)
    up = _upsample(_block(params["up2"], jnp.concatenate([middle, second], -1), emb))
    return _conv(params["out"], _block(params["up1"], jnp.concatenate([up, first], -1), emb))


def parameter_count(params) -> int:
    return int(sum(np.prod(a.shape) for a in jax.tree.leaves(params)))


def from_torch(state: dict) -> dict:
    """E3's ``UNet.state_dict()`` (numpy arrays) as ``init_unet``'s pytree."""

    def conv(prefix):
        return {"w": jnp.asarray(state[f"{prefix}.weight"]).transpose(2, 3, 1, 0), "b": jnp.asarray(state[f"{prefix}.bias"])}

    def linear(prefix):
        return {"w": jnp.asarray(state[f"{prefix}.weight"]).T, "b": jnp.asarray(state[f"{prefix}.bias"])}

    def norm(prefix):
        return {"scale": jnp.asarray(state[f"{prefix}.weight"]), "bias": jnp.asarray(state[f"{prefix}.bias"])}

    def block(name):
        out = {"conv1": conv(f"{name}.conv1"), "conv2": conv(f"{name}.conv2"), "norm1": norm(f"{name}.norm1"),
               "norm2": norm(f"{name}.norm2"), "time": linear(f"{name}.time")}
        if f"{name}.skip.weight" in state:
            out["skip"] = conv(f"{name}.skip")
        return out

    return {"time1": linear("embedding.project.0"), "time2": linear("embedding.project.2"),
            **{n: block(n) for n in ("down1", "down2", "middle", "up2", "up1")}, "out": conv("out")}


# ------------------------------------------------------------------------------- objectives


def predict_eps_x0(output, x_t, abar_t, parameterisation: str):
    """(eps_hat, x0_hat) from the network's output; ``abar_t`` broadcasts against images."""
    a, s = jnp.sqrt(abar_t), jnp.sqrt(1.0 - abar_t)
    if parameterisation == "eps":
        eps = output
        return eps, (x_t - s * eps) / a
    if parameterisation == "x0":
        x0 = output
        return (x_t - a * x0) / s, x0
    if parameterisation == "v":  # x0 = a x_t - s v, eps = s x_t + a v
        return s * x_t + a * output, a * x_t - s * output
    raise ValueError(f"unknown parameterisation {parameterisation!r}")


def loss(params, x0, t, noise, abar, parameterisation: str = "eps"):
    """Mean squared error between the output and its target, as each parameterisation's
    standard objective (for ``eps``, exactly E3's)."""
    abar_t = abar[t][:, None, None, None]
    x_t = jnp.sqrt(abar_t) * x0 + jnp.sqrt(1.0 - abar_t) * noise
    target = {"eps": noise, "x0": x0, "v": jnp.sqrt(abar_t) * noise - jnp.sqrt(1.0 - abar_t) * x0}[parameterisation]
    return jnp.mean((unet(params, x_t, t) - target) ** 2)


# ------------------------------------------------------------------------------- samplers


def timesteps(total: int, steps: int) -> np.ndarray:
    """E3's subsequence: every step if ``steps >= total``, else linspace(T - 1, 0, steps) as int."""
    if steps >= total:
        return np.arange(total - 1, -1, -1)
    return np.linspace(total - 1, 0, steps).astype(np.int64)  # torch .long() truncates toward zero


def sample(params, key, shape, abar, *, steps: int, sampler: str = "ddpm", parameterisation: str = "eps",
           apply=unet):
    """Draw ``shape[0]`` images. One ``lax.scan`` over the retained timesteps."""
    chosen = timesteps(abar.shape[0], steps)
    prev = np.append(chosen[1:], -1)  # -1: the step that lands on x_0
    key, k0 = jax.random.split(key)
    x = jax.random.normal(k0, shape)
    noise_keys = jax.random.split(key, len(chosen))

    def body(x, inputs):
        t, p, k = inputs
        abar_t = abar[t]
        abar_prev = jnp.where(p >= 0, abar[jnp.maximum(p, 0)], 1.0)
        out = apply(params, x, jnp.full((shape[0],), t))
        eps, x0 = predict_eps_x0(out, x, abar_t, parameterisation)
        if sampler.endswith("-clip"):
            # The standard guard: keep the clean-image estimate in the data's range [-1, 1] and
            # recompute the noise estimate from it. Near t = T, x0 = (x_t - sqrt(1 - abar) eps) /
            # sqrt(abar) multiplies the network's error by 1 / sqrt(abar), which is 2e4 on the
            # cosine schedule at T = 1000.
            x0 = jnp.clip(x0, -1.0, 1.0)
            eps = (x - jnp.sqrt(abar_t) * x0) / jnp.sqrt(1.0 - abar_t)
        if sampler.startswith("ddpm"):
            alpha = abar_t / abar_prev
            beta = 1.0 - alpha
            mean = (x - beta / jnp.sqrt(1.0 - abar_t) * eps) / jnp.sqrt(alpha)
            var = jnp.maximum(beta * (1.0 - abar_prev) / (1.0 - abar_t), 0.0)
            x = mean + jnp.where(p >= 0, jnp.sqrt(var), 0.0) * jax.random.normal(k, shape)
        elif sampler.startswith("ddim"):
            x = jnp.sqrt(abar_prev) * x0 + jnp.sqrt(1.0 - abar_prev) * eps
        else:
            raise ValueError(f"unknown sampler {sampler!r}")
        return x, None

    x, _ = jax.lax.scan(body, x, (jnp.asarray(chosen), jnp.asarray(prev), noise_keys))
    return x
