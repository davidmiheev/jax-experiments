"""N8: E3's diffusion model in JAX — schedules, parameterisations, samplers, and the U-Net."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxexp import diffusion as D

LADDER = Path(os.environ.get("ML_LADDER", Path(__file__).resolve().parents[2] / "ml-ladder"))


@pytest.mark.parametrize("kind", ["linear", "cosine"])
@pytest.mark.parametrize("steps", [400, 1000])
def test_schedules_end_at_noise(kind, steps):
    """E3's own check: the forward process must reach N(0, I) (alpha_bar_T < 0.01)."""
    betas = D.make_schedule(kind, steps)
    assert betas.shape == (steps,) and np.all(betas > 0) and np.all(betas <= 0.999)
    assert float(D.alpha_bar(betas)[-1]) < 0.01
    if kind == "linear" and steps == 1000:  # DDPM's own range at T = 1000
        assert betas[0] == pytest.approx(1e-4) and betas[-1] == pytest.approx(0.02)


@pytest.mark.parametrize("p", D.PARAMETERISATIONS)
def test_each_target_converts_back_to_eps_and_x0(p):
    k = jax.random.split(jax.random.key(0), 2)
    x0, eps = jax.random.normal(k[0], (3, 4, 4, 1)), jax.random.normal(k[1], (3, 4, 4, 1))
    abar_t = jnp.asarray([0.9, 0.3, 0.02])[:, None, None, None]
    x_t = jnp.sqrt(abar_t) * x0 + jnp.sqrt(1 - abar_t) * eps
    target = {"eps": eps, "x0": x0, "v": jnp.sqrt(abar_t) * eps - jnp.sqrt(1 - abar_t) * x0}[p]
    e, x = D.predict_eps_x0(target, x_t, abar_t, p)
    np.testing.assert_allclose(e, eps, atol=2e-5)
    np.testing.assert_allclose(x, x0, atol=2e-5)


@pytest.mark.parametrize("sampler", D.SAMPLERS)
@pytest.mark.parametrize("p", D.PARAMETERISATIONS)
@pytest.mark.parametrize("steps", [10, 50, 1000])
def test_samplers_with_an_oracle_denoiser_land_on_the_data(sampler, p, steps):
    """If the data is one point, the exact denoiser is known in closed form. Both samplers,
    every parameterisation and every step count must then end exactly at that point: DDIM by
    construction, DDPM because its last step (abar_prev = 1, no noise) returns x0_hat."""
    target = jnp.linspace(-1, 1, 16).reshape(1, 4, 4, 1)
    abar = D.alpha_bar(D.make_schedule("linear", 1000))

    def oracle(_, x, t):
        a = abar[t][:, None, None, None]
        eps = (x - jnp.sqrt(a) * target) / jnp.sqrt(1 - a)
        return {"eps": eps, "x0": jnp.broadcast_to(target, x.shape),
                "v": jnp.sqrt(a) * eps - jnp.sqrt(1 - a) * target}[p]

    out = D.sample(None, jax.random.key(1), (2, 4, 4, 1), abar, steps=steps, sampler=sampler,
                   parameterisation=p, apply=oracle)
    np.testing.assert_allclose(out, jnp.broadcast_to(target, out.shape), atol=1e-4)


def test_timesteps_are_e3s():
    np.testing.assert_array_equal(D.timesteps(400, 400), np.arange(399, -1, -1))
    t = D.timesteps(1000, 10)
    assert t[0] == 999 and t[-1] == 0 and len(t) == 10 and np.all(np.diff(t) < 0)


def test_unet_shape_and_size():
    params = D.init_unet(jax.random.key(0))
    out = D.unet(params, jnp.zeros((2, 28, 28, 1)), jnp.array([0, 399]))
    assert out.shape == (2, 28, 28, 1)
    assert 240_000 < D.parameter_count(params) < 250_000  # E3: "a 245k-parameter U-Net"


def _e3():
    spec = importlib.util.spec_from_file_location("e3_ddpm", LADDER / "parts" / "e_generative" / "e3_ddpm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch (uv sync --extra torch)")
@pytest.mark.skipif(not (LADDER / "parts" / "e_generative" / "e3_ddpm.py").exists(), reason="needs ../ml-ladder")
def test_unet_matches_e3s_torch_unet():
    import sys

    import torch

    sys.path.insert(0, str(LADDER / "src"))
    e3 = _e3()
    torch.manual_seed(0)
    net = e3.make_denoiser(channels=32, image_channels=1)()
    params = D.from_torch({k: v.detach().numpy() for k, v in net.state_dict().items()})
    assert D.parameter_count(params) == sum(p.numel() for p in net.parameters())
    x = np.random.default_rng(0).standard_normal((3, 1, 28, 28)).astype(np.float32)
    t = np.array([0, 137, 399])
    with torch.no_grad():
        ref = net(torch.from_numpy(x), torch.from_numpy(t)).numpy()
    got = np.asarray(D.unet(params, jnp.asarray(x.transpose(0, 2, 3, 1)), jnp.asarray(t))).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
    # E3's schedules, verbatim
    for kind in ("linear", "cosine"):
        np.testing.assert_array_equal(D.make_schedule(kind, 400), e3.make_schedule(kind, 400))
