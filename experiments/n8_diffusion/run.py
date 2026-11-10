"""N8: diffusion samplers — E3's DDPM in JAX; DDPM vs DDIM, eps / x0 / v, linear vs cosine.

E3's protocol for one seed (``parts/e_generative/e3_ddpm.py``): the same 12,000 MNIST training
and 2,000 test images (``rng.choice`` with ``default_rng(seed)``), scaled to [-1, 1]; the
245k-parameter U-Net; AdamW (lr 2e-3, betas 0.9 / 0.999, wd 1e-5), batch 128, 16 epochs;
500 samples scored against the first 500 test images by Frechet distance in the feature
space of ml-ladder's own scorer (``mlladder.generative``: a small CNN trained 2 epochs on the
same training images with the seed), run here on the CPU with PyTorch.

Arms (``--arms``):

- ``e3``: E3's own setting, T = 400, eps-prediction, linear and cosine, DDPM sampling at
  {10, 25, 50, 100, 400} steps: the reproduction, against ``e-metrics.json``.
- ``e3torch``: the control for ``e3``: E3's own PyTorch model, schedule, sampler and
  training loop on the same GPU, scored by the same scorer instance as the JAX models.
- ``t1000``: T = 1000; {linear, cosine} x {eps, x0, v}; DDPM and DDIM, each also with the x0
  estimate clipped to [-1, 1] (``-clip``), at {10, 20, 50, 100, 1000} steps.

Each sampling run is one ``lax.scan``; its wall time (after compilation, which is timed
separately) is recorded per step count.

    python experiments/n8_diffusion/run.py --seed 0 --arms e3 t1000 --out runs/n8-s0.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

# The e3torch arm trains PyTorch on the same GPU: let XLA allocate as it goes instead of
# reserving most of the card up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from jaxexp import data  # noqa: E402
from jaxexp import diffusion as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def e3_subset(seed: int):
    """E3's images: ``vision.load(limit=...)`` draws train then test indices from one Generator."""
    x_tr = data._idx(data._fetch("train-images-idx3-ubyte.gz"))
    y_tr = data._idx(data._fetch("train-labels-idx1-ubyte.gz")).astype(np.int64)
    x_te = data._idx(data._fetch("t10k-images-idx3-ubyte.gz"))
    rng = np.random.default_rng(seed)
    i_tr = rng.choice(len(x_tr), size=12000, replace=False)
    i_te = rng.choice(len(x_te), size=2000, replace=False)
    to01 = lambda a: (a.astype(np.float32) / 255.0)[:, None]  # noqa: E731  [N, 1, 28, 28] like E3
    return to01(x_tr[i_tr]), y_tr[i_tr], to01(x_te[i_te])


class Scorer:
    """ml-ladder's fd: its feature extractor, trained with the seed on the training images (CPU)."""

    def __init__(self, ladder: Path, x_train, y_train, seed: int):
        sys.path.insert(0, str(ladder / "src"))
        from mlladder import generative

        self.g = generative
        self.model = generative.train_feature_extractor(x_train, y_train, device="cpu", seed=seed, epochs=2)

    def fd(self, real, fake) -> float:
        return float(self.g.distribution_distance(self.model, real, fake, device="cpu"))


def train(params, x_train, abar, parameterisation, seed, epochs=16, batch=128, lr=2e-3):
    opt = optax.adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-5)
    state = opt.init(params)
    total = abar.shape[0]

    @jax.jit
    def step(params, state, x0, key):
        kt, kn = jax.random.split(key)
        t = jax.random.randint(kt, (x0.shape[0],), 0, total)
        noise = jax.random.normal(kn, x0.shape)
        loss, grads = jax.value_and_grad(D.loss)(params, x0, t, noise, abar, parameterisation)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    data_nhwc = jnp.asarray(x_train.transpose(0, 2, 3, 1) * 2 - 1)
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(data_nhwc))
        epoch = []
        for s in range(0, len(order), batch):
            key, k = jax.random.split(key)
            params, state, loss = step(params, state, data_nhwc[order[s : s + batch]], k)
            epoch.append(loss)
        losses.append(float(jnp.mean(jnp.stack(epoch))))
    return params, losses


def run_model(args, scorer, x_train, real, *, total, schedule, parameterisation, samplers, steps_list):
    abar = D.alpha_bar(D.make_schedule(schedule, total))
    params = D.init_unet(jax.random.key(args.seed))
    started = time.time()
    params, losses = train(params, x_train, abar, parameterisation, args.seed, epochs=args.epochs)
    train_seconds = time.time() - started
    shape = (args.samples, 28, 28, 1)
    rows = []
    for sampler in samplers:
        for steps in steps_list:
            fn = jax.jit(lambda p, k, steps=steps, sampler=sampler: D.sample(
                p, k, shape, abar, steps=steps, sampler=sampler, parameterisation=parameterisation))
            key = jax.random.key(1000 + args.seed)
            t0 = time.time()
            fn(params, key).block_until_ready()
            compile_and_run = time.time() - t0
            t0 = time.time()
            drawn = np.asarray(fn(params, key).block_until_ready())
            seconds = time.time() - t0
            samples = np.clip((drawn + 1) / 2, 0, 1).transpose(0, 3, 1, 2).astype(np.float32)
            row = {"seed": args.seed, "T": total, "schedule": schedule, "parameterisation": parameterisation,
                   "sampler": sampler, "framework": "jax", "steps": steps, "fd": scorer.fd(real, samples),
                   "sample_seconds": seconds,
                   "compile_seconds": compile_and_run - seconds, "train_mse": losses[-1],
                   "train_seconds": train_seconds, "parameters": D.parameter_count(params)}
            rows.append(row)
            print(f"[n8] s{args.seed} T={total} {schedule:<6} {parameterisation:<3} {sampler:<4} {steps:>4} steps  "
                  f"fd {row['fd']:7.3f}  {seconds * 1e3:8.1f} ms", flush=True)
            if steps == max(steps_list) and sampler == samplers[0]:
                np.save(Path(args.out).with_name(Path(args.out).stem + f"-samples-T{total}-{schedule}-{parameterisation}.npy"),
                        (samples[:64] * 255).round().astype(np.uint8))
    return rows, losses


def run_torch_e3(args, scorer, x_train, real, *, schedule, steps_list):
    """The control: E3's own PyTorch classes (``make_schedule``, ``Diffusion``, ``make_denoiser``)
    and its training loop, restated from ``run`` without the plots, on this GPU, scored by the
    same scorer instance as the JAX models."""
    import torch

    ladder = Path(args.ml_ladder)
    sys.path.insert(0, str(ladder / "src"))
    spec = importlib.util.spec_from_file_location("e3_ddpm", ladder / "parts" / "e_generative" / "e3_ddpm.py")
    e3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e3)
    from mlladder import torchkit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = e3.Diffusion(e3.make_schedule(schedule, 400), device=device)
    torchkit.seed_torch(args.seed)
    model = e3.make_denoiser(channels=32, image_channels=1)().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    train_tensor = torch.from_numpy(x_train * 2 - 1).to(device)
    started, losses = time.time(), []
    for _ in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(train_tensor), device=device)
        epoch = []
        for start in range(0, len(train_tensor), 128):
            batch = train_tensor[permutation[start : start + 128]]
            t = torch.randint(0, 400, (len(batch),), device=device)
            noise = torch.randn_like(batch)
            loss = torch.nn.functional.mse_loss(model(diffusion.add_noise(batch, t, noise), t), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch.append(loss.item())
        losses.append(float(np.mean(epoch)))
    train_seconds = time.time() - started
    rows = []
    for steps in steps_list:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        t0 = time.time()
        drawn = diffusion.sample(model, shape=(args.samples, 1, 28, 28), device=device, steps=steps, generator=generator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - t0
        samples = ((drawn + 1) / 2).clamp(0, 1).cpu().numpy()
        row = {"seed": args.seed, "T": 400, "schedule": schedule, "parameterisation": "eps", "sampler": "ddpm",
               "framework": "torch", "steps": steps, "fd": scorer.fd(real, samples), "sample_seconds": seconds,
               "train_mse": losses[-1], "train_seconds": train_seconds}
        rows.append(row)
        print(f"[n8] s{args.seed} TORCH T=400 {schedule:<6} eps ddpm {steps:>4} steps  fd {row['fd']:7.3f}  "
              f"{seconds * 1e3:8.1f} ms", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=["e3", "e3torch", "t1000"], choices=["e3", "e3torch", "t1000"])
    parser.add_argument("--samplers", nargs="+", default=list(D.SAMPLERS), choices=list(D.SAMPLERS),
                        help="samplers for the T=1000 arm")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--schedules", nargs="+", default=["linear", "cosine"])
    parser.add_argument("--parameterisations", nargs="+", default=list(D.PARAMETERISATIONS))
    parser.add_argument("--out", default=str(ROOT / "runs" / "n8.json"))
    parser.add_argument("--ml-ladder", default=str(ROOT.parent / "ml-ladder"))
    parser.add_argument("--data-dir", default=None)  # accepted for the Kaggle runner's common arguments
    args = parser.parse_args()
    report = {"device": str(jax.devices()[0]), "environment": {"python": platform.python_version(), "jax": jax.__version__},
              "settings": vars(args), "rows": [], "train_curves": {}}
    x_train, y_train, x_test = e3_subset(args.seed)
    scorer = Scorer(Path(args.ml_ladder), x_train, y_train, args.seed)
    real = x_test[: args.samples]
    report["real_vs_real_fd"] = scorer.fd(real, x_test[args.samples : 2 * args.samples])  # the floor
    plan = []
    if "e3" in args.arms:
        plan += [(400, s, "eps", ["ddpm"], [10, 25, 50, 100, 400]) for s in args.schedules]
    if "t1000" in args.arms:
        plan += [(1000, s, p, list(args.samplers), [10, 20, 50, 100, 1000])
                 for s in args.schedules for p in args.parameterisations]
    if "e3torch" in args.arms:
        for schedule in args.schedules:
            rows, losses = run_torch_e3(args, scorer, x_train, real, schedule=schedule, steps_list=[10, 25, 50, 100, 400])
            report["rows"] += rows
            report["train_curves"][f"torch/T400/{schedule}/eps"] = losses
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=1))
    for total, schedule, p, samplers, steps_list in plan:
        rows, losses = run_model(args, scorer, x_train, real, total=total, schedule=schedule, parameterisation=p,
                                 samplers=samplers, steps_list=steps_list)
        report["rows"] += rows
        report["train_curves"][f"T{total}/{schedule}/{p}"] = losses
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
