"""N8: why seed 2's E3 arm (T=400, eps, linear) samples thin, faint digits (CPU).

Retrains it as run, then with only the training key (timesteps and noise) changed, then with
only the data order changed; scores each with the sampling key as run and another one, plus an
EMA (0.999) of the weights. Ran on the cloud container's 4 CPU cores (~19 min per training).

    PYTHONPATH=src python experiments/n8_diffusion/diag_seed2.py docs/data/n8/seed2-diagnosis.json
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, jax, jax.numpy as jnp, optax
import run as R
from jaxexp import diffusion as D

seed = 2
x_train, y_train, x_test = R.e3_subset(seed)
scorer = R.Scorer(Path(__file__).resolve().parents[3] / "ml-ladder", x_train, y_train, seed)
real = x_test[:500]
abar = D.alpha_bar(D.make_schedule("linear", 400))

def train(train_key, order_seed, epochs=16, snapshots=()):
    params = D.init_unet(jax.random.key(seed))
    opt = optax.adamw(2e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-5)
    state = opt.init(params)
    @jax.jit
    def step(params, state, x0, key):
        kt, kn = jax.random.split(key)
        t = jax.random.randint(kt, (x0.shape[0],), 0, 400)
        noise = jax.random.normal(kn, x0.shape)
        loss, grads = jax.value_and_grad(D.loss)(params, x0, t, noise, abar, "eps")
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss, t
    data = jnp.asarray(x_train.transpose(0, 2, 3, 1) * 2 - 1)
    rng = np.random.default_rng(order_seed); key = jax.random.key(train_key)
    ema = params; ts = []
    for e in range(epochs):
        order = rng.permutation(len(data))
        for s in range(0, len(order), 128):
            key, k = jax.random.split(key)
            params, state, loss, t = step(params, state, data[order[s:s+128]], k)
            ema = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b, ema, params)
            ts.append(np.asarray(t))
    return params, ema, ts

fn = jax.jit(lambda p, k: D.sample(p, k, (500, 28, 28, 1), abar, steps=400, sampler="ddpm", parameterisation="eps"))
def score(p, key):
    x = np.asarray(fn(p, jax.random.key(key)))
    s = np.clip((x + 1) / 2, 0, 1).transpose(0, 3, 1, 2).astype(np.float32)
    return round(scorer.fd(real, s), 3), round(float(s.mean()), 4)

out = {"real_mean": float(real.mean())}
for name, tk, os_ in [("as-run", seed, seed), ("other-train-key", 102, seed), ("other-order", seed, 102)]:
    t0 = time.time()
    p, ema, ts = train(tk, os_)
    last = np.concatenate(ts[-20:])
    out[name] = {"final": score(p, 1000 + seed), "final-other-sample-key": score(p, 7),
                 "ema0.999": score(ema, 1000 + seed),
                 "last20-batches-mean-t": float(last.mean()), "last20-frac-t<50": float((last < 50).mean()),
                 "seconds": time.time() - t0}
    print(name, out[name], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
