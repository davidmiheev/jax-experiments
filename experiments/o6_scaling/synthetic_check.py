"""Synthetic check of O6's analysis: rows from a known law (E=0.9, alpha=0.30, beta=0.28, so a=0.483) at the
planned design (sizes s0-s8, C=3e13..1e16, 2e6 <= D <= the corpus, 3 seeds) with 1% noise.

    python experiments/o6_scaling/synthetic_check.py
"""
import sys, time, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import analyze as A
SIZES = {"s0": (96, 2), "s1": (128, 2), "s2": (160, 3), "s3": (192, 4), "s4": (256, 4), "s5": (320, 5), "s6": (384, 6), "s7": (448, 7), "s8": (512, 8)}
NNE = {"s0": 223680, "s1": 396544, "s2": 927840, "s3": 1779456, "s4": 3159040, "s5": 6164800, "s6": 10646784, "s7": 16899904, "s8": 25219072}
E, Acoef, Bcoef, al, be = 0.9, 40.0, 60.0, 0.30, 0.28
rng = np.random.default_rng(0); rows = []
for C in (3e13, 1e14, 3e14, 1e15, 3e15, 1e16):
    for s, (w, l) in SIZES.items():
        D = C / (6 * NNE[s])
        if not (2e6 <= D <= 3.53e8): continue
        for seed in range(3):
            L = (E + Acoef / NNE[s] ** al + Bcoef / D ** be) * math.exp(rng.normal(0, 0.01))
            rows.append({"size": s, "width": w, "layers": l, "seed": seed, "budget": C, "non_embedding": NNE[s], "tokens": D, "bits_per_byte": L})
print(len(rows), "rows; true a =", be / (al + be))
t = time.time(); print("isoflop", {k: v for k, v in A.isoflop_fit(rows).items() if k != "contours"})
p = A.parametric_fit(rows); print("parametric", {k: round(v, 4) for k, v in p.items() if k not in ("count", "x")}, round(time.time() - t, 1), "s")
print("true cost", A.true_cost_optimum(p))
print("with head", {k: round(v, 4) for k, v in A.parametric_fit(rows, "with_head").items() if k not in ("count", "x")})
A.BOOT = 20; t = time.time(); b = A.bootstrap(rows); print("bootstrap(20)", round(time.time() - t, 1), "s", json.dumps(b))
