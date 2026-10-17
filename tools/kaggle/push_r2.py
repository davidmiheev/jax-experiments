"""Write (and optionally push) an R2/R3 Kaggle kernel: one job per T4, two at once.

The kernel unpacks the ``jaxexp-source`` dataset (``package_source.sh``) into
``/kaggle/temp``, installs ``jaxexp`` without dependencies (the image's JAX
0.11.1 is CUDA-enabled), fetches enwik8 once, then runs ``experiments/r2_enwik8/run.py``
twice in parallel, with ``CUDA_VISIBLE_DEVICES=0`` and ``1``. Only result JSON
and logs go to ``/kaggle/working``.

    uv run python tools/kaggle/push_r2.py calibrate                 # writes runs/kaggle-r2-calibrate/
    uv run python tools/kaggle/push_r2.py calibrate --push          # and pushes it

Presets (each is two job lists, one per GPU):
- ``calibrate``: the compile-only memory study (fp32 / fp16, remat none / dots),
  then 300-step fp16 runs of the GRU and the 25M RoPE transformer at T=1024, to
  measure throughput, compile time and fp16 stability before spending hours.
- ``r2``: all of R2 in one kernel: the 25M RoPE transformer, the GRU and the 4M
  transformer at lr 2e-3, seeds 0-2, 12,000 steps at T=1024, one seed per job (so
  each result is written as it finishes), lanes balanced to ~4 h. No remat: the
  calibration measured 6.85 GiB of temp memory for the 25M step in fp16 on a T4.
- ``r2-gru`` / ``r2-rope``: R2's main arms, 12,000 steps, seeds 0-2, T=1024.
- ``r2-4m``: the 4M transformer at lr 2e-3, T=1024 (target 1.544).
- ``r3``: the copy task.
- ``calibrate-fused``: GPU memory and 300-step throughput with fused attention (ml-ladder's
  SDPA protocol; the default from 2026-10-16), which keeps fp32 scores.
- ``r2-fused``: R2's transformer arms with fused attention, seeds 0-2. ``r2`` ran them with
  explicit attention whose fp16 scores were rounded before the softmax (fixed since).
- ``r3-fused``: R3's transformer arm with fused attention.
- ``n1-calibrate``: N1's vmapped step: compiled memory for K = 1..18 copies of the 4M
  transformer (remat none / dots), and steady throughput for K = 1, 2, 4, 8 against K = 1.
- ``r3-s1``: the transformer's seed 1 at n=512 (0.273 in ``r3``, ml-ladder 1.000), repeated
  as is, in fp32 and with fused attention: is the late escape from the plateau numerics?
- ``r3-s1b``: repeats of the fp32 and fused arms (one run each in ``r3-s1``), and of seed 0 in
  fp16 with explicit attention, to tell a systematic lag from run-to-run nondeterminism.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = "experiments/r2_enwik8/run.py"
COMMON = ["--data-dir", "/kaggle/temp/data/text", "--ml-ladder", "/kaggle/temp/src/ml-ladder"]



def _r2(model: str, seed: int, *extra: str, tag: str | None = None) -> list[str]:
    tag = tag or {"transformer-25M-rope": "rope", "gru": "gru", "transformer-4M": "4m"}[model]
    return ["--studies", "enwik8", "--models", model, "--contexts", "1024", "--seeds", str(seed), "--amp",
            *extra, "--out", f"r2-{tag}-s{seed}.json"]


def _r3s1(tag: str, *extra: str, seed: int = 1) -> list[str]:
    return ["--studies", "copy", "--copy-models", "transformer-4M", "--copy-lengths", "512", "--seeds", str(seed),
            *extra, "--out", f"r3-s{seed}-{tag}.json"]


N1 = "@experiments/n1_vmap_sweep/run.py"

PRESETS = {
    "n1-calibrate": [
        [
            [N1, "--studies", "memory", "--ks", "1", "2", "4", "6", "8", "12", "18", "--out", "n1-memory.json"],
            [N1, "--studies", "throughput", "--ks", "1", "2", "4", "8", "--remats", "none", "--out", "n1-throughput-none.json"],
        ],
        [[N1, "--studies", "throughput", "--ks", "1", "2", "4", "8", "12", "--remats", "dots", "--out", "n1-throughput-dots.json"]],
    ],
    "calibrate-fused": [
        [
            ["--studies", "memory", "--models", "transformer-25M-rope", "transformer-4M", "--contexts", "1024",
             "--amp", "--attention", "fused", "--out", "memory-fp16-fused.json"],
            ["--studies", "memory", "--models", "transformer-25M-rope", "--contexts", "1024", "--amp",
             "--attention", "fused", "--remat", "dots", "--out", "memory-fp16-fused-dots.json"],
            ["--studies", "enwik8", "--models", "transformer-4M", "--contexts", "1024", "--seeds", "0", "--amp",
             "--attention", "fused", "--peak-lr", "2e-3", "--enwik8-steps", "300", "--log-every", "50",
             "--out", "calibrate-4m-fused.json"],
        ],
        [
            ["--studies", "enwik8", "--models", "transformer-25M-rope", "--contexts", "1024", "--seeds", "0",
             "--amp", "--attention", "fused", "--enwik8-steps", "300", "--log-every", "50",
             "--out", "calibrate-rope-fused.json"],
        ],
    ],
    "r2-fused": [
        [_r2("transformer-25M-rope", 0, "--attention", "fused", tag="rope-fused"),
         _r2("transformer-25M-rope", 2, "--attention", "fused", tag="rope-fused")],
        [_r2("transformer-25M-rope", 1, "--attention", "fused", tag="rope-fused")]
        + [_r2("transformer-4M", s, "--peak-lr", "2e-3", "--attention", "fused", tag="4m-fused") for s in (0, 1, 2)],
    ],
    "r3-fused": [
        [["--studies", "copy", "--copy-models", "transformer-4M", "--seeds", "0", "1", "--amp", "--attention", "fused",
          "--out", "r3-transformer-fused-a.json"]],
        [["--studies", "copy", "--copy-models", "transformer-4M", "--seeds", "2", "--amp", "--attention", "fused",
          "--out", "r3-transformer-fused-b.json"]],
    ],
    "r3-s1b": [
        [_r3s1("fp32-b"), _r3s1("fp32-c")],
        [_r3s1("fused-b", "--amp", "--attention", "fused"), _r3s1("fused-c", "--amp", "--attention", "fused"),
         _r3s1("repeat-a", "--amp", seed=0)],
    ],
    "r3-s1": [
        [_r3s1("repeat-a", "--amp"), _r3s1("fp32")],
        [_r3s1("repeat-b", "--amp"), _r3s1("fused", "--amp", "--attention", "fused")],
    ],
    "r2": [
        [_r2("transformer-25M-rope", 0), _r2("transformer-25M-rope", 2), _r2("gru", 2)],
        [_r2("transformer-25M-rope", 1), _r2("gru", 0), _r2("gru", 1)]
        + [_r2("transformer-4M", s, "--peak-lr", "2e-3") for s in (0, 1, 2)],
    ],
    "calibrate": [
        [
            [
                "--studies",
                "memory",
                "--models",
                "transformer-25M-rope",
                "transformer-4M",
                "gru",
                "--contexts",
                "1024",
                "--out",
                "memory-fp32.json",
            ],
            [
                "--studies",
                "memory",
                "--models",
                "transformer-25M-rope",
                "--contexts",
                "1024",
                "--amp",
                "--remat",
                "dots",
                "--out",
                "memory-fp16-dots.json",
            ],
            [
                "--studies",
                "memory",
                "--models",
                "transformer-25M-rope",
                "--contexts",
                "1024",
                "--amp",
                "--out",
                "memory-fp16.json",
            ],
            [
                "--studies",
                "enwik8",
                "--models",
                "gru",
                "--contexts",
                "1024",
                "--seeds",
                "0",
                "--amp",
                "--enwik8-steps",
                "300",
                "--log-every",
                "50",
                "--out",
                "calibrate-gru.json",
            ],
        ],
        [
            [
                "--studies",
                "enwik8",
                "--models",
                "transformer-25M-rope",
                "--contexts",
                "1024",
                "--seeds",
                "0",
                "--amp",
                "--remat",
                "dots",
                "--enwik8-steps",
                "300",
                "--log-every",
                "50",
                "--out",
                "calibrate-rope.json",
            ],
            [
                "--studies",
                "enwik8",
                "--models",
                "transformer-4M",
                "--contexts",
                "1024",
                "--seeds",
                "0",
                "--amp",
                "--peak-lr",
                "2e-3",
                "--enwik8-steps",
                "300",
                "--log-every",
                "50",
                "--out",
                "calibrate-4m.json",
            ],
        ],
    ],
    "r2-gru": [
        [
            [
                "--studies",
                "enwik8",
                "--models",
                "gru",
                "--contexts",
                "1024",
                "--seeds",
                "0",
                "2",
                "--amp",
                "--out",
                "r2-gru-a.json",
            ]
        ],
        [
            [
                "--studies",
                "enwik8",
                "--models",
                "gru",
                "--contexts",
                "1024",
                "--seeds",
                "1",
                "--amp",
                "--out",
                "r2-gru-b.json",
            ]
        ],
    ],
    "r2-rope": [
        [
            [
                "--studies",
                "enwik8",
                "--models",
                "transformer-25M-rope",
                "--contexts",
                "1024",
                "--seeds",
                "0",
                "--amp",
                "--remat",
                "dots",
                "--out",
                "r2-rope-a.json",
            ]
        ],
        [
            [
                "--studies",
                "enwik8",
                "--models",
                "transformer-25M-rope",
                "--contexts",
                "1024",
                "--seeds",
                "1",
                "--amp",
                "--remat",
                "dots",
                "--out",
                "r2-rope-b.json",
            ]
        ],
    ],
    "r2-4m": [
        [
            [
                "--studies",
                "enwik8",
                "--models",
                "transformer-4M",
                "--contexts",
                "1024",
                "--seeds",
                "0",
                "2",
                "--amp",
                "--peak-lr",
                "2e-3",
                "--out",
                "r2-4m-a.json",
            ]
        ],
        [
            [
                "--studies",
                "enwik8",
                "--models",
                "transformer-4M",
                "--contexts",
                "1024",
                "--seeds",
                "1",
                "--amp",
                "--peak-lr",
                "2e-3",
                "--out",
                "r2-4m-b.json",
            ]
        ],
    ],
    "r3": [
        [
            [
                "--studies",
                "copy",
                "--copy-models",
                "transformer-4M",
                "--seeds",
                "0",
                "1",
                "2",
                "--amp",
                "--out",
                "r3-transformer.json",
            ]
        ],
        [["--studies", "copy", "--copy-models", "lstm", "--seeds", "0", "1", "2", "--amp", "--out", "r3-lstm.json"]],
    ],
}

KERNEL = """\
# Generated by tools/kaggle/push_r2.py ({preset}). Do not edit; regenerate.
import os, subprocess, sys, tarfile, time, urllib.request, zipfile, io, json
from pathlib import Path

JOBS = {jobs}
SRC, DATA, OUT = Path("/kaggle/temp/src"), Path("/kaggle/temp/data/text"), Path("/kaggle/working")
SRC.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
import shutil
# Kaggle unpacks uploaded tarballs, so each repo arrives as a directory (or, failing that, a .tar).
MARKERS = {{"jax-experiments": "experiments/r2_enwik8/run.py", "ml-ladder": "tools/gpu/c_context_scaling.py"}}
for name, marker in MARKERS.items():
    found = next((p for p in Path("/kaggle/input").rglob(marker) if p.parents[marker.count("/")].name == name), None)
    if found is not None:
        shutil.copytree(found.parents[marker.count("/")], SRC / name, dirs_exist_ok=True)
    else:
        tar = next(Path("/kaggle/input").rglob(name + ".tar"))
        with tarfile.open(tar) as t:
            t.extractall(SRC)
    assert (SRC / name / marker).exists(), name
commits = next(Path("/kaggle/input").rglob("COMMITS"), None)
print("source:", sorted(p.name for p in SRC.iterdir()), commits.read_text() if commits else "no COMMITS file", flush=True)
pip = [sys.executable, "-m", "pip", "install", "--no-deps", "-q", "-e", str(SRC / "jax-experiments")]
subprocess.run(pip, check=True)
if not (DATA / "enwik8").exists():  # once, before the parallel jobs
    raw = urllib.request.urlopen("https://mattmahoney.net/dc/enwik8.zip", timeout=600).read()
    (DATA / "enwik8").write_bytes(zipfile.ZipFile(io.BytesIO(raw)).read("enwik8"))
common = {common}

def run_lane(gpu, jobs):
    # 0.92 of the card instead of JAX's default 0.75: N1 packs K model copies into one program.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", XLA_PYTHON_CLIENT_MEM_FRACTION="0.92")
    for args in jobs:
        script = "{run}"
        if args[0].startswith("@"):  # a job may name its own script: ["@experiments/.../run.py", ...]
            script, args = args[0][1:], args[1:]
        out = args[args.index("--out") + 1]
        args = args[: args.index("--out")] + ["--out", str(OUT / out)] + args[args.index("--out") + 2 :]
        log = open(OUT / (Path(out).stem + ".log"), "w")
        cmd = [sys.executable, str(SRC / "jax-experiments" / script)] + common + args
        print(f"[gpu{{gpu}}]", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=SRC / "jax-experiments", env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[gpu{{gpu}}] exit {{rc}} for {{out}}", flush=True)

import threading
threads = [threading.Thread(target=run_lane, args=(g, lane)) for g, lane in enumerate(JOBS)]
started = time.time()
[t.start() for t in threads]; [t.join() for t in threads]
print(f"all done in {{time.time() - started:.0f}} s", flush=True)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", choices=list(PRESETS))
    parser.add_argument("--user", default="davidmiheev")
    parser.add_argument("--push", action="store_true", help="push with the kaggle CLI (uses GPU quota)")
    args = parser.parse_args()
    slug = f"jaxexp-{args.preset}"
    out = ROOT / "runs" / f"kaggle-{args.preset}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(
        KERNEL.format(preset=args.preset, jobs=json.dumps(PRESETS[args.preset]), common=json.dumps(COMMON), run=RUN)
    )
    meta = {
        "id": f"{args.user}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [f"{args.user}/jaxexp-source"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(
        textwrap.dedent(f"""\
        wrote {out}/kernel.py and kernel-metadata.json ({args.user}/{slug})
        lanes: {[len(lane) for lane in PRESETS[args.preset]]} jobs on GPU 0 / GPU 1""")
    )
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)


if __name__ == "__main__":
    main()
