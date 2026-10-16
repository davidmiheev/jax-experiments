"""O7: Whisper's log-mel front end five ways, and whether candle's departs enough to change transcripts.

ml-ladder (part I, "Whisper through the candle fork") left one thread open: with
prompts and timestamp rules matched, candle's whisper decoders still disagree
with transformers on about a third of 61 tiny.en transcripts, and the shared
suspect is candle's ``pcm_to_mel``. This experiment measures it.

``stages``  On the same 61 clips, five implementations of the front end:
            transformers' WhisperFeatureExtractor (its torch path, which is what
            ran in ml-ladder, and its NumPy path), torchaudio, ``jaxexp.mel`` in
            ``jnp``, and candle's own ``pcm_to_mel`` (vendored verbatim, run as
            ``tools/candle_mel``). Diffed stage by stage against transformers:
            window, filters, power, mel, log, features. candle exposes only its
            output, so it is matched by a ``jnp`` replica with candle's two
            conventions switched on (frames start at 160 i instead of being
            centred; power in bins 1..199 doubled), and the replica's stages say
            where candle departs and by how much.
``decode``  transformers' tiny.en ``generate`` (timestamps on, greedy, the
            protocol of ml-ladder's ``whisper_hf_timestamps.py``) fed each
            variant's features. Same decoder, only the front end changes. The
            transcripts are compared with ml-ladder's recorded transformers
            tokens (which checks that the clips are rebuilt exactly) and with
            ml-ladder's candle example run (fixed prompt and pairing rule,
            fallback off), which is candle's decoder on candle's features.

    uv run --isolated --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \\
        --index-strategy unsafe-best-match --with-requirements experiments/o7_mel/requirements.txt \\
        --with-editable . python experiments/o7_mel/run.py stages decode plot
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxexp import mel as M  # noqa: E402

DATA = ROOT / "data" / "audio"
OUT = ROOT / "docs" / "data"
FIG = ROOT / "docs" / "figures"
ML_LADDER = Path(os.environ.get("ML_LADDER", ROOT.parent / "ml-ladder"))
CANDLE = ROOT / "tools" / "candle_mel"
CANDLE_FILTERS = CANDLE / "vendor" / "melfilters.bytes"
CANDLE_FILTERS_SHA256 = "85818f156f7e189453901a515e4726d270d307f976e161cf9403e8caab405498"
AUDIO_RS_SHA256 = "ed5f1c81a0bebf0e67f8066f51d87cf8312a48aadb3a9a3a126ab7c09714880b"
FSDD_URL = "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/"
JFK_URL = "https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_jfk.wav"
MODEL = "openai/whisper-tiny.en"

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
NUMERALS = {str(i): WORDS[i] for i in range(10)}
HOMOPHONES = {"to": "two", "too": "two", "for": "four", "won": "one", "ate": "eight", "oh": "zero"}
JFK_WORDS = (
    "and so my fellow americans ask not what your country can do for you ask what you can do for your country".split()
)


# ---------------------------------------------------------------------------------------------
# Clips: ml-ladder's 61, rebuilt (whisper_example_rules.py's list; resample_poly 8 -> 16 kHz)


def _fetch(url: str, path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def _wav_int16(payload: bytes, rate: int) -> np.ndarray:
    from scipy.io import wavfile

    sr, data = wavfile.read(io.BytesIO(payload))
    assert sr == rate and data.dtype == np.int16, (sr, data.dtype)
    return data.astype(np.float32) / 32768.0


def _to_16k_int16(x8k: np.ndarray) -> np.ndarray:
    """resample_poly(2, 1), then the int16 round trip of writing a 16 kHz wav and reading it back
    as the example (symphonia) and ml-ladder's transformers script do: s / 32768."""
    from scipy.signal import resample_poly

    y = resample_poly(x8k, 2, 1).astype(np.float32)
    q = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
    return q.astype(np.float32) / 32768.0


def clips() -> list[dict]:
    digit_names = sorted(f"{d}_yweweler_{i}.wav" for d in range(10) for i in range(5))
    raw = {n: _wav_int16(_fetch(FSDD_URL + n, DATA / "fsdd" / n), 8000) for n in digit_names}
    out = [
        {"name": n, "kind": "digit", "words": [WORDS[int(n[0])]], "audio": _to_16k_int16(raw[n])} for n in digit_names
    ]
    rng = np.random.default_rng(0)
    gap = np.zeros(2000, np.float32)  # 0.25 s at 8 kHz
    for s in range(10):
        chosen = rng.choice(len(digit_names), size=4, replace=False)
        parts = []
        for k, c in enumerate(chosen):
            parts += [gap] * (k > 0) + [raw[digit_names[c]]]
        out.append(
            {
                "name": f"seq_{s}.wav",
                "kind": "sequence",
                "words": [WORDS[int(digit_names[c][0])] for c in chosen],
                "audio": _to_16k_int16(np.concatenate(parts)),
            }
        )
    jfk = _wav_int16(_fetch(JFK_URL, DATA / "samples_jfk.wav"), 16000)
    out.append({"name": "samples_jfk.wav", "kind": "long", "words": JFK_WORDS, "audio": jfk})
    return out


# ---------------------------------------------------------------------------------------------
# The five front ends


def candle_filters() -> np.ndarray:
    raw = CANDLE_FILTERS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CANDLE_FILTERS_SHA256
    return np.frombuffer(raw, "<f4").reshape(80, 201).copy()


def candle_binary() -> Path:
    assert hashlib.sha256((CANDLE / "vendor" / "audio.rs").read_bytes()).hexdigest() == AUDIO_RS_SHA256
    binary = CANDLE / "target" / "release" / "candle-mel"
    if not binary.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=CANDLE, check=True)
    return binary


def candle_features(audio: np.ndarray, binary: Path, filters: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        pcm, fil, out = (Path(tmp) / n for n in ("pcm", "filters", "out"))
        pcm.write_bytes(audio.astype("<f4").tobytes())
        fil.write_bytes(filters.astype("<f4").tobytes())
        subprocess.run([str(binary), str(pcm), str(fil), "80", str(out)], check=True, capture_output=True)
        return np.frombuffer(out.read_bytes(), "<f4").reshape(80, -1).copy()


class Reference:
    """transformers' torch path, stage by stage (and checked against the extractor's own output)."""

    def __init__(self):
        import torch
        from transformers import WhisperFeatureExtractor

        self.torch = torch
        self.fe = WhisperFeatureExtractor.from_pretrained(MODEL)
        self.window = torch.hann_window(M.N_FFT)
        self.filters = torch.from_numpy(self.fe.mel_filters).float()  # [201, 80]

    def stages(self, audio: np.ndarray) -> dict:
        t = self.torch
        x = np.pad(audio[: M.N_SAMPLES], (0, max(0, M.N_SAMPLES - len(audio))))
        stft = t.stft(t.from_numpy(x), M.N_FFT, M.HOP, window=self.window, return_complex=True)
        power = stft[..., :-1].abs() ** 2
        mel = self.filters.T @ power
        log = t.clamp(mel, min=1e-10).log10()
        feats = (t.maximum(log, log.max() - 8.0) + 4.0) / 4.0
        out = {"power": power.T.numpy(), "mel": mel.T.numpy(), "log": log.T.numpy(), "features": feats.numpy()}
        own = self.fe(audio, sampling_rate=M.SAMPLE_RATE, return_tensors="np").input_features[0]
        out["extractor"] = own
        out["extractor_numpy"] = self.fe._np_extract_fbank_features(x[None], "cpu")[0].astype(np.float32)
        return out


class TorchAudio:
    def __init__(self):
        import torch
        import torchaudio

        self.torch = torch
        self.spec = torchaudio.transforms.Spectrogram(n_fft=M.N_FFT, hop_length=M.HOP, power=2.0)
        self.filters = torchaudio.functional.melscale_fbanks(
            M.N_FFT // 2 + 1, 0.0, M.SAMPLE_RATE / 2, 80, M.SAMPLE_RATE, norm="slaney", mel_scale="slaney"
        )  # [201, 80]

    def stages(self, audio: np.ndarray) -> dict:
        t = self.torch
        x = np.pad(audio[: M.N_SAMPLES], (0, max(0, M.N_SAMPLES - len(audio))))
        power = self.spec(t.from_numpy(x))[..., :-1]  # [201, 3000]
        mel = power.T @ self.filters
        log = t.clamp(mel, min=1e-10).log10()
        feats = ((t.maximum(log, log.max() - 8.0) + 4.0) / 4.0).T
        return {"power": power.T.numpy(), "mel": mel.numpy(), "log": log.numpy(), "features": feats.numpy()}


def jnp_stages(audio, filters, **kw) -> dict:
    return {k: np.asarray(v) for k, v in M.stages(audio, filters, **kw).items()}


# ---------------------------------------------------------------------------------------------
# stages


def _rel(a, b) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def _maxabs(a, b) -> float:
    return float(np.abs(a - b).max())


def _feature_diff(a, b, n_speech: int) -> dict:
    d = np.abs(a - b)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "mean_abs_speech": float(d[:, :n_speech].mean()),
        "max_abs_speech": float(d[:, :n_speech].max()),
    }


def run_stages(args) -> dict:
    all_clips = clips()
    ref, ta = Reference(), TorchAudio()
    filters_jnp = M.mel_filters(80)
    filters_candle = candle_filters()
    binary = candle_binary()
    hf_filters = ref.fe.mel_filters.T  # [80, 201] float64

    filt = {
        "jnp_vs_transformers_float64": _maxabs(filters_jnp, hf_filters),
        "candle_file_vs_transformers_float64": _maxabs(filters_candle, hf_filters),
        "torchaudio_vs_transformers_float64": _maxabs(ta.filters.numpy().T, hf_filters),
        "filters_max": float(hf_filters.max()),
    }
    n = np.arange(M.N_FFT)
    hann64 = 0.5 * (1 - np.cos(2 * np.pi * n / M.N_FFT))
    hann_candle = np.float32(0.5) * (np.float32(1) - np.cos((np.float32(2 * np.pi) * n.astype(np.float32)) / np.float32(400)))
    window = {
        "torch_vs_float64": _maxabs(ref.window.numpy(), hann64),
        "jnp_vs_float64": _maxabs(np.asarray(0.5 * (1 - np.cos(2 * np.pi * n / M.N_FFT)), np.float32), hann64),
        "candle_formula_f32_vs_float64": _maxabs(hann_candle, hann64),
    }

    variants = {
        "jnp": {},
        "jnp-candle-filters": {"filters": filters_candle},
        "replica": {"filters": filters_candle, "center": False, "double_bins": True},
        "no-center": {"center": False},
        "double-bins": {"double_bins": True},
    }
    rows, features = [], {}
    started = time.time()
    for c in all_clips:
        audio = c["audio"]
        n_speech = min(M.N_FRAMES, -(-len(audio) // M.HOP))
        r = ref.stages(audio)
        t = ta.stages(audio)
        row = {"clip": c["name"], "kind": c["kind"], "samples": len(audio), "speech_frames": n_speech}
        row["extractor_vs_stages"] = _maxabs(r["extractor"], r["features"])
        row["numpy_path"] = _feature_diff(r["extractor_numpy"], r["features"], n_speech)
        row["torchaudio"] = {
            "power_rel": _rel(t["power"], r["power"]),
            "mel_rel": _rel(t["mel"], r["mel"]),
            "log_max_abs": _maxabs(t["log"], r["log"]),
            "features": _feature_diff(t["features"], r["features"], n_speech),
        }
        js = {name: jnp_stages(audio, kw.get("filters", filters_jnp), **{k: v for k, v in kw.items() if k != "filters"}) for name, kw in variants.items()}
        j = js["jnp"]
        row["jnp"] = {
            "power_rel": _rel(j["power"], r["power"]),
            "mel_rel": _rel(j["mel"], r["mel"]),
            "log_max_abs": _maxabs(j["log"], r["log"]),
            "features": _feature_diff(j["features"], r["features"], n_speech),
        }
        cf = candle_features(audio, binary, filters_candle)
        assert cf.shape == (80, M.N_FRAMES), cf.shape
        row["candle"] = {
            "features": _feature_diff(cf, r["features"], n_speech),
            "vs_replica": _feature_diff(cf, js["replica"]["features"], n_speech),
        }
        for name in ("jnp-candle-filters", "replica", "no-center", "double-bins"):
            row[name] = {"features": _feature_diff(js[name]["features"], r["features"], n_speech)}
        # Where the replica departs: power at the same frame index, and the global max that sets the floor.
        row["replica"]["power_rel_same_index"] = _rel(js["replica"]["power"], r["power"])
        row["replica"]["log_max_shift"] = float(js["replica"]["log"].max() - r["log"].max())
        row["no-center"]["log_max_shift"] = float(js["no-center"]["log"].max() - r["log"].max())
        row["double-bins"]["log_max_shift"] = float(js["double-bins"]["log"].max() - r["log"].max())
        # The doubling, isolated: centred frames, doubled bins, against the reference's own power.
        rp = r["power"].copy()
        rp[:, 1 : M.N_FFT // 2] *= 2
        row["double-bins"]["mel_ratio_median"] = float(
            np.median((rp @ hf_filters.T.astype(np.float32))[:n_speech] / np.maximum(r["mel"][:n_speech], 1e-30))
        )
        rows.append(row)
        features[c["name"]] = {
            "transformers": r["extractor"],
            "transformers-numpy": r["extractor_numpy"],
            "torchaudio": t["features"],
            "candle": cf,
            **{name: js[name]["features"] for name in variants},
        }
        print(
            f"{c['name']:>18}  jnp {row['jnp']['features']['max_abs']:.1e}  ta {row['torchaudio']['features']['max_abs']:.1e}"
            f"  candle {row['candle']['features']['mean_abs_speech']:.3f}  candle-replica {row['candle']['vs_replica']['max_abs']:.1e}",
            flush=True,
        )
    np.savez_compressed(
        ROOT / "runs" / "o7-features.npz",
        **{f"{clip}|{name}": f for clip, d in features.items() for name, f in d.items()},
    )

    def summary(path):
        vals = []
        for row in rows:
            v = row
            for k in path:
                v = v[k]
            vals.append(v)
        vals = np.array(vals)
        return {"median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())}

    keys = {
        "transformers extractor vs its stages": ["extractor_vs_stages"],
        "transformers numpy path, features max_abs": ["numpy_path", "max_abs"],
        "torchaudio power_rel": ["torchaudio", "power_rel"],
        "torchaudio mel_rel": ["torchaudio", "mel_rel"],
        "torchaudio log_max_abs": ["torchaudio", "log_max_abs"],
        "torchaudio features max_abs": ["torchaudio", "features", "max_abs"],
        "jnp power_rel": ["jnp", "power_rel"],
        "jnp mel_rel": ["jnp", "mel_rel"],
        "jnp log_max_abs": ["jnp", "log_max_abs"],
        "jnp features max_abs": ["jnp", "features", "max_abs"],
        "jnp with candle's filter file, features max_abs": ["jnp-candle-filters", "features", "max_abs"],
        "candle features max_abs": ["candle", "features", "max_abs"],
        "candle features mean_abs_speech": ["candle", "features", "mean_abs_speech"],
        "candle features mean_abs (all 3000 frames)": ["candle", "features", "mean_abs"],
        "candle vs replica max_abs": ["candle", "vs_replica", "max_abs"],
        "candle vs replica mean_abs": ["candle", "vs_replica", "mean_abs"],
        "replica features mean_abs_speech": ["replica", "features", "mean_abs_speech"],
        "replica log_max_shift": ["replica", "log_max_shift"],
        "no-center features mean_abs_speech": ["no-center", "features", "mean_abs_speech"],
        "no-center features max_abs": ["no-center", "features", "max_abs"],
        "no-center log_max_shift": ["no-center", "log_max_shift"],
        "double-bins features mean_abs_speech": ["double-bins", "features", "mean_abs_speech"],
        "double-bins features max_abs": ["double-bins", "features", "max_abs"],
        "double-bins log_max_shift": ["double-bins", "log_max_shift"],
        "double-bins mel_ratio_median": ["double-bins", "mel_ratio_median"],
    }
    report = {
        "what": "Whisper log-mel front end five ways on ml-ladder's 61 I-part clips, diffed against transformers' "
        "WhisperFeatureExtractor (torch path) stage by stage; candle's pcm_to_mel run verbatim and matched by a jnp replica",
        "clips": "FSDD yweweler digits 0-9 x 0-4 and ten 4-digit sequences (default_rng(0), 0.25 s gaps at 8 kHz), "
        "resample_poly 8 -> 16 kHz and an int16 round trip; samples_jfk.wav",
        "env": _env(),
        "seconds": round(time.time() - started, 1),
        "window_max_abs_vs_float64": window,
        "filters_max_abs": filt,
        "summary": {k: summary(v) for k, v in keys.items()},
        "rows": rows,
    }
    _write(OUT / "o7-stages.json", report)
    return report


# ---------------------------------------------------------------------------------------------
# decode


def words_of(text: str) -> list[str]:
    out = []
    for token in re.findall(r"[a-z0-9']+", text.lower()):
        if token.isdigit() and len(token) > 1:
            out += [WORDS[int(ch)] for ch in token]
        else:
            out.append(NUMERALS.get(token, token))
    return out


def wer(ref, hyp) -> float:
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return float(d[len(hyp)] / max(len(ref), 1))


def scores(rows: list[dict]) -> dict:
    out = {}
    for kind in ("digit", "sequence", "long"):
        sub = [r for r in rows if r["kind"] == kind]
        hyps = [words_of(r["text"]) for r in sub]
        out[kind] = {"wer": float(np.mean([wer(r["words"], h) for r, h in zip(sub, hyps, strict=True)]))}
        if kind == "digit":
            out[kind]["exact"] = float(
                np.mean([h == r["words"] or [HOMOPHONES.get(w, w) for w in h] == r["words"] for r, h in zip(sub, hyps, strict=True)])
            )
    return out


def _strip_eot(tokens: list[int]) -> list[int]:
    return tokens[:-1] if tokens and tokens[-1] == 50256 else tokens


def run_decode(args) -> dict:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(os.cpu_count() or 4)
    store = np.load(ROOT / "runs" / "o7-features.npz")
    all_clips = {c["name"]: c for c in clips()}
    processor = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL).eval()
    ladder = json.loads((ML_LADDER / "docs" / "writeups" / "data" / "i-whisper-example-rules.json").read_text())
    hf_ref = ladder["arms"]["hf-timestamps"]["transcripts"]
    candle_arms = {k: ladder["arms"][k]["transcripts"] for k in ("prompt+pair-greedy", "prompt+pair")}
    names = sorted({k.split("|")[1] for k in store.files}, key=lambda s: (s != "transformers", s))
    arms, started = {}, time.time()
    for name in names:
        rows = []
        for clip, c in all_clips.items():
            feats = torch.from_numpy(store[f"{clip}|{name}"][None])
            with torch.no_grad():
                ids = model.generate(feats, return_timestamps=True, max_new_tokens=200)
            toks = ids[0].tolist()
            if toks[0] != model.config.decoder_start_token_id:
                toks = [model.config.decoder_start_token_id, *toks]
            text = processor.decode(ids[0], skip_special_tokens=True).strip()
            rows.append({"clip": clip, "kind": c["kind"], "words": c["words"], "text": text, "tokens": toks})
        arms[name] = {"scores": scores(rows), "rows": rows}
        print(name, json.dumps(arms[name]["scores"]), f"{time.time() - started:.0f}s", flush=True)

    def agreement(rows, other: dict) -> dict:
        tok = [r["clip"] for r in rows if _strip_eot(r["tokens"]) == _strip_eot(other[r["clip"]]["tokens"])]
        wrd = [r["clip"] for r in rows if words_of(r["text"]) == words_of(other[r["clip"]]["text"])]
        return {"tokens": len(tok), "words": len(wrd), "words_differ": sorted(set(all_clips) - set(wrd))}

    ours_ref = {r["clip"]: r for r in arms["transformers"]["rows"]}
    comparisons = {}
    for name, arm in arms.items():
        comparisons[name] = {
            "vs_mlladder_transformers": agreement(arm["rows"], hf_ref),
            "vs_this_run_transformers": agreement(arm["rows"], ours_ref),
            **{f"vs_candle_example_{k}": agreement(arm["rows"], v) for k, v in candle_arms.items()},
        }
        print(name, {k: (v["tokens"], v["words"]) for k, v in comparisons[name].items()}, flush=True)
    candle_vs_hf = {
        k: agreement([{"clip": c, **v} for c, v in t.items()], hf_ref) for k, t in candle_arms.items()
    }
    report = {
        "what": "transformers' tiny.en generate (return_timestamps=True, max_new_tokens=200, greedy; ml-ladder's "
        "whisper_hf_timestamps.py protocol) fed each front end's features; only the features change",
        "env": _env(),
        "seconds": round(time.time() - started, 1),
        "references": {
            "mlladder_transformers": "ml-ladder i-whisper-example-rules.json arm hf-timestamps (transformers on the same clips)",
            "candle_example_prompt+pair-greedy": "candle's whisper example with prompt and pairing rule fixed, temperature fallback off (CPU, f32)",
            "candle_example_prompt+pair": "the same with candle's fallback on",
        },
        "candle_example_vs_mlladder_transformers": candle_vs_hf,
        "comparisons": comparisons,
        "arms": arms,
    }
    _write(OUT / "o7-decode.json", report)
    return report


# ---------------------------------------------------------------------------------------------


def run_plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = np.load(ROOT / "runs" / "o7-features.npz")
    clip = "seq_0.wav"
    ref = store[f"{clip}|transformers"]
    cand = store[f"{clip}|candle"]
    n = 300
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, img, title in (
        (axes[0], ref[:, :n], "transformers"),
        (axes[1], cand[:, :n], "candle pcm_to_mel"),
    ):
        im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma", vmin=ref.min(), vmax=ref.max())
        ax.set_title(f"{title}: {clip}, first 3 s")
        ax.set_xlabel("frame (10 ms)")
        ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=axes[:2], shrink=0.8)
    d = cand[:, :n] - ref[:, :n]
    lim = np.abs(d).max()
    im2 = axes[2].imshow(d, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("candle - transformers")
    axes[2].set_xlabel("frame (10 ms)")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "o7-mel-candle-vs-transformers.png", dpi=130)

    stages = json.loads((OUT / "o7-stages.json").read_text())
    decode = json.loads((OUT / "o7-decode.json").read_text())
    order = ["transformers", "transformers-numpy", "torchaudio", "jnp", "jnp-candle-filters", "double-bins", "no-center", "replica", "candle"]
    rows = stages["rows"]
    feat_key = {"transformers-numpy": ("numpy_path",), "torchaudio": ("torchaudio", "features"), "jnp": ("jnp", "features")}

    def mean_abs(name):
        if name == "transformers":
            return 0.0
        path = feat_key.get(name, (name, "features"))
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v[k]
            vals.append(v["mean_abs_speech"])
        return float(np.median(vals))

    agree = [decode["comparisons"][n]["vs_this_run_transformers"]["words"] for n in order]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    x = np.arange(len(order))
    ax[0].bar(x, [max(mean_abs(n), 1e-9) for n in order], color="tab:blue")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, order, rotation=35, ha="right")
    ax[0].set_ylabel("median over clips of mean |features - transformers|\n(speech frames)")
    ax[0].set_title("Feature distance from transformers")
    ax[1].bar(x, agree, color="tab:green")
    ax[1].axhline(decode["candle_example_vs_mlladder_transformers"]["prompt+pair-greedy"]["words"], color="k", ls="--", lw=1)
    ax[1].text(len(order) - 0.5, decode["candle_example_vs_mlladder_transformers"]["prompt+pair-greedy"]["words"] + 0.8,
               "candle's own decoder on candle's features", ha="right", fontsize=8)
    ax[1].set_xticks(x, order, rotation=35, ha="right")
    ax[1].set_ylim(0, 63)
    ax[1].set_ylabel("transcripts equal to transformers' (of 61)")
    ax[1].set_title("Same decoder (transformers), front end swapped")
    fig.savefig(FIG / "o7-front-end-to-transcripts.png", dpi=130)
    print("wrote figures")


# ---------------------------------------------------------------------------------------------


def _env() -> dict:
    import jax
    import scipy
    import torch
    import torchaudio
    import transformers

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "candle_audio_rs": "davidmiheev/candle@9aa02f10 candle-transformers/src/models/whisper/audio.rs, sha256 " + AUDIO_RS_SHA256,
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
    }


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    print("wrote", path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="+", choices=["stages", "decode", "plot"])
    args = parser.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    for part in args.parts:
        {"stages": run_stages, "decode": run_decode, "plot": run_plot}[part](args)


if __name__ == "__main__":
    main()
