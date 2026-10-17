"""O7: jaxexp.mel is Whisper's front end, and its two switches are candle's conventions."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jaxexp import mel as M

ROOT = Path(__file__).resolve().parents[1]
CANDLE = ROOT / "tools" / "candle_mel"


def _chirp(seconds=2.0, seed=0):
    t = np.arange(int(seconds * M.SAMPLE_RATE)) / M.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    x = 0.3 * np.sin(2 * np.pi * (200 + 1500 * t) * t) * np.exp(-t) + 0.01 * rng.standard_normal(t.size)
    return x.astype(np.float32)


def _reference_float64(audio, filters):
    """openai-whisper's recipe in float64 NumPy: centred reflect-padded frames, periodic Hann, |rfft|^2."""
    x = np.pad(audio.astype(np.float64), (0, M.N_SAMPLES - len(audio)))
    x = np.pad(x, (200, 200), mode="reflect")
    idx = np.arange(3000)[:, None] * M.HOP + np.arange(M.N_FFT)[None, :]
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(M.N_FFT) / M.N_FFT))
    power = np.abs(np.fft.rfft(x[idx] * window, axis=-1)) ** 2
    log = np.log10(np.maximum(power @ filters.T.astype(np.float64), 1e-10))
    return ((np.maximum(log, log.max() - 8) + 4) / 4).T


def test_filters_match_candles_file():
    raw = np.frombuffer((CANDLE / "vendor" / "melfilters.bytes").read_bytes(), "<f4").reshape(80, 201)
    np.testing.assert_allclose(M.mel_filters(80), raw, atol=1e-8)


def test_features_match_float64_whisper_recipe():
    audio, filters = _chirp(), M.mel_filters(80)
    ours = np.asarray(M.log_mel(audio, filters))
    assert ours.shape == (80, M.N_FRAMES)
    np.testing.assert_allclose(ours, _reference_float64(audio, filters), atol=1e-4)


def test_double_bins_is_a_uniform_offset():
    """Every Slaney filter is zero at DC and at Nyquist, so doubling bins 1..199 doubles every mel
    energy exactly: the features shift by log10(2) / 4, everywhere (the floor moves with the max)."""
    audio, filters = _chirp(), M.mel_filters(80)
    assert filters[:, 0].max() == 0 and filters[:, -1].max() == 0
    shift = np.asarray(M.log_mel(audio, filters, double_bins=True) - M.log_mel(audio, filters))
    np.testing.assert_allclose(shift, np.log10(2) / 4, atol=2e-5)


def test_uncentred_frames_start_at_hop_multiples():
    audio = _chirp(0.5)
    frames = np.asarray(M.stages(audio, M.mel_filters(80), center=False)["frames"])
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(M.N_FFT) / M.N_FFT))
    for i in (0, 7, 30):
        np.testing.assert_allclose(frames[i], audio[i * M.HOP : i * M.HOP + M.N_FFT] * window, atol=1e-6)


@pytest.mark.skipif(not (CANDLE / "target" / "release" / "candle-mel").exists(), reason="cargo build tools/candle_mel first")
def test_switches_reproduce_candles_pcm_to_mel():
    audio = _chirp()
    filters = M.mel_filters(80)
    with tempfile.TemporaryDirectory() as tmp:
        pcm, fil, out = (Path(tmp) / n for n in ("pcm", "filters", "out"))
        pcm.write_bytes(audio.astype("<f4").tobytes())
        fil.write_bytes(filters.astype("<f4").tobytes())
        subprocess.run([str(CANDLE / "target" / "release" / "candle-mel"), pcm, fil, "80", out], check=True, capture_output=True)
        candle = np.frombuffer(out.read_bytes(), "<f4").reshape(80, -1)
    replica = np.asarray(M.log_mel(audio, filters, center=False, double_bins=True))
    assert candle.shape == replica.shape
    np.testing.assert_allclose(candle, replica, atol=5e-4)
    assert np.abs(candle - np.asarray(M.log_mel(audio, filters))).max() > 0.5  # and the switches matter
