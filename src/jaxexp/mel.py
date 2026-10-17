"""Whisper's log-mel front end in ``jnp``, one stage at a time.

The reference is openai-whisper's ``log_mel_spectrogram``, which
``transformers.WhisperFeatureExtractor`` reproduces in NumPy:

1. pad or trim the waveform to 30 s (480,000 samples at 16 kHz);
2. frame it: ``torch.stft(center=True)`` reflect-pads n_fft/2 = 200 samples at
   each end, so frame i is *centred* on sample 160 i; 3001 frames, the last dropped;
3. multiply by a periodic Hann window of 400;
4. real FFT, 201 bins;
5. power |X|^2;
6. Slaney-normalised mel filter bank (80 or 128 filters, 0-8 kHz, Slaney mel scale);
7. log10 of max(mel, 1e-10);
8. clamp from below at (global max - 8);
9. (x + 4) / 4.

Two switches reproduce what candle's ``whisper/audio.rs`` (copied from an early
whisper.cpp) does differently, so each can be measured alone:
- ``center=False``: frame i *starts* at sample 160 i (no reflect padding), and
  the tail is zero-filled;
- ``double_bins=True``: power in bins 1..199 is doubled (candle adds each
  negative-frequency mirror into its positive bin), but not DC or Nyquist.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

SAMPLE_RATE, N_FFT, HOP, N_SAMPLES, N_FRAMES = 16_000, 400, 160, 480_000, 3000


def _hz_to_mel_slaney(f):
    f = np.asarray(f, np.float64)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, 3.0 * f / 200.0)


def _mel_to_hz_slaney(m):
    m = np.asarray(m, np.float64)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), 200.0 * m / 3.0)


def mel_filters(n_mels: int = 80, n_fft: int = N_FFT, sr: int = SAMPLE_RATE) -> np.ndarray:
    """[n_mels, n_fft // 2 + 1] Slaney-normalised triangular filters on the Slaney mel scale,
    0 Hz to sr/2: librosa.filters.mel's defaults, which Whisper's mel_filters.npz holds."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_points = np.linspace(_hz_to_mel_slaney(0.0), _hz_to_mel_slaney(sr / 2), n_mels + 2)
    hz = _mel_to_hz_slaney(mel_points)
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (hz[2 : n_mels + 2] - hz[:n_mels]))[:, None]
    return weights.astype(np.float32)


def stages(audio, filters, *, center: bool = True, double_bins: bool = False, pad_to_30s: bool = True) -> dict:
    """Every intermediate, as jnp arrays: frames [T, 400] (windowed), power [T, 201],
    mel [T, n_mels], log [T, n_mels], features [n_mels, T] (the model's input)."""
    x = jnp.asarray(audio, jnp.float32)
    if pad_to_30s:
        x = jnp.pad(x[:N_SAMPLES], (0, max(0, N_SAMPLES - x.shape[0])))
    if center:
        x = jnp.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
        n_frames = 1 + (x.shape[0] - N_FFT) // HOP - 1  # torch.stft's count, minus the dropped last frame
    else:
        n_frames = x.shape[0] // HOP
        x = jnp.pad(x, (0, max(0, (n_frames - 1) * HOP + N_FFT - x.shape[0])))
    idx = jnp.arange(n_frames)[:, None] * HOP + jnp.arange(N_FFT)[None, :]
    window = 0.5 * (1 - jnp.cos(2 * jnp.pi * jnp.arange(N_FFT) / N_FFT))  # periodic Hann
    frames = x[idx] * window
    spectrum = jnp.fft.rfft(frames, axis=-1)
    power = jnp.real(spectrum) ** 2 + jnp.imag(spectrum) ** 2
    if double_bins:
        power = power.at[:, 1 : N_FFT // 2].multiply(2.0)
    mel = power @ jnp.asarray(filters).T
    log = jnp.log10(jnp.maximum(mel, 1e-10))
    clamped = jnp.maximum(log, log.max() - 8.0)
    return {
        "frames": frames,
        "power": power,
        "mel": mel,
        "log": log,
        "features": ((clamped + 4.0) / 4.0).T,
    }


def log_mel(audio, filters, **kwargs) -> jnp.ndarray:
    """[n_mels, 3000] Whisper input features."""
    return stages(audio, filters, **kwargs)["features"]
