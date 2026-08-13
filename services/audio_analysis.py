"""Turn a block of samples into named signals.

Pure numpy, no IO. Runs on the capture thread, so nothing here may touch GL,
ImGui or any state the frame loop owns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FFT_SIZE = 2048
HOP = 1024               # frames between analyses; the capture's callback size
N_MEL = 40

SIGNAL_NAMES: tuple[str, ...] = ("bass", "mid", "presence", "hi", "volume")

# The spectrum is read in decibels over this fixed window, chosen so ordinary
# material rests across the middle of the scale and a room's hiss floor does
# not. A linear magnitude puts everything real against the bottom instead, and
# makes each band a spiky ratio of energies rather than a level.
DB_MIN = -90.0
DB_MAX = -20.0

# `volume` measures the whole block at once, which sits far above any single
# bin, so it needs a window of its own.
LEVEL_DB_MIN = -60.0
LEVEL_DB_MAX = -6.0

# The spectrum is averaged over this long before anything reads it. Without it
# a band is one block's sample of a noisy quantity, which reads as vibration.
SMOOTHING_SECONDS = 0.075

# Below this a bin is silence rather than a very negative number of decibels.
_MAG_FLOOR = 1e-9

# The four spectral bands, in Hz. `volume` is RMS and has no band.
BAND_EDGES_HZ: tuple[tuple[str, float, float], ...] = (
    ("bass", 20.0, 250.0),
    ("mid", 250.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("hi", 6000.0, 20000.0),
)

# A band's running peak decays by this factor per block, so auto-gain follows a
# track down as well as up.
_PEAK_DECAY = 0.9995
# Below this the peak is treated as silence rather than divided by.
_PEAK_FLOOR = 1e-6


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def analysis_matrix(sample_rate: float, fft_size: int = FFT_SIZE,
                    n_mel: int = N_MEL) -> np.ndarray:
    """Rows 0..n_mel-1 are a mel filterbank; the last four average one band each.

    Both live in one matrix so a single matmul yields the display spectrum and
    the band signals together.
    """
    n_bins = fft_size // 2 + 1
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    out = np.zeros((n_mel + len(BAND_EDGES_HZ), n_bins), dtype=np.float32)

    nyquist = sample_rate / 2.0
    edges = _mel_to_hz(np.linspace(_hz_to_mel(20.0),
                                   _hz_to_mel(min(16000.0, nyquist)),
                                   n_mel + 2))
    for i in range(n_mel):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs >= lo) & (freqs < ctr)
        falling = (freqs >= ctr) & (freqs < hi)
        if ctr > lo:
            out[i, rising] = (freqs[rising] - lo) / (ctr - lo)
        if hi > ctr:
            out[i, falling] = (hi - freqs[falling]) / (hi - ctr)
        # Each row averages, like the band rows below. A high mel band spans
        # twenty times the bins of a low one, so raw triangles would draw any
        # spectrum at all as a ramp rising to the right.
        total = out[i].sum()
        if total > 0.0:
            out[i] /= total
        else:
            nearest = int(np.argmin(np.abs(freqs - ctr)))
            out[i, nearest] = 1.0

    for j, (_name, lo, hi) in enumerate(BAND_EDGES_HZ):
        sel = (freqs >= lo) & (freqs < min(hi, nyquist))
        count = int(np.count_nonzero(sel))
        if count:
            out[n_mel + j, sel] = 1.0 / count
    return out


@dataclass(frozen=True)
class SignalSnapshot:
    """One analysis result. Immutable so the frame loop may read it without a
    lock while the capture thread builds the next one."""
    signals: dict[str, float]
    mel: np.ndarray
    seq: int


class Analyzer:
    """Stateful across blocks: auto-gain peaks and the sequence counter."""

    def __init__(self, sample_rate: float, fft_size: int = FFT_SIZE,
                 n_mel: int = N_MEL, auto_gain: bool = True,
                 hop: int = HOP) -> None:
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.n_mel = int(n_mel)
        self.auto_gain = bool(auto_gain)
        self._m = analysis_matrix(sample_rate, fft_size, n_mel)
        # The window carries its own coherent gain, so a full-scale sine reads
        # 1.0 at its bin. Without it a band is raw FFT magnitude, which is far
        # above the [0,1] clip for any real input - every band would pin at 1.0
        # and switching auto-gain off would do nothing.
        window = np.hanning(fft_size)
        self._window = (window * (2.0 / max(window.sum(), 1e-9))).astype(
            np.float32)
        self._peaks = np.full(len(BAND_EDGES_HZ) + 1, _PEAK_FLOOR,
                              dtype=np.float32)
        self._spectrum = np.zeros(fft_size // 2 + 1, dtype=np.float32)
        self._level = 0.0
        block_dt = max(1, int(hop)) / max(1.0, self.sample_rate)
        self._smooth_k = (1.0 if SMOOTHING_SECONDS <= 0.0 else
                          1.0 - np.exp(-block_dt / SMOOTHING_SECONDS))
        self._seq = 0

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """raw is [bass, mid, presence, hi, volume]."""
        if not self.auto_gain:
            return np.clip(raw, 0.0, 1.0)
        self._peaks *= _PEAK_DECAY
        np.maximum(self._peaks, raw, out=self._peaks)
        return np.clip(raw / np.maximum(self._peaks, _PEAK_FLOOR), 0.0, 1.0)

    def process(self, block: np.ndarray) -> SignalSnapshot:
        b = np.asarray(block, dtype=np.float32)
        if b.size != self.fft_size:
            b = np.resize(b, self.fft_size)
        mag = np.abs(np.fft.rfft(b * self._window)).astype(np.float32)

        db = 20.0 * np.log10(np.maximum(mag, _MAG_FLOOR))
        spec = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0)
        self._spectrum += (spec.astype(np.float32) -
                           self._spectrum) * self._smooth_k

        rows = self._m @ self._spectrum
        mel = rows[:self.n_mel]
        raw = np.empty(len(BAND_EDGES_HZ) + 1, dtype=np.float32)
        raw[:len(BAND_EDGES_HZ)] = rows[self.n_mel:]
        # Loudness of the block itself, on the same scale. Averaging the
        # spectrum instead would measure the unlit bins, which outnumber the
        # ones a kick drum lights by a hundred to one.
        rms_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(b * b))),
                                     _MAG_FLOOR))
        level = float(np.clip((rms_db - LEVEL_DB_MIN) /
                              (LEVEL_DB_MAX - LEVEL_DB_MIN), 0.0, 1.0))
        self._level += (level - self._level) * self._smooth_k
        raw[-1] = self._level

        # A DC or clipped block can still produce a non-finite magnitude on some
        # inputs; scrub here so nothing downstream has to.
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        norm = self._normalise(raw)

        self._seq += 1
        return SignalSnapshot(
            signals={n: float(v) for n, v in zip(SIGNAL_NAMES, norm)},
            mel=np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0),
            seq=self._seq,
        )
