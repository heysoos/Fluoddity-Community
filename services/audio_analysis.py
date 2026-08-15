"""Turn a block of samples into named signals.

Pure numpy, no IO. Runs on the capture thread, so nothing here may touch GL,
ImGui or any state the frame loop owns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The window stays wide: it sets FREQUENCY resolution, and the bass band is
# only about ten bins wide at 2048. The HOP is what sets time resolution, and
# it is half the window so a transient is analysed twice on its way through.
FFT_SIZE = 2048
HOP = 512                # frames between analyses; the capture's callback size
N_MEL = 40

SIGNAL_NAMES: tuple[str, ...] = ("bass", "mid", "presence", "hi", "volume",
                                 "centroid")

# The DISPLAY spectrum is read in decibels over this fixed window, chosen so
# ordinary material rests across the middle of the scale. It draws the mel bars
# and nothing else - a band signal is measured from linear magnitudes, below.
DB_MIN = -90.0
DB_MAX = -20.0

# `volume` measures the whole block at once, which sits far above any single
# bin, so it needs a window of its own.
LEVEL_DB_MIN = -60.0
LEVEL_DB_MAX = -6.0

# How a band turns its bins into one number. `power` sums LINEAR energy and
# converts once, so a band rests at its floor between hits; `mean_db` averages
# each bin's decibel level, where the hundreds of bins carrying nothing set the
# result - `hi` spans about 600 bins and a cymbal lights a handful of them.
#
# `flux` is the odd one: it measures what is NEW rather than what is there, so
# a sustained note reads zero however loud it is and only the attack registers.
# It is level-dependent like the rest, deliberately - a ratio would put a quiet
# passage's hits at full scale, which is the complaint the others answer.
MEASURES: tuple[str, ...] = ("power", "rms", "peak", "flux", "mean_db")

# Each measure lives on its own scale - a sum over 600 bins is not a mean over
# them - so each carries its own dB window. Measured, see
# `python -m tools.measure_audio_response`.
# The floor is where a band reads zero, and it is set ABOVE a quiet passage
# rather than above the noise floor: a quiet part of a track passing a healthy
# signal is the complaint these measures answer.
MEASURE_WINDOWS: dict[str, tuple[float, float]] = {
    "power": (-45.0, -5.0),
    "rms": (-62.0, -20.0),
    "peak": (-60.0, -12.0),
    # Flux sits higher than the levels: broadband noise is new every block, so
    # the floor has to clear a room's hiss in `hi` as well as its level.
    "flux": (-35.0, 0.0),
    # The historical measure keeps the display window it was defined against.
    "mean_db": (DB_MIN, DB_MAX),
    # Not decibels: where the spectrum's centre of mass sits, in Hz, read on a
    # log axis because pitch is.
    "centroid_hz": (200.0, 8000.0),
    "block_rms": (LEVEL_DB_MIN, LEVEL_DB_MAX),
}

# Signals whose measure is not a choice. `volume` is the block's own RMS in the
# time domain and `centroid` is a ratio across the whole spectrum; neither is
# one band's bins reduced to a number.
FIXED_MEASURES: dict[str, str] = {
    "volume": "block_rms",
    "centroid": "centroid_hz",
}
VOLUME_MEASURE = FIXED_MEASURES["volume"]

# What a measure's window is expressed in, so the panel can label it.
MEASURE_UNITS: dict[str, str] = {"centroid_hz": "Hz"}

# The spectral bands, in order. `volume` and `centroid` are not among them.
BAND_NAMES: tuple[str, ...] = ("bass", "mid", "presence", "hi")

# How far up `volume`'s own scale a block must reach before the centroid is
# read from it at all. Measured: a room's noise floor at -60 to -40 dBFS reads
# 0.08 to 0.45 there, and the centroid of noise WANDERS - a block's spectrum is
# a fresh random draw, so it swings 0.14 on pink and 0.36 on brown while
# nothing is playing. Expressed as a fraction of the scale rather than in dB,
# so `volume`'s floor slider moves the gate with it.
CENTROID_GATE = 0.25

# The DISPLAY spectrum's own smoothing. Not the band release below: the bars
# are there to be read, and an unsmoothed spectrum vibrates.
DISPLAY_SMOOTHING_SECONDS = 0.075

# A band is smoothed ASYMMETRICALLY, because the two directions solve different
# problems. Falling slowly is what stops a band being one block's sample of a
# noisy quantity, which reads as vibration. Rising slowly buys nothing and
# costs the transient: a hi-hat is over in a few milliseconds, so a smoother
# that takes tens of them to respond reports a fraction of its height.
#
# The release is a SETTING (AudioInState.release_seconds). At 0 a band is the
# raw per-block measurement and every shaper in the rig sees it unsmoothed.
ATTACK_SECONDS = 0.0
SMOOTHING_SECONDS = 0.075

# Below this a bin is silence rather than a very negative number of decibels.
_MAG_FLOOR = 1e-9
_POWER_FLOOR = _MAG_FLOOR * _MAG_FLOOR

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


def _coeff(seconds: float, dt: float) -> float:
    """One-pole coefficient reaching ~63% of a step in `seconds`.

    A time constant at or below the block period is `1.0`: the smoother cannot
    resolve anything faster than one analysis, so asking for it means "follow".
    """
    if seconds <= 0.0:
        return 1.0
    return float(min(1.0, 1.0 - np.exp(-dt / seconds)))


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def default_band(name: str) -> dict:
    """The measure and window a signal uses until the user changes them."""
    measure = FIXED_MEASURES.get(name, "power")
    lo, hi = MEASURE_WINDOWS[measure]
    return {"measure": measure, "floor": lo, "ceiling": hi}


def effective_bands(stored: dict | None = None) -> dict[str, dict]:
    """Every signal's settings, with anything stored overriding the defaults.

    A rig saved before these existed stores nothing, so it opens on the
    defaults rather than on a band that measures nothing.
    """
    out = {name: default_band(name) for name in SIGNAL_NAMES}
    for name, value in (stored or {}).items():
        if name not in out or not isinstance(value, dict):
            continue
        measure = value.get("measure")
        allowed = ((FIXED_MEASURES[name],) if name in FIXED_MEASURES
                   else MEASURES)
        if measure in allowed:
            out[name]["measure"] = measure
        for key in ("floor", "ceiling"):
            if isinstance(value.get(key), (int, float)):
                out[name][key] = float(value[key])
    return out


def band_bins(sample_rate: float, fft_size: int = FFT_SIZE) -> dict:
    """Each band's half-open bin range into the rfft, by name."""
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    nyquist = sample_rate / 2.0
    out = {}
    for name, lo, hi in BAND_EDGES_HZ:
        sel = np.nonzero((freqs >= lo) & (freqs < min(hi, nyquist)))[0]
        out[name] = ((int(sel[0]), int(sel[-1]) + 1) if sel.size else (0, 0))
    return out


def mel_matrix(sample_rate: float, fft_size: int = FFT_SIZE,
               n_mel: int = N_MEL) -> np.ndarray:
    """A mel filterbank for the display bars, one row per bar.

    Bands are NOT in here: they are measured from linear magnitudes over the
    bin ranges `band_bins` gives, because a mean over a row of this matrix is
    the one measure that cannot rest at zero.
    """
    n_bins = fft_size // 2 + 1
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    out = np.zeros((n_mel, n_bins), dtype=np.float32)

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
    return out


def mel_bar_bands(sample_rate: float, n_mel: int = N_MEL) -> np.ndarray:
    """Which of BAND_EDGES_HZ each mel bar's centre falls in, as an index.

    The display colours a bar by the band it belongs to, and only the analyser
    knows where the mel axis was placed for this device's rate.
    """
    nyquist = sample_rate / 2.0
    edges = _mel_to_hz(np.linspace(_hz_to_mel(20.0),
                                   _hz_to_mel(min(16000.0, nyquist)),
                                   n_mel + 2))
    centres = edges[1:n_mel + 1]
    out = np.zeros(n_mel, dtype=np.int8)
    for i, hz in enumerate(centres):
        for j, (_name, lo, hi) in enumerate(BAND_EDGES_HZ):
            if lo <= hz < hi:
                out[i] = j
                break
        else:
            out[i] = len(BAND_EDGES_HZ) - 1
    return out


@dataclass(frozen=True)
class SignalSnapshot:
    """One analysis result. Immutable so the frame loop may read it without a
    lock while the capture thread builds the next one."""
    signals: dict[str, float]
    mel: np.ndarray
    seq: int
    mel_bands: np.ndarray
    # Signals this block did not measure, and is reporting their last value
    # for. A held value is not a small one, so a shaper that integrates would
    # run on forever without being told. Default for anything older.
    held: frozenset = frozenset()


def spectral_centroid_hz(power: np.ndarray, freqs: np.ndarray) -> float:
    """Where the spectrum's centre of mass sits, weighted by energy.

    A RATIO, so it says nothing about level: a quiet bright break reads high
    and a loud bass-only drop reads low. That is the point - no band can tell
    those apart.
    """
    total = float(power.sum())
    if total <= _POWER_FLOOR:
        return 0.0
    return float((power * freqs).sum() / total)


def centroid_value(setting: dict, hz: float) -> float:
    """The centroid over its window, read on a LOG axis because pitch is.

    An octave is the same distance anywhere on the scale, so a filter opening
    from 400 to 800 Hz moves it as far as one from 4k to 8k.
    """
    floor = max(1.0, float(setting["floor"]))
    ceiling = max(floor + 1.0, float(setting["ceiling"]))
    if hz <= 0.0:
        return 0.0
    span = np.log(ceiling) - np.log(floor)
    return float(np.clip((np.log(max(hz, 1.0)) - np.log(floor)) / span,
                         0.0, 1.0))


def band_value(setting: dict, mag: np.ndarray, power: np.ndarray,
               db: np.ndarray, lo: int, hi: int,
               rise: np.ndarray | None = None) -> float:
    """One band, as 0..1 over its own dB window.

    `power` and `rms` sum LINEAR energy and take decibels once at the end, so a
    bin carrying nothing contributes nothing. `mean_db` maps each bin on its
    own first and averages the results, which is the historical measure and the
    reason a band never returned to zero.
    """
    if hi <= lo:
        return 0.0
    floor = float(setting["floor"])
    span = max(1e-6, float(setting["ceiling"]) - floor)
    if setting["measure"] == "mean_db":
        return float(np.clip((db[lo:hi] - floor) / span, 0.0, 1.0).mean())
    level = band_level_db(setting["measure"], mag, power, lo, hi, rise)
    return float(np.clip((level - floor) / span, 0.0, 1.0))


def band_level_db(measure: str, mag: np.ndarray, power: np.ndarray,
                  lo: int, hi: int, rise: np.ndarray | None = None) -> float:
    """A band's level in dB, before any window is applied.

    Defined apart from `band_value` so the calibration tool reads the same
    number the analyser does rather than a second copy of the formula.
    `rise` is the half-wave rectified change since the last block, which only
    `flux` reads.
    """
    if measure == "power":
        return 10.0 * np.log10(max(float(power[lo:hi].sum()), _POWER_FLOOR))
    if measure == "rms":
        return 10.0 * np.log10(max(float(power[lo:hi].mean()), _POWER_FLOOR))
    if measure == "flux":
        if rise is None:
            return 20.0 * np.log10(_MAG_FLOOR)
        return 20.0 * np.log10(max(float(rise[lo:hi].sum()), _MAG_FLOOR))
    return 20.0 * np.log10(max(float(mag[lo:hi].max()), _MAG_FLOOR))


class Analyzer:
    """Stateful across blocks: auto-gain peaks and the sequence counter."""

    def __init__(self, sample_rate: float, fft_size: int = FFT_SIZE,
                 n_mel: int = N_MEL, auto_gain: bool = True,
                 hop: int = HOP, bands: dict | None = None,
                 release: float | None = None) -> None:
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.n_mel = int(n_mel)
        self.auto_gain = bool(auto_gain)
        self._m = mel_matrix(sample_rate, fft_size, n_mel)
        self._stored_bands = dict(bands or {})
        self._bands = effective_bands(bands)
        self._bins = band_bins(sample_rate, fft_size)
        # The window carries its own coherent gain, so a full-scale sine reads
        # 1.0 at its bin. Without it a band is raw FFT magnitude, which is far
        # above the [0,1] clip for any real input - every band would pin at 1.0
        # and switching auto-gain off would do nothing.
        window = np.hanning(fft_size)
        self._window = (window * (2.0 / max(window.sum(), 1e-9))).astype(
            np.float32)
        # Auto-gain covers the levels only. Dividing a ratio by its own running
        # peak means nothing, so `centroid` is left out of it.
        self._gained = len(BAND_EDGES_HZ) + 1
        self._peaks = np.full(self._gained, _PEAK_FLOOR, dtype=np.float32)
        self._spectrum = np.zeros(fft_size // 2 + 1, dtype=np.float32)
        self._values = np.zeros(len(SIGNAL_NAMES), dtype=np.float32)
        self._freqs = np.fft.rfftfreq(fft_size,
                                      d=1.0 / self.sample_rate).astype(
                                          np.float32)
        # None until a second block arrives: the first one has nothing to be
        # new against, and calling all of it new would fire every flux mapping
        # the moment capture starts.
        self._prev_mag = None
        # Held through silence rather than snapping to an end of its range -
        # there is no brightness to report when nothing is playing.
        self._centroid = 0.0
        self._mel_bands = mel_bar_bands(sample_rate, n_mel)
        self._block_dt = max(1, int(hop)) / max(1.0, self.sample_rate)
        self._display_k = _coeff(DISPLAY_SMOOTHING_SECONDS, self._block_dt)
        self.release = (SMOOTHING_SECONDS if release is None
                        else max(0.0, float(release)))
        self._release_k = _coeff(self.release, self._block_dt)
        self._attack_k = _coeff(ATTACK_SECONDS, self._block_dt)
        self._seq = 0

    def set_bands(self, bands: dict | None) -> None:
        """Retune a RUNNING analyser; ignored when nothing actually changed."""
        stored = dict(bands or {})
        if stored == self._stored_bands:
            return
        self._stored_bands = stored
        self._bands = effective_bands(stored)

    def set_release(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if seconds == self.release:
            return
        self.release = seconds
        self._release_k = _coeff(seconds, self._block_dt)

    def set_auto_gain(self, on: bool) -> None:
        """Switch the gain while blocks are arriving.

        The peaks are dropped on any change - they record a level the switch
        has just made meaningless.
        """
        on = bool(on)
        if on == self.auto_gain:
            return
        self.auto_gain = on
        self._peaks.fill(_PEAK_FLOOR)

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """raw is SIGNAL_NAMES in order; only the levels are gained."""
        out = np.clip(raw, 0.0, 1.0)
        if not self.auto_gain:
            return out
        n = self._gained
        self._peaks *= _PEAK_DECAY
        np.maximum(self._peaks, raw[:n], out=self._peaks)
        out[:n] = np.clip(raw[:n] / np.maximum(self._peaks, _PEAK_FLOOR),
                          0.0, 1.0)
        return out

    def process(self, block: np.ndarray) -> SignalSnapshot:
        b = np.asarray(block, dtype=np.float32)
        if b.size != self.fft_size:
            b = np.resize(b, self.fft_size)
        mag = np.abs(np.fft.rfft(b * self._window)).astype(np.float32)

        db = 20.0 * np.log10(np.maximum(mag, _MAG_FLOOR)).astype(np.float32)

        # The display bars, and only them: their smoothing is fixed because
        # they are there to be read.
        spec = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0)
        delta = spec.astype(np.float32) - self._spectrum
        self._spectrum += delta * np.where(delta > 0.0, 1.0, self._display_k)
        mel = self._m @ self._spectrum

        power = (mag * mag).astype(np.float32)
        rise = (np.zeros_like(mag) if self._prev_mag is None
                else np.maximum(mag - self._prev_mag, 0.0))
        self._prev_mag = mag

        raw = np.empty(len(SIGNAL_NAMES), dtype=np.float32)
        for i, name in enumerate(BAND_NAMES):
            lo, hi = self._bins[name]
            raw[i] = band_value(self._bands[name], mag, power, db, lo, hi,
                                rise)
        # Loudness of the block itself. Measuring the spectrum instead would
        # count the unlit bins, which outnumber the ones a kick drum lights by
        # a hundred to one.
        vol = self._bands["volume"]
        rms_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(b * b))),
                                     _MAG_FLOOR))
        span = max(1e-6, float(vol["ceiling"]) - float(vol["floor"]))
        level = float(np.clip((rms_db - float(vol["floor"])) / span, 0.0, 1.0))
        raw[len(BAND_NAMES)] = level

        # Brightness, held while the block is too quiet to read one from: a
        # ratio over a room's noise floor is not a dark sound or a bright one,
        # it is no sound - and it wanders, because noise is a fresh spectrum
        # every block.
        live = level >= CENTROID_GATE
        if live:
            self._centroid = centroid_value(
                self._bands["centroid"],
                spectral_centroid_hz(power, self._freqs))
        raw[len(BAND_NAMES) + 1] = self._centroid

        # A DC or clipped block can still produce a non-finite magnitude on some
        # inputs; scrub here so nothing downstream has to.
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        norm = self._normalise(raw)

        # The release, last, so it means what the slider says whatever measure
        # produced the number. The rise is never smoothed.
        d = norm - self._values
        self._values += d * np.where(d > 0.0, self._attack_k, self._release_k)

        self._seq += 1
        return SignalSnapshot(
            held=frozenset() if live else frozenset({"centroid"}),
            signals={n: float(v) for n, v in zip(SIGNAL_NAMES, self._values)},
            mel=np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0),
            seq=self._seq,
            mel_bands=self._mel_bands,
        )
