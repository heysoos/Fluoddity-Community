"""A microphone's ultrasonic noise must not become the signal.

Measured on a real USB mic (`Microphone (USBAudio2.0)`): 92.5% of its total
energy sits in 22-24 kHz, right at Nyquist. That is ADC noise shaping - it is
inaudible, and nothing in the app reports on it: `hi` tops out at 20 kHz, the
mel display at 16 kHz, the centroid's window at 8 kHz.

Two signals integrated it anyway. `volume` is a time-domain RMS over the whole
block and `centroid` is an energy-weighted mean over the whole spectrum, so
both read the noise instead of the music: on a silent room the mic gave
volume 0.32 and centroid pinned at 1.000, and neither moved for anything
played into it.

The profile here is synthetic but its proportions are the measured ones.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.audio_analysis import Analyzer, CENTROID_GATE

RATE = 48000.0
FFT = 2048


def _shelf(n, rng, level_db=-42.4, share=0.925):
    """Noise shaped like the measured mic: `share` of its energy at 22-24 kHz.

    Built in the frequency domain so the split is exact rather than a filter's
    approximation of one.
    """
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    spec = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size))
    audible = freqs < 20000.0
    ultra = freqs >= 22000.0
    spec[~(audible | ultra)] = 0.0
    # Scale the two halves to the measured energy split.
    spec[audible] *= np.sqrt((1.0 - share) / max(audible.sum(), 1))
    spec[ultra] *= np.sqrt(share / max(ultra.sum(), 1))
    x = np.fft.irfft(spec, n=n)
    x *= 10 ** (level_db / 20.0) / max(float(np.sqrt(np.mean(x * x))), 1e-12)
    return x.astype(np.float32)


def _tone(n, hz=1000.0, level_db=-30.0):
    t = np.arange(n) / RATE
    amp = 10 ** (level_db / 20.0) * np.sqrt(2.0)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _run(x):
    """-> the last snapshot after streaming x through a fresh Analyzer."""
    an = Analyzer(RATE, auto_gain=False)
    snap = None
    for i in range(0, x.size - FFT, 512):
        snap = an.process(x[i:i + FFT])
    assert snap is not None
    return snap


@pytest.fixture
def rng():
    return np.random.default_rng(20260827)


def test_an_ultrasonic_shelf_does_not_pin_the_centroid(rng):
    """It sits above the centroid's own 8 kHz ceiling, so unfiltered it can
    only push the value to 1.0 - and hold it there for good."""
    # A quiet tone, which is the case that matters: music IS present and the
    # noise still decides the answer.
    snap = _run(_shelf(FFT * 40, rng) + _tone(FFT * 40, level_db=-40.0))
    assert snap.signals["centroid"] < 0.95, (
        "the centroid is pinned at the top of its range by inaudible noise")


def test_an_ultrasonic_shelf_does_not_inflate_the_level(rng):
    """Measured: a silent room read volume 0.32 through this mic, which is
    also above CENTROID_GATE - so the hold that exists for exactly this case
    never engaged either."""
    snap = _run(_shelf(FFT * 40, rng))
    assert snap.signals["volume"] < CENTROID_GATE, (
        f"a silent room reads {snap.signals['volume']:.3f}, at or above the "
        f"{CENTROID_GATE} gate, so the centroid is never held")


def test_the_centroid_reports_brightness_not_loudness(rng):
    """A ratio must not track level. Unfiltered it read 0.999 / 0.638 / 0.466
    for one 1 kHz tone at -40 / -30 / -20 dBFS - that is the tone's level
    against the noise, not its brightness."""
    n = FFT * 40
    shelf = _shelf(n, rng)
    values = [_run(shelf + _tone(n, level_db=db)).signals["centroid"]
              for db in (-40.0, -30.0, -20.0)]
    spread = max(values) - min(values)
    assert spread < 0.15, (
        f"the centroid moved {spread:.3f} across a 20 dB level change: "
        f"{[round(v, 3) for v in values]}")


def test_a_clean_signal_is_left_alone(rng):
    """The fix must be a no-op for a healthy device.

    Measured: the other two devices on the same machine had 0.0% of their
    energy above 16 kHz, so band-limiting must not move their readings.
    """
    n = FFT * 40
    clean = (_tone(n, hz=440.0, level_db=-20.0)
             + _tone(n, hz=3000.0, level_db=-26.0)).astype(np.float32)
    snap = _run(clean)
    # A 440 + 3000 Hz pair sits inside the centroid's window and well above
    # volume's floor; both must read as ordinary content.
    assert 0.0 < snap.signals["centroid"] < 1.0
    assert snap.signals["volume"] > CENTROID_GATE
    for band in ("bass", "mid", "presence"):
        assert 0.0 <= snap.signals[band] <= 1.0
