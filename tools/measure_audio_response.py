"""The two things the band smoother has to buy at once.

SNAP: a hi-hat is over in a few milliseconds, so a smoother that takes tens of
them to respond reports a fraction of its height. CALM: a steady note must not
draw a fuzzy hash. They pull in opposite directions, which is why the smoothing
is asymmetric - and why changing either constant needs both columns, not one.

    python -m tools.measure_audio_response
"""
from __future__ import annotations

import numpy as np
import ui  # noqa: F401  - imported first, or services/ hits a circular import

from services import audio_analysis as aa

SR = 48000.0

# (fft_size, hop, attack_s, release_s)
CASES = (
    (2048, 1024, 0.075, 0.075),
    (2048, 1024, 0.000, 0.075),
    (2048, 1024, 0.010, 0.075),
    (2048, 512, 0.000, 0.075),
    (2048, 512, 0.010, 0.075),
    (1024, 512, 0.000, 0.075),
)


def _analyzer(fft, hop, attack, release):
    """The constants are read in __init__, so they are set before it runs."""
    aa.ATTACK_SECONDS, aa.SMOOTHING_SECONDS = attack, release
    return aa.Analyzer(SR, fft_size=fft, auto_gain=False, hop=hop)


def transient(fft, hop, attack, release, blocks=90, onset=30):
    """Height the `hi` band reports for one hi-hat, and the ms to reach it."""
    rng = np.random.default_rng(0)
    an = _analyzer(fft, hop, attack, release)
    n = hop * (blocks - onset)
    t = np.arange(n) / SR
    hit = (rng.standard_normal(n) * np.exp(-t * 90.0) * 0.6).astype(np.float32)
    tail = np.zeros(fft, dtype=np.float32)
    seen = []
    for i in range(blocks):
        chunk = np.zeros(hop, dtype=np.float32)
        if i >= onset:
            j = (i - onset) * hop
            chunk = hit[j:j + hop]
        tail = np.roll(tail, -hop)
        tail[fft - hop:] = chunk
        seen.append(an.process(tail).signals["hi"])
    a = np.asarray(seen)
    return a.max(), (int(np.argmax(a)) - onset) * hop / SR * 1000.0


def steady(fft, hop, attack, release, blocks=240):
    """Mean change per analysis of `hi` on a steady note plus hiss."""
    rng = np.random.default_rng(1)
    an = _analyzer(fft, hop, attack, release)
    tail = np.zeros(fft, dtype=np.float32)
    seen = []
    for i in range(blocks):
        t = (np.arange(hop) + i * hop) / SR
        chunk = (0.10 * np.sin(2 * np.pi * 9000.0 * t)
                 + 0.02 * rng.standard_normal(hop)).astype(np.float32)
        tail = np.roll(tail, -hop)
        tail[fft - hop:] = chunk
        seen.append(an.process(tail).signals["hi"])
    a = np.asarray(seen[blocks // 3:])
    return float(np.abs(np.diff(a)).mean())


def main() -> None:
    shipped = (aa.FFT_SIZE, aa.HOP, aa.ATTACK_SECONDS, aa.SMOOTHING_SECONDS)
    print(f"{'fft/hop':>10} {'attack':>8} {'release':>8} | "
          f"{'hat peak':>9} {'at':>8} | {'steady step':>12}")
    print("-" * 66)
    try:
        for case in CASES:
            fft, hop, att, rel = case
            peak, at = transient(*case)
            tag = "  <- shipped" if case == shipped else ""
            print(f"{fft:5d}/{hop:<4d} {att * 1000:7.0f}m {rel * 1000:7.0f}m | "
                  f"{peak:9.3f} {at:7.1f}m | {steady(*case):12.5f}{tag}")
    finally:
        aa.ATTACK_SECONDS, aa.SMOOTHING_SECONDS = shipped[2], shipped[3]
    print("\nhat peak: taller is snappier.  steady step: lower is calmer.")


if __name__ == "__main__":
    main()
