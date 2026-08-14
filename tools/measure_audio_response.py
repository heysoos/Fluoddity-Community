"""The two things the band smoother has to buy at once, and what auto-gain costs.

SNAP: a hi-hat is over in a few milliseconds, so a smoother that takes tens of
them to respond reports a fraction of its height. CALM: a steady note must not
draw a fuzzy hash. They pull in opposite directions, which is why the smoothing
is asymmetric - and why changing either constant needs both columns, not one.

The second table is the one auto-gain has to answer for. It divides each band
by its own running peak, and `np.maximum` puts that peak AT the signal, so any
steady input normalises to its own top whatever its level. Read the silence
rows against the music row: a gap there is the whole feature.

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


# (label, dBFS, is the material dynamic)
GAIN_CASES = (
    ("digital silence", None, False),
    ("room hiss", -70.0, False),
    ("room hiss", -50.0, False),
    ("steady note", -30.0, False),
    ("kick loop", -12.0, True),
)


def _material(db, dynamic, hop, i):
    """One block of test signal at `db` dBFS."""
    if db is None:
        return np.zeros(hop, dtype=np.float32)
    amp = 10 ** (db / 20.0)
    rng = np.random.default_rng(i)
    if not dynamic:
        if db <= -50.0:
            return (rng.standard_normal(hop) * amp).astype(np.float32)
        t = (np.arange(hop) + i * hop) / SR
        return (amp * np.sin(2 * np.pi * 800.0 * t)).astype(np.float32)
    # A hit every 24 blocks, so the peak follower has something to fall from.
    phase = i % 24
    env = np.exp(-(np.arange(hop) + phase * hop) / (SR * 0.05))
    t = (np.arange(hop) + i * hop) / SR
    return (amp * env * (np.sin(2 * np.pi * 90.0 * t)
                         + 0.3 * rng.standard_normal(hop))).astype(np.float32)


def gain_response(db, dynamic, gain, blocks=900, hop=None):
    """`mid` with auto-gain on or off: (last value, low, high) once settled."""
    hop = hop or aa.HOP
    an = aa.Analyzer(SR, auto_gain=gain, hop=hop)
    tail = np.zeros(aa.FFT_SIZE, dtype=np.float32)
    seen = []
    for i in range(blocks):
        tail = np.roll(tail, -hop)
        tail[aa.FFT_SIZE - hop:] = _material(db, dynamic, hop, i)
        seen.append(an.process(tail).signals["mid"])
    settled = np.asarray(seen[blocks // 3:])
    return settled[-1], settled.min(), settled.max()


def auto_gain_table() -> None:
    print(f"\n{'material':22} | {'raw mid':>8} | {'gained mid':>10} "
          f"{'low':>7} {'high':>7}")
    print("-" * 62)
    for label, db, dynamic in GAIN_CASES:
        raw, _lo, _hi = gain_response(db, dynamic, False)
        got, lo, hi = gain_response(db, dynamic, True)
        name = f"{label} {db:.0f} dB" if db is not None else label
        print(f"{name:22} | {raw:8.3f} | {got:10.3f} {lo:7.3f} {hi:7.3f}")
    print("\nEvery non-silent row reading near 1.0 is the defect: auto-gain "
          "cannot\ntell a room's hiss from music, which is why it ships off.")


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
    auto_gain_table()


if __name__ == "__main__":
    main()
