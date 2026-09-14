"""Does a frame loop that holds the interpreter lose audio?

Opens the real default input through `AudioCapture`, taps it into a sidecar
exactly as a recording does, holds the GIL on the main thread for a while,
and compares the audio that reached the tap against the wall clock.

    python -m tools.measure_audio_stall [hold_seconds]

Nothing here is a unit test: it needs a microphone, and the answer is a
property of the machine as much as of the code.
"""
from __future__ import annotations

import re
import sys
import time

import ui  # noqa: F401 - primes the package order the app uses
from services.audio_capture import AudioCapture
from services.audio_track import AudioTrackWriter


def hold_gil(seconds: float) -> float:
    """Catastrophic regex backtracking runs in C and never releases the GIL."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        re.match(r"(a+)+b", "a" * 24)
    return time.perf_counter() - t0


def main(hold: float) -> int:
    cap = AudioCapture()
    t0 = time.perf_counter()
    cap.prewarm()
    print(f"prewarm: {1000 * (time.perf_counter() - t0):.1f} ms on the caller")
    t0 = time.perf_counter()
    if not cap.start(None, auto_gain=False):
        print("could not open the default input:", cap.last_error)
        return 1
    print(f"start:   {1000 * (time.perf_counter() - t0):.1f} ms "
          f"({cap.sample_rate:.0f} Hz, {cap.channels} ch)")
    track = AudioTrackWriter(sys.argv[0] + ".pcm", cap.sample_rate,
                             cap.channels)
    time.sleep(0.5)                     # let the device settle
    cap.tap = track.write
    wall0 = time.perf_counter()
    time.sleep(1.0)
    held = hold_gil(hold)
    time.sleep(1.0)
    time.sleep(0.2)                     # let the reader drain the pipe
    cap.tap = None
    wall = time.perf_counter() - wall0 - 0.2
    track.close()
    got = track.duration_seconds
    print(f"held the GIL {held:.2f}s; wall {wall:.2f}s, audio tapped "
          f"{got:.2f}s, missing {wall - got:+.3f}s, dropped blocks "
          f"{cap.dropped}")
    cap.shutdown()
    import os
    os.remove(track.path)
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.5))
