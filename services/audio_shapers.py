"""Per-mapping shaping of a signal before the modulation maths.

Every shaper takes a value in [0,1] and returns one in [0,1], and NONE of them
moves on its own while the band is silent. That is not the same as answering
silence with zero: a stopped integrator, like a latched sample-and-hold, holds
a perfectly good non-zero value. What no shaper may do is generate motion from
nothing - which includes a signal that is merely being HELD, so `apply` takes
`live`. State is per mapping, because the same band commonly drives one target
directly and another through an envelope.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SHAPER_KINDS: tuple[str, ...] = ("none", "smooth", "gate", "envelope", "phase",
                                 "sample_hold")


@dataclass
class ShaperParams:
    kind: str = "none"
    attack: float = 0.02        # seconds to rise
    release: float = 0.25       # seconds to fall
    threshold: float = 0.5      # gate / envelope / sample-hold trigger level
    hold: float = 0.05          # seconds a gate stays open after falling below
    rate: float = 1.0           # phase cycles per second at a full-scale band
    wave: str = "sine"          # "sine" | "triangle" | "ramp"


def _coeff(seconds: float, dt: float) -> float:
    """One-pole coefficient reaching ~63% of a step in `seconds`."""
    if seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / seconds)


def _wave(phase: float, wave: str) -> float:
    """Every shape starts at ZERO, so a rig that has heard nothing contributes
    nothing."""
    if wave == "triangle":
        return 1.0 - abs(2.0 * (phase % 1.0) - 1.0)
    if wave == "ramp":
        return phase % 1.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


class ShaperState:
    """Carried between blocks for one mapping."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._value = 0.0
        self._phase = 0.0
        self._hold_left = 0.0
        self._open = False
        self._above = False
        self._latched = 0.0

    def apply(self, x: float, dt: float, p: ShaperParams,
              live: bool = True, rate_scale: float = 1.0) -> float:
        """`live` is False while the signal is being HELD rather than measured.

        A held signal is a perfectly good non-zero number, which is why the
        integrator below has to be told: without this, a centroid parked at
        0.73 by a paused track kept the wave travelling forever.

        `rate_scale` multiplies FREQUENCY and nothing else. Applying it to
        `dt` would reach every attack, release and hold as well, which are
        durations - stretching those is not a change of tempo.
        """
        x = 0.0 if x != x else min(1.0, max(0.0, float(x)))
        dt = max(1e-6, float(dt))
        kind = p.kind

        if kind == "smooth":
            k = _coeff(p.attack if x > self._value else p.release, dt)
            self._value += (x - self._value) * k
            return min(1.0, max(0.0, self._value))

        if kind == "gate":
            if x >= p.threshold:
                self._open = True
                self._hold_left = p.hold
            elif self._open:
                self._hold_left -= dt
                if self._hold_left <= 0.0:
                    self._open = False
            return 1.0 if self._open else 0.0

        if kind == "envelope":
            crossed = x >= p.threshold and not self._above
            self._above = x >= p.threshold
            if crossed:
                self._value = 1.0
            else:
                self._value -= self._value * _coeff(p.release, dt)
            return min(1.0, max(0.0, self._value))

        if kind == "phase":
            # The band is INTEGRATED: it drives how fast the phase moves, not
            # where the phase is. So it only ever lurches forward, a held note
            # keeps the wave cycling, and silence stops it dead wherever it had
            # got to rather than dragging the parameter back.
            self._phase = (self._phase + (x if live else 0.0)
                           * p.rate * rate_scale * dt) % 1.0
            return min(1.0, max(0.0, _wave(self._phase, p.wave)))

        if kind == "sample_hold":
            crossed = x >= p.threshold and not self._above
            self._above = x >= p.threshold
            if crossed:
                self._latched = x
            return self._latched

        # "none", and anything a newer build wrote that this one does not know.
        return x
