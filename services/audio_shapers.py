"""Per-mapping shaping of a signal before the modulation maths.

Every shaper takes a value in [0,1] and returns one in [0,1]. State is per
mapping, because the same band commonly drives one target directly and another
through an oscillator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SHAPER_KINDS: tuple[str, ...] = ("none", "smooth", "gate", "envelope", "lfo",
                                 "sample_hold")


@dataclass
class ShaperParams:
    kind: str = "none"
    attack: float = 0.02        # seconds to rise
    release: float = 0.25       # seconds to fall
    threshold: float = 0.5      # gate / envelope / sample-hold trigger level
    hold: float = 0.05          # seconds a gate stays open after falling below
    rate_min: float = 0.5       # LFO Hz at signal 0
    rate_max: float = 12.0      # LFO Hz at signal 1
    wave: str = "sine"          # "sine" | "triangle" | "ramp"


def _coeff(seconds: float, dt: float) -> float:
    """One-pole coefficient reaching ~63% of a step in `seconds`."""
    if seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / seconds)


def _wave(phase: float, wave: str) -> float:
    if wave == "triangle":
        return 1.0 - abs(2.0 * (phase % 1.0) - 1.0)
    if wave == "ramp":
        return phase % 1.0
    return 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)


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

    def apply(self, x: float, dt: float, p: ShaperParams) -> float:
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

        if kind == "lfo":
            rate = p.rate_min + (p.rate_max - p.rate_min) * x
            self._phase = (self._phase + rate * dt) % 1.0
            return min(1.0, max(0.0, _wave(self._phase, p.wave)))

        if kind == "sample_hold":
            crossed = x >= p.threshold and not self._above
            self._above = x >= p.threshold
            if crossed:
                self._latched = x
            return self._latched

        # "none", and anything a newer build wrote that this one does not know.
        return x
