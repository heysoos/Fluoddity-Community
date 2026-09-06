"""What may be modulated, and by how much.

Pure. Takes a dict of signal values and returns a dict of modulated values; it
cannot tell an FFT band from a shaper output from anything added later.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from services.audio_shapers import ShaperParams, ShaperState
from services.cohort_audio import full_mask

MODES: tuple[str, ...] = ("add", "subtract", "multiply")

_uid_counter = itertools.count(1)


def _next_uid() -> int:
    return next(_uid_counter)


@dataclass(frozen=True)
class TargetDef:
    """One modulatable parameter. Physics and brain rows share this shape."""
    key: str
    label: str
    group: str                 # "physics" | "brain"
    lo: float                  # modulation scale, from the live slider range
    hi: float
    hard_lo: float | None      # clamp, when the parameter declares one
    hard_hi: float | None


@dataclass
class Mapping:
    signal: str
    target: str
    mode: str = "add"
    depth: float = 0.5
    gain: float = 1.0
    shaper: ShaperParams = field(default_factory=ShaperParams)
    enabled: bool = True
    # Names this mapping's shaper state between frames; id() cannot, being
    # handed to the next mapping the moment this one is deleted. Copied with
    # the mapping, never persisted, and an ordinary field so that a rig diffed
    # by value sees a delete-and-re-add.
    uid: int = field(default_factory=_next_uid)
    # Which cohorts this mapping drives, over the normalised cohort axis. Full
    # means every cohort, which is what a rig written before this comes to.
    cohorts: np.ndarray = field(default_factory=full_mask)

    def __eq__(self, other) -> bool:
        # Explicit, because the generated one compares the mask with `==` and
        # returns an array rather than a bool.
        if not isinstance(other, Mapping):
            return NotImplemented
        return (
            (self.signal, self.target, self.mode, self.depth, self.gain,
             self.shaper, self.enabled, self.uid)
            == (other.signal, other.target, other.mode, other.depth,
                other.gain, other.shaper, other.enabled, other.uid)
            and bool(np.array_equal(self.cohorts, other.cohorts))
        )


@dataclass
class Channel:
    """One audio input to the brain, as the rig sees it.

    The channel's identity is its INDEX - the weight column it drives - and
    the name is a label saved with the rig. `range` is what a full-scale sum
    reads as, in the brain's own input units.
    """
    name: str = ""
    range: float = 0.25
    uid: int = field(default_factory=_next_uid)


def channel_key(k: int) -> str:
    return f"AUDIO_IN_{int(k)}"


def channel_targets(layout, channels) -> list[TargetDef]:
    """One target per audio input the LAYOUT declares. A channel list shorter
    than that reads defaults for the rest; a longer one keeps its extras
    waiting, as brain_mappings waits for its modality."""
    k = int(getattr(layout, "audio_inputs", 0) or 0) if layout is not None else 0
    out = []
    for i in range(k):
        c = channels[i] if i < len(channels) else Channel()
        r = max(float(c.range), 1e-6)
        out.append(TargetDef(channel_key(i), c.name or f"A{i + 1}", "channel",
                             -r, r, -r, r))
    return out


def physics_targets(sim_state) -> list[TargetDef]:
    """Every physics slider except those whose maximum switches them off."""
    from ui.physics_params import PHYSICS_PARAMS

    out = []
    for p in PHYSICS_PARAMS:
        if p.off_at_max:
            continue
        rng = sim_state.slider_ranges.get(p.label)
        lo, hi = (rng[0], rng[1]) if rng else (p.default_min, p.default_max)
        out.append(TargetDef(p.name, p.label, "physics", float(lo), float(hi),
                             p.hard_min, p.hard_max))
    return out


def brain_targets(modality, layout) -> list[TargetDef]:
    """The active modality's decode scales.

    Structural settings change the parameter count, so modulating one would
    reshape the brain buffer every frame.
    """
    from services import brains

    out = []
    for s in modality.settings_schema():
        if s.kind in brains.STRUCTURAL_KINDS:
            continue
        out.append(TargetDef(s.key, s.label, "brain", float(s.lo), float(s.hi),
                             float(s.lo), float(s.hi)))
    return out


def deaf_targets(sim_state) -> set[str]:
    """Parameters a sweep has made unreachable.

    calculate_setting() returns slider_value only when every sweep is zero, so
    a swept parameter cannot be modulated at all.
    """
    deaf = set()
    for axis in ("x_sweeps", "y_sweeps", "cohort_sweeps"):
        for name, value in getattr(sim_state, axis, {}).items():
            if value:
                deaf.add(name)
    return deaf


def modulate(bases: dict[str, float], targets, mappings, signals,
             states: dict[int, ShaperState], strengths: dict[str, float],
             global_strength: float, dt: float,
             deaf: set[str], shaped: dict | None = None,
             apply_shapers: bool = True, held=(),
             rate_scale: float = 1.0) -> dict[str, float]:
    """Modulated values for the targets that have an enabled mapping.

    `bases` is read and never written. Targets with no mapping, and targets a
    sweep has made deaf, are absent from the result. When `shaped` is given,
    each applied mapping's post-shaper signal is recorded in it under `m.uid` -
    the panel draws that, since the shaper is the whole point of the drawer and
    the raw band shows none of its effect.

    `states` is keyed by `m.uid` too, and the caller owns it: nothing here
    removes the entry of a mapping the user has deleted.

    `apply_shapers=False` takes every signal at face value. Every shaper
    answers a full-scale input with full scale eventually, so this is what asks
    how far a mapping could reach without waiting out an attack or an
    oscillator's phase.

    `held` names the signals reporting a remembered value rather than a
    measured one. A held value is not a small one, so without this a shaper
    that integrates keeps running after the music stops.
    """
    by_target = {t.key: t for t in targets}
    adds: dict[str, list] = {}
    muls: dict[str, list] = {}

    for m in mappings:
        if not m.enabled or m.target not in by_target or m.target in deaf:
            continue
        if m.signal not in signals:
            continue
        (muls if m.mode == "multiply" else adds).setdefault(m.target, []).append(m)

    out: dict[str, float] = {}
    for key in set(adds) | set(muls):
        t = by_target[key]
        base = float(bases.get(key, 0.0))
        v = base
        span = t.hi - t.lo

        for m in adds.get(key, ()):
            s = min(1.0, max(0.0, signals[m.signal] * m.gain))
            if apply_shapers:
                s = states.setdefault(m.uid, ShaperState()).apply(
                    s, dt, m.shaper, m.signal not in held, rate_scale)
            if shaped is not None:
                shaped[m.uid] = s
            sign = -1.0 if m.mode == "subtract" else 1.0
            v += sign * s * m.depth * span

        for m in muls.get(key, ()):
            s = min(1.0, max(0.0, signals[m.signal] * m.gain))
            if apply_shapers:
                s = states.setdefault(m.uid, ShaperState()).apply(
                    s, dt, m.shaper, m.signal not in held, rate_scale)
            if shaped is not None:
                shaped[m.uid] = s
            v *= 1.0 + s * m.depth

        v = base + (v - base) * strengths.get(key, 1.0) * global_strength

        lo = t.hard_lo if t.hard_lo is not None else t.lo
        hi = t.hard_hi if t.hard_hi is not None else t.hi
        out[key] = min(hi, max(lo, v))
    return out
