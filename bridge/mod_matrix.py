"""Beat-locked parameter modulation, evaluated at frame rate.

The VJ rig broadcasts a global clock: ``Tick`` (seconds per beat, 84 receivers)
and ``|FULL AUTO|`` (the rig-wide modulation gate). Every fx module in it does
``R (Value) Tick`` -> ``eRand (Value Animation)`` / ``Tempo (VJ)``. This module
is the Fluoddity-side equivalent, so the particles move on the same clock as
everything else on screen.

Division of labour. vvvv decides *what* modulates *what*, *how deep*, and *when
the beat is*; this module evaluates the result once per rendered frame. That
matters because OSC arrives at vvvv's mainloop rate in discrete steps, whereas
the simulation wants a continuous value -- interpolating here is what turns a
stepped fader into a smooth sweep. Modulating on both sides would double up, so
vvvv sends the matrix and leaves the evaluation alone.

Phase discipline. ``beat_phase`` advances by ``dt / tick`` every frame and snaps
to zero on each bang. Free-running between bangs gives frame-rate smoothness;
snapping on them means it can never drift from the show, however wrong the
frame timing gets.

Addresses, all under the app prefix::

    /fluoddity/clock/tick    <float>   seconds per beat (vvvv's Tick)
    /fluoddity/clock/bang    <float>   nonzero = beat boundary
    /fluoddity/clock/auto    <float>   global gate (vvvv's |FULL AUTO|)
    /fluoddity/clock/audio   <float>   0..1 envelope for SRC_AUDIO
    /fluoddity/mod/base      "a,b,.."  12 slices, normalized 0..1
    /fluoddity/mod/depth     "a,b,.."  12 slices, 0..1
    /fluoddity/mod/source    "a,b,.."  12 slices, source index
    /fluoddity/mod/period    "a,b,.."  12 slices, beats per cycle
    /fluoddity/mod/lo        "a,b,.."  12 slices, normalized safety floor
    /fluoddity/mod/hi        "a,b,.."  12 slices, normalized safety ceiling

Every slice list is positional over the physics registry, the same order the
existing ``/fluoddity/n/params`` bulk address uses.

Safety property, and the reason the design is shaped this way: with all depths
at zero the output is exactly ``base``, which is exactly what
``/fluoddity/n/params`` already delivers. Sending no ``/mod/`` message at all
therefore leaves behaviour bit-identical to the pre-modulation bridge.
"""
import math
import time

# Modulation sources. Numbering is part of the vvvv-facing contract -- the
# patch sends these as integers -- so append, never reorder.
SRC_OFF = 0
SRC_SAW = 1
SRC_SINE = 2
SRC_TRIANGLE = 3
SRC_RANDOM_HOLD = 4      # eRand: a new value each period, held
SRC_RANDOM_SMOOTH = 5    # the same values, interpolated
SRC_BEAT_ENVELOPE = 6    # 1 at the period start, decaying -- bursts
SRC_AUDIO = 7

SOURCE_NAMES = [
    "Off", "Saw", "Sine", "Triangle", "Random Hold",
    "Random Smooth", "Beat Envelope", "Audio",
]

CLOCK_TICK = "clock/tick"
CLOCK_BANG = "clock/bang"
CLOCK_AUTO = "clock/auto"
CLOCK_AUDIO = "clock/audio"

MATRIX_ROWS = ("base", "depth", "source", "period", "lo", "hi")
RAW_ADDRESSES = [CLOCK_TICK, CLOCK_BANG, CLOCK_AUTO, CLOCK_AUDIO] + [
    f"mod/{row}" for row in MATRIX_ROWS]

# Guards. A zero or negative tick/period would divide by zero, and a runaway
# tick from a garbled message would freeze all modulation.
MIN_TICK = 0.01
MAX_TICK = 10.0
MIN_PERIOD = 0.01
DEFAULT_TICK = 0.5      # 120 BPM
# Longest frame gap still treated as real time passing. A config load or a
# shader recompile can genuinely stall a frame for over a second; beyond this
# the clock source itself is suspect, so the interval is dropped instead.
MAX_DT = 2.0
ENVELOPE_DECAY = 4.0    # e-folds per period; ~2% left at the period end


def _hash01(a: float, b: float) -> float:
    """Deterministic 0..1 hash of two numbers.

    Deliberately not :mod:`random`: the value for a given (step, param) pair
    must be identical on every run and independent of evaluation order, so a
    reload or a dropped frame cannot change what the audience sees.
    """
    x = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return x - math.floor(x)


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def evaluate_source(source: int, beats: float, period: float,
                    index: int, audio: float) -> float:
    """One modulator's value in 0..1.

    Args:
        source: one of the ``SRC_*`` constants.
        beats: continuous position on the beat timeline.
        period: cycle length in beats.
        index: parameter index, so parallel random sources decorrelate.
        audio: the current audio envelope, for :data:`SRC_AUDIO`.

    Returns:
        0..1. Bipolar shapes centre on 0.5 so a depth sweep is symmetric
        around the base value.
    """
    period = max(float(period), MIN_PERIOD)
    pos = beats / period
    phase = pos - math.floor(pos)

    if source == SRC_SAW:
        return phase
    if source == SRC_SINE:
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
    if source == SRC_TRIANGLE:
        return 1.0 - abs(2.0 * phase - 1.0)
    if source == SRC_RANDOM_HOLD:
        return _hash01(math.floor(pos), index * 7.3 + 1.7)
    if source == SRC_RANDOM_SMOOTH:
        step = math.floor(pos)
        a = _hash01(step, index * 7.3 + 1.7)
        b = _hash01(step + 1.0, index * 7.3 + 1.7)
        return a + (b - a) * _smoothstep(phase)
    if source == SRC_BEAT_ENVELOPE:
        return math.exp(-ENVELOPE_DECAY * phase)
    if source == SRC_AUDIO:
        return min(max(audio, 0.0), 1.0)
    return 0.5  # SRC_OFF and anything unrecognised: no offset from base


class BeatClock:
    """Free-running beat position, re-synced by bangs from vvvv.

    Kept separate from :class:`ModMatrix` so it can be tested against a
    synthetic time source without any OSC involved.
    """

    def __init__(self, tick: float = DEFAULT_TICK):
        self.tick = tick
        self.beats = 0.0
        self.auto = 1.0
        self.audio = 0.0
        self._last_time: float | None = None
        self._last_bang = 0.0

    def set_tick(self, tick: float) -> None:
        self.tick = min(max(float(tick), MIN_TICK), MAX_TICK)

    def bang(self, value: float) -> None:
        """Handle a bang pin. Only a rising edge counts, since vvvv holds the
        value at 1 for the whole frame the beat lands on."""
        rising = value > 0.5 and self._last_bang <= 0.5
        self._last_bang = value
        if rising:
            # Snap forward to the next whole beat rather than truncating back,
            # so the timeline never runs backwards. A backwards jump would make
            # a random-hold source re-fire a value it had already moved past.
            self.beats = math.floor(self.beats) + 1.0

    def advance(self, now: float | None = None) -> float:
        """Move the clock to ``now`` and return the new beat position."""
        if now is None:
            now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return self.beats
        dt = now - self._last_time
        self._last_time = now
        # A negative or absurd dt means the clock source hiccupped (a paused
        # debugger, a wrapped counter). Skip rather than jump the timeline.
        if 0.0 < dt <= MAX_DT:
            self.beats += dt / self.tick
        return self.beats


class ModMatrix:
    """Per-parameter modulation over a parameter registry.

    Args:
        specs: the physics registry, in the same order the vvvv patch sends
            its slices. Anything not in this list is passed through untouched.
    """

    def __init__(self, specs):
        self.specs = list(specs)
        self.clock = BeatClock()
        count = len(self.specs)
        self.base = [0.0] * count
        self.depth = [0.0] * count
        self.source = [SRC_OFF] * count
        self.period = [4.0] * count
        self.lo = [0.0] * count
        self.hi = [1.0] * count
        self._base_seen = [False] * count

    # --- configuration ---

    def set_row(self, row: str, values: list[float]) -> None:
        """Write one matrix row. Short lists leave later slices untouched."""
        target = getattr(self, row, None)
        if target is None or not isinstance(target, list):
            return
        for i, value in enumerate(values[:len(target)]):
            if row == "source":
                target[i] = int(round(value))
            else:
                target[i] = float(value)
            if row == "base":
                self._base_seen[i] = True

    def ingest(self, osc) -> None:
        """Pull clock and matrix updates from an :class:`OscControl`."""
        tick = osc.raw(CLOCK_TICK)
        if tick:
            self.clock.set_tick(tick[0])
        bang = osc.raw(CLOCK_BANG)
        if bang:
            self.clock.bang(bang[0])
        auto = osc.raw(CLOCK_AUTO)
        if auto:
            self.clock.auto = min(max(auto[0], 0.0), 1.0)
        audio = osc.raw(CLOCK_AUDIO)
        if audio:
            self.clock.audio = min(max(audio[0], 0.0), 1.0)
        for row in MATRIX_ROWS:
            values = osc.raw(f"mod/{row}")
            if values:
                self.set_row(row, values)

    # --- evaluation ---

    def is_active(self) -> bool:
        """True if any parameter would actually be modulated."""
        return self.clock.auto > 0.0 and any(
            d > 0.0 and s != SRC_OFF for d, s in zip(self.depth, self.source))

    def evaluate(self, normalized: dict[str, float] | None = None,
                 now: float | None = None) -> dict[str, float]:
        """Compute absolute values for every modulated parameter.

        Args:
            normalized: last 0..1 values seen on the normalized OSC path, from
                ``OscControl.normalized()``. Used as the base for any parameter
                the ``/mod/base`` row has not set, so the plain
                ``/fluoddity/n/params`` address keeps working as the base
                source and the matrix is purely additive on top.
            now: monotonic seconds; defaults to ``time.monotonic()``.

        Returns:
            ``{PARAM_NAME: absolute_value}`` for modulated parameters only.
            Unmodulated ones are absent, so whatever OSC already staged for
            them stands.
        """
        beats = self.clock.advance(now)
        auto = self.clock.auto
        out: dict[str, float] = {}
        for i, spec in enumerate(self.specs):
            depth = self.depth[i]
            source = self.source[i]
            if depth <= 0.0 or source == SRC_OFF or auto <= 0.0:
                continue
            if self._base_seen[i]:
                base = self.base[i]
            elif normalized and spec.name in normalized:
                base = normalized[spec.name]
            else:
                continue  # no base to modulate around yet
            mod = evaluate_source(source, beats, self.period[i], i,
                                  self.clock.audio)
            value = base + auto * depth * (mod - 0.5) * 2.0
            lo, hi = self.lo[i], self.hi[i]
            if lo > hi:
                lo, hi = hi, lo
            value = min(max(value, lo), hi)
            out[spec.name] = spec.from_normalized(value)
        return out

    def apply(self, osc, target, targets: dict[str, object] | None = None
              ) -> dict[str, float]:
        """Drain OSC onto the target(s), then overwrite with modulated values.

        A drop-in replacement for ``OscControl.apply``. Order matters: OSC is
        applied first so that unmodulated parameters, absolute-address writes
        and every non-physics group land normally, then modulation overrides
        only the parameters it actually drives.
        """
        applied = osc.apply(target, targets)
        self.ingest(osc)
        modulated = self.evaluate(osc.normalized())
        for name, value in modulated.items():
            setattr(target, name, value)
        applied.update(modulated)
        return applied
