"""The frame loop's half of the MIDI knobs.

`update` drains the device and nudges the real state - a physics slider on
ui_state.sim or an audio setting on ui_state.audio - so the slider moves and
Save writes it. `feedback` lights the rings from the value the sim actually
ran under, which for a physics slider is the audio-modulated copy.
"""
import math
import time
from dataclasses import dataclass

from services import midi_control as mc

GESTURE_SECONDS = 0.5      # how long after the last detent a twist is still one undo step
RESCAN_SECONDS = 1.0       # how often the port list is polled for a device
ACTIVE_BAND = 0.005        # fraction of the range past which audio counts as bending


@dataclass(frozen=True)
class KnobTarget:
    key: str          # what a KnobBinding.param names
    label: str
    holder: str       # "sim" or "audio": which ui_state member carries it
    attr: str
    lo: float
    hi: float
    log: bool = False
    hard_lo: float | None = None
    hard_hi: float | None = None


AUDIO_KNOB_TARGETS = (
    KnobTarget("audio:global_strength", "Audio Strength", "audio",
               "global_strength", 0.0, 2.0, False, 0.0, 2.0),
    KnobTarget("audio:rate_scale", "Rate Scale", "audio",
               "rate_scale", 0.05, 8.0, True, 0.05, 8.0),
)


def _param_defs():
    from ui.physics_params import PARAM_BY_NAME
    return PARAM_BY_NAME


def slider_range(sim_state, param: str) -> tuple[float, float]:
    """The live slider range of `param`, falling back to the registry's default."""
    p = _param_defs().get(param)
    if p is None:
        return (0.0, 1.0)
    rng = sim_state.slider_ranges.get(p.label)
    if rng:
        return float(rng[0]), float(rng[1])
    return float(p.default_min), float(p.default_max)


def knob_targets(sim_state) -> list[KnobTarget]:
    """Every physics slider, then the audio settings, in the order the panel lists them."""
    out = []
    for p in _param_defs().values():
        lo, hi = slider_range(sim_state, p.name)
        out.append(KnobTarget(p.name, p.label, "sim", p.name, lo, hi, False,
                              p.hard_min, p.hard_max))
    out.extend(AUDIO_KNOB_TARGETS)
    return out


def resolve(ui_state, key: str) -> KnobTarget | None:
    for t in AUDIO_KNOB_TARGETS:
        if t.key == key:
            return t
    p = _param_defs().get(key)
    if p is None or not hasattr(ui_state.sim, key):
        return None
    lo, hi = slider_range(ui_state.sim, key)
    return KnobTarget(key, p.label, "sim", key, lo, hi, False, p.hard_min, p.hard_max)


def _holder(ui_state, t: KnobTarget):
    return ui_state.sim if t.holder == "sim" else ui_state.audio


def _fraction(t: KnobTarget, value: float) -> float:
    """Where `value` sits in the target's range, 0..1, log-aware."""
    if t.log and t.lo > 0 and t.hi > t.lo:
        v = min(max(value, t.lo), t.hi)
        return math.log(v / t.lo) / math.log(t.hi / t.lo)
    span = t.hi - t.lo
    if span == 0:
        return 0.0
    return min(max((value - t.lo) / span, 0.0), 1.0)


def _from_fraction(t: KnobTarget, f: float) -> float:
    f = min(max(f, 0.0), 1.0)
    if t.log and t.lo > 0 and t.hi > t.lo:
        return t.lo * (t.hi / t.lo) ** f
    return t.lo + f * (t.hi - t.lo)


def _clamp_hard(t: KnobTarget, value: float) -> float:
    if t.hard_lo is not None:
        value = max(value, t.hard_lo)
    if t.hard_hi is not None:
        value = min(value, t.hard_hi)
    return value


class MidiRuntime:
    def __init__(self, open_port=None, clock=time.monotonic, list_devices=None):
        self._open_port = open_port or mc.MidiPort
        self._list_devices = list_devices or mc.list_devices
        self._clock = clock
        self.port = None
        self._wanted = ""
        self._last_scan = -1e9
        self._last_event = -1e9
        self._centres: dict[str, float] = {}
        self._sent: dict[tuple[int, int], int] = {}
        self.last_label = ""

    # --- connection -------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.port is not None

    def _connect(self, ui_state, name: str) -> None:
        now = self._clock()
        if now - self._last_scan < RESCAN_SECONDS:
            return
        self._last_scan = now
        if name not in self._list_devices():
            ui_state.perform.midi_status = f"'{name}' is not connected."
            return
        try:
            self.port = self._open_port(name)
        except Exception as e:
            ui_state.perform.midi_status = f"Could not open '{name}': {e}"
            return
        self._sent.clear()
        ui_state.perform.midi_status = f"Listening to {name}."
        # What is on screen now is the point a push returns to, until a
        # preset or a rig says otherwise.
        if not self._centres:
            self.recentre(ui_state, "sim")
            self.recentre(ui_state, "audio")
        self._blank_unbound(ui_state)

    def _drop(self, ui_state, why: str) -> None:
        if self.port is not None:
            self.port.close()
        self.port = None
        self._sent.clear()
        if why:
            ui_state.perform.midi_status = why

    def close(self) -> None:
        if self.port is not None:
            self.port.close()
        self.port = None

    # --- input ------------------------------------------------------------

    def update(self, ui_state) -> None:
        """Reconcile the device with the preference, then apply every event."""
        perform = ui_state.perform
        wanted = ui_state.preferences.midi_device
        if wanted != self._wanted:
            self._drop(ui_state, "" if wanted else "No device chosen.")
            self._wanted = wanted
            self._last_scan = -1e9
        if not wanted:
            return
        if self.port is None:
            self._connect(ui_state, wanted)
            if self.port is None:
                return

        if perform.recentre_knobs:
            perform.recentre_knobs = False
            self.recentre(ui_state, "sim")
        if perform.recentre_audio_knobs:
            perform.recentre_audio_knobs = False
            self.recentre(ui_state, "audio")

        try:
            events = self.port.poll()
        except Exception as e:
            self._drop(ui_state, f"Lost '{wanted}': {e}")
            return
        sl = perform.set_list
        for channel, control, value in events:
            knob = sl.knob_for_cc(control)
            if knob is None:
                continue
            if channel == mc.ENCODER_CHANNEL:
                self._nudge(ui_state, knob, mc.decode_relative(value))
            elif channel == mc.SWITCH_CHANNEL and value == mc.SWITCH_PRESSED:
                self._reset(ui_state, knob)

    def recentre(self, ui_state, holder: str) -> None:
        """Remember every target of `holder` as the point a push returns to."""
        for t in knob_targets(ui_state.sim):
            if t.holder != holder:
                continue
            obj = _holder(ui_state, t)
            if hasattr(obj, t.attr):
                self._centres[t.key] = float(getattr(obj, t.attr))

    def _nudge(self, ui_state, knob, detents: int) -> None:
        t = resolve(ui_state, knob.param)
        if not detents or t is None:
            return
        obj = _holder(ui_state, t)
        f = _fraction(t, float(getattr(obj, t.attr))) + detents * knob.step
        setattr(obj, t.attr, _clamp_hard(t, _from_fraction(t, f)))
        self._touched(ui_state, t)

    def _reset(self, ui_state, knob) -> None:
        t = resolve(ui_state, knob.param)
        centre = self._centres.get(knob.param)
        if t is None or centre is None:
            return
        setattr(_holder(ui_state, t), t.attr, centre)
        self._touched(ui_state, t)

    def _touched(self, ui_state, t: KnobTarget) -> None:
        self._last_event = self._clock()
        self.last_label = t.label
        ui_state.undo_tag = t.label

    def gesture_active(self) -> bool:
        """Is a twist still in progress? Undo waits for it to end."""
        return self._clock() - self._last_event < GESTURE_SECONDS

    # --- output -----------------------------------------------------------

    def _send(self, ui_state, channel: int, control: int, value: int) -> None:
        key = (channel, control)
        if self._sent.get(key) == value:
            return
        try:
            self.port.send_cc(channel, control, value)
        except Exception as e:
            self._drop(ui_state, f"Lost '{self._wanted}': {e}")
            return
        self._sent[key] = value

    def _blank_unbound(self, ui_state) -> None:
        bound = {k.cc for k in ui_state.perform.set_list.knobs if k.param}
        for cc in range(mc.ENCODER_COUNT):
            if cc in bound or self.port is None:
                continue
            self._send(ui_state, mc.ENCODER_CHANNEL, cc, 0)
            self._send(ui_state, mc.RGB_ANIM_CHANNEL, cc, mc.RGB_DIMMEST)

    def feedback(self, ui_state, effective_sim) -> None:
        """Light every bound ring from the value the sim ran under."""
        if self.port is None or not self.port.can_send:
            return
        from services.audio_mapping import deaf_targets
        deaf = deaf_targets(ui_state.sim)
        for knob in ui_state.perform.set_list.knobs:
            t = resolve(ui_state, knob.param)
            if t is None:
                continue
            base = float(getattr(_holder(ui_state, t), t.attr))
            if t.holder == "sim" and hasattr(effective_sim, t.attr):
                live = float(getattr(effective_sim, t.attr))
            else:
                live = base
            bending = abs(_fraction(t, live) - _fraction(t, base)) > ACTIVE_BAND
            self._send(ui_state, mc.ENCODER_CHANNEL, knob.cc,
                       int(round(_fraction(t, live) * 127)))
            self._send(ui_state, mc.SWITCH_CHANNEL, knob.cc,
                       knob.active_colour if bending else knob.rest_colour)
            dim = t.holder == "sim" and t.attr in deaf
            self._send(ui_state, mc.RGB_ANIM_CHANNEL, knob.cc,
                       mc.RGB_DIM if dim else mc.RGB_BRIGHTEST)
            if self.port is None:
                return
