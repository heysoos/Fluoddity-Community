"""The frame loop's half of the MIDI knobs.

`update` drains the device and nudges ui_state.sim - the real state, so the
slider moves and Save writes it. `feedback` lights the rings from the value
the sim actually ran under, which is the audio-modulated copy.
"""
import time

from services import midi_control as mc

GESTURE_SECONDS = 0.5      # how long after the last detent a twist is still one undo step
RESCAN_SECONDS = 1.0       # how often the port list is polled for a device
ACTIVE_BAND = 0.005        # fraction of the range past which audio counts as bending


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


def clamp_to_hard(param: str, value: float) -> float:
    p = _param_defs().get(param)
    if p is None:
        return value
    if p.hard_min is not None:
        value = max(value, p.hard_min)
    if p.hard_max is not None:
        value = min(value, p.hard_max)
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
            self.recentre(ui_state.sim)

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

    def recentre(self, sim_state) -> None:
        """Remember every parameter's value as the point a push returns to."""
        self._centres = {name: float(getattr(sim_state, name))
                         for name in _param_defs()
                         if hasattr(sim_state, name)}

    def _nudge(self, ui_state, knob, detents: int) -> None:
        if not detents or not hasattr(ui_state.sim, knob.param):
            return
        lo, hi = slider_range(ui_state.sim, knob.param)
        value = float(getattr(ui_state.sim, knob.param))
        value += detents * knob.step * (hi - lo)
        value = min(max(value, lo), hi)
        value = clamp_to_hard(knob.param, value)
        setattr(ui_state.sim, knob.param, value)
        self._touched(ui_state, knob)

    def _reset(self, ui_state, knob) -> None:
        centre = self._centres.get(knob.param)
        if centre is None or not hasattr(ui_state.sim, knob.param):
            return
        setattr(ui_state.sim, knob.param, centre)
        self._touched(ui_state, knob)

    def _touched(self, ui_state, knob) -> None:
        self._last_event = self._clock()
        p = _param_defs().get(knob.param)
        self.last_label = p.label if p else knob.param
        ui_state.undo_tag = self.last_label

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
            if not knob.param or not hasattr(effective_sim, knob.param):
                continue
            lo, hi = slider_range(ui_state.sim, knob.param)
            base = float(getattr(ui_state.sim, knob.param))
            live = float(getattr(effective_sim, knob.param))
            bending = abs(live - base) > ACTIVE_BAND * abs(hi - lo)
            self._send(ui_state, mc.ENCODER_CHANNEL, knob.cc,
                       mc.ring_value(live, lo, hi))
            self._send(ui_state, mc.SWITCH_CHANNEL, knob.cc,
                       knob.active_colour if bending else knob.rest_colour)
            self._send(ui_state, mc.RGB_ANIM_CHANNEL, knob.cc,
                       mc.RGB_DIM if knob.param in deaf else mc.RGB_BRIGHTEST)
            if self.port is None:
                return
