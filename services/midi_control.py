"""One MIDI device: encoder events in, LED feedback out.

`mido` and `python-rtmidi` are optional and imported lazily, so a machine
without them loses this section and nothing else. The device is chosen by
name; Windows appends a port index ("Midi Fighter Twister 0") which
`port_base_name` strips, so one name covers the input and the output.
"""
import re
import threading
from collections import deque

_PORT_INDEX = re.compile(r"\s+\d+$")

# Midi Fighter Twister, channels 0-indexed as mido reports them. MIDI IN to
# the device cannot be remapped, so these are fixed whatever the utility says.
ENCODER_CHANNEL = 0        # encoder turns in; ring position out
SWITCH_CHANNEL = 1         # encoder push in (value 127); RGB colour out
RGB_ANIM_CHANNEL = 2       # RGB brightness and animation out
RING_ANIM_CHANNEL = 5      # ring brightness and animation out
SWITCH_PRESSED = 127
ENCODER_COUNT = 64         # four banks of sixteen
RGB_DIMMEST = 17
RGB_BRIGHTEST = 47
RGB_DIM = 25


def port_base_name(name: str) -> str:
    """'Midi Fighter Twister 0' -> 'Midi Fighter Twister'."""
    return _PORT_INDEX.sub("", str(name)).strip()


def decode_relative(value: int) -> int:
    """Signed detents from a relative CC: 65 is +1, 63 is -1, 64 is nothing."""
    return int(value) - 64


def ring_value(value: float, lo: float, hi: float) -> int:
    """Where `value` sits in [lo, hi] as a 0..127 ring position."""
    span = hi - lo
    if span == 0:
        return 0
    t = (float(value) - lo) / span
    return int(round(min(max(t, 0.0), 1.0) * 127))


def available() -> bool:
    try:
        import mido  # noqa: F401
        import rtmidi  # noqa: F401
    except ImportError:
        return False
    return True


def list_devices() -> list[str]:
    """Base names of every device offering an input port, deduplicated."""
    if not available():
        return []
    import mido
    seen = []
    try:
        names = mido.get_input_names()
    except Exception:
        return []
    for n in names:
        base = port_base_name(n)
        if base and base not in seen:
            seen.append(base)
    return seen


def _find_port(names, base: str) -> str | None:
    for n in names:
        if port_base_name(n) == base:
            return n
    return None


class MidiPort:
    """An open device. Events queue on rtmidi's thread; `poll` drains them."""

    def __init__(self, base_name: str):
        import mido
        self.base_name = base_name
        self._events = deque()
        self._lock = threading.Lock()
        in_name = _find_port(mido.get_input_names(), base_name)
        if in_name is None:
            raise OSError(f"'{base_name}' has no input port.")
        out_name = _find_port(mido.get_output_names(), base_name)
        self._in = mido.open_input(in_name, callback=self._on_message)
        self._out = mido.open_output(out_name) if out_name else None

    def _on_message(self, msg):
        if msg.type != "control_change":
            return
        with self._lock:
            self._events.append((msg.channel, msg.control, msg.value))

    @property
    def can_send(self) -> bool:
        return self._out is not None

    def poll(self) -> list[tuple[int, int, int]]:
        """Every (channel, control, value) received since the last call."""
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    def send_cc(self, channel: int, control: int, value: int) -> None:
        if self._out is None:
            return
        import mido
        self._out.send(mido.Message("control_change", channel=channel,
                                    control=control & 127, value=value & 127))

    def close(self) -> None:
        for port in (self._in, self._out):
            if port is None:
                continue
            try:
                port.close()
            except Exception:
                pass
        self._in = self._out = None
