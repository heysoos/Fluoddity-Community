"""A knob walks the archive gallery: one detent, one entry, wrapping at the ends.

The runtime writes a STEP and the command handler resolves it against the
gallery as it is sorted and filtered, so the knob shows exactly what the
browser shows. A push returns to the first entry.
"""
import numpy as np
import pytest

from command_handler import CommandHandler
from services import midi_control as mc
from services.midi_runtime import ARCHIVE_KNOB_TARGET, MidiRuntime, knob_targets
from state.archive_state import ArchiveState
from state.perform_state import PerformState
from state.preferences_state import PreferencesState
from state.set_list import KnobBinding, SetList
from state.sim_state import SimState

DEVICE = "Midi Fighter Twister"


class FakePort:
    def __init__(self, name):
        self.name = name
        self.queued = []
        self.sent = []
        self.can_send = True

    def poll(self):
        out, self.queued = self.queued, []
        return out

    def send_cc(self, channel, control, value):
        self.sent.append((channel, control, value))

    def close(self):
        pass


class State:
    def __init__(self):
        from state.audio_in_state import AudioInState
        self.sim = SimState()
        self.audio = AudioInState()
        self.perform = PerformState()
        self.preferences = PreferencesState()
        self.archive = ArchiveState()
        self.undo_tag = ""


@pytest.fixture
def rig():
    ports = []
    rt = MidiRuntime(open_port=lambda n: ports.append(FakePort(n)) or ports[-1],
                     clock=lambda: 100.0, list_devices=lambda: [DEVICE])
    st = State()
    st.preferences.midi_device = DEVICE
    st.perform.set_list = SetList(knobs=[
        KnobBinding(cc=5, param=ARCHIVE_KNOB_TARGET.key, step=0.1)])
    rt.update(st)
    return rt, st, ports[0]


# ---- the runtime's half ------------------------------------------------------

def test_the_archive_entry_is_offered_as_a_knob_target():
    assert ARCHIVE_KNOB_TARGET.key in [t.key for t in knob_targets(SimState())]


def test_a_detent_asks_for_one_step_and_a_push_asks_for_the_first_entry(rig):
    rt, st, port = rig
    port.queued = [(mc.ENCODER_CHANNEL, 5, 65), (mc.ENCODER_CHANNEL, 5, 65),
                   (mc.ENCODER_CHANNEL, 5, 63)]
    rt.update(st)
    assert st.archive.step_entry == 1
    assert st.undo_tag == ARCHIVE_KNOB_TARGET.label
    port.queued = [(mc.SWITCH_CHANNEL, 5, mc.SWITCH_PRESSED)]
    rt.update(st)
    assert st.archive.knob_entry_reset is True


def test_the_ring_shows_where_in_the_gallery_the_knob_is(rig):
    rt, st, port = rig
    st.archive.knob_entry_pos = 0.5
    rt.feedback(st, st.sim)
    assert (mc.ENCODER_CHANNEL, 5, 64) in port.sent


def test_recentring_the_physics_knobs_leaves_the_archive_knob_alone(rig):
    rt, st, _ = rig
    rt.recentre(st, "sim")
    assert ARCHIVE_KNOB_TARGET.key not in rt._centres


# ---- the handler's half ------------------------------------------------------

class _Entry:
    def __init__(self, i, layout="", novelty=0.5):
        # Ids restart per brain directory, so they are deliberately NOT the
        # rows here: a knob that handed an id to a row-addressed one-shot
        # loaded the wrong entry.
        self.id = 100 + i
        self.layout = layout
        self.novelty = novelty
        self.pinned = False
        self.source = "expansion"
        self.goal = ""
        self.liveness = 0.02
        self.ts = float(i)


class _Archive:
    signature = "fourier-n10+c2"

    def __init__(self, layouts):
        self.entries = [_Entry(i, lay) for i, lay in enumerate(layouts)]

    def layout_at(self, i):
        return self.entries[i].layout or self.signature

    def __len__(self):
        return len(self.entries)


def _handler(archive):
    h = object.__new__(CommandHandler)
    h.archive = archive
    h._knob_entry_row = -1
    return h


def _state(brain=""):
    st = State()
    st.archive.sort_by, st.archive.sort_desc = "id", False
    st.archive.brain_filter = brain
    return st


def test_steps_walk_the_gallery_in_its_order_and_wrap():
    h = _handler(_Archive(["", "", ""]))
    st = _state()
    st.archive.preview_entry_id = 7          # a hover the browser left standing
    st.archive.step_entry = 1
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == 0
    assert st.archive.preview_entry_id == -1
    assert st.archive.step_entry == 0
    assert st.archive.knob_entry_pos == 0.0
    assert st.perform.recentre_knobs is True
    for want, pos in ((1, 0.5), (2, 1.0), (0, 0.0)):
        st.archive.step_entry = 1
        h._handle_entry_knob(st)
        assert st.archive.load_entry_id == want
        assert st.archive.knob_entry_pos == pytest.approx(pos)
    st.archive.step_entry = -1
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == 2


def test_the_walk_respects_the_brain_filter():
    h = _handler(_Archive(["", "mlp-n16-a0+c2", "", "mlp-n16-a0+c2"]))
    st = _state(brain="mlp-n16-a0+c2")
    st.archive.step_entry = 1
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == 1
    st.archive.step_entry = 1
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == 3
    st.archive.step_entry = 1
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == 1


def test_a_push_goes_to_the_first_entry():
    h = _handler(_Archive(["", "", ""]))
    st = _state()
    st.archive.step_entry = 2
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == 1
    st.archive.knob_entry_reset = True
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == 0
    assert st.archive.knob_entry_reset is False


def test_nothing_to_walk_clears_the_request_and_loads_nothing():
    h = _handler(_Archive([]))
    st = _state()
    st.archive.step_entry = 3
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == -1
    assert st.archive.step_entry == 0
    h = _handler(None)
    st.archive.step_entry = 3
    h._handle_entry_knob(st)
    assert st.archive.load_entry_id == -1
    assert st.archive.step_entry == 0
