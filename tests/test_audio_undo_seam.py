"""Audio and undo meet at ui_state, and neither may write the other's half.

The undo journal commits a step whenever a declared field differs from the
step it holds, once per frame, forever. A rig running against a physics target
moves that target sixty times a second, so if the modulation reached
`ui_state.sim` the journal would fill with one step per frame and push every
real change off the end within seconds. It does not, because the runtime
modulates a COPY - and this is the test that says so from the journal's side.
"""
import numpy as np
import pytest

from services import undo_history as uh
from services.audio_mapping import Mapping
from services.audio_runtime import AudioRuntime
from services.undo_history import UndoHistory
from state import UIState
from state import preferences_state


class FakeSnapshot:
    def __init__(self, **signals):
        self.signals = signals
        self.mel = np.zeros(40, dtype=np.float32)
        self.seq = 1


def rig_and_runtime(**signals):
    st = UIState()
    st.audio.enabled = True
    st.audio.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN",
                                     mode="add", depth=0.5))
    st.sim.SENSOR_GAIN = 1.0
    rt = AudioRuntime()
    rt.capture._snapshot = FakeSnapshot(**signals)
    return st, rt


def test_a_running_rig_commits_no_undo_steps():
    st, rt = rig_and_runtime(bass=1.0)
    history = UndoHistory()
    history.commit(uh.capture(st, None, None))

    for _ in range(120):
        out, _brain = rt.update(st, 1 / 60, None, None)
        assert out is not st.sim, "the rig stopped modulating; test proves nothing"
        snap = uh.capture(st, None, None)
        assert uh.same(history.current(), snap)
    assert len(history.steps) == 1


def test_the_modulated_copy_is_not_what_the_journal_reads():
    """The copy carries the movement; the snapshot carries the slider."""
    st, rt = rig_and_runtime(bass=1.0)
    out, _brain = rt.update(st, 1 / 60, None, None)
    held = uh.capture(st, None, None).fields["sim"]["SENSOR_GAIN"]
    assert out.SENSOR_GAIN > 1.0
    assert held == pytest.approx(1.0)


def test_the_rig_itself_is_outside_undo():
    """Deliberate: the rig has its own file and its own named presets, and a
    Ctrl+Z aimed at a slider must not silently rewire the audio."""
    assert "audio" not in {name for name, _module, _panel in uh.CONTAINERS}


@pytest.mark.parametrize("field", ("show_audio_window", "show_archive_browser",
                                   "show_undo_window", "show_history_window",
                                   "record_notice"))
def test_transient_and_visibility_preferences_are_not_undoable(field):
    """Which windows are open, and the last take's message, are not state a
    creature is made of - an undo aimed at a slider must not reopen a panel."""
    assert field in preferences_state.PreferencesState.__dataclass_fields__
    assert field not in preferences_state.UNDOABLE_FIELDS
