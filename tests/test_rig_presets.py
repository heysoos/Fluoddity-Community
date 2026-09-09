"""Named rigs, and the rule that stops one instance wiping another's.

The rig file is shared by every copy of Fluoddity on the machine. Saving it
worked all along; what lost people's work was writing it at exit whether or
not this session had touched it.
"""
from __future__ import annotations

import json

import pytest

from command_handler import CommandHandler
from main import App, rig_to_dict
from services import audio_rig_io as rig_io
from services.audio_mapping import Mapping
from state.audio_in_state import AudioInState


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point every rig path at a scratch Documents/Fluoddity."""
    monkeypatch.setattr(rig_io, "get_user_data_dir", lambda: tmp_path)
    return tmp_path


def _ui_state(audio):
    brain = type("Brain", (), {"reroll_audio_requested": False})()
    return type("UIState", (), {"audio": audio, "brain": brain})()


# --- the folder -------------------------------------------------------------

def test_rigs_have_their_own_folder(data_dir):
    """Not the configs folder: everything there shows up in File > Load."""
    assert rig_io.rigs_dir() == data_dir / "audio_rigs"
    assert rig_io.rigs_dir() != data_dir / "physics_configs"


def test_listing_a_folder_that_is_not_there_yet_is_empty(data_dir):
    assert rig_io.list_rigs() == []


def test_the_names_are_sorted_and_carry_no_extension(data_dir):
    rig_io.rigs_dir().mkdir(parents=True)
    for name in ("zither", "alpha", "middle"):
        (rig_io.rigs_dir() / f"{name}.json").write_text("{}", encoding="utf-8")
    assert rig_io.list_rigs() == ["alpha", "middle", "zither"]


def test_a_separator_in_the_name_cannot_write_outside_the_folder(data_dir):
    for name in ("../escape", "sub/rig", r"..\escape"):
        assert rig_io.preset_path(name).parent == rig_io.rigs_dir()


# --- round trip -------------------------------------------------------------

def test_a_named_rig_round_trips(data_dir):
    st = AudioInState()
    st.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN", depth=0.7))
    st.global_strength = 1.5
    assert rig_io.save_rig(st, rig_io.preset_path("reef")) is True

    back = AudioInState()
    assert rig_io.load_rig(back, rig_io.preset_path("reef")) is True
    assert [m.target for m in back.mappings] == ["SENSOR_GAIN"]
    assert back.mappings[0].depth == pytest.approx(0.7)
    assert back.global_strength == pytest.approx(1.5)


# --- the save dialog's end of it --------------------------------------------

def test_saving_a_preset_writes_it_and_says_so(data_dir):
    st = AudioInState()
    st.mappings.append(Mapping(signal="hi", target="SENSOR_GAIN"))
    ui_state = _ui_state(st)

    CommandHandler._save_audio_rig(None, ui_state, "reef")

    assert (rig_io.rigs_dir() / "reef.json").exists()
    assert st.preset_name == "reef"
    assert "reef" in st.notice


def test_a_save_that_fails_is_not_silent(data_dir, monkeypatch):
    """A save that only prints to the console reads as a save that worked."""
    monkeypatch.setattr(rig_io, "save_rig", lambda *a, **k: False)
    st = AudioInState()
    CommandHandler._save_audio_rig(None, _ui_state(st), "reef")
    assert "could not" in st.warning.lower()
    assert st.notice == "", "a failure must not read as a save"


# --- loading a preset -------------------------------------------------------

def test_loading_a_preset_applies_it_and_clears_the_request(data_dir):
    source = AudioInState()
    source.mappings.append(Mapping(signal="mid", target="SENSOR_DISTANCE"))
    rig_io.save_rig(source, rig_io.preset_path("reef"))

    st = AudioInState()
    st.preset_name = "reef"
    st.request_load_preset = True
    CommandHandler._handle_audio_preset(None, _ui_state(st))

    assert [m.target for m in st.mappings] == ["SENSOR_DISTANCE"]
    assert st.request_load_preset is False


def test_loading_a_preset_leaves_the_input_device_alone(data_dir):
    """The device belongs to the machine, not to the rig.

    A named rig still records the device it was built on - it is written by
    the one serialiser the last-used rig uses - but nothing reads it back
    here, so loading one cannot move the input.
    """
    source = AudioInState()
    source.device_name = "Some Other Machine's Mic"
    source.mappings.append(Mapping(signal="mid", target="SENSOR_DISTANCE"))
    rig_io.save_rig(source, rig_io.preset_path("reef"))

    st = AudioInState()
    st.device_name = "Microphone (K66)"
    st.preset_name = "reef"
    st.request_load_preset = True
    CommandHandler._handle_audio_preset(None, _ui_state(st))

    assert st.device_name == "Microphone (K66)"
    assert [m.target for m in st.mappings] == ["SENSOR_DISTANCE"]


def test_a_preset_that_will_not_read_reports_rather_than_retrying(data_dir):
    st = AudioInState()
    st.preset_name = "gone"
    st.request_load_preset = True
    CommandHandler._handle_audio_preset(None, _ui_state(st))
    assert st.request_load_preset is False, "it would retry every frame"
    assert "gone" in st.warning


def test_nothing_happens_without_a_request(data_dir):
    st = AudioInState()
    st.preset_name = "reef"
    CommandHandler._handle_audio_preset(None, _ui_state(st))
    assert (st.notice, st.warning) == ("", "")


# --- the wipe ---------------------------------------------------------------

class _Session:
    """An App wearing the real exit-time rig write."""

    _save_last_rig = App._save_last_rig

    def __init__(self, audio):
        self.audio = audio
        self._rig_at_start = rig_to_dict(audio)

    def exit(self):
        return self._save_last_rig(_ui_state(self.audio))


def _stored_rig(data_dir):
    return rig_io.rig_path().read_text(encoding="utf-8")


def test_an_untouched_session_writes_nothing(data_dir):
    """The regression that matters: a second copy of the app, opened and
    closed without going near audio, used to overwrite the rig you had just
    built in the first one."""
    rig_io.rig_path().parent.mkdir(parents=True, exist_ok=True)
    rig_io.rig_path().write_text(
        json.dumps({"mappings": [{"signal": "bass", "target": "SENSOR_GAIN",
                                  "mode": "add"}]}), encoding="utf-8")
    before = _stored_rig(data_dir)

    st = AudioInState()
    rig_io.load_rig(st, rig_io.rig_path())
    session = _Session(st)

    assert session.exit() is False
    assert _stored_rig(data_dir) == before


def test_a_session_that_changed_the_rig_does_write(data_dir):
    st = AudioInState()
    session = _Session(st)
    st.mappings.append(Mapping(signal="hi", target="SENSOR_GAIN"))

    assert session.exit() is True
    stored = json.loads(_stored_rig(data_dir))
    assert [m["target"] for m in stored["mappings"]] == ["SENSOR_GAIN"]


def test_loading_a_preset_counts_as_a_change(data_dir):
    """A preset loaded this session becomes the last-used rig."""
    source = AudioInState()
    source.mappings.append(Mapping(signal="mid", target="SENSOR_DISTANCE"))
    rig_io.save_rig(source, rig_io.preset_path("reef"))

    st = AudioInState()
    session = _Session(st)
    st.preset_name = "reef"
    st.request_load_preset = True
    CommandHandler._handle_audio_preset(None, _ui_state(st))

    assert session.exit() is True
    stored = json.loads(_stored_rig(data_dir))
    assert [m["target"] for m in stored["mappings"]] == ["SENSOR_DISTANCE"]


def test_a_slider_moved_and_moved_back_writes_nothing(data_dir):
    """The comparison is on the rig's contents, not on whether it was edited."""
    st = AudioInState()
    session = _Session(st)
    st.global_strength = 1.7
    st.global_strength = 1.0
    assert session.exit() is False


def test_the_window_being_open_is_not_a_rig_change(data_dir):
    """show_window is not part of the rig, and opening the panel to look at it
    must not make this instance the one that writes."""
    st = AudioInState()
    session = _Session(st)
    st.show_window = True
    st.open_target = "SENSOR_GAIN"
    assert session.exit() is False


def test_a_named_rig_carries_its_cohort_masks(data_dir):
    """The two-band split has to survive Save and Load by name, not just the
    dict helpers - this is the path the preset row actually uses."""
    import numpy as np

    from services import cohort_audio as ca

    st = AudioInState()
    lo = Mapping(signal="bass", target="SENSOR_GAIN", mode="add")
    hi = Mapping(signal="hi", target="SENSOR_GAIN", mode="subtract")
    lo.cohorts[:] = False
    hi.cohorts[:] = False
    for cell in range(32):
        ca.paint(lo.cohorts, cell, 64, True)
    for cell in range(32, 64):
        ca.paint(hi.cohorts, cell, 64, True)
    st.mappings = [lo, hi]

    assert rig_io.save_rig(st, rig_io.preset_path("split"))

    back = AudioInState()
    assert rig_io.load_rig(back, rig_io.preset_path("split"))
    assert [m.signal for m in back.mappings] == ["bass", "hi"]
    assert np.array_equal(back.mappings[0].cohorts, lo.cohorts)
    assert np.array_equal(back.mappings[1].cohorts, hi.cohorts)
    # The halves land the right way round, not merely somewhere.
    assert ca.covers(back.mappings[0].cohorts, 0, 64)
    assert not ca.covers(back.mappings[0].cohorts, 63, 64)
    assert ca.covers(back.mappings[1].cohorts, 63, 64)


def test_painting_a_mask_makes_this_instance_the_one_that_writes(data_dir):
    """Or the session that scoped a band would not save it at exit."""
    from services import cohort_audio as ca

    st = AudioInState()
    st.mappings = [Mapping(signal="bass", target="SENSOR_GAIN")]
    session = _Session(st)
    ca.paint(st.mappings[0].cohorts, 0, 64, False)
    assert session.exit() is True
