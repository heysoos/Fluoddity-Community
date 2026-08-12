"""One global rig file. Not attached to a physics config."""
import json

import pytest

from services.audio_mapping import Mapping
from services.audio_rig_io import load_rig, rig_path, save_rig
from state.audio_in_state import AudioInState


def test_the_rig_lives_beside_the_other_user_files():
    assert rig_path().name == "audio_rig.json"


def test_a_saved_rig_loads_back(tmp_path):
    path = tmp_path / "audio_rig.json"
    st = AudioInState()
    st.mappings.append(Mapping(signal="mid", target="DRAG", mode="multiply"))
    st.global_strength = 1.25
    assert save_rig(st, path) is True

    fresh = AudioInState()
    assert load_rig(fresh, path) is True
    assert fresh.mappings[0].target == "DRAG"
    assert fresh.global_strength == pytest.approx(1.25)


def test_loading_a_missing_file_is_not_an_error(tmp_path):
    st = AudioInState()
    assert load_rig(st, tmp_path / "nothing.json") is False
    assert st.mappings == []


def test_a_corrupt_file_does_not_raise(tmp_path):
    path = tmp_path / "audio_rig.json"
    path.write_text("{not json", encoding="utf-8")
    st = AudioInState()
    assert load_rig(st, path) is False


def test_enabled_is_not_written_to_disk(tmp_path):
    path = tmp_path / "audio_rig.json"
    st = AudioInState()
    st.enabled = True
    save_rig(st, path)
    assert "enabled" not in json.loads(path.read_text(encoding="utf-8"))


def test_loading_never_starts_capture(tmp_path):
    path = tmp_path / "audio_rig.json"
    save_rig(AudioInState(), path)
    st = AudioInState()
    load_rig(st, path)
    assert st.enabled is False


def test_saving_into_a_missing_directory_creates_it(tmp_path):
    path = tmp_path / "deep" / "deeper" / "audio_rig.json"
    assert save_rig(AudioInState(), path) is True
    assert path.exists()
