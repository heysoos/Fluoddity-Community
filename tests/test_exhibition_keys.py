"""The exhibition lock and the set-list keys, driven through the real
UI.key_callback and the real CommandHandler dispatch.

A source-level reading cannot see the ordering fault this guards: a slot press
that reaches _handle_file_load while a File > Load hover is live takes the
finalise branch and loads the HOVERED preset, so the slot load runs after the
preview handler and clears the borrow itself.
"""
import glfw
import pytest
from imgui_bundle import imgui

from command_handler import CommandHandler
from services.config_saver import ConfigSaver, PhysicsConfig
from services.rule_manager import RuleManager
from state.set_list import SetList, Slot
from state.ui_state import UIState
from ui.core import UI
from utilities.keybinding_management import KeybindingManager
from utilities.paths import get_default_keyboard_controls_path


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800, 600)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    imgui.new_frame()
    imgui.render()
    yield
    imgui.destroy_context()


class _Keys(UI):
    """Only the key handling, over a real KeybindingManager."""

    def __init__(self, tmp_path):
        self.state = UIState()
        self.keybindings = KeybindingManager(
            config_path=tmp_path / "keys.json",
            default_config_path=get_default_keyboard_controls_path())
        self.imgui_key_callback = None
        self._keys_pressed = set()
        self._request_undo = False
        self._request_redo = False
        self._request_full_reset = False
        self._request_randomize_mutations = False
        self._request_reload = False

    key_callback = UI.key_callback
    _slot_for_key = UI._slot_for_key
    _request_slot = UI._request_slot


def _press(ui, key, mods=0):
    ui.key_callback(None, key, 0, glfw.PRESS, mods)


@pytest.fixture
def ui(gui, tmp_path):
    u = _Keys(tmp_path)
    u.state.perform.set_list = SetList(slots=[Slot("1", "Core", "reef"),
                                              Slot("Q", "Custom", "karst")])
    return u


def test_a_free_key_loads_its_slot_even_unlocked(ui):
    _press(ui, glfw.KEY_1)
    assert ui.state.perform.slot_load == ("reef", "Core", "")


def test_a_key_the_app_uses_keeps_its_binding_while_unlocked(ui):
    """Q is camera_out; a rehearsal does not steal it."""
    _press(ui, glfw.KEY_Q)
    assert ui.state.perform.slot_load is None
    assert glfw.KEY_Q in ui._keys_pressed


def test_the_lock_hands_that_key_to_the_set_list_and_drops_the_rest(ui):
    ui.state.perform.exhibition_locked = True
    _press(ui, glfw.KEY_Q)
    assert ui.state.perform.slot_load == ("karst", "Custom", "")
    assert ui._keys_pressed == set(), "a held camera key must not move the view"
    ui.state.perform.slot_load = None
    _press(ui, glfw.KEY_Z)
    _press(ui, glfw.KEY_F11)
    _press(ui, glfw.KEY_W)
    assert ui.state.perform.slot_load is None
    assert ui._request_full_reset is False
    assert ui.state.perform.enabled is False
    assert ui._keys_pressed == set()


def test_a_locked_key_never_reaches_imgui(ui):
    seen = []
    ui.imgui_key_callback = lambda *a: seen.append(a)
    ui.state.perform.exhibition_locked = True
    _press(ui, glfw.KEY_A)
    assert seen == []
    ui.state.perform.exhibition_locked = False
    _press(ui, glfw.KEY_A)
    assert len(seen) == 1


def test_only_the_chord_unlocks(ui):
    ui.state.perform.exhibition_locked = True
    _press(ui, glfw.KEY_F11, glfw.MOD_CONTROL)
    assert ui.state.perform.exhibition_locked is True
    _press(ui, glfw.KEY_F11, glfw.MOD_CONTROL | glfw.MOD_SHIFT)
    assert ui.state.perform.exhibition_locked is False
    assert ui.state.perform.enabled is False, "the chord is not the F11 toggle"


def test_a_ctrl_number_is_not_a_slot(ui):
    _press(ui, glfw.KEY_1, glfw.MOD_CONTROL)
    assert ui.state.perform.slot_load is None


# ---- the dispatch ----------------------------------------------------------

class _Sim:
    def __init__(self):
        self.rules = []
        # A rig load rerolls the audio weights, which asks the layout how
        # many inputs it has; a deaf layout is left alone.
        self.brain_layout = type("L", (), {"audio_inputs": 0,
                                           "signature": lambda _s: "fourier-n10+c2"})()

    def apply_rule(self, rule):
        self.rules.append(rule)


class _UI:
    def __init__(self, root):
        self.root = root
        self.loaded = []

    def _get_config_path(self, filename, category):
        return self.root / f"{filename}.json"

    def update_physics_defaults(self, filename):
        self.loaded.append(filename)


def _handler(tmp_path):
    ch = CommandHandler(_Sim(), None, _UI(tmp_path), RuleManager(), None,
                        None, ConfigSaver(), None, tmp_path)
    ch.apply_brain_layout = None
    return ch


def _write(tmp_path, stem, gain):
    d = PhysicsConfig().to_dict()
    d["physics"]["sensor_gain"] = gain
    cfg = PhysicsConfig.from_dict(d)
    (tmp_path / f"{stem}.json").write_text(cfg.to_json(), encoding="utf-8")


def test_a_slot_press_loads_on_the_cold_path_and_recentres_the_knobs(tmp_path):
    ch = _handler(tmp_path)
    _write(tmp_path, "reef", 3.0)
    ui = UIState()
    ui.perform.slot_load = ("reef", "Core", "")

    ch.process_commands(ui, False)

    assert ui.perform.slot_load is None
    assert ui.sim.SENSOR_GAIN == pytest.approx(3.0)
    assert ui.perform.recentre_knobs is True
    assert ui.undo_tag == "Load reef"
    assert ch.ui.loaded == ["reef"]


def test_a_slot_press_under_a_live_hover_loads_its_own_preset(tmp_path):
    ch = _handler(tmp_path)
    _write(tmp_path, "hovered", 1.0)
    _write(tmp_path, "reef", 3.0)
    ui = UIState()
    # A hover: the preview handler pushed a rule and left the flag up.
    ui.request_preview_config = True
    ui.preview_filename = "hovered"
    ui.preview_category = "Core"
    ch.process_commands(ui, False)
    assert ch.preview_rule_active

    ui = UIState()
    ui.perform.slot_load = ("reef", "Core", "")
    ch.process_commands(ui, False)

    assert ch.preview_rule_active is False
    assert ui.sim.SENSOR_GAIN == pytest.approx(3.0)
    assert ch.ui.loaded[-1] == "reef"


def test_a_slot_loads_its_rig_through_the_audio_panels_own_path(tmp_path, monkeypatch):
    from services import audio_rig_io
    from state.audio_in_state import AudioInState
    monkeypatch.setattr(audio_rig_io, "rigs_dir", lambda: tmp_path / "rigs")
    rig = AudioInState()
    rig.global_strength = 0.25
    audio_rig_io.save_rig(rig, audio_rig_io.preset_path("loud"))
    ch = _handler(tmp_path)
    _write(tmp_path, "reef", 3.0)
    ui = UIState()
    ui.audio.global_strength = 1.0
    ui.perform.slot_load = ("reef", "Core", "loud")

    ch.process_commands(ui, False)

    assert ui.sim.SENSOR_GAIN == pytest.approx(3.0)
    assert ui.audio.global_strength == pytest.approx(0.25)
    assert ui.audio.preset_name == "loud"
    assert ui.audio.request_load_preset is False


def test_a_slot_with_only_a_rig_leaves_the_physics_alone(tmp_path, monkeypatch):
    from services import audio_rig_io
    from state.audio_in_state import AudioInState
    monkeypatch.setattr(audio_rig_io, "rigs_dir", lambda: tmp_path / "rigs")
    audio_rig_io.save_rig(AudioInState(), audio_rig_io.preset_path("quiet"))
    ch = _handler(tmp_path)
    ui = UIState()
    ui.sim.SENSOR_GAIN = 7.0
    ui.perform.slot_load = ("", "", "quiet")

    ch.process_commands(ui, False)

    assert ui.sim.SENSOR_GAIN == 7.0
    assert ui.audio.preset_name == "quiet"
    assert ch.ui.loaded == []
