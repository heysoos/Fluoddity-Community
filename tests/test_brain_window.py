"""The Brain window's logic, tested apart from ImGui.

The render call needs a GL context, so the decisions it makes live in free
functions and are tested directly - the same split tests/test_shader_source.py
uses for shaders.
"""
import numpy as np
import pytest

from services.brains import REGISTRY, BrainLayout
from state import BrainState
from ui.brain_window import (SATURATION_Z, layout_change_needed,
                             layout_for, saturation_fraction)


def test_every_modality_is_offered():
    assert {"fourier", "gabor", "lenia", "mlp"} <= set(REGISTRY)


def test_a_count_setting_needs_a_layout_change():
    assert layout_change_needed({"centers": 10}, {"centers": 16}) is True
    assert layout_change_needed({"filters": 12}, {"filters": 8}) is True
    assert layout_change_needed({"bumps": 12}, {"bumps": 24}) is True
    assert layout_change_needed({"hidden": 16}, {"hidden": 32}) is True


def test_activation_needs_a_layout_change():
    """tanh and sin are different function families - a genome evolved under one
    means nothing under the other, so they must not share an archive."""
    assert layout_change_needed({"activation": 0}, {"activation": 1}) is True


def test_a_scale_setting_is_free():
    """Freq Scale and friends only move the squash; the genome keeps its width,
    so there is nothing to reset and no archive to switch."""
    assert layout_change_needed({"centers": 10}, {"freq_scale": 2.0}) is False
    assert layout_change_needed({"centers": 10}, {"centers": 10}) is False


def test_saturation_counts_params_near_the_rails():
    z = np.array([0.0, 0.1, 5.0, -5.0], dtype=np.float32)
    assert saturation_fraction(z) == 0.5


def test_saturation_of_nothing_is_zero_not_a_crash():
    assert saturation_fraction(np.zeros(0, dtype=np.float32)) == 0.0
    assert saturation_fraction(None) == 0.0


def test_the_saturation_threshold_is_one_percent_from_the_rail():
    assert np.tanh(SATURATION_Z) == pytest.approx(0.99, abs=0.005)


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia", "mlp"])
def test_layout_for_uses_defaults_when_settings_are_empty(name):
    got = layout_for(name, {})
    want = REGISTRY[name].layout_from_settings({})
    assert got == want


def test_layout_for_survives_an_unknown_modality():
    """A config written by a newer build must not stop the app from starting."""
    assert layout_for("no-such-brain", {}).modality == "fourier"


def test_layout_for_survives_a_setting_the_modality_does_not_have():
    layout = layout_for("lenia", {"hidden": 32, "bumps": 6})
    assert layout.modality == "lenia" and layout.shape[0] == 6


# ---- the state object ------------------------------------------------------

def test_brain_state_defaults_to_fourier():
    s = BrainState()
    assert s.modality == "fourier"
    assert s.settings == {}
    assert s.request_layout_change is False


def test_brain_state_settings_are_not_shared_between_instances():
    a, b = BrainState(), BrainState()
    a.settings["centers"] = 16
    assert b.settings == {}, "mutable default leaked between instances"


def test_the_mixin_is_registered_on_the_ui_class():
    """Writing the mixin is not enough - ui/core.py must inherit it and the
    render loop must call it, or the window silently never appears."""
    from ui import UI

    assert hasattr(UI, "render_brain_window")


def test_the_render_loop_calls_it():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "ui" / "core.py").read_text()
    assert "self.render_brain_window()" in src


def test_the_menu_can_open_it():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "ui" / "menu_bar.py").read_text()
    assert "self.state.brain.enabled" in src


def test_ui_state_carries_the_brain_group():
    from state import UIState

    assert isinstance(UIState().brain, BrainState)


def test_the_orchestrator_consumes_and_clears_the_flag():
    """A one-shot flag nobody clears would fire the switch every frame."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    src = (root / "command_handler.py").read_text()
    i = src.index("def _handle_brain_layout")
    body = src[i:i + 1200]
    assert "request_layout_change = False" in body, "the flag is never cleared"
    assert "apply_brain_layout" in body
    # and the orchestrator must actually supply the callback
    assert "apply_brain_layout = self._apply_brain_layout" in (
        root / "main.py").read_text()


def test_the_layout_round_trips_through_the_state():
    s = BrainState(modality="mlp", settings={"hidden": 32, "activation": 2})
    layout = layout_for(s.modality, s.settings)
    assert layout == BrainLayout("mlp", (32, 2), 9 * 32 + 4)
    assert layout.signature() == "mlp-n32-a2"
