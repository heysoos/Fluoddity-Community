"""The per-frame layout apply must be a no-op when nothing changed.

CommandHandler._handle_brain_layout calls App._apply_brain_layout EVERY frame,
not just on the one-shot flag - a scale slider commits immediately and still has
to reach the decode. That is only safe because _apply_brain_layout early-returns
when the layout matches, and the heavy path it guards is destructive: it calls
sim.apply_rule(None), which DISCARDS the loaded brain. If the layout the UI
derives ever disagreed with sim.brain_layout at rest, the user's rule would be
wiped 60 times a second.

The disagreement is easy to introduce by accident, because BrainLayout compares
with scales EXCLUDED (compare=False) while the light path compares scales
explicitly. A default_layout() that dropped its scales would still be `==` to
the UI's layout and would still take the light path every frame.
"""
import pytest

import main
from services.brains import REGISTRY, default_layout
from state import UIState
from ui.brain_window import layout_for


class _StubSim:
    """Records what was done to it. Mirrors the real methods' contracts,
    including set_brain_scales raising on a width change."""

    def __init__(self, layout):
        self._layout = layout
        self.calls: list[str] = []

    @property
    def brain_layout(self):
        return self._layout

    def set_brain_scales(self, layout):
        if layout.length != self._layout.length:
            raise ValueError("width changed; use realloc_brain_buffers")
        self._layout = layout
        self.calls.append("set_brain_scales")

    def realloc_brain_buffers(self, layout):
        self._layout = layout
        self.calls.append("realloc")

    def apply_rule(self, genome):
        self.calls.append("apply_rule")


class _StubApp:
    """App's collaborators, all absent - _apply_brain_layout guards each with
    `is not None`, so this exercises the real control flow."""

    auto_service = imgep_driver = archive = None
    goal_list = archive_store = thumb_cache = None

    def __init__(self, layout=None):
        self.sim = _StubSim(layout or default_layout())
        self.specs: list[tuple] = []

    def _refresh_driver_specs(self, layout, reset: bool = False):
        self.specs.append((layout, reset))

    def _build_archive_set(self, path):
        raise AssertionError("unreachable: archive and store are both None")

    def apply(self, layout, ui_state):
        return main.App._apply_brain_layout(self, layout, ui_state)


def _ui(modality="fourier", settings=None):
    st = UIState()
    st.brain.modality = modality
    st.brain.settings = dict(settings or {})
    return st


def test_the_startup_state_is_already_settled():
    """The single most important assertion here: at rest, frame after frame,
    the apply does nothing at all."""
    app, ui_state = _StubApp(), _ui()
    layout = layout_for(ui_state.brain.modality, ui_state.brain.settings)

    for _ in range(3):
        assert app.apply(layout, ui_state) is False
    assert app.sim.calls == [], "the idle path touched the GPU"
    assert app.specs == [], "the idle path reset the optimizer's spec"


def test_the_defaults_the_ui_derives_match_the_sim_exactly():
    """Equality is not enough - scales are compare=False, so two layouts can be
    `==` and still send every frame down the light path."""
    ui_state = _ui()
    ui_layout = layout_for(ui_state.brain.modality, ui_state.brain.settings)
    sim_layout = default_layout()
    assert ui_layout == sim_layout
    assert tuple(ui_layout.scales) == tuple(sim_layout.scales)


@pytest.mark.parametrize("modality", ["fourier", "gabor", "lenia", "mlp"])
def test_a_switch_settles_on_the_very_next_frame(modality):
    """The switch itself is destructive; the frame after it must not be.

    This is what catches a modality whose layout the sim stores differently
    from the one the UI derives - the switch would repeat forever, and every
    repeat calls apply_rule(None).
    """
    app, ui_state = _StubApp(), _ui(modality)
    layout = layout_for(modality, ui_state.brain.settings)

    first = app.apply(layout, ui_state)
    app.sim.calls.clear()
    app.specs.clear()

    assert app.apply(layout, ui_state) is False
    assert app.sim.calls == [], f"{modality} re-applies every frame"
    if modality != "fourier":
        assert first is True, "a real switch should have reported a change"


@pytest.mark.parametrize("modality,key,value", [
    ("fourier", "freq_scale", 1.0),
    ("gabor", "envelope_width", 0.3),
    ("lenia", "mu_scale", 0.5),
])
def test_a_scale_change_takes_the_light_path(modality, key, value):
    """No realloc, and above all no apply_rule: changing a squash must not cost
    the user the brain they are looking at."""
    tuned = REGISTRY[modality].layout_from_settings({key: value})
    app = _StubApp(REGISTRY[modality].layout_from_settings({}))

    assert app.apply(tuned, _ui(modality, {key: value})) is True
    assert app.sim.calls == ["set_brain_scales"]
    assert app.specs == [(tuned, False)], "a scale change reset the optimizer"


def test_a_scale_change_settles_too():
    tuned = REGISTRY["gabor"].layout_from_settings({"envelope_width": 0.3})
    app = _StubApp(REGISTRY["gabor"].layout_from_settings({}))
    app.apply(tuned, _ui("gabor", {"envelope_width": 0.3}))
    app.sim.calls.clear()
    assert app.apply(tuned, _ui("gabor", {"envelope_width": 0.3})) is False
    assert app.sim.calls == []


def test_a_count_change_takes_the_heavy_path():
    """A different width cannot go through set_brain_scales - it would raise -
    so the guard must route it to the realloc."""
    wide = REGISTRY["gabor"].layout_from_settings({"filters": 24})
    app = _StubApp(REGISTRY["gabor"].layout_from_settings({"filters": 12}))
    assert app.apply(wide, _ui("gabor", {"filters": 24})) is True
    assert app.sim.calls == ["realloc", "apply_rule"]
    assert app.specs == [(wide, True)], "a width change must reset the search"


def test_the_handler_still_clears_the_flag_before_applying():
    """Clearing after the apply would leave the flag set if the apply raised,
    firing the destructive path again on the next frame."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "command_handler.py").read_text()
    i = src.index("def _handle_brain_layout")
    body = src[i:src.index("def ", i + 10)]
    assert body.index("request_layout_change = False") < body.index(
        "self.apply_brain_layout("), "the flag is cleared after the apply"
