"""Render the Brain window for real, headless.

Only the imgui BACKEND needs GL; the widget layer is pure CPU, so a context plus
new_frame() is enough to execute every binding call.

Believing otherwise is why `imgui.collapsing_header("Inspector")[0]` shipped:
the one-argument overload returns a bare bool and only the p_visible overload
returns a tuple, so opening the window raised TypeError and took the app down
with the whole render path uncovered.

These tests call render_brain_window itself, so the binding checks its own
argument and return types. Verified by re-breaking the fix: with the `[0]` back,
the run fails.

Two nearby suspects turned out NOT to be bugs, which is only knowable by
executing them - pybind converts a plain tuple to ImVec2 and ImVec4, so
progress_bar and text_colored were fine with one. They are spelled explicitly
now to match the rest of ui/, not because they were broken.
"""
import numpy as np
import pytest

from services.brain_preview import AXES, CHANNELS, BrainPreview
from services.brains import REGISTRY
from state import UIState
from ui.brain_window import BrainWindowMixin

MODALITIES = ["fourier", "gabor", "lenia", "mlp"]


@pytest.fixture
def gui():
    """A live imgui frame with no renderer backend behind it."""
    imgui = pytest.importorskip("imgui_bundle").imgui
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 720)
    io.delta_time = 1.0 / 60.0
    # NewFrame otherwise asserts on an unbuilt font atlas: with no backend, no
    # one ever calls GetTexDataAsRGBA32. This flag says the backend uploads
    # font textures itself, which is vacuously true when there is none.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    imgui.new_frame()
    try:
        yield imgui
    finally:
        # A test that raises mid-window leaves begin/end unbalanced, and
        # end_frame would assert over the top of the real failure.
        try:
            imgui.end_frame()
        except Exception:
            pass
        imgui.destroy_context(ctx)


class _StubUI(BrainWindowMixin):
    """The mixin's `self`, carrying only what render_brain_window reads."""

    def __init__(self, modality="fourier", settings=None, *, preview=True,
                 best_z=None):
        self.state = UIState()
        self.state.brain.enabled = True
        self.state.brain.modality = modality
        self.state.brain.settings = dict(settings or {})
        # On BrainState, not on the UI: it is a display-only value the
        # orchestrator pushes, exactly like archive_entries. As a loose UI
        # attribute it was assigned by this stub and by nothing else in the app,
        # so the saturation readout was permanently 0%.
        self.state.brain.best_z = best_z
        # The real geometry helpers, with the GL half left unbuilt: unit_count
        # is a staticmethod and uv_for/grid read only _grid, so this exercises
        # the actual tile maths rather than a stand-in that cannot disagree.
        if preview:
            self.brain_preview = object.__new__(BrainPreview)
            self.brain_preview._grid = 4
            self.brain_preview_tex = _FakeTexture()
        else:
            self.brain_preview = None
            self.brain_preview_tex = None


class _FakeTexture:
    """imgui.ImTextureRef wants an integer handle; it never dereferences it
    without a renderer backend."""
    glo = 1


@pytest.mark.parametrize("modality", MODALITIES)
def test_the_window_renders(gui, modality):
    """Every modality's settings loop, including MLP's `choice` combo."""
    _StubUI(modality).render_brain_window()


@pytest.mark.parametrize("modality", MODALITIES)
def test_the_inspector_renders(gui, modality):
    """The crash was here, and only behind an OPEN header - so force it open.

    Left closed, this test passes against the broken code.
    """
    ui = _StubUI(modality)
    gui.set_next_item_open(True)
    ui.render_brain_window()


def test_the_saturation_warning_renders(gui):
    """This branch only runs past 10% saturation, so nothing else on the
    default path ever reaches it."""
    ui = _StubUI(best_z=np.full(80, 9.0, dtype=np.float32))
    ui.render_brain_window()
    from ui.brain_window import saturation_fraction

    assert saturation_fraction(ui.state.brain.best_z) == 1.0


def test_the_inspector_degrades_without_a_preview(gui):
    """A GL failure disables the preview mid-run; the window must still draw."""
    ui = _StubUI(preview=False)
    gui.set_next_item_open(True)
    ui.render_brain_window()


@pytest.mark.parametrize("axes", range(len(AXES)))
def test_every_slice_choice_renders(gui, axes):
    ui = _StubUI()
    ui.state.brain.preview_axes = axes
    gui.set_next_item_open(True)
    ui.render_brain_window()


@pytest.mark.parametrize("channel", range(len(CHANNELS)))
def test_every_output_choice_renders(gui, channel):
    ui = _StubUI()
    ui.state.brain.preview_channel = channel
    gui.set_next_item_open(True)
    ui.render_brain_window()


def test_nothing_opens_a_modal(gui):
    """There is no confirmation dialog, by request. A count change is deferred
    to slider release instead, which fixes the problem the modal existed for -
    a drag firing one archive rebuild per frame - without interrupting anyone.
    """
    from pathlib import Path

    ui = _StubUI("gabor", {"filters": 12})
    ui.render_brain_window()
    assert not gui.is_popup_open("", gui.PopupFlags_.any_popup_id)
    src = (Path(__file__).resolve().parent.parent
           / "ui" / "brain_window.py").read_text()
    assert "imgui.open_popup(" not in src
    assert "imgui.begin_popup_modal(" not in src


def test_a_pending_count_commits_once_the_slider_is_released(gui):
    """The failure this must never have: a deferred value that never lands.

    Nothing is held in a headless frame, so is_item_active() is false and the
    draft is due. If this ever returns {"filters": 12} the count sliders are
    dead - which is exactly the class of bug the scales wiring already had once
    (every non-count slider was decorative), and it is invisible on screen
    because the slider still moves.
    """
    ui = _StubUI("gabor", {"filters": 12})
    ui._brain_draft[("gabor", "filters")] = 30
    ui.render_brain_window()
    assert ui.state.brain.settings == {"filters": 30}
    assert ui._brain_draft == {}, "the draft outlived its commit"


def test_the_deferral_is_gated_on_the_slider_being_held(gui):
    """The other half cannot be driven headlessly - it needs a mouse held down
    across frames - so it is pinned at the source. The gate must be `is the
    widget still active`, not a one-shot event that can be missed."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "ui" / "brain_window.py").read_text()
    assert "not imgui.is_item_active()" in src


def test_a_scale_slider_commits_live(gui):
    """The other half: scales take the light path, so they must track the
    slider rather than wait for release."""
    ui = _StubUI("gabor")
    ui.render_brain_window()
    assert ("gabor", "envelope_width") not in ui._brain_draft


def test_switching_modality_clears_the_draft(gui):
    """A draft is keyed by (modality, setting); a stale one from the previous
    modality must not be committed into the new one on the next release."""
    ui = _StubUI("gabor", {"filters": 12})
    ui._brain_draft[("gabor", "filters")] = 30
    ui.state.brain.modality = "lenia"
    ui.state.brain.settings = {}
    ui.render_brain_window()
    assert ui.state.brain.settings == {}


@pytest.mark.parametrize("modality", MODALITIES)
def test_a_widest_layout_renders(gui, modality):
    """The tile loop is sized from the unit count; run it at the maximum each
    modality allows, which is where a grid or slot index would go wrong."""
    m = REGISTRY[modality]
    counts = {s.key: s.hi for s in m.settings_schema() if s.kind == "int"}
    ui = _StubUI(modality, counts)
    ui.brain_preview._grid = 8
    gui.set_next_item_open(True)
    ui.render_brain_window()


def test_a_closed_window_draws_nothing(gui):
    ui = _StubUI()
    ui.state.brain.enabled = False
    ui.render_brain_window()
