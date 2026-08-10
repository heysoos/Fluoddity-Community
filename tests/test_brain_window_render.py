"""Render the Brain window for real, headless.

test_brain_window.py asserts on the decisions the window makes and states that
"the render call needs a GL context". It does not. Only the imgui BACKEND needs
GL; the widget layer is pure CPU, so a context plus new_frame() is enough to
execute every binding call.

That wrong premise is exactly why `imgui.collapsing_header("Inspector")[0]`
shipped: the one-argument overload returns a bare bool and only the p_visible
overload returns a tuple, so opening the window raised TypeError and took the
app down. Nothing in the suite had ever called the function - the whole window
was covered only by tests of the free functions beside it.

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
        self.brain_best_z = best_z
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

    assert saturation_fraction(ui.brain_best_z) == 1.0


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


def test_the_layout_popup_renders(gui):
    """The confirm modal draws through a different begin_popup_modal path,
    which DOES return a tuple - the opposite of collapsing_header."""
    ui = _StubUI()
    ui._pending_brain = ("gabor", {"filters": 8})
    gui.open_popup("Change brain layout?")
    ui.render_brain_window()


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
