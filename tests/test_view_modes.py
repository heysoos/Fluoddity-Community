"""The view indices have one home, and the shader has to agree with it.

`view_mode` is a uniform the fragment shader branches on by number, so the
Python constants and the GLSL literals are one fact in two files. Removing the
brush view shifted all six at once - across seven files - which is exactly the
sweep these names exist to make unnecessary next time.

The numbers are not persisted (neither PhysicsConfig nor PreferencesState
carries one), so a renumber costs no migration; it costs a silent behaviour
change at every literal that was missed.
"""
import re
from pathlib import Path

from state import view_modes as vm

ROOT = Path(__file__).resolve().parents[1]
FRAG = (ROOT / "shaders" / "frame_assembly.frag").read_text(encoding="utf-8")
BODY = re.sub(r"//[^\n]*", "", FRAG)          # comments are prose, not branches


def test_the_indices_are_contiguous_and_start_at_zero():
    names = [vm.CANVAS, vm.CAMERA, vm.CAMERA_TILED,
             vm.FORCE_FIELD, vm.STRAFE_FIELD, vm.CAMERA_TRAILS]
    assert names == list(range(len(names)))


def test_watercolor_views_are_a_subset_of_the_camera_views():
    """Watercolour has no meaning over the trail overlay, so the two sets are
    deliberately different - which is why they are two names."""
    assert set(vm.WATERCOLOR_VIEWS) < set(vm.CAMERA_VIEWS)
    assert vm.CAMERA_TRAILS not in vm.WATERCOLOR_VIEWS


def test_the_shader_draws_the_trail_overlay_for_the_right_view():
    assert f"view_mode == {vm.CAMERA_TRAILS} && TRAIL_OVERLAY_STRENGTH" in BODY


def test_the_shader_reads_the_strafe_field_from_the_right_view():
    assert f"else if (view_mode == {vm.STRAFE_FIELD})" in BODY


def test_the_shader_agrees_on_which_views_the_camera_draws():
    """The draw-target overlay is drawn over the camera views and no others."""
    expected = "||".join(f"view_mode=={i}" for i in sorted(vm.CAMERA_VIEWS))
    assert expected in BODY.replace(" ", "")


def test_the_shader_agrees_on_which_views_are_raw_textures():
    """Canvas and the two field views show a texture rather than the sim, so
    they are the ones needing the canvas-to-screen mapping."""
    raw = sorted({vm.CANVAS, *vm.FIELD_VIEWS})
    expected = "||".join(f"view_mode=={i}" for i in raw)
    assert BODY.replace(" ", "").count(expected) == 2, (
        "both overlay branches must use it")


def test_nothing_still_names_the_brush_view():
    assert "brush_tex" not in FRAG
    for name in ("sim.py", "camera.py", "main.py", "simulation_runner.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "brush_tex" not in src, f"{name} still refers to the brush texture"
