"""The exposure feedback loop cannot mint a NaN.

frame_assembly.frag's is_first_frame branch undoes the PREVIOUS frame's asinh
compression with a sinh, so exposure can blend two frames in linear space. That
branch runs on every frame (is_first_frame is sample 0, not frame 0) and reads
the buffer it writes, so it is a feedback loop whose gain is EXPOSURE - and the
sinh was its one unbounded step.

sinh overflows float32 at an argument of 89.4. TONEMAP_SOFTNESS reaches 5.0 on
its slider, so an accumulated length of ~18 is enough; at the default 1.0 it
takes ~89. Once a component is inf, safenorm divides inf by inf and yields NaN,
and that NaN lives in the ACCUMULATION buffer - which the canvas-side scrub in
canvas.frag never touches, because it is not the canvas. That is why the sim
looks healthy while the view rots.

Pure arithmetic over the shader's own formula. A GL test would report that some
pixels went dark; it would not say which step did it.
"""
from __future__ import annotations

import numpy as np
import pytest

f32 = np.float32

# The slider bounds in ui/preferences_window.py. If these move, the reachable
# trigger moves with them.
SOFTNESS_MIN, SOFTNESS_MAX = 0.1, 5.0
EXPOSURE_MAX = 1.0

# shaders/frame_assembly.frag
SINH_MAX_ARG = 88.0
SAFE_MAX = 3.0e38


def safenorm(v):
    n = f32(np.linalg.norm(v))
    return v / n if n > 0 else np.zeros(3, np.float32)


def undo_tonemap(previous, softness, clamp):
    """The is_first_frame branch's inverse-tonemap step, both ways.

    The bound is on the QUOTIENT: the divide by softness comes after the sinh,
    so clamping only the argument still overflows for softness below 1.
    """
    previous = np.asarray(previous, dtype=np.float32)
    plen = f32(np.linalg.norm(previous))
    with np.errstate(over="ignore", invalid="ignore"):
        if clamp:
            arg = min(plen * f32(softness), f32(SINH_MAX_ARG))
            stretched = min(f32(np.sinh(arg)) / f32(softness), f32(SAFE_MAX))
        else:
            stretched = f32(np.sinh(plen * f32(softness))) / f32(softness)
        return safenorm(previous) * stretched


@pytest.mark.parametrize("softness", [SOFTNESS_MIN, 1.0, 3.0, SOFTNESS_MAX])
@pytest.mark.parametrize("plen", [0.0, 1.0, 10.0, 50.0, 200.0, 1e4, 1e12])
def test_the_clamped_form_is_always_finite(softness, plen):
    out = undo_tonemap([plen, 0.0, 0.0], softness, clamp=True)
    assert np.all(np.isfinite(out)), f"softness={softness} len={plen} -> {out}"


def test_the_unclamped_form_really_did_overflow():
    """Without this the test above could pass against a no-op change."""
    bad = undo_tonemap([20.0, 0.0, 0.0], SOFTNESS_MAX, clamp=False)
    assert not np.all(np.isfinite(bad)), (
        "the unclamped sinh must actually overflow, or this bug was elsewhere")


def test_the_overflow_produced_a_nan_not_merely_an_inf():
    """safenorm divides inf by inf. An inf would still be a bounded-looking
    value the tonemap could squash; a NaN poisons everything it touches."""
    bad = undo_tonemap([20.0, 0.0, 0.0], SOFTNESS_MAX, clamp=False)
    assert np.any(np.isnan(bad)), f"expected a NaN, got {bad}"


def test_the_trigger_was_reachable_from_the_sliders():
    """How much accumulated brightness it took, at the top of the track."""
    trigger = np.arcsinh(np.finfo(np.float32).max) / SOFTNESS_MAX
    assert trigger < 20.0, (
        f"needs an accumulated length of {trigger:.1f} at softness "
        f"{SOFTNESS_MAX} - well inside what a long exposure reaches")


@pytest.mark.parametrize("softness", [SOFTNESS_MIN, 1.0, 3.0, SOFTNESS_MAX])
def test_the_clamp_changes_nothing_that_already_worked(softness):
    """The round trip must stay exact wherever it did not overflow, or this
    fix would alter every preset's look."""
    for plen in (0.0, 0.5, 1.0, 5.0):
        if plen * softness >= SINH_MAX_ARG:
            continue
        a = undo_tonemap([plen, 0.0, 0.0], softness, clamp=False)
        b = undo_tonemap([plen, 0.0, 0.0], softness, clamp=True)
        assert np.allclose(a, b, rtol=0, atol=0), (
            f"clamp perturbed a working value at softness={softness} len={plen}")


def test_a_saturated_value_stays_large_rather_than_going_dark():
    """Saturating must read as 'very bright', not as the black hole the NaN
    produced - the whole complaint was the picture going black."""
    out = undo_tonemap([1e6, 0.0, 0.0], SOFTNESS_MAX, clamp=True)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 1e12, f"expected a huge finite value, got {out}"


# The divide by softness comes AFTER the sinh, so clamping the argument alone
# still overflows once sinh(88)/S passes float32's ceiling - that is S below
# 0.2427, and the slider's floor is 0.1. This is the defect the first version
# of the fix had.
ARG_ONLY_FAILS_BELOW = float(np.sinh(SINH_MAX_ARG)) / float(np.finfo(np.float32).max)


@pytest.mark.parametrize("softness", [SOFTNESS_MIN, 0.2])
def test_clamping_only_the_argument_would_not_have_been_enough(softness):
    """Why the bound is on the quotient rather than the argument."""
    assert softness < ARG_ONLY_FAILS_BELOW, "pick a softness that actually fails"
    plen = 1e6
    arg = min(plen * f32(softness), f32(SINH_MAX_ARG))
    with np.errstate(over="ignore"):
        arg_only = f32(np.sinh(arg)) / f32(softness)
    assert not np.isfinite(arg_only), (
        f"softness={softness} must overflow when only the argument is clamped")
    assert np.all(np.isfinite(undo_tonemap([plen, 0, 0], softness, clamp=True)))


def test_the_argument_only_defect_was_reachable_from_the_slider():
    """It is a narrow window - the bottom quarter of the softness track - so
    it must be pinned, or a future 'simplification' back to an argument clamp
    would look harmless."""
    assert SOFTNESS_MIN < ARG_ONLY_FAILS_BELOW < 1.0
