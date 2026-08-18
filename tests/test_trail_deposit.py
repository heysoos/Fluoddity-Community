"""The trail is ONE moving average split across two passes.

canvas.frag keeps `trail_persistence` of what was there; brush.frag adds
`1 - trail_persistence` of this step's deposit. There is no intermediate brush
texture any more, so the two shaders each compute that number for themselves -
and they have to agree at every texel or the trail has a gain that varies with
nothing.

Agreeing is not a matter of passing the same uniform. `trail_persistence` is
POSITION-dependent: sweeps and jitter make it vary across the canvas, so a
single value for the whole frame is wrong wherever a sweep is on. Upstream does
exactly that. Both shaders therefore carry the same copy of calculate_setting,
and these tests are what keep the copies the same.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = (ROOT / "shaders" / "canvas.frag").read_text(encoding="utf-8")
BRUSH = (ROOT / "shaders" / "brush.frag").read_text(encoding="utf-8")


def grab(src, pattern):
    m = re.search(pattern, src, re.S)
    assert m, f"not found: {pattern}"
    return m.group(0)


def test_the_two_shaders_share_one_physics_setting_struct():
    pat = r"struct PhysicsSetting \{.*?\n\};"
    assert grab(CANVAS, pat) == grab(BRUSH, pat)


def test_the_two_shaders_share_one_calculate_setting():
    """Character-identical, deliberately. entity_update.glsl's copy differs on
    purpose - it has a real hash() and its own cohort source - but these two
    are evaluated at the same texel on the same frame and must not diverge by
    so much as a constant."""
    pat = r"float calculate_setting\(PhysicsSetting setting, vec2 pos, float cohort\)\s*\{.*?\n\}"
    assert grab(CANVAS, pat) == grab(BRUSH, pat)


def test_both_clamp_and_time_scale_the_same_way():
    for src in (CANVAS, BRUSH):
        assert re.search(r"clamp\(\s*\w+\s*,\s*0\.0\s*,\s*0\.999\s*\)", src)
        assert re.search(r"if \(TIME_SCALE != 1\.0\)", src)


def test_the_canvas_pass_only_decays():
    """It used to read the brush texture and mix. If it starts adding again
    while brush.frag also adds, every deposit lands twice."""
    assert "brush_tex" not in CANVAS
    assert "can_out = can_color * trail_persistence;" in CANVAS


def test_the_deposit_is_additive_and_unweighted_by_alpha():
    src = (ROOT / "sim.py").read_text(encoding="utf-8")
    assert "moderngl.ONE, moderngl.ONE" in src
    assert "moderngl.SRC_ALPHA, moderngl.ONE" not in src


def test_the_brush_deposits_into_the_framebuffer_it_was_given():
    """It must not bind one of its own: update() binds the canvas write buffer
    for the decay pass and the deposit has to land in that same target."""
    src = (ROOT / "sim.py").read_text(encoding="utf-8")
    body = grab(src, r"    def brush_update\(self.*?\n(?=    def )")
    assert ".use()" not in body
    assert "ctx.clear" not in body


import pytest


@pytest.mark.gpu
def test_the_brush_program_actually_receives_the_persistence():
    """The half the source-level tests above cannot see.

    Both shaders declaring calculate_setting proves nothing if only one of the
    two PROGRAMS is ever handed the struct. The brush read an all-zero one,
    which is a valid struct meaning slider_value 0 - so (1 - p) was 1 and every
    deposit landed 30x to 1000x too strong, with nothing raising anywhere. It
    took comparing canvas energy against the old pipeline to see it.
    """
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"no GL context: {exc}")

    from sim import Sim
    from state import SimState

    st = SimState()
    st.TRAIL_PERSISTENCE = 0.9427
    st.TIME_SCALE = 0.5
    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    sim.apply_state(st)
    sim.reset()
    sim.update(ctx)
    ctx.finish()

    brush = sim.brush_update_program
    assert brush['TRAIL_PERSISTENCE_SETTING.slider_value'].value == pytest.approx(0.9427)
    assert brush['TIME_SCALE'].value == pytest.approx(0.5)
    # ...and the same number the canvas decayed by.
    canvas = sim.canvas_update_program
    for field in ('slider_value', 'min_value', 'max_value',
                  'x_sweep', 'y_sweep', 'cohort_sweep', 'jitter'):
        name = f'TRAIL_PERSISTENCE_SETTING.{field}'
        assert brush[name].value == pytest.approx(canvas[name].value), name

    ctx.release()
