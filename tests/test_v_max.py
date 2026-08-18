"""V Max: the per-step speed limit.

It is a preset parameter, not a searched one, so the things worth guarding are
the format's backward compatibility and the two raw std430 writers that have to
agree with the GLSL struct.
"""
import inspect
import json
import re
from pathlib import Path

import pytest

from services.config_saver import ConfigSaver, PhysicsConfig
from state import SimState
from ui.physics_params import PARAM_BY_NAME, PHYSICS_PARAMS

ROOT = Path(__file__).resolve().parent.parent
SHADER = (ROOT / "shaders" / "entity_update.glsl").read_text(encoding="utf-8")

UNCONSTRAINED = PARAM_BY_NAME["V_MAX"].default_max

# The preset library's step distance in V Max units: the slowest preset's
# median, and the fastest preset's p99.
# Re-derive with `python -m tools.measure_speed`.
LIBRARY_FLOOR = 0.000036
LIBRARY_PEAK = 0.0461


# --- the preset format -------------------------------------------------

def test_a_preset_saved_before_v_max_existed_is_unconstrained():
    """Every config on disk predates this parameter. A missing key must mean
    'no limit', or loading one silently changes the physics it was saved with."""
    data = json.loads(PhysicsConfig().to_json())
    del data["physics"]["v_max"]

    cfg = PhysicsConfig.from_dict(data)
    assert cfg.v_max == UNCONSTRAINED

    state = SimState()
    state.V_MAX = 0.004
    ConfigSaver().apply_config(cfg, state)
    assert state.V_MAX == UNCONSTRAINED


def test_a_real_preset_on_disk_still_loads():
    cfg = ConfigSaver().load_from_file(ROOT / "physics_configs" / "Core" / "_Default.json")
    assert cfg is not None
    assert cfg.v_max == UNCONSTRAINED


def test_v_max_round_trips_through_json():
    state = SimState()
    state.V_MAX = 0.0123
    saver = ConfigSaver()

    cfg = PhysicsConfig.from_json(saver.create_config(state, None).to_json())
    assert cfg.v_max == pytest.approx(0.0123)

    out = SimState()
    saver.apply_config(cfg, out)
    assert out.V_MAX == pytest.approx(0.0123)


def test_v_max_round_trips_through_the_clipboard():
    state = SimState()
    state.V_MAX = 0.006
    saver = ConfigSaver()

    out = SimState()
    assert saver.load_from_string(saver.save_to_string(state, None), out) is not None
    assert out.V_MAX == pytest.approx(0.006)


# --- the shader and the two writers that feed it -----------------------

def _glsl_config_members() -> list[str]:
    body = re.search(r"struct MultiLoadConfig \{(.*?)\n\};", SHADER, re.S).group(1)
    return re.findall(r"PhysicsSetting\s+(\w+);", body)


def test_the_glsl_struct_and_both_ssbo_writers_agree():
    """Both writers pack this struct by offset, so a member the shader has and
    Python does not shifts every field after it into the wrong slot."""
    moderngl = pytest.importorskip("moderngl")  # noqa: F841 - sim imports it
    from sim import MULTI_LOAD_CONFIG_SIZE, Sim

    members = _glsl_config_members()
    assert members == [n.lower() for n, _lo, _hi in Sim._TOURNAMENT_PHYSICS_ORDER]

    src = inspect.getsource(Sim._write_multi_load_ssbo)
    packed = re.findall(r"\('(\w+)', '[A-Z_]+', ", src)
    assert packed == members

    assert MULTI_LOAD_CONFIG_SIZE == len(members) * 7 * 4 + 6 * 4 + 3 * 4


def test_v_max_reaches_the_shader():
    assert "uniform PhysicsSetting V_MAX_SETTING;" in SHADER
    assert "configs[idx].v_max : V_MAX_SETTING" in SHADER


def test_the_limit_covers_strafe_and_not_just_velocity():
    """Strafe is added STRAIGHT TO POSITION. Capping e.vel alone caps nothing a
    strafing preset does - with V Max at zero its particles keep flying."""
    accelerate = SHADER.index("e.vel = e.vel*drag_ts + force*force_gain;")
    # Anchored on the statement, not on what computes its scalar - this asserts
    # ORDER, and a wrapper around the setting is not a change of order.
    hop = SHADER.index("vec2 hop = strafe*")
    assert "get_particle_strafe_power(" in SHADER[hop:hop + 200]
    delta = SHADER.index("vec2 step_delta = e.vel + hop;")
    clamp = SHADER.index("if(vraw < vm.max_value && smag > vlim)")
    move = SHADER.index("e.pos += step_delta*TIME_SCALE;")
    assert accelerate < hop < delta < clamp < move
    assert "e.pos += e.vel;" not in SHADER, "the unclamped move is back"


def test_the_limit_covers_the_advanced_drawing_field():
    """The brush's strafe field is another straight-to-position term, and its
    force field feeds a velocity the cap has to reach as well."""
    delta = SHADER.index("vec2 step_delta = e.vel + hop;")
    brush = SHADER.index("vec4 draw_sample = get_field(")
    clamp = SHADER.index("if(vraw < vm.max_value && smag > vlim)")
    assert delta < brush < clamp
    assert "step_delta += .01*strafe_field_strength*draw_sample.zw;" in SHADER
    assert "e.pos += .01*strafe_field_strength" not in SHADER


def test_the_limit_also_scales_the_stored_velocity():
    """Otherwise speed piles up behind the cap and lurches when it is raised."""
    body = SHADER[SHADER.index("if(vraw < vm.max_value && smag > vlim)"):]
    body = body[:body.index("e.pos += step_delta*TIME_SCALE;")]
    assert "step_delta *= k;" in body and "e.vel *= k;" in body


def test_the_clock_scales_the_step_AFTER_the_cap():
    """V Max is a distance per unit TIME, not per step. Scaled before the
    clamp, halving the clock would halve the speed a preset is allowed."""
    clamp = SHADER.index("if(vraw < vm.max_value && smag > vlim)")
    move = SHADER.index("e.pos += step_delta*TIME_SCALE;")
    assert clamp < move
    assert "float smag = length(step_delta*TIME_SCALE)" not in SHADER


# --- the naming rule the SSBO writers depend on ------------------------

def test_every_slider_label_is_the_title_case_of_its_field():
    """Both SSBO writers look a custom slider range up by
    `NAME.replace('_', ' ').title()`. A label that breaks that rule does not
    raise - it silently falls back to the default range for that tile."""
    for p in PHYSICS_PARAMS:
        assert p.label == p.name.replace("_", " ").title(), p.name


def test_v_max_is_a_registered_slider():
    p = PARAM_BY_NAME["V_MAX"]
    assert p.group == "forces"
    assert p.is_power_scaled          # useful values sit near the bottom
    assert (p.hard_min, p.hard_max) == (0.0, p.default_max)
    assert SimState().V_MAX == p.default_max
    assert PhysicsConfig().v_max == p.default_max


def test_the_top_of_the_track_is_off_rather_than_a_large_number():
    """"Above every preset" is a measurement, and a hand-tuned config can beat
    it. Off is a fact."""
    p = PARAM_BY_NAME["V_MAX"]
    assert p.off_at_max
    assert SimState().V_MAX >= p.default_max
    assert "vraw < vm.max_value" in SHADER


def test_the_readout_says_off_where_the_shader_switches_off():
    """A slider showing 0.10000 while the limit is disabled is the same lie
    the measured-headroom version told."""
    src = (ROOT / "ui" / "slider_widgets.py").read_text(encoding="utf-8")
    assert '"Off" if (pdef.off_at_max and value >= pdef.default_max)' in src


def test_the_default_is_above_anything_the_preset_library_reaches():
    """The default has to be a no-op, or loading a preset saved before this
    parameter existed would quietly brake it."""
    assert UNCONSTRAINED > LIBRARY_PEAK


def test_the_library_spans_most_of_the_slider():
    """value = max * t**exponent. If the library crowds into the bottom of
    the track the control reads as doing nothing."""
    p = PARAM_BY_NAME['V_MAX']
    slowest = (LIBRARY_FLOOR / p.default_max) ** (1.0 / p.power_exponent)
    peak = (LIBRARY_PEAK / p.default_max) ** (1.0 / p.power_exponent)
    assert slowest < 0.25 and peak > 0.6, (slowest, peak)


@pytest.mark.gpu
def test_a_zero_limit_actually_stops_a_strafing_preset():
    """The source tests above all passed while V Max limited only e.vel and
    particles kept moving. Only running the sim catches that."""
    pytest.importorskip("moderngl")
    import moderngl
    import numpy as np

    from sim import SIZE_OF_ENTITY_STRUCT, Sim

    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    cfg = ConfigSaver().load_from_file(
        ROOT / "physics_configs" / "Core" / "_Default.json")
    state = SimState()
    rule = ConfigSaver().apply_config(cfg, state)
    assert state.STRAFE_POWER > 0, "this preset must strafe or the test proves nothing"

    def moved(v_max):
        state.V_MAX = v_max
        sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
        sim.apply_state(state)
        sim.apply_rule(rule)
        sim.reset()
        for _ in range(80):
            sim.apply_state(state)
            sim.update(ctx)
        ctx.finish()

        def pos():
            raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
            return raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)[:, 0:2].copy()

        a = pos()
        sim.apply_state(state)
        sim.update(ctx)
        ctx.finish()
        d = np.linalg.norm((pos() - a).astype(np.float64), axis=1)
        return float(np.median(d[d < 0.5]))

    free = moved(UNCONSTRAINED)
    pinned = moved(1e-9)
    ctx.release()

    assert free > 1e-5, "the preset has to move at all for this to mean anything"
    assert pinned < free / 100.0, (
        f"V Max near zero still moved {pinned:.7f}/step against {free:.7f} free")


# --- the decision recorded in the archive's width ----------------------

def test_v_max_is_not_in_the_optimizer_search_space():
    """Widening the physics genome widens Archive._phys, and every vectors.npz
    already written holds the current width."""
    from services.physics_genome import PHYSICS_DIM, PHYSICS_PARAMS as SEARCHED

    assert "V_MAX" not in [n for n, _g, _lo, _hi in SEARCHED]
    assert PHYSICS_DIM == 8
