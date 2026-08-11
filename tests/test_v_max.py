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

# The fastest particle anywhere in the preset library, in V Max units.
# Re-derive with `python -m tools.measure_speed`.
LIBRARY_PEAK = 0.0204


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


def test_the_limit_applies_before_the_particle_moves():
    """Clamping after `e.pos += e.vel` would let a step land anywhere and only
    tidy up the velocity afterwards."""
    accelerate = SHADER.index("e.vel = e.vel*calculate_setting(get_particle_drag()")
    clamp = SHADER.index("if(vmag > vlim)")
    move = SHADER.index("e.pos += e.vel;")
    assert accelerate < clamp < move


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


def test_the_default_is_above_anything_the_preset_library_reaches():
    """The default has to be a no-op, or loading a preset saved before this
    parameter existed would quietly brake it."""
    assert UNCONSTRAINED > LIBRARY_PEAK


def test_the_library_spans_most_of_the_slider():
    """value = max * t**exponent. If the library crowds into the bottom of
    the track the control reads as doing nothing."""
    p = PARAM_BY_NAME['V_MAX']
    slowest = (0.00004 / p.default_max) ** (1.0 / p.power_exponent)
    peak = (LIBRARY_PEAK / p.default_max) ** (1.0 / p.power_exponent)
    assert slowest < 0.25 and peak > 0.6, (slowest, peak)


# --- the decision recorded in the archive's width ----------------------

def test_v_max_is_not_in_the_optimizer_search_space():
    """Widening the physics genome widens Archive._phys, and every vectors.npz
    already written holds the current width."""
    from services.physics_genome import PHYSICS_DIM, PHYSICS_PARAMS as SEARCHED

    assert "V_MAX" not in [n for n, _g, _lo, _hi in SEARCHED]
    assert PHYSICS_DIM == 8
