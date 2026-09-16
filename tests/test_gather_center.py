"""Gather to centre: every particle back to the middle, in one shot.

A fourth initial-conditions mode, Centre, puts every cohort on the one small
disk at the origin. The gather action resets the sim under that mode for the
reset step alone, so a preset's own spawn mode is untouched and its hazard
respawns keep landing where they always did.
"""
import glfw
import numpy as np
import pytest

from command_handler import CommandHandler
from state.sim_state import IC_CENTER, SimState
from state.ui_state import UIState
from tests.test_exhibition_keys import _Keys, _press, gui  # noqa: F401


# ---- the key --------------------------------------------------------------

@pytest.fixture
def ui(gui, tmp_path):
    u = _Keys(tmp_path)
    u._request_gather_center = False
    return u


def test_the_gather_key_is_bound_by_default(ui):
    assert ui.keybindings.get_key("gather_center") == glfw.KEY_M


def test_the_key_asks_for_a_gather_unlocked_and_locked(ui):
    _press(ui, glfw.KEY_M)
    assert ui._request_gather_center is True
    ui._request_gather_center = False
    ui.state.perform.exhibition_locked = True
    _press(ui, glfw.KEY_M)
    assert ui._request_gather_center is True


# ---- the handler ------------------------------------------------------------

class _Sim:
    def __init__(self):
        self.resets = []

    def reset(self, mode=None):
        self.resets.append(mode)


def test_the_handler_resets_under_centre_and_leaves_the_setting_alone():
    h = object.__new__(CommandHandler)
    h.sim = _Sim()
    st = UIState()
    st.sim.initial_conditions = 2
    st.request_gather_center = True
    h._handle_gather_center(st)
    assert h.sim.resets == [IC_CENTER]
    assert st.sim.initial_conditions == 2


# ---- the shader -------------------------------------------------------------

def _positions(sim, ctx):
    from sim import SIZE_OF_ENTITY_STRUCT
    ctx.finish()
    raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
    return raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)[:, 0:2].copy()


@pytest.mark.gpu
def test_a_gather_lands_every_particle_on_the_centre_and_the_next_reset_does_not():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")
    from sim import Sim

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    state = SimState()
    state.initial_conditions = 1            # Random: spread over the canvas
    state.num_cohorts = 8
    sim.apply_state(state)

    sim.reset(IC_CENTER)
    sim.entity_update(ctx)
    gathered = _positions(sim, ctx)
    assert np.abs(gathered).max() < 0.05, "a gather must land on the centre disk"

    sim.reset()
    sim.entity_update(ctx)
    spread = _positions(sim, ctx)
    assert np.abs(spread).max() > 0.5, "the preset's own mode must be back"
    ctx.release()
