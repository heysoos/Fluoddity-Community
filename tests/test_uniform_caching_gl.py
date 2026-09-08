"""Frame-constant uniforms are uploaded once per FRAME, not once per step.

At speedmult 60-100 the physics runs that many steps between two rendered
frames, and none of the physics-setting structs can change across them - they
are read from the SimState that apply_state() handed over before the first
step. Uploading them per step is over a hundred tryset() calls of pure CPU per
step, measured at a fifth of the step.

What makes this safe is where the "already uploaded" mark lives. It is the
PROGRAM OBJECT, not a boolean: sim keeps one compiled entity-update program per
set of shader defines and swaps between them when the brain layout changes, so
a bare flag would hand a program that has been sitting in the cache for minutes
whatever uniforms it last happened to receive.
"""
import pytest

import sim as sim_module
from state import SimState

ENTITY_ONLY = "AXIAL_FORCE_SETTING.slider_value"
# Trail PERSISTENCE now goes to two programs - canvas.frag decays by it and
# brush.frag weights the deposit by it - so it is no longer a one-program
# probe. Diffusion still is.
CANVAS_ONLY = "TRAIL_DIFFUSION_SETTING.slider_value"


@pytest.fixture(scope="module")
def ctx():
    moderngl = pytest.importorskip("moderngl")
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                        # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    yield c
    c.release()


@pytest.fixture
def rig(ctx, monkeypatch):
    """A stepping sim plus a count of every uniform upload it makes."""
    s = sim_module.Sim(ctx, world_size=0.05, canvas_aspect_ratio="1:1",
                       particle_density=0.1)
    state = SimState()
    s.apply_state(state)
    s.reset()

    counts = {}
    real = sim_module.tryset

    def spy(prog, name, value):
        counts[name] = counts.get(name, 0) + 1
        return real(prog, name, value)

    monkeypatch.setattr(sim_module, "tryset", spy)

    def frame(steps):
        s.apply_state(state)
        for _ in range(steps):
            s.update(ctx)

    return s, state, counts, frame


def test_one_frame_uploads_the_physics_settings_once_however_many_steps(rig):
    _, _, counts, frame = rig
    frame(6)
    assert counts[ENTITY_ONLY] == 1
    assert counts[CANVAS_ONLY] == 1


def test_the_next_frame_uploads_them_again(rig):
    """A slider moved between frames has to reach the shader."""
    _, _, counts, frame = rig
    for _ in range(3):
        frame(4)
    assert counts[ENTITY_ONLY] == 3


def test_the_per_step_uniforms_still_run_every_step(rig):
    """frame_count advances, and WRITE_RULES is true for one step only."""
    _, _, counts, frame = rig
    frame(5)
    assert counts["WRITE_RULES"] == 5
    assert counts["frame_count"] == 15          # entity, brush and canvas programs


def test_swapping_the_program_re_uploads_without_a_new_frame(rig):
    """The brain-layout program cache is why the mark is the program object.

    realloc_brain_buffers hands back a program built for a different set of
    defines - possibly minutes ago, holding another creature's physics.
    """
    from services.brains import layout_from_signature

    s, _, counts, frame = rig
    frame(2)
    assert counts[ENTITY_ONLY] == 1
    before = s.entity_update_program

    s.realloc_brain_buffers(layout_from_signature("mlp-n16.16-a0.0"))
    assert s.entity_update_program is not before, (
        "this test is vacuous unless the program really changed")

    s.update(ctx=s.ctx)                          # same frame, no apply_state
    assert counts[ENTITY_ONLY] == 2


def test_what_does_not_come_through_apply_state_is_never_cached(rig):
    """The cache covers exactly what apply_state() hands over, and no more.

    reset() moves the seed and apply_tournament() rewrites the grid, both
    without going near apply_state - so caching either leaves the shader on the
    previous frame's value with nothing on screen to say so. Widening the block
    to cover them breaks test_reset_seed and test_tournament_cohort_colour,
    which is a long way from here.
    """
    _, _, counts, frame = rig
    frame(5)
    assert counts["RESET_SEED"] == 5
    assert counts["TILE_MODE"] == 15             # entity, brush and canvas programs
    assert counts["TOURNAMENT_MODE"] == 5        # entity only: brains and per-tile physics
