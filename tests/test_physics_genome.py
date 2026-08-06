"""Physics as part of the search space."""
import numpy as np
import pytest

from services.auto_tournament_service import AutoTournamentService
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.physics_genome import (
    PHYSICS_DIM,
    PHYSICS_PARAMS,
    decode_physics,
    encode_physics,
    midpoint_z,
)
from services.tournament_service import TournamentService


def test_every_parameter_decodes_inside_its_slider_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        vals = decode_physics(rng.normal(0, 3, PHYSICS_DIM))
        for name, _glsl, lo, hi in PHYSICS_PARAMS:
            assert lo <= vals[name] <= hi, f"{name} escaped [{lo}, {hi}]"


def test_extreme_z_saturates_rather_than_escaping():
    """tanh, not clipping: no repair bias at the boundary."""
    hot = decode_physics(np.full(PHYSICS_DIM, 50.0))
    cold = decode_physics(np.full(PHYSICS_DIM, -50.0))
    for name, _g, lo, hi in PHYSICS_PARAMS:
        assert hot[name] == pytest.approx(hi, abs=1e-4)
        assert cold[name] == pytest.approx(lo, abs=1e-4)


def test_zero_is_the_midpoint_of_every_range():
    vals = decode_physics(midpoint_z())
    for name, _g, lo, hi in PHYSICS_PARAMS:
        assert vals[name] == pytest.approx((lo + hi) / 2)


def test_encode_decode_roundtrips():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, PHYSICS_DIM)
    vals = decode_physics(z)
    assert np.allclose(encode_physics(vals), z, atol=1e-3)


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError):
        decode_physics(np.zeros(PHYSICS_DIM + 1))


def test_spec_dimension_grows_by_exactly_the_physics_block():
    assert BRAIN_PHYSICS_SPEC.dim == BRAIN_SPEC.dim + PHYSICS_DIM
    assert BRAIN_PHYSICS_SPEC.signature() == f"brain:80,physics:{PHYSICS_DIM}"


# -- service integration -----------------------------------------------------


class FakeScorer:
    def set_prompt(self, text, distractors=None):
        pass

    def score(self, images):
        return np.linspace(0.1, 0.5, len(images)).astype(np.float32)


def make(physics):
    ts = TournamentService(grid=2)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=FakeScorer(), logger=None)
    svc.configure(steps_per_gen=20, snapshots_per_gen=1, sim_steps_per_frame=20,
                  physics_enabled=physics)
    return svc, ts


def test_brain_only_produces_no_per_tile_physics():
    svc, _ = make(False)
    svc.start("coral")
    assert svc.tile_physics == []
    assert svc.spec is BRAIN_SPEC


def test_physics_search_produces_one_block_per_tile():
    svc, ts = make(True)
    svc.start("coral")
    assert svc.spec is BRAIN_PHYSICS_SPEC
    assert len(svc.tile_physics) == ts.tiles
    for block in svc.tile_physics:
        assert set(block) == {n for n, _g, _lo, _hi in PHYSICS_PARAMS}


def test_tiles_actually_differ_in_physics():
    """If every tile got the same values the search would be pointless."""
    svc, _ = make(True)
    svc.start("coral")
    first = PHYSICS_PARAMS[0][0]
    values = [b[first] for b in svc.tile_physics]
    assert len(set(values)) > 1


def test_brains_are_still_the_right_shape_with_physics_on():
    """The brain block must not absorb the physics genes."""
    svc, ts = make(True)
    svc.start("coral")
    assert all(g.shape == (10, 8) for g in ts.population)
    assert len(ts.population) == ts.tiles


def test_toggling_physics_rebuilds_the_optimizer():
    """cmaes fixes the dimension at construction; carrying it across would
    feed 88-D solutions to an 80-D instance."""
    svc, _ = make(False)
    svc.start("coral")
    assert svc.optimizer.state_dict()["dim"] == BRAIN_SPEC.dim

    svc.physics_enabled = True
    svc._begin_generation()
    assert svc.optimizer.state_dict()["dim"] == BRAIN_PHYSICS_SPEC.dim

    svc.physics_enabled = False
    svc._begin_generation()
    assert svc.optimizer.state_dict()["dim"] == BRAIN_SPEC.dim


def test_reset_clears_the_per_tile_physics():
    svc, _ = make(True)
    svc.start("coral")
    assert svc.tile_physics
    svc.reset()
    assert svc.tile_physics == []


def test_checkpoint_carries_the_physics_flag():
    svc, _ = make(True)
    svc.start("coral")
    st = svc.checkpoint_state()
    assert st["settings"]["physics_enabled"] is True
    assert st["genome_spec_signature"] == BRAIN_PHYSICS_SPEC.signature()

    svc2, _ = make(False)
    svc2.restore(st)
    assert svc2.physics_enabled is True


@pytest.mark.gpu
def test_ssbo_write_matches_the_glsl_struct_size():
    """A raw std430 write: if the packing drifts from the shader struct, every
    tile reads garbage physics and the search silently ranks noise."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    from sim import Sim

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    sim.apply_state(__import__("state", fromlist=["SimState"]).SimState())

    blocks = [decode_physics(np.full(PHYSICS_DIM, float(i) - 8))
              for i in range(16)]
    sim.write_tournament_physics(blocks)   # must not raise or overflow

    # 10 PhysicsSetting * 7 floats + 6 ints + 3 floats
    assert sim.multi_load_buffer.size >= 64 * 316
    ctx.release()


def test_starting_with_physics_already_enabled_does_not_crash():
    """start() builds the optimizer before _begin_generation runs, so the spec
    has to be resolved first or x0 is sized for the brain alone."""
    svc, ts = make(True)
    svc.physics_origin = {n: (lo + hi) / 2 for n, _g, lo, hi in PHYSICS_PARAMS}
    svc.start("coral")                      # must not raise
    assert svc.spec is BRAIN_PHYSICS_SPEC
    assert len(svc.tile_physics) == ts.tiles


def test_search_is_centred_on_the_loaded_preset_not_the_midpoint():
    """z=0 at the midpoint means AXIAL_FORCE=0, LATERAL_FORCE=0, DRAG=0 - no
    propulsion at all, which is where CMA-ES would centre its search."""
    svc, _ = make(True)
    origin = {n: lo + 0.8 * (hi - lo) for n, _g, lo, hi in PHYSICS_PARAMS}
    svc.physics_origin = origin
    svc.start("coral")
    x0 = svc._physics_x0()
    assert x0 is not None and x0.shape == (BRAIN_PHYSICS_SPEC.dim,)
    assert np.allclose(x0[:BRAIN_SPEC.dim], 0.0), "brain still starts at zero"
    recovered = decode_physics(x0[BRAIN_SPEC.dim:])
    for name, _g, _lo, _hi in PHYSICS_PARAMS:
        assert recovered[name] == pytest.approx(origin[name], rel=1e-3)
