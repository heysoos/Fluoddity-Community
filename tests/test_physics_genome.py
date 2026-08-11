"""Physics as part of the search space."""
import numpy as np
import pytest

from services.auto_tournament_service import AutoTournamentService
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.physics_genome import (
    PHYSICS_DIM,
    PHYSICS_PARAMS,
    SPAN_FRACTION,
    decode_physics,
    default_origin,
    encode_physics,
    midpoint_z,
)
from services.tournament_service import TournamentService

# A preset that sits OUTSIDE several nominal ranges, as real presets do.
# HungryHungryHippos stores global_force_mult = -0.341 against a nominal
# (0.0, 2.0), which is what broke the original absolute parameterisation.
HHH_ORIGIN = {
    "SENSOR_DISTANCE": -1.0521,
    "SENSOR_ANGLE": -0.7286,
    "SENSOR_GAIN": 0.9078,
    "AXIAL_FORCE": 0.3593,
    "LATERAL_FORCE": -1.2937,
    "STRAFE_POWER": 0.0762,
    "GLOBAL_FORCE_MULT": -0.3407,
    "DRAG": 0.5608,
}


def test_zero_reproduces_the_origin_exactly():
    """The whole point: ticking the toggle must change nothing until the
    optimizer moves. Previously -0.341 decoded to 0.0002, silently switching
    the forces off before generation 1."""
    vals = decode_physics(midpoint_z(), HHH_ORIGIN)
    for name, value in HHH_ORIGIN.items():
        assert vals[name] == pytest.approx(value, abs=1e-9)


def test_an_out_of_range_preset_value_is_not_at_a_saturated_edge():
    """-0.341 used to encode to z = -4.605, where the tanh gradient is 4e-4 and
    CMA-ES could never move the gene again."""
    z = encode_physics(HHH_ORIGIN, HHH_ORIGIN)
    assert np.allclose(z, 0.0, atol=1e-6)
    gradient = 1.0 - np.tanh(z) ** 2
    assert gradient.min() > 0.99, "origin must sit where the map is steepest"


def test_every_parameter_stays_within_one_span_of_the_origin():
    rng = np.random.default_rng(0)
    for _ in range(200):
        vals = decode_physics(rng.normal(0, 3, PHYSICS_DIM), HHH_ORIGIN)
        for name, _glsl, lo, hi in PHYSICS_PARAMS:
            span = SPAN_FRACTION * (hi - lo)
            assert abs(vals[name] - HHH_ORIGIN[name]) <= span + 1e-6


def test_extreme_z_saturates_rather_than_escaping():
    """tanh, not clipping: no repair bias at the boundary."""
    hot = decode_physics(np.full(PHYSICS_DIM, 50.0), HHH_ORIGIN)
    cold = decode_physics(np.full(PHYSICS_DIM, -50.0), HHH_ORIGIN)
    for name, _g, lo, hi in PHYSICS_PARAMS:
        span = SPAN_FRACTION * (hi - lo)
        assert hot[name] == pytest.approx(HHH_ORIGIN[name] + span, abs=1e-3)
        assert cold[name] == pytest.approx(HHH_ORIGIN[name] - span, abs=1e-3)


def test_without_an_origin_zero_is_the_nominal_midpoint():
    """Fallback for callers that have no preset to hand."""
    vals = decode_physics(midpoint_z())
    for name, _g, lo, hi in PHYSICS_PARAMS:
        assert vals[name] == pytest.approx((lo + hi) / 2)
    assert default_origin()["GLOBAL_FORCE_MULT"] == pytest.approx(1.0)


def test_encode_decode_roundtrips():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, PHYSICS_DIM)
    vals = decode_physics(z, HHH_ORIGIN)
    assert np.allclose(encode_physics(vals, HHH_ORIGIN), z, atol=1e-3)


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


# `is BRAIN_SPEC` was the wrong assertion, and it enshrined a real bug: the
# module-level constants are permanently Fourier, so a service that satisfied it
# was one that ignored the selected brain. The space is what matters, and
# same_space_as is what compares it.

def test_brain_only_produces_no_per_tile_physics():
    svc, _ = make(False)
    svc.start("coral")
    assert svc.tile_physics == []
    assert svc.spec.same_space_as(BRAIN_SPEC)


def test_physics_search_produces_one_block_per_tile():
    svc, ts = make(True)
    svc.start("coral")
    assert svc.spec.same_space_as(BRAIN_PHYSICS_SPEC)
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

    from sim import MULTI_LOAD_CONFIG_SIZE
    assert sim.multi_load_buffer.size >= 64 * MULTI_LOAD_CONFIG_SIZE
    # Every block the writer emits must be exactly one struct wide, or tile i
    # reads tile i-1's tail.
    assert len(Sim._TOURNAMENT_PHYSICS_ORDER) * 7 * 4 + 6 * 4 + 3 * 4 \
        == MULTI_LOAD_CONFIG_SIZE
    ctx.release()


def test_starting_with_physics_already_enabled_does_not_crash():
    """start() builds the optimizer before _begin_generation runs, so the spec
    has to be resolved first or x0 is sized for the brain alone."""
    svc, ts = make(True)
    svc.physics_origin = {n: (lo + hi) / 2 for n, _g, lo, hi in PHYSICS_PARAMS}
    svc.start("coral")                      # must not raise
    assert svc.spec.same_space_as(BRAIN_PHYSICS_SPEC)
    assert len(svc.tile_physics) == ts.tiles


def test_the_service_decodes_against_the_loaded_preset():
    """Origin-relative means the search starts AT the preset, so the optimizer
    perturbs it rather than replacing it."""
    ts = TournamentService(grid=4)          # popsize 16, so the mean is stable
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=FakeScorer(), logger=None)
    svc.configure(steps_per_gen=20, snapshots_per_gen=1, sim_steps_per_frame=20,
                  physics_enabled=True)
    svc.physics_origin = dict(HHH_ORIGIN)
    svc.start("coral")

    # z is drawn symmetrically about 0, and z=0 IS the preset, so the
    # population centres on the preset rather than on the nominal midpoint
    # (which for GLOBAL_FORCE_MULT would be +1.0).
    gfm = np.array([b["GLOBAL_FORCE_MULT"] for b in svc.tile_physics])
    assert abs(gfm.mean() - HHH_ORIGIN["GLOBAL_FORCE_MULT"]) < 0.4
    assert abs(gfm.mean() - 1.0) > 0.8, "must not be centred on the midpoint"


def test_enabling_physics_search_does_not_move_the_preset_on_its_own():
    """Regression: the absolute parameterisation decoded HungryHungryHippos'
    global_force_mult of -0.341 to 0.0002, switching the forces off before a
    single generation had run."""
    svc, _ = make(True)
    svc.physics_origin = dict(HHH_ORIGIN)
    at_origin = decode_physics(np.zeros(PHYSICS_DIM), svc.physics_origin)
    assert at_origin["GLOBAL_FORCE_MULT"] == pytest.approx(-0.3407, abs=1e-6)


def test_a_stale_smaller_population_must_not_enable_per_tile_physics():
    """get_particle_config_index() returns the home tile, so a tile with no
    block written reads a zeroed config - zero force, zero drag, zero sensor
    gain - and renders black. Growing the grid between generations leaves
    tile_physics holding the old, smaller population."""
    ts = TournamentService(grid=2)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=FakeScorer(), logger=None)
    svc.configure(steps_per_gen=20, snapshots_per_gen=1, sim_steps_per_frame=20,
                  physics_enabled=True)
    svc.start("coral")
    assert len(svc.tile_physics) == 4

    ts.set_grid(4)                     # 16 tiles, blocks still sized for 4
    assert len(svc.tile_physics) < ts.tiles, (
        "stale coverage is the condition the orchestrator must refuse")

    svc._begin_generation()            # after a rebuild it covers them again
    assert len(svc.tile_physics) == ts.tiles
