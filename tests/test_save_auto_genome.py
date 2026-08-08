"""Saving a genome must work while physics is part of the search.

Reported 2026-08-08: pressing Save during an Auto run with physics search on
killed the app mid-frame with

    ValueError: cannot reshape array of size 88 into shape (20,4)

88 is 80 brain genes plus 8 physics genes. export_genome decodes a BRAIN, so
handing it the whole search vector cannot work - and because command handlers
run inside orchestrate_frame, the exception took the whole app down rather than
printing a warning.

The dimension is only half of it. A genome evolved with physics search is a
brain AND a set of physics values; writing the brain against whatever the
sliders happen to say would save a file that does not reproduce what was on
screen when the user pressed the button.
"""
import numpy as np
import pytest

from services.config_saver import ConfigSaver
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC, DIM, decode
from services.physics_genome import PHYSICS_DIM, PHYSICS_PARAMS, decode_physics
from state import SimState


class _FakeOptimizer:
    def __init__(self, z):
        self._z = np.asarray(z, dtype=np.float32)

    def best(self):
        return self._z, 0.5


class _FakeService:
    def __init__(self, z, physics=False, origin=None):
        z = np.asarray(z, dtype=np.float32)
        self.spec = BRAIN_PHYSICS_SPEC if physics else BRAIN_SPEC
        self.optimizer = _FakeOptimizer(z)
        self.current_z = z[None, :]
        self.physics_enabled = physics
        self.physics_origin = dict(origin or {})
        self.generation = 12
        self.prompt = "coral reef"
        self.algorithm = "CMA-ES"
        self.tile_mutation_enabled = False
        self.tile_mutation_strength = 0.1
        self.variants_per_tile = 4


class _FakeSim:
    def get_canvas_dimensions(self):
        return (1024, 1024)


def save(tmp_path, svc, sim_state, tile=None):
    """CommandHandler._save_auto_genome with only the attributes it touches."""
    from command_handler import CommandHandler

    handler = object.__new__(CommandHandler)
    handler.sim = _FakeSim()
    handler.user_configs_dir = tmp_path

    ui_state = type("UI", (), {"sim": sim_state})()
    CommandHandler._save_auto_genome(handler, svc, ui_state, tile)
    written = sorted(tmp_path.glob("*.json"))
    assert len(written) == 1, f"expected one file, got {written}"
    return ConfigSaver().load_from_file(written[0])


def brain_physics_z(seed=0, physics_scale=0.8):
    rng = np.random.default_rng(seed)
    brain = rng.normal(0, 0.5, DIM).astype(np.float32)
    phys = (physics_scale * rng.normal(0, 1.0, PHYSICS_DIM)).astype(np.float32)
    return np.concatenate([brain, phys]).astype(np.float32)


# ---- the crash ----------------------------------------------------------

def test_saving_during_a_physics_search_does_not_crash(tmp_path):
    z = brain_physics_z()
    cfg = save(tmp_path, _FakeService(z, physics=True), SimState())
    assert cfg is not None


def test_the_saved_brain_is_the_brain_half_of_the_genome(tmp_path):
    z = brain_physics_z()
    cfg = save(tmp_path, _FakeService(z, physics=True), SimState())
    assert np.allclose(cfg.rule, decode(z[:DIM]), atol=1e-5)


def test_the_saved_config_carries_the_evolved_physics(tmp_path):
    """Not the sliders'. The file has to reproduce what was on screen."""
    z = brain_physics_z()
    sim_state = SimState()
    svc = _FakeService(z, physics=True)
    expected = decode_physics(z[DIM:], svc.physics_origin)

    cfg = save(tmp_path, svc, sim_state)

    for field, cfg_field, _lo, _hi in PHYSICS_PARAMS:
        assert getattr(cfg, cfg_field) == pytest.approx(expected[field], abs=1e-5), field
    # and it really is different from what the sliders said
    assert cfg.sensor_distance != pytest.approx(sim_state.SENSOR_DISTANCE, abs=1e-6)


def test_saving_does_not_move_the_users_sliders(tmp_path):
    """Export is a read. Writing the evolved physics onto the live SimState
    would change the running simulation as a side effect of pressing Save."""
    z = brain_physics_z()
    sim_state = SimState()
    before = {f: getattr(sim_state, f) for f, _g, _lo, _hi in PHYSICS_PARAMS}

    save(tmp_path, _FakeService(z, physics=True), sim_state)

    assert {f: getattr(sim_state, f) for f, _g, _lo, _hi in PHYSICS_PARAMS} == before


def test_the_physics_origin_is_respected(tmp_path):
    """z is relative to the loaded preset, so decoding against the wrong origin
    silently saves a different creature."""
    z = np.concatenate([np.zeros(DIM), np.zeros(PHYSICS_DIM)]).astype(np.float32)
    origin = {f: 0.25 for f, _g, _lo, _hi in PHYSICS_PARAMS}
    svc = _FakeService(z, physics=True, origin=origin)

    cfg = save(tmp_path, svc, SimState())

    # z = 0 reproduces the origin exactly
    assert cfg.sensor_distance == pytest.approx(0.25, abs=1e-5)


def test_a_tile_genome_saves_too(tmp_path):
    z = brain_physics_z(seed=3)
    cfg = save(tmp_path, _FakeService(z, physics=True), SimState(), tile=0)
    assert np.allclose(cfg.rule, decode(z[:DIM]), atol=1e-5)


# ---- brain-only searches are unaffected ---------------------------------

def test_a_brain_only_genome_still_saves(tmp_path):
    z = np.random.default_rng(1).normal(0, 0.5, DIM).astype(np.float32)
    cfg = save(tmp_path, _FakeService(z, physics=False), SimState())
    assert np.allclose(cfg.rule, decode(z), atol=1e-5)


def test_a_brain_only_save_uses_the_sliders_physics(tmp_path):
    """Nothing evolved the physics, so the config takes it from the UI - the
    behaviour this fix must not change."""
    z = np.zeros(DIM, dtype=np.float32)
    sim_state = SimState()
    sim_state.SENSOR_DISTANCE = 1.75

    cfg = save(tmp_path, _FakeService(z, physics=False), sim_state)

    assert cfg.sensor_distance == pytest.approx(1.75, abs=1e-6)


def test_nothing_is_written_when_there_is_no_genome_yet(tmp_path):
    from command_handler import CommandHandler

    svc = _FakeService(np.zeros(DIM, dtype=np.float32))
    svc.optimizer = None
    handler = object.__new__(CommandHandler)
    handler.sim = _FakeSim()
    handler.user_configs_dir = tmp_path

    CommandHandler._save_auto_genome(
        handler, svc, type("UI", (), {"sim": SimState()})(), None)

    assert list(tmp_path.glob("*.json")) == []
