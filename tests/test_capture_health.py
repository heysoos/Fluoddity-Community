import numpy as np

from services.capture_health import (
    check_capture,
    is_viable_tile,
    sweeping_parameters,
)
from state import SimState


def test_healthy_capture_returns_none():
    rng = np.random.default_rng(0)
    crops = rng.integers(20, 200, (4, 224, 224, 3), dtype=np.uint8)
    assert check_capture(crops) is None


def test_all_zero_capture_is_reported():
    """An unbound or never-drawn FBO produces a perfectly flat fitness
    landscape, which is easy to mistake for 'CLIP has no signal'."""
    msg = check_capture(np.zeros((4, 224, 224, 3), dtype=np.uint8))
    assert msg is not None and "black" in msg.lower()


def test_nearly_black_capture_is_reported():
    crops = np.ones((4, 224, 224, 3), dtype=np.uint8)
    msg = check_capture(crops)
    assert msg is not None and "black" in msg.lower()


def test_blown_out_capture_is_reported():
    msg = check_capture(np.full((4, 224, 224, 3), 255, dtype=np.uint8))
    assert msg is not None and "blown" in msg.lower()


def test_empty_capture_is_reported():
    assert check_capture(np.zeros((0, 224, 224, 3), dtype=np.uint8)) is not None


def test_a_clean_sim_state_has_no_sweeping_parameters():
    assert sweeping_parameters(SimState()) == []


def test_a_spatial_sweep_is_named():
    """Tiles partition position space, so any x/y/cohort sweep makes tiles
    run different physics and the comparison is confounded."""
    s = SimState()
    s.parameter_sweeps_enabled = True
    s.x_sweeps["SENSOR_DISTANCE"] = 0.8
    s.cohort_sweeps["TRAIL_DIFFUSION"] = -1.0
    got = sweeping_parameters(s)
    assert "SENSOR_DISTANCE" in got
    assert "TRAIL_DIFFUSION" in got


def test_mutation_scale_is_excluded():
    """Tournament mode owns Mutation Scale, so a sweep on it is neutralised
    in sim.py rather than warned about."""
    s = SimState()
    s.parameter_sweeps_enabled = True
    s.x_sweeps["MUTATION_SCALE"] = 1.0
    assert sweeping_parameters(s) == []


def test_stored_sweeps_do_not_count_while_the_master_toggle_is_off():
    """_assign_physics_setting sends 0.0 for every sweep when
    parameter_sweeps_enabled is False, so the GPU never sees these values and
    the tiles ARE comparable. Presets routinely carry sweep values with the
    toggle off; warning about them cries wolf."""
    s = SimState()
    s.parameter_sweeps_enabled = False
    s.x_sweeps["AXIAL_FORCE"] = 1.0
    s.y_sweeps["GLOBAL_FORCE_MULT"] = -1.0
    s.cohort_sweeps["DRAG"] = 0.5
    assert sweeping_parameters(s) == []


def test_toggling_sweeps_back_on_restores_the_warning():
    s = SimState()
    s.x_sweeps["AXIAL_FORCE"] = 1.0
    assert sweeping_parameters(s) == []
    s.parameter_sweeps_enabled = True
    assert sweeping_parameters(s) == ["AXIAL_FORCE"]


def test_jitter_alone_is_not_a_confound():
    """Jitter is per-particle noise hashed on position, not a systematic
    gradient across the canvas, so it does not bias one tile against another.
    It is also applied regardless of the master toggle."""
    s = SimState()
    s.parameter_sweeps_enabled = True
    s.jitters["AXIAL_FORCE"] = 0.5
    assert sweeping_parameters(s) == []


# --- per-tile viability, for the exploration archive's admission gate -------


def test_viable_tile_accepts_an_ordinary_crop():
    crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    assert is_viable_tile(crop) is True


def test_viable_tile_rejects_pure_black():
    assert is_viable_tile(np.zeros((224, 224, 3), dtype=np.uint8)) is False


def test_viable_tile_rejects_nearly_black():
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    crop[0, 0] = 255                      # max > 0 but the mean is ~0.02
    assert is_viable_tile(crop) is False


def test_viable_tile_rejects_blown_out():
    assert is_viable_tile(np.full((224, 224, 3), 255, dtype=np.uint8)) is False


def test_viable_tile_rejects_empty():
    assert is_viable_tile(np.zeros((0, 224, 3), dtype=np.uint8)) is False


def test_viable_tile_agrees_with_check_capture_on_single_tile_batches():
    """One dead tile must be rejectable on its own; discarding the whole
    generation would throw away 15 useful samples with it."""
    for value in (0, 1, 128, 254, 255):
        crop = np.full((224, 224, 3), value, dtype=np.uint8)
        batch = crop[None, ...]
        assert is_viable_tile(crop) == (check_capture(batch) is None), value
