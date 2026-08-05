import numpy as np

from services.capture_health import check_capture, sweeping_parameters
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
    s.x_sweeps["SENSOR_DISTANCE"] = 0.8
    s.cohort_sweeps["TRAIL_DIFFUSION"] = -1.0
    got = sweeping_parameters(s)
    assert "SENSOR_DISTANCE" in got
    assert "TRAIL_DIFFUSION" in got


def test_mutation_scale_is_excluded():
    """Tournament mode owns Mutation Scale, so a sweep on it is neutralised
    in sim.py rather than warned about."""
    s = SimState()
    s.x_sweeps["MUTATION_SCALE"] = 1.0
    assert sweeping_parameters(s) == []
