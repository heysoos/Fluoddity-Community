"""Phase-diagram features against fields whose answer is known by construction.

Every case here is a field built to score a specific value, so a feature that
silently stops measuring what its name says fails rather than merely changing.
No GPU: the module under test is pure numpy, which is the whole reason it is
separate from tools/phase_diagram.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from services import phase_metrics as pm

H = W = 64


def _blob(sigma=1.5):
    y, x = np.mgrid[0:H, 0:W]
    return np.exp(-(((x - W / 2) ** 2 + (y - H / 2) ** 2) / (2 * sigma ** 2)))


def _noise(seed=0):
    return np.abs(np.random.default_rng(seed).standard_normal((H, W)))


def _stripes(wavelength=8.0, phase=0.0):
    x = np.arange(W)
    return np.tile(1.0 + np.cos(2 * np.pi * (x + phase) / wavelength), (H, 1))


def _field_from(rho, angle=0.3):
    return np.stack([rho * np.cos(angle), rho * np.sin(angle)], axis=-1)


# --- participation ratio: the die/explode axis ------------------------------

def test_a_blob_has_a_participation_ratio_near_zero():
    assert pm.participation_ratio(_blob()) < 0.01


def test_a_uniform_field_has_a_participation_ratio_of_one():
    assert pm.participation_ratio(np.ones((H, W))) == pytest.approx(1.0)


def test_participation_ratio_is_the_occupied_fraction():
    a = np.zeros(1000)
    a[:250] = 1.0
    assert pm.participation_ratio(a) == pytest.approx(0.25)


def test_participation_ratio_ignores_overall_scale():
    a = _blob()
    assert pm.participation_ratio(a * 1000.0) == pytest.approx(
        pm.participation_ratio(a))


def test_an_empty_field_does_not_divide_by_zero():
    assert pm.participation_ratio(np.zeros((H, W))) == 0.0


# --- structure: coherent vs white noise ------------------------------------

def test_noise_has_no_structure():
    assert pm.structure(_noise()) < 0.1


def test_a_lattice_has_full_structure():
    assert pm.structure(_stripes()) > 0.95


def test_a_fine_lattice_is_not_mistaken_for_noise():
    """Lag 1 alone scores a 3px and a 4px lattice exactly what noise scores.

    This is the false negative the multi-lag choice exists to prevent, and it
    lands on precisely the fine, complex patterns the feature must not punish.
    """
    for wavelength in (3.0, 4.0):
        assert pm.structure(_stripes(wavelength)) > 0.9


# --- spectrum: characteristic length scale ---------------------------------

def test_the_spectral_peak_finds_the_constructed_wavelength():
    for wavelength in (4.0, 8.0, 16.0):
        found, _ = pm.spectral_features(_stripes(wavelength))
        assert found == pytest.approx(wavelength, rel=0.02)


def test_the_peak_resolves_a_wavelength_between_two_bins():
    """Wavelength is width/k, so the bins are crowded at large scales and a
    bin-only argmax puts a family of distinguishable patterns on one terrace."""
    wavelength = 10.0                       # k = 6.4 in a 64-wide image
    found, _ = pm.spectral_features(_stripes(wavelength))
    nearest_bin = W / round(W / wavelength)  # 10.667, what argmax alone gives
    assert abs(found - wavelength) < abs(nearest_bin - wavelength)
    assert found == pytest.approx(wavelength, rel=0.03)


def test_noise_is_broadband_and_a_lattice_is_not():
    _, noisy = pm.spectral_features(_noise())
    _, sharp = pm.spectral_features(_stripes())
    assert noisy > 0.9
    assert sharp < 0.3


def test_a_field_with_no_spectrum_is_undefined_rather_than_zero():
    """Zero is a legal-looking wavelength - the finest possible scale - so a
    sentinel of 0 would read as the opposite of what it means."""
    for dead in (np.ones((H, W)), np.zeros((H, W))):
        assert np.isnan(pm.spectral_features(dead)).all()


# --- order parameters ------------------------------------------------------

def test_particles_moving_together_are_fully_ordered():
    vel = np.tile([0.3, -0.4], (500, 1))
    assert pm.polar_order(vel) == pytest.approx(1.0)


def test_isotropic_particles_are_disordered():
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    vel = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    assert pm.polar_order(vel) < 0.01


def test_polar_order_ignores_speed_and_counts_only_direction():
    theta = np.full(500, 0.7)
    speed = np.linspace(0.01, 10.0, 500)
    vel = np.stack([speed * np.cos(theta), speed * np.sin(theta)], axis=-1)
    assert pm.polar_order(vel) == pytest.approx(1.0)


def test_particles_at_rest_carry_no_direction():
    assert pm.polar_order(np.zeros((100, 2))) == 0.0


def test_field_order_matches_polar_order_on_an_aligned_field():
    assert pm.field_order(_field_from(_blob())) == pytest.approx(1.0)


def test_field_order_cancels_on_an_antisymmetric_field():
    half = np.ones((H // 2, W, 2))
    field = np.concatenate([half, -half], axis=0)
    assert pm.field_order(field) < 1e-9


# --- clustering ------------------------------------------------------------

def test_uniform_particles_are_unclustered():
    rng = np.random.default_rng(1)
    pos = rng.uniform(-1.0, 1.0, size=(200_000, 2))
    assert pm.particle_pr(pos, bins=32) > 0.9


def test_a_tight_clump_is_clustered():
    rng = np.random.default_rng(2)
    pos = rng.normal(0.0, 0.01, size=(20_000, 2))
    assert pm.particle_pr(pos, bins=32) < 0.05


# --- change: the motion term ----------------------------------------------

def test_identical_frames_show_no_change():
    rho = _blob()
    assert pm.change(rho, rho) == 0.0


def test_the_first_probe_has_nothing_to_compare_against():
    assert pm.change(_blob(), None) == 0.0


def test_a_translating_pattern_changes_while_its_structure_does_not():
    """The case that separates motion from disorder.

    A rigid shift is invisible to every instantaneous feature - same histogram,
    same spectrum, same autocorrelation - so if `change` did not see it, a
    travelling structure would be indistinguishable from a frozen one.
    """
    before, after = _stripes(phase=0.0), _stripes(phase=4.0)
    assert pm.structure(after) == pytest.approx(pm.structure(before), abs=1e-6)
    assert pm.participation_ratio(after) == pytest.approx(
        pm.participation_ratio(before), rel=1e-6)
    assert pm.change(after, before) > 0.3


# --- alive_steps: the Lenia transcription ----------------------------------

def test_a_run_that_stays_in_the_bracket_survives_the_whole_budget():
    steps = np.arange(0, 500, 50)
    pr = np.full(len(steps), 0.4)
    assert pm.alive_steps(steps, pr, 0.05, 0.95, 500) == 500.0


def test_the_first_departure_is_what_is_reported():
    steps = np.array([0, 50, 100, 150, 200])
    pr = np.array([0.4, 0.4, 0.99, 0.4, 0.001])
    assert pm.alive_steps(steps, pr, 0.05, 0.95, 200) == 100.0


def test_collapse_and_saturation_are_both_deaths():
    steps = np.array([0, 50, 100])
    assert pm.alive_steps(steps, np.array([0.4, 0.001, 0.4]), .05, .95, 100) == 50.0
    assert pm.alive_steps(steps, np.array([0.4, 0.999, 0.4]), .05, .95, 100) == 50.0


# --- assembly --------------------------------------------------------------

def test_a_cell_row_matches_its_declared_names():
    rng = np.random.default_rng(3)
    pos = rng.uniform(-1, 1, size=(1000, 2))
    vel = rng.normal(0, 0.01, size=(1000, 2))
    field = _field_from(_blob())

    frames = np.stack([pm.frame_row(pos, vel, field, 0.01) for _ in range(3)])
    steps = np.arange(0, 500, 50)
    series = np.stack([pm.probe_row(pm.rho_of(field), None, 0.01)
                       for _ in steps])
    row = pm.cell_row(frames, steps, series, 0.05, 0.95, 500)

    assert frames.shape[1] == len(pm.FRAME_NAMES)
    assert series.shape[1] == len(pm.PROBE_NAMES)
    assert row.shape == (len(pm.CELL_NAMES),)
    assert np.isfinite(row).all()


def test_the_cell_row_averages_its_frames_rather_than_taking_the_last():
    """Past the transient a cell's features do not converge, they fluctuate, so
    one late frame samples the fluctuation rather than the state."""
    rng = np.random.default_rng(4)
    pos = rng.uniform(-1, 1, size=(500, 2))
    slow = pm.frame_row(pos, np.full((500, 2), 1.0), _field_from(_blob()), .01)
    fast = pm.frame_row(pos, np.full((500, 2), 3.0), _field_from(_blob()), .01)

    steps = np.array([0, 50])
    series = np.zeros((2, len(pm.PROBE_NAMES)))
    row = pm.cell_row(np.stack([slow, fast]), steps, series, 0.05, 0.95, 50)
    # |(1,1)| and |(3,3)| are sqrt2 and 3*sqrt2; the mean is 2*sqrt2, and the
    # last frame alone would be 3*sqrt2.
    assert row[pm.CELL_NAMES.index("speed_p50")] == pytest.approx(
        2.0 * np.sqrt(2.0), rel=1e-6)


def test_a_single_frame_still_works():
    """The pilot and any M=1 run hand cell_row one row, not a stack."""
    rng = np.random.default_rng(5)
    pos = rng.uniform(-1, 1, size=(200, 2))
    one = pm.frame_row(pos, np.full((200, 2), 1.0), _field_from(_blob()), .01)
    row = pm.cell_row(one[None, :], np.array([50]),
                      np.zeros((1, len(pm.PROBE_NAMES))), 0.05, 0.95, 50)
    assert row[pm.CELL_NAMES.index("speed_p50")] == pytest.approx(np.sqrt(2.0))


def test_change_rate_reads_the_tail_and_not_the_opening_transient():
    """Every cell has a large transient; averaging it in would hide the
    difference between a settled pattern and a churning one."""
    steps = np.arange(0, 800, 50)
    series = np.zeros((len(steps), len(pm.PROBE_NAMES)))
    ci = pm.PROBE_NAMES.index("change")
    series[:, ci] = 0.0
    series[:4, ci] = 10.0                       # the transient
    series[:, pm.PROBE_NAMES.index("participation_ratio")] = 0.4

    frames = np.zeros((1, len(pm.FRAME_NAMES)))
    row = pm.cell_row(frames, steps, series, 0.05, 0.95, 800)
    assert row[pm.CELL_NAMES.index("change_rate")] == pytest.approx(0.0)
