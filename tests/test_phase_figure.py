"""The figure tool renders, and says what it drew.

A render test, in the manner of the UI smoke tests: it cannot judge whether the
picture is good, but it fails when a schema change silently stops the tool
opening a sweep at all. The sweep is synthesised here rather than loaded, so
these need no GPU and no hours-long run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from services import phase_metrics as pm

figure = pytest.importorskip("tools.phase_figure")


GRID, PROBES, MEASURES = 8, 6, 2


@pytest.fixture
def sweep(tmp_path):
    """A minimal but complete sweep directory."""
    rng = np.random.default_rng(0)
    feats = rng.random((GRID, GRID, len(pm.CELL_NAMES))).astype(np.float32)
    series = np.zeros((GRID, GRID, PROBES, len(pm.PROBE_NAMES)), np.float32)
    series[..., pm.PROBE_NAMES.index("participation_ratio")] = 0.4
    series[..., pm.PROBE_NAMES.index("rho_mean")] = 0.01
    series[..., pm.PROBE_NAMES.index("change")] = 0.5

    np.savez_compressed(
        tmp_path / "features.npz",
        cell_features=feats,
        frame_features=np.zeros((GRID, GRID, MEASURES, len(pm.FRAME_NAMES)),
                                np.float32),
        probe_series=series,
        done=np.ones((GRID, GRID), bool),
        noise_strip=np.zeros((0, len(pm.CELL_NAMES)), np.float32),
        x_values=np.linspace(-1, 1, GRID), y_values=np.linspace(-1, 1, GRID),
        probe_steps=np.arange(1, PROBES + 1) * 50,
        snapshot_steps=np.array([50, 300]),
        measure_steps=np.array([250, 300]),
        raw_index=np.zeros(0, int),
        cell_names=np.array(pm.CELL_NAMES),
        frame_names=np.array(pm.FRAME_NAMES),
        probe_names=np.array(pm.PROBE_NAMES),
        meta=np.array(json.dumps({
            "preset": "x", "preset_name": "testpreset",
            "brain_layout": "mlp-n16-a0",
            "x_param": "AXIAL_FORCE", "y_param": "LATERAL_FORCE",
            "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0],
            "grid": GRID, "steps": 300, "probe_every": 50, "measure_frames": 2,
            "raw_stride": 0, "seed": 1.0, "world_size": 0.1,
            "particle_density": 1.0, "canvas": [323, 323], "entity_count": 100,
            "coverage_threshold": 0.01, "pr_bracket": [0.02, 0.98],
            "rho_floor": 1e-4, "particle_subsample": 16,
            "config": {"physics": {"axial_force": 0.371,
                                   "lateral_force": -0.707},
                       "settings": {"num_cohorts": 6}},
        })))
    return tmp_path


def test_a_single_feature_renders_at_the_declared_size(sweep):
    img, name = figure.make(sweep, "participation_ratio", "viridis", 2, 98,
                            300, "testpreset")
    assert name == "participation_ratio"
    assert img.size == (300 + figure.Figure.PAD_L + figure.Figure.PAD_R,
                        300 + figure.Figure.PAD_T + figure.Figure.PAD_B)


def test_every_declared_feature_can_be_drawn(sweep):
    """A new feature must not land in the diagram with no way to plot it."""
    for name in pm.CELL_NAMES:
        img, _ = figure.make(sweep, name, "viridis", 2, 98, 120, "p")
        assert img.size[0] > 0


def test_every_feature_has_a_stated_meaning():
    """The colourbar's end labels are what make it readable; a feature missing
    from MEANING silently renders as bare numbers."""
    missing = [n for n in pm.CELL_NAMES if n not in figure.MEANING]
    assert not missing, f"no MEANING entry for {missing}"


def test_an_unknown_feature_is_refused_by_name(sweep):
    with pytest.raises(SystemExit, match="no feature"):
        figure.make(sweep, "not_a_feature", "viridis", 2, 98, 120, "p")


def test_the_contact_sheet_is_taller_than_its_panels(sweep):
    """It carries a shared header and footer the panels no longer repeat."""
    picks = ["field_order", "change_rate", "structure", "spec_entropy"]
    sheet = figure.contact(sweep, picks, "viridis", 120, "testpreset")
    panel, _ = figure.make(sweep, "field_order", "viridis", 2, 98, 120, "p",
                           compact=True)
    rows = 2
    assert sheet.size[1] > rows * panel.size[1]


def test_ramps_are_lightness_monotonic():
    """Magnitude must read as brightness, so the picture survives greyscale,
    colour-vision deficiency and a bad projector. A rainbow ramp would not."""
    for name, table in figure.RAMPS.items():
        lum = table @ np.array([0.2126, 0.7152, 0.0722])
        assert np.all(np.diff(lum) > 0), f"{name} is not monotonic in lightness"


def test_ticks_land_on_round_numbers_inside_the_range():
    ticks = figure.nice_ticks(-1.0, 1.0)
    assert 0.0 in ticks
    assert min(ticks) >= -1.0 and max(ticks) <= 1.0
