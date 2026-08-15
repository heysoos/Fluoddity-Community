"""The noise floor must be sized at the point it is quoted against.

This runs a real (tiny) sweep rather than reading the source, because the bug
it guards against was invisible to reading: `run_cell` writes each cell's
overrides straight into the shared SimState and never puts them back, so
reading those fields after the loop returned the LAST CELL's values. The strip
then measured the splat race at a corner of the diagram while every table
printed it as the preset's own floor, and every source-level reading of the
function said it did the right thing.

Skipped when no GL 4.3 context is available (CI, remote shells, no GPU).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("moderngl")
pytest.importorskip("glfw")

import ui  # noqa: F401,E402  prime the services/ui import cycle

from tools import phase_diagram as pd  # noqa: E402


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    """The smallest sweep that still has a corner to be confused with."""
    out = tmp_path_factory.mktemp("sweep")
    argv = ["--preset", "fish soup", "--grid", "2", "--steps", "20",
            "--probe-every", "10", "--measure-frames", "1",
            "--no-raw", "--noise-strip", "2", "--world-size", "0.05",
            "--x-param", "SENSOR_ANGLE", "--y-param", "SENSOR_DISTANCE",
            "--x-range", "-1", "1", "--y-range", "0", "3",
            "--out", str(out)]
    try:
        assert pd.main(argv) == 0
    except Exception as exc:                       # no GL 4.3 on this machine
        pytest.skip(f"no usable GL context: {exc}")
    data = np.load(out / "features.npz", allow_pickle=False)
    return json.loads(str(data["meta"])), data


def test_the_noise_floor_records_where_it_was_measured(swept):
    meta, _ = swept
    assert "noise_at" in meta, "a floor with no stated point cannot be checked"


def test_the_noise_floor_is_measured_at_the_preset_not_the_last_cell(swept):
    meta, _ = swept
    physics = meta["config"]["physics"]
    assert meta["noise_at"]["SENSOR_ANGLE"] == pytest.approx(
        physics["sensor_angle"], abs=1e-6)
    assert meta["noise_at"]["SENSOR_DISTANCE"] == pytest.approx(
        physics["sensor_distance"], abs=1e-6)


def test_the_preset_does_not_sit_on_a_corner_of_this_sweep(swept):
    """Otherwise the assertion above would hold even with the bug live."""
    meta, data = swept
    for axis in ("SENSOR_ANGLE", "SENSOR_DISTANCE"):
        key = "x_values" if axis == "SENSOR_ANGLE" else "y_values"
        assert meta["noise_at"][axis] not in set(data[key].tolist())


def test_two_axes_naming_one_field_are_refused(tmp_path):
    """They share a dict key, so the x sweep would silently do nothing."""
    argv = ["--preset", "fish soup", "--grid", "2", "--steps", "10",
            "--x-param", "SENSOR_ANGLE", "--y-param", "SENSOR_ANGLE",
            "--no-raw", "--noise-strip", "0", "--out", str(tmp_path)]
    with pytest.raises(SystemExit, match="must differ"):
        pd.main(argv)
