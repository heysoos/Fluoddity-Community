"""A config downloaded from fluoddity.com loads as the creature it was saved as.

The web app writes version 8: a `world` block plus a `configs` list, with the
parameters regrouped and renamed, its own enum numbering for the boundary and
spawn modes, an absolute sensor jitter where the desktop's is relative, and a
colour term that is the desktop's LEGACY colouring.
"""
import json

import numpy as np
import pytest

from services.config_saver import ConfigSaver, PhysicsConfig
from services.web_config import is_web_config, translate_web_config


def web_config(**over):
    rule = [float(i) / 10 for i in range(80)]
    cfg = {
        "rule": rule,
        "sensor": {"gain": 3.5, "angle": -0.18, "distance": 2.0,
                   "mutation_scale": 0.2},
        "force": {"global_mult": 0.29, "drag": 0.5, "strafe": 0.22,
                  "axial": 0.12},
        "misc": {"lateral": 0.51, "hazard_rate": 0.01, "cohorts": 16,
                 "mutation_seed": 0.543},
        "force2": {"gravity_force": 0.0, "gravity_strafe": 0.0,
                   "initial_conditions": 0, "cohort_fences": False},
        "misc2": {"color_sensitivity": -0.42, "color_by_cohort": False,
                  "sensor_angle_jitter": 0.0, "sensor_distance_jitter": 0.25},
        "misc3": {"radial_gravity": False},
    }
    for key, val in over.items():
        group, name = key.split(".")
        cfg[group][name] = val
    return {
        "version": 8,
        "world": {"trail_persistence": 0.938, "trail_diffusion": 1,
                  "boundary_conditions": 1},
        "configs": [cfg],
    }


def test_a_native_config_is_not_mistaken_for_a_web_one():
    assert not is_web_config(PhysicsConfig().to_dict())
    assert is_web_config(web_config())


def test_every_parameter_lands_on_its_desktop_name():
    d = translate_web_config(web_config())[0]
    p = d["physics"]
    assert p["sensor_gain"] == 3.5
    assert p["sensor_angle"] == -0.18
    assert p["sensor_distance"] == 2.0
    assert p["mutation_scale"] == 0.2
    assert p["global_force_mult"] == 0.29
    assert p["drag"] == 0.5
    assert p["strafe_power"] == 0.22
    assert p["axial_force"] == 0.12
    assert p["lateral_force"] == 0.51
    assert p["hazard_rate"] == 0.01
    assert p["trail_persistence"] == 0.938
    assert p["trail_diffusion"] == 1
    assert d["settings"]["num_cohorts"] == 16
    assert d["settings"]["rule_seed"] == 0.543
    assert d["appearance"]["color_by_cohort"] is False
    assert d["rule"] == web_config()["configs"][0]["rule"]
    assert d["brain_layout"] == "fourier-n10"


@pytest.mark.parametrize("web, desktop", [(0, 0), (1, 2), (2, 1)])
def test_the_boundary_enum_is_renumbered(web, desktop):
    src = web_config()
    src["world"]["boundary_conditions"] = web
    assert translate_web_config(src)[0]["settings"]["boundary_conditions"] == desktop


@pytest.mark.parametrize("web, desktop", [(0, 0), (1, 1), (3, 2), (2, 0)])
def test_the_spawn_enum_is_renumbered_and_centre_falls_back_to_grid(web, desktop):
    d = translate_web_config(web_config(**{"force2.initial_conditions": web}))[0]
    assert d["settings"]["initial_conditions"] == desktop


def test_the_web_colour_is_the_legacy_colouring():
    d = translate_web_config(web_config())[0]
    assert d["appearance"]["hue_sensitivity"] == -0.42
    assert d["appearance"]["legacy_color"] is True


def test_sensor_jitter_is_rescaled_from_absolute_to_relative():
    # Web: distance += 5 * j * U(-1, 1). Desktop: distance += J * distance * U.
    d = translate_web_config(web_config())[0]
    assert d["jitters"]["SENSOR_DISTANCE"] == pytest.approx(5.0 * 0.25 / 2.0)
    d = translate_web_config(web_config(**{"misc2.sensor_angle_jitter": 0.09}))[0]
    assert d["jitters"]["SENSOR_ANGLE"] == pytest.approx(0.09 / 0.18)


def test_a_slider_range_grows_to_hold_an_out_of_range_value():
    d = translate_web_config(web_config(**{"sensor.distance": 4.5}))[0]
    lo, hi, _, _ = d["slider_ranges"]["Sensor Distance"]
    assert lo <= 4.5 <= hi


def test_features_the_desktop_lacks_are_named_in_the_notes():
    d = translate_web_config(web_config(
        **{"force2.gravity_force": 0.3, "force2.cohort_fences": True,
           "misc3.radial_gravity": True}))[0]
    assert "gravity" in d["notes"].lower()
    assert "fence" in d["notes"].lower()
    # Below the web's dead zone the pull is exactly zero, so it is not a loss.
    d = translate_web_config(web_config(**{"force2.gravity_force": 0.04}))[0]
    assert "gravity" not in d["notes"].lower()


def test_a_multi_config_file_yields_one_desktop_config_each():
    src = web_config()
    src["configs"].append(dict(src["configs"][0], sensor={
        "gain": 1.0, "angle": 0.1, "distance": 1.0, "mutation_scale": 0.0}))
    out = translate_web_config(src)
    assert len(out) == 2
    assert out[1]["physics"]["sensor_gain"] == 1.0
    assert out[1]["settings"]["boundary_conditions"] == 2


def test_the_loader_reads_a_web_file_straight_off_disk(tmp_path):
    path = tmp_path / "Weave.json"
    path.write_text(json.dumps(web_config()))
    cfg = ConfigSaver().load_from_file(path)
    assert cfg is not None
    assert cfg.boundary_conditions == 2
    assert cfg.legacy_color is True
    assert cfg.num_cohorts == 16
    # The 80-float rule is a legacy Fourier and is widened on the way in.
    assert cfg.brain_layout == "fourier-n10+c2"
    assert cfg.rule.shape == (100,)
    np.testing.assert_array_equal(cfg.rule.reshape(10, 10)[:, :8].reshape(-1),
                                  np.array(web_config()["configs"][0]["rule"],
                                           dtype=np.float32))
