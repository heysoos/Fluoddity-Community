"""Configs saved by the web app (fluoddity.com), read as desktop configs.

The web app writes version 8: a `world` block and a `configs` list, each entry
grouping the parameters under `sensor`, `force`, `misc`, `force2`, `misc2` and
`misc3`. The physics is the desktop's, so the translation is a renaming plus
four things that genuinely differ:

- the boundary and spawn enums are numbered differently;
- `mutation_seed` is `rule_seed`;
- sensor jitter is an absolute wobble, where the desktop's is relative;
- the colour term is the desktop's LEGACY colouring.

Gravity and cohort fences have no desktop counterpart and are reported in the
config's notes.
"""
from __future__ import annotations

from ui.physics_params import DEFAULT_SLIDER_RANGES

WEB_VERSION = 8

# Web: 0 Bounce, 1 Wrap, 2 Reset.  Desktop: 0 Bounce, 1 Reset, 2 Wrap.
BOUNDARY_MAP = {0: 0, 1: 2, 2: 1}
# Web: 0 Grid, 1 Random, 2 Centre, 3 Ring.  Desktop: 0 Grid, 1 Random,
# 2 Ring, 3 Centre.
SPAWN_MAP = {0: 0, 1: 1, 2: 3, 3: 2}
# The web adds SENSOR_DISTANCE_SPAN * jitter to the distance; the desktop adds
# jitter * value.
SENSOR_DISTANCE_SPAN = 5.0
# |gravity| at or below this is exactly zero on the web.
GRAVITY_KNEE = 0.05


def is_web_config(data: dict) -> bool:
    """Is this JSON dict in the web app's format?"""
    return (isinstance(data, dict) and data.get("version") == WEB_VERSION
            and isinstance(data.get("configs"), list))


def _relative_jitter(absolute: float, value: float) -> float:
    if absolute == 0.0:
        return 0.0
    if value == 0.0:
        return 0.0
    return abs(absolute / value)


def _ranges_holding(physics: dict) -> dict[str, list[float]]:
    """Default slider ranges, each widened to hold its value."""
    by_label = {
        "Sensor Gain": "sensor_gain", "Sensor Angle": "sensor_angle",
        "Sensor Distance": "sensor_distance", "Mutation Scale": "mutation_scale",
        "Global Force Mult": "global_force_mult", "Drag": "drag",
        "Axial Force": "axial_force", "Lateral Force": "lateral_force",
        "Strafe Power": "strafe_power", "Trail Persistence": "trail_persistence",
        "Trail Diffusion": "trail_diffusion", "Hazard Rate": "hazard_rate",
    }
    out = {}
    for label, (lo, hi) in DEFAULT_SLIDER_RANGES.items():
        cur_lo, cur_hi = lo, hi
        field = by_label.get(label)
        if field is not None:
            v = float(physics[field])
            cur_lo, cur_hi = min(lo, v), max(hi, v)
        out[label] = [cur_lo, cur_hi, lo, hi]
    return out


def _translate_one(world: dict, cfg: dict, notes: str) -> dict:
    sensor = cfg.get("sensor", {})
    force = cfg.get("force", {})
    misc = cfg.get("misc", {})
    force2 = cfg.get("force2", {})
    misc2 = cfg.get("misc2", cfg.get("appearance", {}))
    misc3 = cfg.get("misc3", {})

    physics = {
        "axial_force": force.get("axial", 0.371),
        "lateral_force": misc.get("lateral", -0.707),
        "sensor_gain": sensor.get("gain", 0.116),
        "mutation_scale": sensor.get("mutation_scale", 0.0),
        "drag": force.get("drag", 0.504),
        "strafe_power": force.get("strafe", 0.224),
        "sensor_angle": sensor.get("angle", 0.45),
        "global_force_mult": force.get("global_mult", 1.0),
        "sensor_distance": sensor.get("distance", 1.0),
        "v_max": 0.1,
        "trail_persistence": world.get("trail_persistence", 0.938),
        "trail_diffusion": world.get("trail_diffusion", 1.0),
        "hazard_rate": misc.get("hazard_rate", 0.0),
    }

    jitters = {
        "SENSOR_ANGLE": _relative_jitter(
            misc2.get("sensor_angle_jitter", 0.0), physics["sensor_angle"]),
        "SENSOR_DISTANCE": _relative_jitter(
            SENSOR_DISTANCE_SPAN * misc2.get("sensor_distance_jitter", 0.0),
            physics["sensor_distance"]),
    }

    lost = []
    if (abs(force2.get("gravity_force", 0.0)) > GRAVITY_KNEE
            or abs(force2.get("gravity_strafe", 0.0)) > GRAVITY_KNEE):
        lost.append("gravity (force %.3f, strafe %.3f%s)" % (
            force2.get("gravity_force", 0.0), force2.get("gravity_strafe", 0.0),
            ", radial" if misc3.get("radial_gravity") else ""))
    if force2.get("cohort_fences"):
        lost.append("cohort fences")
    ic = int(force2.get("initial_conditions", 2))
    if lost:
        line = "Web-only features not reproduced: " + "; ".join(lost) + "."
        notes = (notes + "\n" + line) if notes else line

    return {
        "version": 7,
        "physics": physics,
        "slider_ranges": _ranges_holding(physics),
        "jitters": jitters,
        "parameter_sweeps_enabled": False,
        "settings": {
            "boundary_conditions": BOUNDARY_MAP.get(
                int(world.get("boundary_conditions", 1)), 2),
            "initial_conditions": SPAWN_MAP.get(ic, 0),
            "num_cohorts": int(misc.get("cohorts", 64)),
            "rule_seed": float(misc.get("mutation_seed",
                                        misc.get("rule_seed", 0.42))),
        },
        "appearance": {
            "hue_sensitivity": misc2.get("color_sensitivity", 0.5),
            "color_by_cohort": bool(misc2.get("color_by_cohort", False)),
            "legacy_color": True,
        },
        "rule": list(cfg.get("rule", [])),
        "brain_layout": "fourier-n10",
        "notes": notes,
    }


def translate_web_config(data: dict) -> list[dict]:
    """The version-7 dicts a web file holds, one per entry in its `configs`."""
    world = data.get("world", {})
    notes = data.get("notes", "") if isinstance(data.get("notes"), str) else ""
    return [_translate_one(world, cfg, notes) for cfg in data["configs"]]
