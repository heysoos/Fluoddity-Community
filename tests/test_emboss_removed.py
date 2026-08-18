"""Emboss is gone, and every config written while it existed must still load.

The effect read the canvas `.z`, a coverage channel the RG canvas no longer
has; it was re-solved once against flow magnitude and then cut. What survives
it is the DATA: 131 shipped presets carry `emboss_mode`/`emboss_intensity`/
`emboss_smoothness` in their appearance block, two of them with it switched on,
and a legacy binary `.frs` carries the same three values in the middle of a
fixed-offset struct.
"""
import json
import struct
from pathlib import Path

import pytest

from services.config_saver import ConfigSaver, PhysicsConfig
from state.sim_state import SimState

ROOT = Path(__file__).resolve().parents[1]


def test_the_fields_are_gone_from_both_state_and_config():
    for name in ("emboss_mode", "emboss_intensity", "emboss_smoothness"):
        assert name not in SimState.__dataclass_fields__, name
        assert name not in PhysicsConfig.__dataclass_fields__, name


def test_a_json_config_carrying_emboss_still_loads():
    """The keys are ignored, not rejected: from_dict reads what it knows."""
    raw = json.loads((ROOT / "physics_configs" / "Core" / "Adrift.json")
                     .read_text(encoding="utf-8"))
    assert raw["appearance"]["emboss_mode"] != 0, "pick a preset that HAD it on"
    cfg = PhysicsConfig.from_dict(raw)
    state = SimState()
    ConfigSaver().apply_config(cfg, state)
    assert not hasattr(state, "emboss_mode")


@pytest.mark.parametrize("rel", [p.relative_to(ROOT / "physics_configs").as_posix()
                                 for p in sorted((ROOT / "physics_configs").rglob("*.json"))])
def test_every_shipped_preset_still_loads(rel):
    raw = json.loads((ROOT / "physics_configs" / rel).read_text(encoding="utf-8"))
    ConfigSaver().apply_config(PhysicsConfig.from_dict(raw), SimState())


def test_a_saved_config_no_longer_writes_the_keys():
    out = json.loads(PhysicsConfig().to_json())
    assert not [k for k in out.get("appearance", {}) if "emboss" in k]


def test_the_legacy_binary_offsets_did_not_move():
    """The three values are still READ and discarded. Drop them from the struct
    and every field after them decodes from the wrong bytes."""
    saver = ConfigSaver()
    body = bytes(360)
    body += struct.pack('??', True, False)                       # 360:362
    body += struct.pack('3i', 1, 2, 32)                          # 362:374
    body += struct.pack('f', 7.5)                                # 374:378
    # 378:404 - the appearance block, emboss in the middle of it
    body += struct.pack('<fff??ffi', 0.0, 3.25, 0.75, True, False, 0.5, 0.1, 2)
    body += struct.pack('?', True)                               # 404:405
    body += bytes(3)   # three sweeps, each an empty name
    cfg = saver._from_legacy_bytes(body, version=6)

    assert cfg.ink_weight == pytest.approx(3.25)
    assert cfg.hue_sensitivity == pytest.approx(0.75)
    assert cfg.color_by_cohort is True
    assert cfg.watercolor_mode is False
    assert cfg.num_cohorts == 32
    assert cfg.rule_seed == pytest.approx(7.5)
    # THE assertion: this byte sits immediately AFTER the emboss values, so
    # it is the one that decodes from the wrong place if they stop being read.
    assert cfg.parameter_sweeps_enabled is True
