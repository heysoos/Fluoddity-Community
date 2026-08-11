"""A config must round-trip the brain it was saved under.

PhysicsConfig.from_dict reshaped the rule to Fourier's (10, 8) unconditionally,
so saving a tournament tile under any other brain wrote a file that could never
be read back:

    Failed to load config from ...tournament_tile8.json:
    cannot reshape array of size 168 into shape (10,8)

Keeping the rule flat fixes the crash but not the ambiguity: 80 floats are a
10-centre Fourier OR an 8-bump Lenia, and 168 are a 21-centre Fourier OR a
12-filter Gabor. So the layout signature is written alongside, and a file
without one is Fourier - by history, not by assumption, since every config
predating swappable brains is Fourier.
"""
import json

import numpy as np
import pytest

from services.brains import REGISTRY
from services.config_saver import ConfigSaver, PhysicsConfig
from services.genome_spec import random_genome_for
from state import SimState

MODALITIES = sorted(REGISTRY)


def layout(name, **settings):
    return REGISTRY[name].layout_from_settings(settings)


def a_genome(name):
    lay = layout(name)
    g = random_genome_for(np.random.default_rng(4), lay)
    return lay, np.asarray(g, dtype=np.float32).reshape(-1)


@pytest.mark.parametrize("name", MODALITIES)
def test_a_saved_rule_survives_the_round_trip(name):
    lay, genome = a_genome(name)
    cfg = ConfigSaver().create_config(SimState(), genome, layout=lay)
    back = PhysicsConfig.from_dict(json.loads(cfg.to_json()))
    assert np.asarray(back.rule).reshape(-1).size == lay.length
    assert np.allclose(np.asarray(back.rule).reshape(-1), genome, atol=1e-6)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_file_says_which_brain_it_holds(name):
    lay, genome = a_genome(name)
    cfg = ConfigSaver().create_config(SimState(), genome, layout=lay)
    assert json.loads(cfg.to_json())["brain_layout"] == lay.signature()
    assert PhysicsConfig.from_dict(json.loads(cfg.to_json())).brain_layout == \
        lay.signature()


@pytest.mark.parametrize("name", MODALITIES)
def test_the_file_carries_the_decode_scales_too(name):
    """The signature deliberately excludes them - they must not split an
    archive - so a file that stored only the signature could not put the Brain
    window back where the creature was authored."""
    from services.brains import settings_of
    from ui.brain_window import layout_for

    lay, genome = a_genome(name)
    cfg = ConfigSaver().create_config(SimState(), genome, layout=lay)
    back = PhysicsConfig.from_dict(json.loads(cfg.to_json()))
    assert back.brain_settings == settings_of(lay)
    rebuilt = layout_for(back.brain_layout.split("-")[0], back.brain_settings)
    assert rebuilt.signature() == lay.signature()
    assert tuple(rebuilt.scales) == tuple(lay.scales)


@pytest.mark.parametrize("name", ["gabor", "fourier", "lenia"])
def test_a_non_default_scale_is_not_lost(name):
    """The case that motivated this. Gabor's Input Scale is a property of the
    PRESET and runs ~900x across the library, so a saved brain is very likely
    to be reloaded under a different one."""
    from services.brains import REGISTRY

    m = REGISTRY[name]
    scale = [s for s in m.settings_schema() if s.kind == "float"][0]
    tuned = m.layout_from_settings({scale.key: scale.lo})
    genome = np.asarray(random_genome_for(np.random.default_rng(7), tuned),
                        dtype=np.float32).reshape(-1)
    cfg = ConfigSaver().create_config(SimState(), genome, layout=tuned)
    back = PhysicsConfig.from_dict(json.loads(cfg.to_json()))
    assert back.brain_settings[scale.key] == pytest.approx(scale.lo)


@pytest.mark.parametrize("name", MODALITIES)
def test_a_saved_file_actually_loads_from_disk(name, tmp_path):
    """The reported failure was at load_from_file, which swallows the exception
    and returns None - so the tile simply never came back."""
    lay, genome = a_genome(name)
    saver = ConfigSaver()
    path = tmp_path / f"tournament_tile8_{name}.json"
    saver.save_to_file(saver.create_config(SimState(), genome, layout=lay), path)
    back = saver.load_from_file(path)
    assert back is not None
    assert np.allclose(np.asarray(back.rule).reshape(-1), genome, atol=1e-6)


def test_two_brains_of_one_width_are_told_apart():
    """The reason a width check alone is not enough. Fourier at 21 centres and
    Gabor at 12 filters are both 168 floats."""
    f, g = layout("fourier", centers=21), layout("gabor", filters=12)
    assert f.length == g.length
    assert f.signature() != g.signature()

    saver = ConfigSaver()
    made = saver.create_config(SimState(),
                               np.zeros(g.length, dtype=np.float32) + 0.25,
                               layout=g)
    back = PhysicsConfig.from_dict(json.loads(made.to_json()))
    assert back.brain_layout == g.signature()


@pytest.mark.parametrize("name", MODALITIES)
def test_loading_puts_the_brain_window_back(name):
    """_handle_brain_layout applies whatever it finds in ui_state.brain every
    frame, so writing the state IS applying it - there is no request to make."""
    from command_handler import CommandHandler
    from state.ui_state import UIState

    lay, genome = a_genome(name)
    cfg = ConfigSaver().create_config(SimState(), genome, layout=lay)

    ui_state = UIState()
    ui_state.brain.modality = "fourier"
    ui_state.brain.settings = {}
    CommandHandler._restore_brain_settings(
        CommandHandler.__new__(CommandHandler), cfg, ui_state)

    from ui.brain_window import layout_for

    assert ui_state.brain.modality == name
    got = layout_for(ui_state.brain.modality, ui_state.brain.settings)
    assert got.signature() == lay.signature()
    assert tuple(got.scales) == tuple(lay.scales)


def test_loading_a_legacy_file_leaves_the_brain_window_alone():
    """No signature means a pre-modality config, which says nothing about the
    brain - so it must not stamp Fourier over whatever is selected."""
    from command_handler import CommandHandler
    from state.ui_state import UIState

    d = PhysicsConfig().to_dict()
    del d["brain_layout"]
    cfg = PhysicsConfig.from_dict(d)

    ui_state = UIState()
    ui_state.brain.modality = "gabor"
    ui_state.brain.settings = {"filters": 20}
    CommandHandler._restore_brain_settings(
        CommandHandler.__new__(CommandHandler), cfg, ui_state)

    assert ui_state.brain.modality == "gabor"
    assert ui_state.brain.settings == {"filters": 20}


def test_a_legacy_file_with_no_signature_still_loads():
    """Every config written before brains were swappable is an 80-float
    Fourier, and those must keep opening."""
    d = PhysicsConfig().to_dict()
    d["rule"] = [0.1] * 80
    del d["brain_layout"]
    back = PhysicsConfig.from_dict(d)
    assert back.brain_layout == ""
    assert np.asarray(back.rule).reshape(-1).size == 80
