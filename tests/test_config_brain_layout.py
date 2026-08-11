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
    cfg = ConfigSaver().create_config(SimState(), genome,
                                      brain_layout=lay.signature())
    back = PhysicsConfig.from_dict(json.loads(cfg.to_json()))
    assert np.asarray(back.rule).reshape(-1).size == lay.length
    assert np.allclose(np.asarray(back.rule).reshape(-1), genome, atol=1e-6)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_file_says_which_brain_it_holds(name):
    lay, genome = a_genome(name)
    cfg = ConfigSaver().create_config(SimState(), genome,
                                      brain_layout=lay.signature())
    assert json.loads(cfg.to_json())["brain_layout"] == lay.signature()
    assert PhysicsConfig.from_dict(json.loads(cfg.to_json())).brain_layout == \
        lay.signature()


@pytest.mark.parametrize("name", MODALITIES)
def test_a_saved_file_actually_loads_from_disk(name, tmp_path):
    """The reported failure was at load_from_file, which swallows the exception
    and returns None - so the tile simply never came back."""
    lay, genome = a_genome(name)
    saver = ConfigSaver()
    path = tmp_path / f"tournament_tile8_{name}.json"
    saver.save_to_file(saver.create_config(SimState(), genome,
                                           brain_layout=lay.signature()), path)
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
                               brain_layout=g.signature())
    back = PhysicsConfig.from_dict(json.loads(made.to_json()))
    assert back.brain_layout == g.signature()


def test_a_legacy_file_with_no_signature_still_loads():
    """Every config written before brains were swappable is an 80-float
    Fourier, and those must keep opening."""
    d = PhysicsConfig().to_dict()
    d["rule"] = [0.1] * 80
    del d["brain_layout"]
    back = PhysicsConfig.from_dict(d)
    assert back.brain_layout == ""
    assert np.asarray(back.rule).reshape(-1).size == 80
