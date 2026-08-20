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


def test_loading_a_legacy_file_moves_the_window_to_the_fourier_it_holds():
    """A pre-modality config names no brain, and its GENOME says which one it
    is anyway: every unsigned file in the library carries a fourier-n10 rule.
    Leaving the window on gabor was what let the whole historical library load
    its physics around whatever creature was already running."""
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

    assert ui_state.brain.modality == "fourier"


def test_a_legacy_file_at_an_unknown_width_still_leaves_it_alone():
    """The width is checked rather than assumed. An unsigned file that is NOT
    Fourier-shaped does not say which brain it wants, and guessing would be the
    same defect pointed the other way."""
    from command_handler import CommandHandler
    from state.ui_state import UIState
    from ui.brain_window import layout_for

    d = PhysicsConfig().to_dict()
    del d["brain_layout"]
    cfg = PhysicsConfig.from_dict(d)
    cfg.rule = [0.0] * layout_for("mlp", {}).length

    ui_state = UIState()
    ui_state.brain.modality = "gabor"
    ui_state.brain.settings = {"filters": 20}
    CommandHandler._restore_brain_settings(
        CommandHandler.__new__(CommandHandler), cfg, ui_state)

    assert ui_state.brain.modality == "gabor"
    assert ui_state.brain.settings == {"filters": 20}


DEEP_LAYERS = [[8, 0], [6, 1], [4, 2]]


def test_a_deep_mlp_config_round_trips():
    """The signature is the only one with a variable-length body, and it is what
    the archive directory, the checkpoint guard and every hover path travel on."""
    from ui.brain_window import layout_for

    lay = REGISTRY["mlp"].layout_from_settings({"layers": DEEP_LAYERS})
    genome = np.asarray(random_genome_for(np.random.default_rng(2), lay),
                        dtype=np.float32).reshape(-1)
    back = PhysicsConfig.from_dict(json.loads(
        ConfigSaver().create_config(SimState(), genome, layout=lay).to_json()))

    assert back.brain_layout == "mlp-n8.6.4-a0.1.2"
    # The decode scales ride along with the structure: a saved rule is decoded,
    # and anything re-encoding it divides by them.
    assert back.brain_settings["layers"] == DEEP_LAYERS
    assert back.brain_settings["w_scale"] == pytest.approx(2.0)
    assert np.allclose(np.asarray(back.rule).reshape(-1), genome, atol=1e-6)
    assert layout_for("mlp", back.brain_settings) == lay


def test_a_config_written_before_the_stack_still_loads():
    """8 user configs are stamped mlp-n16-a0 with hidden/activation settings.
    They must open as a one-layer stack with no migration step."""
    from command_handler import CommandHandler
    from state.ui_state import UIState
    from ui.brain_window import layout_for

    d = PhysicsConfig().to_dict()
    d["rule"] = [0.05] * 148
    d["brain_layout"] = "mlp-n16-a0"
    d["brain_settings"] = {"hidden": 16, "activation": 0}
    cfg = PhysicsConfig.from_dict(d)

    ui_state = UIState()
    CommandHandler._restore_brain_settings(
        CommandHandler.__new__(CommandHandler), cfg, ui_state)

    assert ui_state.brain.modality == "mlp"
    got = layout_for("mlp", ui_state.brain.settings)
    assert got.signature() == "mlp-n16-a0" and got.length == 148


@pytest.mark.parametrize("sig", ["mlp-n8.6.4-a0.1.2", "gabor-n7",
                                 "lenia-n20", "fourier-n21"])
def test_the_signature_wins_when_the_settings_disagree(sig):
    """A file whose settings do not rebuild its signature used to load the
    SETTINGS, so a non-default layout with missing settings came back as the
    modality's defaults - the wrong brain, silently, and then apply_rule
    refuses the genome on width. The signature is what the rule was decoded
    under, so it is what wins; the settings only carry the scales it omits."""
    from command_handler import CommandHandler
    from state.ui_state import UIState
    from ui.brain_window import layout_for

    d = PhysicsConfig().to_dict()
    d["brain_layout"] = sig
    d["brain_settings"] = {}
    cfg = PhysicsConfig.from_dict(d)

    ui_state = UIState()
    CommandHandler._restore_brain_settings(
        CommandHandler.__new__(CommandHandler), cfg, ui_state)

    assert layout_for(ui_state.brain.modality,
                      ui_state.brain.settings).signature() == sig


def test_a_signature_this_build_cannot_rebuild_says_so(capsys):
    """And leaves the settings alone rather than inventing a layout."""
    from command_handler import CommandHandler
    from state.ui_state import UIState

    d = PhysicsConfig().to_dict()
    d["brain_layout"] = "mlp-n16-a9"
    d["brain_settings"] = {"layers": [[16, 0]]}
    cfg = PhysicsConfig.from_dict(d)

    ui_state = UIState()
    CommandHandler._restore_brain_settings(
        CommandHandler.__new__(CommandHandler), cfg, ui_state)

    assert ui_state.brain.settings == {"layers": [[16, 0]]}
    assert "cannot rebuild" in capsys.readouterr().out


def test_a_legacy_file_with_no_signature_still_loads():
    """Every config written before brains were swappable is an 80-float
    Fourier, and those must keep opening."""
    d = PhysicsConfig().to_dict()
    d["rule"] = [0.1] * 80
    del d["brain_layout"]
    back = PhysicsConfig.from_dict(d)
    assert back.brain_layout == ""
    assert np.asarray(back.rule).reshape(-1).size == 80
