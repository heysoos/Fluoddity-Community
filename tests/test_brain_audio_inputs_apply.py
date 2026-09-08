"""Changing ONLY the audio input count carries the loaded rule across, and
Reroll redraws the audio weights under it.

Every other structural change still drops the rule, as it always has. See
docs/superpowers/specs/2026-09-06-brain-audio-inputs-design.md.
"""
import numpy as np
import pytest

import main
from command_handler import CommandHandler
from services.brains import REGISTRY, audio_weight_index, default_layout
from services.brains.layout_moves import grow_inputs
from state import UIState
from tests.test_brain_layout_apply import _StubApp, _ui

FOURIER = REGISTRY["fourier"]


class _RuleManager:
    def __init__(self, rule):
        self.stack = [np.asarray(rule, dtype=np.float32).reshape(-1)]
        self.seeds: list = []

    def push_rule(self, rule, seed):
        self.stack.append(np.asarray(rule, dtype=np.float32).reshape(-1))
        self.seeds.append(seed)

    def get_current_rule(self):
        return self.stack[-1]


class _App(_StubApp):
    """A stub app that also holds a rule, and records what apply_rule got."""

    def __init__(self, layout=None, rule=None):
        super().__init__(layout)
        self.rule_manager = _RuleManager(rule) if rule is not None else None
        self.applied: list = []
        sim = self.sim
        original = sim.apply_rule

        def apply_rule(genome):
            original(genome)
            self.applied.append(None if genome is None
                                else np.asarray(genome, np.float32).reshape(-1))
        sim.apply_rule = apply_rule


def _rule(layout, seed=0):
    return np.asarray(FOURIER.random(np.random.default_rng(seed), layout),
                      dtype=np.float32).reshape(-1)


def _keep(layout):
    return np.setdiff1d(np.arange(layout.length), audio_weight_index(layout))


# ---- the apply path -------------------------------------------------------

def test_growing_the_inputs_carries_the_rule_across():
    base = default_layout()
    rule = _rule(base)
    app, ui_state = _App(base, rule), _ui()
    wide = grow_inputs(base, 3)

    assert app.apply(wide, ui_state) is True
    assert "realloc" in app.sim.calls
    got = app.applied[-1]
    assert got is not None and got.shape == (wide.length,)
    np.testing.assert_array_equal(got[_keep(wide)], rule)
    assert np.any(got[audio_weight_index(wide)] != 0.0)
    # And the rule stack holds the widened brain, so Z and Save see it.
    np.testing.assert_array_equal(app.rule_manager.get_current_rule(), got)


def test_the_audio_weights_come_from_the_rigs_seed():
    base = default_layout()
    rule = _rule(base)
    wide = grow_inputs(base, 2)
    outs = []
    for seed in (0.25, 0.25, 0.75):
        app, ui_state = _App(base, rule), _ui()
        ui_state.audio.audio_seed = seed
        app.apply(wide, ui_state)
        outs.append(app.applied[-1])
    np.testing.assert_array_equal(outs[0], outs[1])
    assert not np.array_equal(outs[0], outs[2])


def test_shrinking_the_inputs_leaves_the_deaf_brain():
    base = default_layout()
    wide = grow_inputs(base, 3)
    rule = _rule(wide)
    app, ui_state = _App(wide, rule), _ui()

    assert app.apply(base, ui_state) is True
    got = app.applied[-1]
    assert got.shape == (base.length,)
    np.testing.assert_array_equal(got, rule[_keep(wide)])


def test_any_other_structural_change_still_drops_the_rule():
    base = default_layout()
    rule = _rule(base)
    app, ui_state = _App(base, rule), _ui()
    other = FOURIER.layout_from_settings({"centers": 11, "audio_inputs": 2})

    app.apply(other, ui_state)
    assert app.applied[-1] is None


def test_with_no_rule_loaded_nothing_is_carried():
    base = default_layout()
    app, ui_state = _App(base, None), _ui()
    app.apply(grow_inputs(base, 2), ui_state)
    assert app.applied[-1] is None


def test_the_frame_after_settles():
    base = default_layout()
    app, ui_state = _App(base, _rule(base)), _ui()
    wide = grow_inputs(base, 2)
    app.apply(wide, ui_state)
    app.sim.calls.clear()
    assert app.apply(wide, ui_state) is False
    assert app.sim.calls == []


# ---- reroll ---------------------------------------------------------------

class _Sim:
    def __init__(self, layout, tournament=False):
        self.brain_layout = layout
        self.brain_per_cohort = False
        self.tournament_enabled = tournament
        self.tournament_grid = 4
        self.applied = []

    def apply_rule(self, params):
        self.applied.append(np.asarray(params, dtype=np.float32).reshape(-1))

    def cohort_brain(self, i):
        return None


def _handler(sim, rule):
    h = object.__new__(CommandHandler)
    h.sim = sim
    h.rule_manager = _RuleManager(rule)
    h._borrow = None
    h.apply_brain_layout = None
    h.archive = None
    h.auto_service = None
    h.imgep_driver = None
    ui_state = UIState()
    ui_state.brain.modality = "fourier"
    ui_state.brain.settings = {"audio_inputs": sim.brain_layout.audio_inputs}
    ui_state.brain.preview_tile0 = sim.tournament_enabled
    return h, ui_state


def test_reroll_redraws_the_audio_weights_from_the_rigs_seed():
    """The panel moves the seed; the handler only ever applies it, so typing
    a seed back in gives the weights it gave before."""
    from services.brains.layout_moves import transfer_audio_inputs

    wide = grow_inputs(default_layout(), 2)
    rule = _rule(wide)
    h, ui_state = _handler(_Sim(wide), rule)
    ui_state.audio.audio_seed = 0.9

    ui_state.brain.reroll_audio_requested = True
    h._handle_brain_source(ui_state)

    assert ui_state.brain.reroll_audio_requested is False
    got = h.sim.applied[-1]
    np.testing.assert_array_equal(got, transfer_audio_inputs(rule, wide, wide, 0.9))
    np.testing.assert_array_equal(got[_keep(wide)], rule[_keep(wide)])
    np.testing.assert_array_equal(h.rule_manager.get_current_rule(), got)


def test_reroll_is_refused_under_a_grid_and_during_a_borrow():
    wide = grow_inputs(default_layout(), 2)
    rule = _rule(wide)
    for tournament, borrow in ((True, None), (False, object())):
        h, ui_state = _handler(_Sim(wide, tournament=tournament), rule)
        h._borrow = borrow
        ui_state.brain.reroll_audio_requested = True
        h._handle_brain_source(ui_state)
        assert ui_state.brain.reroll_audio_requested is False
        assert h.sim.applied == []


def test_reroll_on_a_deaf_brain_does_nothing():
    base = default_layout()
    h, ui_state = _handler(_Sim(base), _rule(base))
    ui_state.brain.reroll_audio_requested = True
    h._handle_brain_source(ui_state)
    assert h.sim.applied == []


# ---- the rig owns the count ------------------------------------------------

def test_the_rig_owns_the_input_count():
    """A preset or a brain change puts the count back to the rig's."""
    base = default_layout()
    h, ui_state = _handler(_Sim(base), _rule(base))
    seen = []
    h.apply_brain_layout = lambda layout, ui: seen.append(layout) or True
    ui_state.audio.audio_inputs = 3
    h._handle_brain_layout(ui_state)
    assert ui_state.brain.settings["audio_inputs"] == 3
    assert seen[-1].audio_inputs == 3
    # Under a grid slot 0 is the tournament's, so the count is left alone.
    ui_state.brain.settings["audio_inputs"] = 0
    ui_state.tournament.enabled = True
    h._handle_brain_layout(ui_state)
    assert ui_state.brain.settings["audio_inputs"] == 0


class _Pending:
    """What a just-loaded config leaves for the layout switch."""
    take_pending_brain_rule = CommandHandler.take_pending_brain_rule
    take_pending_brain_rule_any = CommandHandler.take_pending_brain_rule_any

    def __init__(self, rule, sig):
        self._pending_brain_rule = (np.asarray(rule, np.float32).reshape(-1),
                                    sig)


def test_a_loaded_deaf_config_lands_widened_to_the_rigs_count():
    """File > Load with the rig at K: the config's rule is carried into the
    wide layout in the frame it arrives, so the creature is the preset's."""
    from services.brains.layout_moves import transfer_audio_inputs

    base = default_layout()
    rule = _rule(base)
    other = REGISTRY["gabor"].layout_from_settings({})
    app, ui_state = _App(other, rule), _ui()
    app.command_handler = _Pending(rule, base.signature())
    ui_state.audio.audio_seed = 0.3
    wide = grow_inputs(base, 2)
    assert app.apply(wide, ui_state) is True
    want = transfer_audio_inputs(rule, base, wide, 0.3)
    np.testing.assert_array_equal(app.applied[-1], want)
    np.testing.assert_array_equal(app.rule_manager.get_current_rule(), want)


# ---- the Inspector hears the channels ------------------------------------

def test_the_inspector_is_handed_the_live_channels():
    from pathlib import Path
    src = (Path(main.__file__)).read_text(encoding="utf-8")
    assert "audio=bst.audio_live" in src
    assert "audio_live" in UIState().brain.__dataclass_fields__


# ---- a same-brain preset under the rig's count ---------------------------

def _menu_handler(layout, rule):
    """CommandHandler wired to the REAL _apply_brain_layout of a stub App, so
    the frame runs the way the app runs it: load, then _handle_brain_layout."""
    from tests.test_menu_cross_brain_load import _MenuUI, _Saver

    app = _App(layout, rule)
    h = object.__new__(CommandHandler)
    for a in ("field_handler", "param_lock_service", "archive", "archive_store",
              "auto_service", "imgep_driver", "multi_load_service"):
        setattr(h, a, None)
    h.sim = app.sim
    h.rule_manager = app.rule_manager
    h.ui = _MenuUI({})
    h.config_saver = _Saver(h.ui)
    h._pending_brain_rule = None
    h._borrow = None
    h.preview_rule_active = False
    h._preview_rule_was_pushed = False
    app.command_handler = h
    h.apply_brain_layout = app.apply
    return app, h


class _DeafConfig:
    rule_seed = 0.0

    def __init__(self, layout, rule):
        self.rule = rule
        self.brain_layout = layout.signature()
        self.brain_settings = {}


@pytest.mark.parametrize("hover", [False, True], ids=["click", "hover+click"])
def test_a_same_brain_deaf_preset_loads_widened_under_the_rig(hover):
    """The sim is already on the rig's wide layout of the SAME brain, so no
    switch happens - and the preset's creature has to land anyway, widened
    from the rig's seed. With a hover first, what the click shows must be
    the creature the hover showed, plus its ears."""
    from services.brains.layout_moves import transfer_audio_inputs
    from tests.test_menu_cross_brain_load import _click, _hover

    base = default_layout()
    wide = grow_inputs(base, 2)
    old = _rule(wide, seed=1)
    app, h = _menu_handler(wide, old)
    sim = app.sim
    ui_state = _ui()
    ui_state.audio.audio_inputs = 2
    ui_state.audio.audio_seed = 0.3
    ui_state.brain.settings = {"audio_inputs": 2}

    preset = _rule(base, seed=7)
    h.ui.configs["p"] = _DeafConfig(base, preset)
    if hover:
        _hover(h, ui_state, "p")
    _click(h, ui_state, "p")
    h._handle_brain_layout(ui_state)

    want = transfer_audio_inputs(preset, base, wide, 0.3)
    np.testing.assert_array_equal(app.applied[-1], want)
    np.testing.assert_array_equal(h.rule_manager.get_current_rule(), want)
    assert sim.brain_layout == wide
