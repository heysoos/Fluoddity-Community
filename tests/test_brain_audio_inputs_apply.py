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


def test_the_audio_weights_come_from_the_config_seed():
    base = default_layout()
    rule = _rule(base)
    wide = grow_inputs(base, 2)
    outs = []
    for seed in (0.25, 0.25, 0.75):
        app, ui_state = _App(base, rule), _ui()
        ui_state.sim.audio_seed = seed
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
    ui_state = UIState()
    ui_state.brain.modality = "fourier"
    ui_state.brain.settings = {"audio_inputs": sim.brain_layout.audio_inputs}
    ui_state.brain.preview_tile0 = sim.tournament_enabled
    return h, ui_state


def test_reroll_redraws_only_the_audio_weights_and_moves_the_seed():
    wide = grow_inputs(default_layout(), 2)
    rule = _rule(wide)
    h, ui_state = _handler(_Sim(wide), rule)
    before = ui_state.sim.audio_seed

    ui_state.brain.reroll_audio_requested = True
    h._handle_brain_source(ui_state)

    assert ui_state.brain.reroll_audio_requested is False
    assert ui_state.sim.audio_seed != before
    got = h.sim.applied[-1]
    np.testing.assert_array_equal(got[_keep(wide)], rule[_keep(wide)])
    assert not np.array_equal(got[audio_weight_index(wide)],
                              rule[audio_weight_index(wide)])
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
    seed = ui_state.sim.audio_seed
    ui_state.brain.reroll_audio_requested = True
    h._handle_brain_source(ui_state)
    assert h.sim.applied == [] and ui_state.sim.audio_seed == seed


# ---- the Inspector hears the channels ------------------------------------

def test_the_inspector_is_handed_the_live_channels():
    from pathlib import Path
    src = (Path(main.__file__)).read_text(encoding="utf-8")
    assert "audio=bst.audio_live" in src
    assert "audio_live" in UIState().brain.__dataclass_fields__
