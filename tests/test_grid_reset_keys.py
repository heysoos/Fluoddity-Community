"""Z and G under a tournament.

Both went through sim.apply_rule, which writes SLOT 0 - tile 0 of a grid, and
overwritten by the next generation. Pressed under a tournament they changed one
square in the bottom-left corner and nothing else, which is neither key's
meaning.
"""
import numpy as np

from command_handler import CommandHandler
from state.archive_state import ArchiveState
from state.auto_tournament_state import AutoTournamentState
from state.sim_state import SimState
from state.tournament_state import TournamentState


class _Sim:
    def __init__(self):
        self.rules = []
        self.resets = 0

    def apply_rule(self, rule):
        self.rules.append(np.asarray(rule).copy())

    def reset(self):
        self.resets += 1


class _RuleManager:
    def __init__(self, current=None):
        self.current = current
        self.pushed = []

    def get_current_rule(self):
        return self.current

    def push_rule(self, rule, seed):
        self.pushed.append((np.asarray(rule).copy(), seed))


class _Owner:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _UIState:
    def __init__(self, tournament=False, auto=False, explore=False):
        self.sim = SimState()
        self.tournament = TournamentState()
        self.tournament.enabled = tournament
        self.auto_tournament = AutoTournamentState()
        self.auto_tournament.enabled = auto
        self.archive = ArchiveState()
        self.archive.enabled = explore


class _Handler:
    """Only the attributes the two methods under test touch."""

    def __init__(self, tournament_service=None, auto_service=None):
        self.sim = _Sim()
        self.rule_manager = _RuleManager(np.ones((10, 8), dtype=np.float32))
        self.tournament_service = tournament_service
        self.auto_service = auto_service

    def _grid_owner(self, ui):
        return CommandHandler._grid_owner(self, ui)


def full_reset(h, ui):
    CommandHandler._handle_full_reset(h, ui)


def randomize(h, ui):
    CommandHandler._handle_randomize_mutations(h, ui)


# ---- outside a tournament: unchanged --------------------------------------

def test_z_outside_a_tournament_still_zeroes_the_single_brain():
    h, ui = _Handler(), _UIState()
    full_reset(h, ui)
    assert h.sim.resets == 1
    assert len(h.sim.rules) == 1
    assert not h.sim.rules[0].any(), "the zero rule is the 'no brain' marker"


def test_g_outside_a_tournament_still_repushes_the_current_rule():
    h, ui = _Handler(), _UIState()
    randomize(h, ui)
    assert len(h.sim.rules) == 1
    assert len(h.rule_manager.pushed) == 1


def test_g_with_no_rule_loaded_writes_nothing_but_still_moves_the_seed():
    h, ui = _Handler(), _UIState()
    h.rule_manager.current = None
    before = ui.sim.rule_seed
    randomize(h, ui)
    assert h.sim.rules == []
    assert ui.sim.rule_seed != before


# ---- manual tournament: the whole grid ------------------------------------

def test_z_under_a_manual_tournament_resets_the_GRID():
    ts = _Owner()
    h, ui = _Handler(tournament_service=ts), _UIState(tournament=True)
    full_reset(h, ui)
    assert ts.resets == 1
    assert h.sim.rules == [], "slot 0 is tile 0, and the grid owns every tile"


def test_g_under_a_manual_tournament_writes_no_tile():
    ts = _Owner()
    h, ui = _Handler(tournament_service=ts), _UIState(tournament=True)
    before = ui.sim.rule_seed
    randomize(h, ui)
    assert h.sim.rules == []
    assert ui.sim.rule_seed != before, "a fresh crop of mutations is the seed"


# ---- Auto and Explore: the optimizer owns it ------------------------------

def test_z_under_auto_resets_the_SEARCH():
    """Re-randomising the grid alone would be undone by the next generation."""
    auto, ts = _Owner(), _Owner()
    h = _Handler(tournament_service=ts, auto_service=auto)
    full_reset(h, _UIState(tournament=True, auto=True))
    assert (auto.resets, ts.resets) == (1, 0)


def test_z_under_explore_resets_the_SEARCH_too():
    auto, ts = _Owner(), _Owner()
    h = _Handler(tournament_service=ts, auto_service=auto)
    full_reset(h, _UIState(tournament=True, explore=True))
    assert (auto.resets, ts.resets) == (1, 0)


def test_a_closed_tournament_window_hands_the_keys_back():
    """render_tournament_window clears both sub-modes when it is shut, so the
    keys must follow tournament.enabled rather than the sub-mode flags."""
    auto, ts = _Owner(), _Owner()
    h = _Handler(tournament_service=ts, auto_service=auto)
    full_reset(h, _UIState(tournament=False, auto=True))
    assert (auto.resets, ts.resets) == (0, 0)
    assert h.sim.resets == 1


def test_auto_mode_with_no_service_falls_back_to_the_grid():
    """The tab can be open before the optional packages have loaded."""
    ts = _Owner()
    h = _Handler(tournament_service=ts, auto_service=None)
    full_reset(h, _UIState(tournament=True, auto=True))
    assert ts.resets == 1
