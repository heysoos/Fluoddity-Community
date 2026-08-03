from state.tournament_state import TournamentState
from state.ui_state import UIState


def test_tournament_state_defaults():
    ts = TournamentState()
    assert ts.enabled is False
    assert ts.mutation_strength == 0.15
    assert ts.inject_randoms == 1
    assert ts.crossover_enabled is False
    assert ts.clicked_tile == -1
    assert ts.next_gen_requested is False
    assert ts.undo_requested is False
    assert ts.reset_requested is False
    assert ts.save_requested is False


def test_uistate_has_independent_tournament_states():
    a = UIState()
    b = UIState()
    a.tournament.enabled = True
    assert b.tournament.enabled is False   # default_factory, not shared
