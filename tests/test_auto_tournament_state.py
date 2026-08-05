from dataclasses import fields

from state import UIState
from state.auto_tournament_state import AutoTournamentState


def test_defaults_match_the_spec():
    s = AutoTournamentState()
    assert s.enabled is False
    assert s.running is False
    assert s.prompt == ""
    assert s.algorithm == "CMA-ES"
    assert s.grid == 4
    assert s.steps_per_gen == 300
    assert s.sim_steps_per_frame == 10
    assert s.snapshots_per_gen == 4
    assert s.sigma0 == 0.5
    assert s.autosave_every == 10
    assert s.tile_mutation_enabled is False
    assert s.variants_per_tile == 4
    assert s.tile_mutation_strength == 0.1


def test_all_one_shot_flags_default_false_or_empty():
    s = AutoTournamentState()
    assert s.start_requested is False
    assert s.pause_requested is False
    assert s.reset_requested is False
    assert s.prompt_changed is False
    assert s.grid_changed is False
    assert s.save_checkpoint_requested is False
    assert s.save_best_requested is False
    assert s.save_tile_requested == -1
    assert s.load_checkpoint_path == ""
    assert s.load_genome_path == ""
    assert s.download_model_requested is False


def test_ui_state_exposes_auto_tournament():
    assert any(f.name == "auto_tournament" for f in fields(UIState))
    assert isinstance(UIState().auto_tournament, AutoTournamentState)


def test_state_is_not_shared_between_instances():
    a, b = UIState(), UIState()
    a.auto_tournament.prompt = "coral"
    assert b.auto_tournament.prompt == ""


def test_clear_auto_flags_resets_one_shots_but_not_settings():
    """Cleared on the consuming side: UI.get_state() returns the live object,
    so clearing there would wipe flags before CommandHandler read them."""
    from command_handler import CommandHandler

    s = AutoTournamentState()
    s.prompt = "coral"
    s.grid = 6
    s.warning = "capture is black"
    s.start_requested = True
    s.save_tile_requested = 3
    s.load_genome_path = "x.json"

    CommandHandler._clear_auto_flags(s)

    assert s.start_requested is False
    assert s.save_tile_requested == -1
    assert s.load_genome_path == ""
    # settings and the warning banner survive
    assert s.prompt == "coral"
    assert s.grid == 6
    assert s.warning == "capture is black"
