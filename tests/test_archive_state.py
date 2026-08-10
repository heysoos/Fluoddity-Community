from dataclasses import fields

from state.archive_state import ArchiveState
from state.ui_state import UIState


ONE_SHOTS = (
    "start_requested", "pause_requested", "reset_requested",
    "add_goal_requested", "remove_goal_index", "move_goal_index",
    "move_goal_delta", "chase_tile", "pin_tile",
    "seed_entry_id", "delete_entry_id", "refit_projection_requested",
    "grid_changed",
)


def test_defaults_match_the_spec_table():
    s = ArchiveState()
    assert s.enabled is False
    assert s.steps_per_gen == 2000
    assert s.snapshots_per_gen == 6
    assert s.seed_n == 256
    assert s.sigma0 == 0.5
    assert s.sigma_expand == 0.15
    assert s.alpha == 4.0
    assert s.k == 10
    assert s.expansion_between == 25
    assert s.expedition_gens == 50
    assert s.expedition_sigma == 0.1
    assert s.latent_share == 0.5
    assert s.liveness_min == 0.002
    assert s.capacity == 20000
    assert s.refresh_sweep_gens == 10
    assert s.goal_order == "round_robin"


def test_one_shot_flags_default_to_inert():
    s = ArchiveState()
    for name in ONE_SHOTS:
        v = getattr(s, name)
        assert v in (False, -1), f"{name} defaults to {v!r}"


def test_the_new_goal_text_buffer_starts_empty():
    assert ArchiveState().new_goal_text == ""


def test_ui_state_carries_an_archive_block():
    assert isinstance(UIState().archive, ArchiveState)


def test_every_declared_one_shot_actually_exists():
    names = {f.name for f in fields(ArchiveState)}
    assert set(ONE_SHOTS) <= names


def test_the_driver_settings_the_command_handler_pushes_all_exist():
    """CommandHandler copies these by name from ArchiveState onto ImgepDriver.
    A typo on either side is a silent no-op at runtime, so pin both ends."""
    from services.imgep_driver import ImgepDriver

    pushed = ("sigma_expand", "alpha", "k", "seed_n", "liveness_min",
              "refresh_sweep_gens", "expansion_between", "expedition_gens",
              "expedition_sigma", "latent_share", "goal_order")
    names = {f.name for f in fields(ArchiveState)}
    driver_attrs = set(vars(ImgepDriver(None, None, [], rng=None)))
    for name in pushed:
        assert name in names, f"ArchiveState has no {name}"
        assert name in driver_attrs, f"ImgepDriver has no {name}"


# ---- archive management ----------------------------------------------------

def test_the_active_archive_defaults_to_default():
    assert ArchiveState().archive_name == "default"


def test_the_cached_listing_starts_empty_and_is_per_instance():
    a, b = ArchiveState(), ArchiveState()
    a.archive_list.append({"name": "x"})
    assert b.archive_list == []


def test_the_management_one_shots_start_clear():
    ast = ArchiveState()
    assert ast.switch_archive_name == ""
    assert ast.new_archive_requested is False
    assert ast.clear_archive_requested is False
    assert ast.delete_archive_requested is False
    assert ast.refresh_archive_list_requested is False


def test_the_modal_buffers_start_empty():
    ast = ArchiveState()
    assert ast.new_archive_name == ""
    assert ast.confirm_delete_text == ""


def test_the_active_archive_survives_a_preferences_round_trip(tmp_path):
    """Without this, every launch reverts to 'default' regardless of what the
    user was working in."""
    from state.preferences_state import (PreferencesState, load_preferences,
                                         save_preferences)

    p = tmp_path / "prefs.json"
    prefs = PreferencesState()
    prefs.archive_name = "dense-trails"
    save_preferences(prefs, p)
    assert load_preferences(p).archive_name == "dense-trails"
