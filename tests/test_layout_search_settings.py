"""The settings that turn layout search on and bound where it may roam."""
from __future__ import annotations

import pytest

from state.archive_state import PERSISTED_FIELDS, ArchiveState

FIELDS = ("layout_search", "layout_move_chance", "layout_max_depth",
          "layout_max_width", "layout_max_floats", "layout_modalities")


@pytest.mark.parametrize("name", FIELDS)
def test_every_bound_persists(name):
    """PERSISTED_FIELDS is an explicit allowlist: a field not named there
    silently does not survive reopening the archive."""
    assert name in PERSISTED_FIELDS


def test_layout_search_ships_off():
    """Opening the app must never start changing brain under anyone."""
    assert ArchiveState().layout_search is False


def test_every_bound_defaults_to_the_limit_this_build_allows():
    """ONE convention for all three: 0 means 'the limit already in force'. A
    literal bound of zero would forbid every layout, so it can never mean
    itself - and it keeps services out of state/, which imports none of it."""
    ast = ArchiveState()
    assert ast.layout_max_depth == 0
    assert ast.layout_max_width == 0
    assert ast.layout_max_floats == 0


def test_a_zero_float_bound_resolves_to_the_hard_ceiling():
    from services.brains import MAX_BRAIN_FLOATS, default_layout
    from services.brains.layout_moves import bounds_from

    b = bounds_from(0, 0, 0, "", default_layout())
    assert b.max_floats == MAX_BRAIN_FLOATS


def test_a_cross_modality_jump_is_opt_in():
    """It is a restart, so opting into one should be a decision."""
    assert ArchiveState().layout_modalities == ""


def test_the_settings_round_trip():
    ast = ArchiveState()
    ast.layout_search = True
    ast.layout_modalities = "mlp"
    ast.layout_move_chance = 0.4
    other = ArchiveState()
    other.apply_settings(ast.to_settings())
    assert other.layout_search is True
    assert other.layout_modalities == "mlp"
    assert other.layout_move_chance == pytest.approx(0.4)


# ---- reaching the driver ------------------------------------------------

class _Drv:
    """Only what the push writes."""
    name = "imgep"

    def __init__(self):
        from services.brains.layout_moves import LayoutBounds

        self.layout_search = False
        self.layout_move_chance = 0.0
        self.layout_bounds = LayoutBounds()


def _push(ast, layout):
    from command_handler import CommandHandler

    drv = _Drv()
    CommandHandler._push_layout_search(drv, ast, layout)
    return drv


def test_the_checkbox_reaches_the_driver():
    from services.brains import default_layout

    ast = ArchiveState()
    ast.layout_search = True
    ast.layout_move_chance = 0.3
    drv = _push(ast, default_layout())
    assert drv.layout_search is True
    assert drv.layout_move_chance == pytest.approx(0.3)


def test_the_bounds_reach_the_driver():
    from services.brains import default_layout

    ast = ArchiveState()
    ast.layout_max_width = 12
    ast.layout_max_floats = 400
    ast.layout_modalities = "mlp"
    b = _push(ast, default_layout()).layout_bounds
    assert b.max_width == 12
    assert b.max_floats == 400
    assert b.modalities == ("mlp",)


def test_the_running_modality_is_dropped_from_the_jump_list():
    from services.brains import REGISTRY

    ast = ArchiveState()
    ast.layout_modalities = "mlp,gabor"
    b = _push(ast, REGISTRY["mlp"].layout_from_settings({})).layout_bounds
    assert b.modalities == ("gabor",)


def test_a_layout_with_no_brain_at_all_is_not_an_error():
    """_brain_layout() answers None where no sim has been built."""
    from services.brains import default_layout

    ast = ArchiveState()
    ast.layout_search = True
    assert _push(ast, None).layout_bounds is not None
    assert _push(ast, default_layout()).layout_search is True
