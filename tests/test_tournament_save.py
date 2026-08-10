"""Saving tournament tiles under a chosen name.

The reported behaviour: "Save Selected" wrote tournament_tile3.json with no
dialog, no feedback, and silent overwriting of the tile 3 you saved five
generations ago.
"""
import numpy as np
import pytest

from services.config_saver import ConfigSaver
from state.sim_state import SimState
from state.tournament_state import TournamentState


class _FakeService:
    def __init__(self, n=16):
        rng = np.random.default_rng(0)
        self.population = [rng.normal(0, 0.5, (10, 8)).astype(np.float32)
                           for _ in range(n)]
        self.selected = set()


def _handler(tmp_path, svc):
    from command_handler import CommandHandler

    h = object.__new__(CommandHandler)
    h.tournament_service = svc
    h.config_saver = ConfigSaver()
    h.user_configs_dir = tmp_path
    return h


def _ui(tiles=()):
    return type("UI", (), {"sim": SimState(),
                           "tournament": TournamentState(),
                           "save_tiles": tuple(tiles)})()


def save(tmp_path, tiles, name="reef", svc=None):
    from command_handler import CommandHandler

    svc = svc or _FakeService()
    svc.selected = set(tiles)
    ui = _ui(tiles)
    CommandHandler._save_tournament_selection(_handler(tmp_path, svc), ui, name)
    return ui, sorted(p.name for p in tmp_path.glob("*.json"))


def test_one_tile_is_saved_under_exactly_the_name_given(tmp_path):
    _ui_state, written = save(tmp_path, [3], name="my creature")
    assert written == ["my creature.json"]


def test_several_tiles_get_distinct_files(tmp_path):
    """Not one shared name: collapsing them would reintroduce the clobbering
    this feature exists to fix, inside a single click."""
    _ui_state, written = save(tmp_path, [1, 4], name="reef")
    assert written == ["reef_tile1.json", "reef_tile4.json"]


def test_the_saved_config_is_that_tile_s_genome(tmp_path):
    svc = _FakeService()
    save(tmp_path, [7], name="seven", svc=svc)
    cfg = ConfigSaver().load_from_file(tmp_path / "seven.json")
    assert np.allclose(np.asarray(cfg.rule, dtype=np.float32),
                       svc.population[7], atol=1e-5)


def test_the_user_is_told_where_the_files_went(tmp_path):
    """A console print is not feedback: the configs folder is not on screen."""
    ui, _written = save(tmp_path, [2], name="reef")
    assert "reef.json" in ui.tournament.notice
    assert "Load" in ui.tournament.notice


def test_saving_nothing_says_so_instead_of_failing_silently(tmp_path):
    ui, written = save(tmp_path, [], name="reef")
    assert written == []
    assert "no tiles" in ui.tournament.notice.lower()


def test_the_named_tiles_win_over_a_selection_that_moved(tmp_path):
    """The dialog names a set of tiles; a click landing while it is open must
    not change which genomes get written under that name."""
    from command_handler import CommandHandler

    svc = _FakeService()
    svc.selected = {11}            # the user clicked again since
    ui = _ui([3])                  # but the dialog was opened for tile 3
    CommandHandler._save_tournament_selection(_handler(tmp_path, svc), ui, "reef")

    cfg = ConfigSaver().load_from_file(tmp_path / "reef.json")
    assert np.allclose(np.asarray(cfg.rule, dtype=np.float32),
                       svc.population[3], atol=1e-5)


def test_a_tile_outside_the_population_is_dropped(tmp_path):
    """The grid can shrink between opening the dialog and pressing Save."""
    svc = _FakeService(n=4)
    ui = _ui([1, 99])
    from command_handler import CommandHandler

    CommandHandler._save_tournament_selection(_handler(tmp_path, svc), ui, "reef")
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["reef.json"]


@pytest.mark.parametrize("kind_tiles", [[0], [0, 1], [0, 1, 2, 3]])
def test_every_selection_size_writes_one_file_per_tile(tmp_path, kind_tiles):
    _ui_state, written = save(tmp_path, kind_tiles, name="x")
    assert len(written) == len(kind_tiles)
    assert len(set(written)) == len(written)
