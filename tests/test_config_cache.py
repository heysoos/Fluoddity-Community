"""Opening File > Load re-parses only the configs whose files changed.

Every open used to parse the whole library - 182 files, ~90 ms - and under a
busy worker that one frame stretched past a second. See CLAUDE.md.
"""
import os
import time

import numpy as np

from services.config_saver import ConfigSaver
from state.sim_state import SimState
from ui.config_browser import ConfigBrowserMixin


class _Browser(ConfigBrowserMixin):
    pass


def _write(saver, path, seed=0.0):
    state = SimState()
    state.rule_seed = seed
    saver.save_to_file(saver.create_config(state, np.zeros((10, 8), np.float32)),
                       path)


def _make(tmp_path, n_core=3, n_custom=2):
    b = _Browser()
    b.app_configs_dir = tmp_path / "app"
    b.user_configs_dir = tmp_path / "user"
    (b.app_configs_dir / "Core").mkdir(parents=True)
    (b.app_configs_dir / "Advanced").mkdir()
    b.user_configs_dir.mkdir()
    b.config_saver = ConfigSaver()
    b.config_files = []
    b.config_files_by_category = {}
    b.cached_configs = {}
    for i in range(n_core):
        _write(b.config_saver, b.app_configs_dir / "Core" / f"c{i}.json")
    for i in range(n_custom):
        _write(b.config_saver, b.user_configs_dir / f"u{i}.json")
    loads = {"n": 0}
    real = b.config_saver.load_from_file

    def counting(path):
        loads["n"] += 1
        return real(path)
    b.config_saver.load_from_file = counting
    return b, loads


def test_the_first_open_parses_everything_and_the_second_parses_nothing(tmp_path):
    b, loads = _make(tmp_path)
    b._cache_all_configs()
    assert loads["n"] == 5
    assert set(b.cached_configs) == {"Core/c0", "Core/c1", "Core/c2",
                                     "Custom/u0", "Custom/u1"}
    b._cache_all_configs()
    assert loads["n"] == 5
    assert len(b.cached_configs) == 5


def test_a_changed_file_is_the_only_one_re_read(tmp_path):
    b, loads = _make(tmp_path)
    b._cache_all_configs()
    path = b.user_configs_dir / "u1.json"
    _write(b.config_saver, path, seed=7.0)
    later = time.time() + 5
    os.utime(path, (later, later))
    b._cache_all_configs()
    assert loads["n"] == 6
    assert b.cached_configs["Custom/u1"].rule_seed == 7.0


def test_a_deleted_file_leaves_the_cache_and_a_new_one_joins_it(tmp_path):
    b, loads = _make(tmp_path)
    b._cache_all_configs()
    (b.user_configs_dir / "u0.json").unlink()
    _write(b.config_saver, b.user_configs_dir / "u9.json")
    b._cache_all_configs()
    assert "Custom/u0" not in b.cached_configs
    assert "Custom/u9" in b.cached_configs
    assert loads["n"] == 6


def test_the_click_that_empties_the_visible_cache_does_not_force_a_reparse(tmp_path):
    """The Load click clears cached_configs; the next open must still be
    stats only."""
    b, loads = _make(tmp_path)
    b._cache_all_configs()
    b.cached_configs = {}
    b._cache_all_configs()
    assert loads["n"] == 5
    assert len(b.cached_configs) == 5
