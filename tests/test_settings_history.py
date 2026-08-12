"""A settings change is recorded once, with enough context to place it.

settings.json is rewritten wholesale, so the value that admitted a given entry
is unrecoverable the moment a slider moves. This is the record that makes it
recoverable.
"""
from services.archive_io import ArchiveStore
from services.settings_history import diff


def _store(tmp_path):
    return ArchiveStore(tmp_path / "arc")


# ---- the diff -----------------------------------------------------------

def test_no_change_produces_no_diff():
    assert diff({"a": 1}, {"a": 1}) == {}


def test_a_change_records_old_and_new():
    assert diff({"a": 1}, {"a": 2}) == {"a": [1, 2]}


def test_a_new_field_records_none_as_its_old_value():
    assert diff({}, {"a": 2}) == {"a": [None, 2]}


def test_a_removed_field_is_not_a_change():
    """PERSISTED_FIELDS shrinking is a code change, not a user action, and
    recording it would put a phantom row in every archive on the next run."""
    assert diff({"a": 1}, {}) == {}


def test_only_the_moved_fields_appear():
    got = diff({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 9, "c": 3})
    assert got == {"b": [2, 9]}


def test_a_float_that_did_not_move_is_not_a_change():
    """Sliders write floats every frame; equality has to hold or every
    generation would append a row."""
    assert diff({"sigma": 0.35}, {"sigma": 0.35}) == {}


# ---- the store ----------------------------------------------------------

def test_an_empty_history_reports_version_minus_one(tmp_path):
    """Nothing written yet, so the next row is version 0."""
    assert _store(tmp_path).latest_version() == -1


def test_version_zero_carries_the_whole_block(tmp_path):
    s = _store(tmp_path)
    s.append_history({"v": 0, "ts": 1.0, "gen": 0, "entries": 0,
                      "full": {"min_separation": 0.02}})
    rows = s.load_history()
    assert rows[0]["v"] == 0 and rows[0]["full"]["min_separation"] == 0.02


def test_versions_increase_and_latest_reports_the_last(tmp_path):
    s = _store(tmp_path)
    for v in range(3):
        s.append_history({"v": v, "ts": float(v), "gen": v, "entries": 0,
                          "changed": {}})
    assert s.latest_version() == 2
    assert [r["v"] for r in s.load_history()] == [0, 1, 2]


def test_the_history_lives_beside_goals_not_under_the_layout(tmp_path):
    """One archive holds every brain layout and they share one settings file."""
    s = _store(tmp_path)
    assert s.history_path.parent == s.goals_path.parent
    assert s.history_path.parent != s.root


def test_a_torn_trailing_line_costs_one_row_not_the_file(tmp_path):
    s = _store(tmp_path)
    s.append_history({"v": 0, "ts": 1.0, "gen": 0, "entries": 0, "full": {}})
    with open(s.history_path, "a", encoding="utf-8") as fh:
        fh.write('{"v": 1, "ts"')
    assert len(s.load_history()) == 1


def test_a_missing_history_file_is_an_empty_list(tmp_path):
    assert _store(tmp_path).load_history() == []


def test_a_disabled_store_writes_nothing_and_does_not_raise(tmp_path):
    """A disk problem must never stop the search."""
    s = _store(tmp_path)
    s.enabled = False
    s.append_history({"v": 0, "ts": 1.0, "gen": 0, "entries": 0, "full": {}})
    assert s.load_history() == []


# ---- replay -------------------------------------------------------------

def test_replay_folds_diffs_onto_the_base():
    from services.settings_history import replay

    rows = [{"v": 0, "full": {"a": 1, "b": 2}},
            {"v": 1, "changed": {"a": [1, 5]}},
            {"v": 2, "changed": {"b": [2, 7]}}]
    assert replay(rows) == {"a": 5, "b": 7}


def test_replay_of_nothing_is_empty():
    from services.settings_history import replay

    assert replay([]) == {}


# ---- the version an entry was admitted under ----------------------------

def _archive(tmp_path):
    from services.archive import Archive

    return Archive(store=_store(tmp_path))


def test_the_first_record_writes_version_zero_with_the_full_block(tmp_path):
    a = _archive(tmp_path)
    assert a.record_settings({"min_separation": 0.02}, gen=0) == 0
    rows = a.store.load_history()
    assert rows[0]["full"]["min_separation"] == 0.02
    assert rows[0]["full"]["encoder"] == a.encoder


def test_an_unchanged_generation_writes_nothing(tmp_path):
    a = _archive(tmp_path)
    a.record_settings({"min_separation": 0.02}, gen=0)
    assert a.record_settings({"min_separation": 0.02}, gen=1) == 0
    assert len(a.store.load_history()) == 1


def test_a_change_bumps_the_version_and_records_the_move(tmp_path):
    a = _archive(tmp_path)
    a.record_settings({"min_separation": 0.02}, gen=0)
    assert a.record_settings({"min_separation": 0.03}, gen=7) == 1
    row = a.store.load_history()[1]
    assert row["changed"]["min_separation"] == [0.02, 0.03]
    assert row["gen"] == 7


def test_a_reopened_archive_diffs_against_its_own_history(tmp_path):
    """The version in force is what the LOG says, not what this session
    happens to have written - otherwise reopening rewrites version 0."""
    a = _archive(tmp_path)
    a.record_settings({"min_separation": 0.02}, gen=0)
    a.record_settings({"min_separation": 0.03}, gen=5)

    b = _archive(tmp_path)
    assert b.cfg_version == 1
    assert b.record_settings({"min_separation": 0.03}, gen=9) == 1
    assert len(b.store.load_history()) == 2


def test_a_storeless_archive_records_nothing(tmp_path):
    from services.archive import Archive

    a = Archive(store=None)
    assert a.record_settings({"min_separation": 0.02}, gen=0) == 0


def test_an_entry_row_without_a_version_reads_as_zero():
    """Every entry admitted before this existed."""
    from services.archive import ArchiveEntry

    e = ArchiveEntry(id=1, novelty=0.5, liveness=0.1, pinned=False,
                     source="expansion", spec="", goal="", run_id="",
                     gen=0, tile=0, ts=0.0)
    assert e.cfg == 0
