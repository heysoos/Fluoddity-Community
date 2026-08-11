import json

import numpy as np

from services.archive_io import FORMAT_VERSION, ArchiveStore


def _arrays(n=3, dim=4):
    return (
        np.arange(n, dtype=np.int64),
        np.eye(n, dim, dtype=np.float32),
        np.zeros((n, 10, 8), dtype=np.float32),
        np.ones((n, 8), dtype=np.float32),
    )


def test_creates_the_directory_layout(tmp_path):
    s = ArchiveStore(tmp_path / "archive")
    assert s.enabled
    assert (tmp_path / "archive" / "thumbs").is_dir()
    s.close()


def test_append_index_writes_one_json_object_per_line(tmp_path):
    s = ArchiveStore(tmp_path)
    s.append_index({"id": 0, "novelty": 0.5})
    s.append_index({"id": 1, "novelty": 0.6})
    s.close()
    lines = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["id"] for x in lines] == [0, 1]


def test_vectors_roundtrip(tmp_path):
    s = ArchiveStore(tmp_path)
    ids, emb, brains, phys = _arrays()
    s.flush_vectors(ids, emb, brains, phys)
    for i in range(3):
        s.append_index({"id": int(i)})
    s.close()

    s2 = ArchiveStore(tmp_path)
    rows, arrays = s2.load()
    assert [r["id"] for r in rows] == [0, 1, 2]
    assert np.array_equal(arrays["ids"], ids)
    assert np.allclose(arrays["embeddings"].astype(np.float32), emb, atol=1e-3)
    assert arrays["embeddings"].dtype == np.float16, "embeddings are stored fp16"
    assert arrays["brains"].shape == (3, 10, 8)
    assert arrays["physics"].shape == (3, 8)
    s2.close()


def test_flush_leaves_no_temp_file(tmp_path):
    s = ArchiveStore(tmp_path)
    s.flush_vectors(*_arrays())
    assert list(tmp_path.glob("*.tmp")) == []
    s.close()


def test_a_failed_flush_leaves_the_previous_vectors_intact(tmp_path, monkeypatch):
    """A crash mid-write must never destroy a good archive."""
    s = ArchiveStore(tmp_path)
    ids, emb, brains, phys = _arrays()
    s.flush_vectors(ids, emb, brains, phys)
    good = (tmp_path / "vectors.npz").read_bytes()

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("numpy.savez", boom)
    s.flush_vectors(ids, emb * 0, brains, phys)
    assert (tmp_path / "vectors.npz").read_bytes() == good
    s.close()


def test_a_corrupt_vectors_file_is_quarantined_not_overwritten(tmp_path):
    (tmp_path / "index.jsonl").write_text('{"id": 0}\n', encoding="utf-8")
    (tmp_path / "vectors.npz").write_bytes(b"not an npz at all")
    s = ArchiveStore(tmp_path)
    rows, arrays = s.load()
    assert rows == [{"id": 0}]
    assert arrays == {}
    assert not (tmp_path / "vectors.npz").exists()
    assert list(tmp_path.glob("vectors.npz.bad-*")), "the bad file must be kept"
    s.close()


def test_a_format_version_mismatch_is_quarantined(tmp_path):
    ids, emb, brains, phys = _arrays()
    with open(tmp_path / "vectors.npz", "wb") as fh:
        np.savez(fh, format_version=np.array(FORMAT_VERSION + 1), ids=ids,
                 embeddings=emb.astype(np.float16), brains=brains, physics=phys)
    s = ArchiveStore(tmp_path)
    _rows, arrays = s.load()
    assert arrays == {}
    assert list(tmp_path.glob("vectors.npz.bad-*"))
    s.close()


def test_a_torn_trailing_index_line_costs_one_entry_not_the_file(tmp_path):
    (tmp_path / "index.jsonl").write_text(
        '{"id": 0}\n{"id": 1}\n{"id": 2, "nov', encoding="utf-8")
    s = ArchiveStore(tmp_path)
    rows, _ = s.load()
    assert [r["id"] for r in rows] == [0, 1]
    s.close()


def test_load_on_an_empty_directory_is_empty_not_an_error(tmp_path):
    s = ArchiveStore(tmp_path)
    rows, arrays = s.load()
    assert rows == []
    assert arrays == {}
    s.close()


def test_write_thumb_returns_the_filename_and_writes_160px(tmp_path):
    from PIL import Image

    s = ArchiveStore(tmp_path)
    crop = np.full((224, 224, 3), 200, dtype=np.uint8)
    name = s.write_thumb(7, crop)
    assert name == "000007.jpg"
    with Image.open(s.thumb_path(name)) as img:
        assert img.size == (160, 160)
    s.close()


def test_goals_roundtrip(tmp_path):
    s = ArchiveStore(tmp_path)
    items = [{"text": "coral reef", "enabled": True},
             {"text": "lightning", "enabled": False}]
    s.save_goals(items)
    s.close()
    s2 = ArchiveStore(tmp_path)
    assert s2.load_goals() == items
    s2.close()


def test_missing_goals_file_loads_as_empty(tmp_path):
    s = ArchiveStore(tmp_path)
    assert s.load_goals() == []
    s.close()


def test_an_unwritable_root_disables_persistence_without_raising(tmp_path):
    """A disk problem must never block evolution."""
    blocker = tmp_path / "archive"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    s = ArchiveStore(blocker)
    assert s.enabled is False
    # every operation must be a no-op rather than an exception
    s.append_index({"id": 0})
    s.flush_vectors(*_arrays())
    assert s.write_thumb(0, np.zeros((8, 8, 3), np.uint8)) == ""
    assert s.load_goals() == []
    s.close()


# ---- the physics a run was carried out under -------------------------------

def test_a_run_config_round_trips(tmp_path):
    s = ArchiveStore(tmp_path)
    assert s.save_run_config("20260811-021952", '{"physics": {"drag": 0.5}}')
    assert s.load_run_config("20260811-021952") == '{"physics": {"drag": 0.5}}'


def test_a_run_that_was_never_recorded_reads_as_none(tmp_path):
    s = ArchiveStore(tmp_path)
    assert s.load_run_config("20260810-010606") is None
    assert s.load_run_config("") is None


def test_a_run_id_names_one_set_of_physics(tmp_path):
    """A resume writing a second config would silently reinterpret every entry
    already filed under that run id."""
    s = ArchiveStore(tmp_path)
    s.save_run_config("r1", '"first"')
    assert s.save_run_config("r1", '"second"')      # succeeds, changes nothing
    assert s.load_run_config("r1") == '"first"'


def test_a_run_id_cannot_write_outside_the_archive(tmp_path):
    s = ArchiveStore(tmp_path)
    s.save_run_config("../../escaped", '"x"')
    assert not (tmp_path.parent.parent / "escaped.json").exists()
    assert list((tmp_path / "runs").glob("*.json"))


def test_run_configs_do_not_disturb_the_index_or_vectors(tmp_path):
    """runs/ is additive: an archive written before it existed still loads."""
    s = ArchiveStore(tmp_path)
    s.save_run_config("r1", '"x"')
    rows, arrays = s.load()
    assert rows == [] and arrays == {}
