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


def _stocked(tmp_path, **kw):
    """A store whose layout directory exists, as one admission would leave it.

    It is made on first WRITE now, and the tests below hand-craft the files an
    earlier run would have written into it."""
    s = ArchiveStore(tmp_path, **kw)
    s._ensure_root()
    return s


def test_creates_the_directory_layout(tmp_path):
    """The store owns its own directory - one level below the named archive,
    under the brain layout's signature. Tests ask it for paths rather than
    rebuilding them, so the signature can change without touching them.

    The ARCHIVE is made on construction; the layout directory waits for the
    first entry, so a brain that was only visited leaves nothing behind."""
    s = ArchiveStore(tmp_path / "archive")
    assert s.enabled
    assert s.root.parent == tmp_path / "archive"
    assert (tmp_path / "archive").is_dir()
    assert not s.root.exists()
    s.append_index({"id": 0})
    assert (s.root / "thumbs").is_dir()
    s.close()


def test_append_index_writes_one_json_object_per_line(tmp_path):
    s = ArchiveStore(tmp_path)
    s.append_index({"id": 0, "novelty": 0.5})
    s.append_index({"id": 1, "novelty": 0.6})
    s.close()
    lines = s.index_path.read_text(encoding="utf-8").splitlines()
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
    assert list(s.root.glob("*.tmp")) == []
    s.close()


def test_a_failed_flush_leaves_the_previous_vectors_intact(tmp_path, monkeypatch):
    """A crash mid-write must never destroy a good archive."""
    s = ArchiveStore(tmp_path)
    ids, emb, brains, phys = _arrays()
    s.flush_vectors(ids, emb, brains, phys)
    good = s.vectors_path.read_bytes()

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("numpy.savez", boom)
    s.flush_vectors(ids, emb * 0, brains, phys)
    assert s.vectors_path.read_bytes() == good
    s.close()


def test_a_corrupt_vectors_file_is_quarantined_not_overwritten(tmp_path):
    s = _stocked(tmp_path)
    s.index_path.write_text('{"id": 0}\n', encoding="utf-8")
    s.vectors_path.write_bytes(b"not an npz at all")
    rows, arrays = s.load()
    assert rows == [{"id": 0}]
    assert arrays == {}
    assert not s.vectors_path.exists()
    assert list(s.root.glob("vectors.npz.bad-*")), "the bad file must be kept"
    s.close()


def test_a_format_version_mismatch_is_quarantined(tmp_path):
    s = _stocked(tmp_path)
    ids, emb, brains, phys = _arrays()
    with open(s.vectors_path, "wb") as fh:
        np.savez(fh, format_version=np.array(FORMAT_VERSION + 1), ids=ids,
                 embeddings=emb.astype(np.float16), brains=brains, physics=phys)
    _rows, arrays = s.load()
    assert arrays == {}
    assert list(s.root.glob("vectors.npz.bad-*"))
    s.close()


def test_a_torn_trailing_index_line_costs_one_entry_not_the_file(tmp_path):
    s = _stocked(tmp_path)
    s.index_path.write_text(
        '{"id": 0}\n{"id": 1}\n{"id": 2, "nov', encoding="utf-8")
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
    # s.base, not s.root: a run's physics belongs to the ARCHIVE, not to one
    # of its brains, so it sits above the signature directory.
    assert list((s.base / "runs").glob("*.json"))


def test_run_configs_do_not_disturb_the_index_or_vectors(tmp_path):
    """runs/ is additive: an archive written before it existed still loads."""
    s = ArchiveStore(tmp_path)
    s.save_run_config("r1", '"x"')
    rows, arrays = s.load()
    assert rows == [] and arrays == {}


# --- a layout that was only VISITED leaves nothing behind --------------------
#
# The signature directory used to be made in __init__, so merely pointing the
# archive at a layout created it, gave it an index.jsonl and a thumbs/ folder,
# and left it there. Dragging the Brain window's layer sliders walks through
# every intermediate stack, so one real archive grew from 15 layout directories
# to 27 without a single admission - and it reads as an archive holding 27
# brains when it holds 11, each one costing an open file handle on load.

def test_visiting_a_layout_creates_no_directory(tmp_path):
    """Constructing a store is not admitting anything to it."""
    s = ArchiveStore(tmp_path / "archive", signature="mlp-n8.8.8-a0.0.0")
    assert s.enabled, "persistence must not be disabled by staying lazy"
    assert not s.root.exists()
    assert (tmp_path / "archive").is_dir(), "the ARCHIVE itself is still made"
    s.close()


def test_the_first_entry_is_what_creates_it(tmp_path):
    s = ArchiveStore(tmp_path / "archive")
    assert not s.root.exists()
    s.append_index({"id": 0, "novelty": 0.5})
    assert s.root.is_dir()
    assert (s.root / "thumbs").is_dir()
    s.close()
    assert json.loads(s.index_path.read_text(encoding="utf-8"))["id"] == 0


def test_vectors_create_it_too(tmp_path):
    s = ArchiveStore(tmp_path / "archive")
    ids, emb, brains, phys = _arrays()
    s.flush_vectors(ids, emb, brains, phys)
    assert s.vectors_path.is_file()
    s.close()


def test_a_thumbnail_creates_it_too(tmp_path):
    s = ArchiveStore(tmp_path / "archive")
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    assert s.write_thumb(0, crop) == "000000.jpg"
    assert s.thumb_path("000000.jpg").is_file()
    s.close()


def test_a_visited_layout_is_not_listed_as_one_of_the_archives_brains(tmp_path):
    """signature_dirs is what load_from_store walks, and what the browser
    counts. A layout nothing was ever admitted to is not one of them."""
    from services.archive_io import signature_dirs

    base = tmp_path / "archive"
    written = ArchiveStore(base, signature="fourier-n10")
    written.append_index({"id": 0, "novelty": 1.0})
    written.close()
    ArchiveStore(base, signature="mlp-n8.8.8-a0.0.0").close()

    assert [d.name for d in signature_dirs(base)] == ["fourier-n10"]


def test_reading_a_layout_that_was_never_written_is_empty_not_an_error(tmp_path):
    """Its directory does not exist at all now, and load() has always had to
    tolerate a missing index."""
    s = ArchiveStore(tmp_path / "archive", signature="lenia-n12")
    rows, arrays = s.load()
    assert rows == []
    assert not arrays
    s.close()
