"""A GENOME belongs to one brain layout; an archive holds every brain.

A stored genome is a bare float vector, and what those floats mean is decided
entirely by the layout that produced them. Mixing two layouts in one DIRECTORY
does not error - it silently decodes a Gabor brain through the Fourier squash
and scores the result, which is worse than a crash because the numbers look
plausible all the way through. So the signature is part of the path.

The archive above those directories is one archive: novelty, separation and
admission all run on the CLIP embedding, and none of them reads a brain.
"""
import numpy as np
import pytest

from services.archive import Archive
from services.archive_io import ArchiveStore, migrate_archive
from services.archive_library import list_archives
from services.brains import BrainLayout, default_layout


def test_store_root_is_the_layout_signature(tmp_path):
    store = ArchiveStore(tmp_path, default_layout())
    assert store.root.name == "fourier-n10"
    assert store.root.parent == tmp_path


def test_two_layouts_get_separate_directories(tmp_path):
    a = ArchiveStore(tmp_path, BrainLayout("gabor", (12,), 168))
    b = ArchiveStore(tmp_path, BrainLayout("gabor", (8,), 112))
    c = ArchiveStore(tmp_path, BrainLayout("lenia", (12,), 120))
    assert len({a.root, b.root, c.root}) == 3


def test_archive_brain_column_is_flat_and_follows_the_layout(tmp_path):
    layout = BrainLayout("gabor", (12,), 168)
    arc = Archive(store=ArchiveStore(tmp_path, layout), layout=layout)
    assert arc._brain.shape[1:] == (168,), (
        "the brain column must be flat and layout-wide; a hardcoded (10, 8) "
        "cannot hold any other modality"
    )


def test_default_layout_still_gives_the_fourier_width(tmp_path):
    arc = Archive(store=ArchiveStore(tmp_path, default_layout()))
    assert arc._brain.shape[1:] == (80,)


# ---- migration -------------------------------------------------------------

def _write_legacy(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.jsonl").write_text('{"id": 0}\n', encoding="utf-8")
    (d / "thumbs").mkdir(exist_ok=True)
    (d / "thumbs" / "000000.jpg").write_bytes(b"x")
    with open(d / "vectors.npz", "wb") as fh:
        np.savez(fh, format_version=np.array(1), ids=np.array([0]),
                 embeddings=np.zeros((1, 512), np.float16),
                 brains=np.zeros((1, 10, 8), np.float32),
                 physics=np.zeros((1, 8), np.float32))


def test_a_legacy_archive_moves_into_the_fourier_signature(tmp_path):
    legacy = tmp_path / "default"
    _write_legacy(legacy)

    moved = migrate_archive(legacy)

    assert moved == legacy / "fourier-n10"
    assert (moved / "index.jsonl").read_text() == '{"id": 0}\n'
    assert (moved / "vectors.npz").is_file()
    assert (moved / "thumbs" / "000000.jpg").is_file()
    # Nothing may be left at the top level, or the next open sees both.
    assert not (legacy / "index.jsonl").exists()
    assert not (legacy / "vectors.npz").exists()


def test_migration_is_idempotent(tmp_path):
    legacy = tmp_path / "default"
    _write_legacy(legacy)
    first = migrate_archive(legacy)
    assert migrate_archive(legacy) is None
    assert (first / "index.jsonl").is_file()


def test_migration_of_an_empty_or_new_archive_does_nothing(tmp_path):
    empty = tmp_path / "fresh"
    empty.mkdir()
    assert migrate_archive(empty) is None


def test_migration_refuses_to_clobber_an_existing_signature_dir(tmp_path):
    """Restoring an old backup beside an already-migrated archive must not
    swallow it - the same rule utilities.paths.migrate_legacy_archive follows."""
    legacy = tmp_path / "default"
    _write_legacy(legacy)
    (legacy / "fourier-n10").mkdir()
    (legacy / "fourier-n10" / "index.jsonl").write_text("keep\n", encoding="utf-8")

    assert migrate_archive(legacy) is None
    assert (legacy / "fourier-n10" / "index.jsonl").read_text() == "keep\n"
    assert (legacy / "index.jsonl").is_file()      # left where it was


# ---- the browser listing ---------------------------------------------------

def test_the_listing_counts_entries_inside_signature_dirs(tmp_path):
    """list_archives walks <root>/<name>/; the entries now live one level
    deeper, and counting only the top level reports every archive as empty."""
    _write_legacy(tmp_path / "default")
    migrate_archive(tmp_path / "default")

    rows = {r["name"]: r for r in list_archives(tmp_path)}
    assert rows["default"]["entries"] == 1


def test_the_listing_sums_across_layouts(tmp_path):
    """One named archive can hold several layouts side by side. The folder
    listing reports what the FOLDER holds; the live count for the active layout
    comes from the Archive itself."""
    root = tmp_path / "mixed"
    for sig, n in (("fourier-n10", 2), ("gabor-n12", 3)):
        d = root / sig
        d.mkdir(parents=True)
        with open(d / "vectors.npz", "wb") as fh:
            np.savez(fh, format_version=np.array(1), ids=np.arange(n),
                     embeddings=np.zeros((n, 512), np.float16),
                     brains=np.zeros((n, 8), np.float32),
                     physics=np.zeros((n, 8), np.float32))

    rows = {r["name"]: r for r in list_archives(tmp_path)}
    assert rows["mixed"]["entries"] == 5


# ---- round trip ------------------------------------------------------------

def test_legacy_three_dim_brains_load_as_flat_rows(tmp_path):
    """Archives written before the rewrite stored (N, 10, 8). They must read
    back as 80-float rows rather than being quarantined."""
    layout = default_layout()
    store = ArchiveStore(tmp_path, layout)
    brains = np.arange(80, dtype=np.float32).reshape(1, 10, 8)
    store.flush_vectors(np.array([0]), np.zeros((1, 512), np.float32),
                        brains, np.zeros((1, 8), np.float32))
    arrays = store.load()[1]
    assert arrays["brains"].reshape(1, -1).shape == (1, 80)


@pytest.mark.parametrize("layout", [
    default_layout(),
    BrainLayout("gabor", (12,), 168),
    BrainLayout("mlp", (16, 0), 148),
])
def test_a_brain_survives_a_store_round_trip(tmp_path, layout):
    store = ArchiveStore(tmp_path / layout.modality, layout)
    want = np.arange(layout.length, dtype=np.float32).reshape(1, -1)
    store.flush_vectors(np.array([0]), np.zeros((1, 512), np.float32),
                        want, np.zeros((1, 8), np.float32))
    got = store.load()[1]["brains"].reshape(1, -1)
    assert np.array_equal(got, want)


# ---- what belongs to the archive rather than to one of its brains ---------

def _sig_dir(root, sig="fourier-n10"):
    d = root / sig
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    (d / "index.jsonl").write_text('{"id": 0}\n', encoding="utf-8")
    return d


def test_goals_settings_and_runs_move_up_out_of_a_layout(tmp_path):
    """They belong to the ARCHIVE. One archive holds every brain, so a copy
    filed beside each one is several answers to a question that has one."""
    root = tmp_path / "arc"
    d = _sig_dir(root)
    (d / "goals.json").write_text("[]", encoding="utf-8")
    (d / "settings.json").write_text('{"grid": 8}', encoding="utf-8")
    (d / "runs").mkdir()
    (d / "runs" / "r1.json").write_text('"x"', encoding="utf-8")

    migrate_archive(root)

    assert (root / "goals.json").is_file()
    assert (root / "settings.json").read_text(encoding="utf-8") == '{"grid": 8}'
    assert (root / "runs" / "r1.json").is_file()
    for name in ("goals.json", "settings.json", "runs"):
        assert not (d / name).exists(), f"{name} was copied, not moved"


def test_the_store_reads_them_from_the_archive_not_the_layout(tmp_path):
    """The paths and the migration have to agree, or it moves them somewhere
    nothing looks."""
    s = ArchiveStore(tmp_path / "arc", default_layout())
    assert s.goals_path.parent == s.base
    assert s.settings_path.parent == s.base
    assert s.run_config_path("r1").parent.parent == s.base
    assert s.index_path.parent == s.root
    assert s.vectors_path.parent == s.root
    assert s.thumb_path("000000.jpg").parent.parent == s.root


def test_the_busiest_layout_s_settings_are_the_ones_the_archive_keeps(tmp_path,
                                                                      capsys):
    """NOT the newest. Switching brain creates an empty sibling and writes its
    settings, so the directories with nothing in them are reliably the most
    recently touched - mtime alone hands the archive the settings of a layout
    that was never worked in."""
    import os
    import time

    root = tmp_path / "arc"
    worked_in = _sig_dir(root, "fourier-n10")
    worked_in.joinpath("index.jsonl").write_text(
        "".join('{"id": %d}\n' % i for i in range(50)), encoding="utf-8")
    passed_through = _sig_dir(root, "gabor-n12")     # one row, created later
    (worked_in / "settings.json").write_text('{"grid": 4}', encoding="utf-8")
    (passed_through / "settings.json").write_text('{"grid": 8}', encoding="utf-8")
    past = time.time() - 600
    os.utime(worked_in, (past, past))

    migrate_archive(root)

    assert (root / "settings.json").read_text(encoding="utf-8") == '{"grid": 4}'
    assert (passed_through / "settings.json").is_file(), "the loser stays put"
    assert "stay where they are" in capsys.readouterr().out


def test_a_legacy_archive_lands_with_its_goals_at_the_top(tmp_path):
    """Both directions in one pass: entries down, goals up."""
    legacy = tmp_path / "default"
    _write_legacy(legacy)
    (legacy / "goals.json").write_text("[]", encoding="utf-8")

    migrate_archive(legacy)

    assert (legacy / "fourier-n10" / "index.jsonl").is_file()
    assert (legacy / "goals.json").is_file()
    assert not (legacy / "fourier-n10" / "goals.json").exists()
