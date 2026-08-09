"""An archive belongs to ONE brain layout.

A stored genome is a bare float vector; what those floats mean is decided
entirely by the layout that produced them. Mixing two layouts in one archive
does not error - it silently decodes a Gabor brain through the Fourier squash
and scores the result, which is worse than a crash because the numbers look
plausible all the way through.

So the layout signature is part of the path, and every archive is opened
alongside the layout it was written with.
"""
import numpy as np
import pytest

from services.archive import Archive
from services.archive_io import ArchiveStore, migrate_to_signature_dir
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

    moved = migrate_to_signature_dir(legacy)

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
    first = migrate_to_signature_dir(legacy)
    assert migrate_to_signature_dir(legacy) is None
    assert (first / "index.jsonl").is_file()


def test_migration_of_an_empty_or_new_archive_does_nothing(tmp_path):
    empty = tmp_path / "fresh"
    empty.mkdir()
    assert migrate_to_signature_dir(empty) is None


def test_migration_refuses_to_clobber_an_existing_signature_dir(tmp_path):
    """Restoring an old backup beside an already-migrated archive must not
    swallow it - the same rule utilities.paths.migrate_legacy_archive follows."""
    legacy = tmp_path / "default"
    _write_legacy(legacy)
    (legacy / "fourier-n10").mkdir()
    (legacy / "fourier-n10" / "index.jsonl").write_text("keep\n", encoding="utf-8")

    assert migrate_to_signature_dir(legacy) is None
    assert (legacy / "fourier-n10" / "index.jsonl").read_text() == "keep\n"
    assert (legacy / "index.jsonl").is_file()      # left where it was


# ---- the browser listing ---------------------------------------------------

def test_the_listing_counts_entries_inside_signature_dirs(tmp_path):
    """list_archives walks <root>/<name>/; the entries now live one level
    deeper, and counting only the top level reports every archive as empty."""
    _write_legacy(tmp_path / "default")
    migrate_to_signature_dir(tmp_path / "default")

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
