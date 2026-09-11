"""Archives written before the colour outputs are backed up, then widened and
renamed in place, and a second launch changes nothing."""
import json

import numpy as np

from services.archive_io import FORMAT_VERSION
from services.brains import (COLOUR_GAIN, axial_index, colour_index, get,
                             with_colour_outputs)
from services.colour_migration import migrate_colour_outputs

LEGACY = with_colour_outputs(get("gabor").layout_from_settings({}), 0)
NEW = with_colour_outputs(LEGACY, 2)


def _legacy_archive(root, name="a", n=3):
    archive = root / name
    d = archive / LEGACY.signature()
    (d / "thumbs").mkdir(parents=True)
    rng = np.random.default_rng(0)
    brains = np.stack([get("gabor").random(rng, LEGACY) for _ in range(n)])
    (d / "index.jsonl").write_text(
        "".join(json.dumps({"id": i}) + "\n" for i in range(n)))
    with open(d / "vectors.npz", "wb") as fh:
        np.savez(fh, format_version=np.array(FORMAT_VERSION),
                 ids=np.arange(n, dtype=np.int64),
                 embeddings=np.zeros((n, 4), np.float16),
                 brains=brains.astype(np.float32),
                 physics=np.zeros((n, 8), np.float32),
                 novelty=np.zeros(n, np.float32),
                 novelty_n=np.array(n, dtype=np.int64))
    (archive / "settings.json").write_text(
        json.dumps({"layout_signature": LEGACY.signature(), "k": 5}))
    (archive / "layouts.jsonl").write_text(json.dumps(
        {"parent": LEGACY.signature(), "child": "lenia-n12", "op": "x"}) + "\n")
    return archive, brains


def test_a_legacy_archive_is_backed_up_widened_and_renamed(tmp_path):
    archive, brains = _legacy_archive(tmp_path / "archives")
    legacy_root = tmp_path / "archives_legacy"

    assert migrate_colour_outputs(tmp_path / "archives", legacy_root) == 1

    assert (legacy_root / "a" / LEGACY.signature() / "vectors.npz").is_file()
    assert not (archive / LEGACY.signature()).exists()
    with np.load(archive / NEW.signature() / "vectors.npz") as z:
        got = z["brains"]
        assert int(z["novelty_n"]) == 3
    assert got.shape == (3, NEW.length)
    hue, sat = colour_index(NEW)
    keep = np.setdiff1d(np.arange(NEW.length), np.concatenate([hue, sat]))
    np.testing.assert_array_equal(got[:, keep], brains)
    np.testing.assert_array_equal(got[:, hue],
                                  COLOUR_GAIN * got[:, axial_index(NEW)])
    assert not got[:, sat].any()

    settings = json.loads((archive / "settings.json").read_text())
    assert settings == {"layout_signature": NEW.signature(), "k": 5}
    row = json.loads((archive / "layouts.jsonl").read_text())
    assert row["parent"] == NEW.signature()
    assert row["child"] == "lenia-n12+c2"


def test_a_second_launch_changes_nothing(tmp_path):
    _legacy_archive(tmp_path / "archives")
    legacy_root = tmp_path / "archives_legacy"
    migrate_colour_outputs(tmp_path / "archives", legacy_root)
    stamp = (legacy_root / "a" / "settings.json").read_text()

    assert migrate_colour_outputs(tmp_path / "archives", legacy_root) == 0
    assert (legacy_root / "a" / "settings.json").read_text() == stamp


def test_a_taken_name_is_skipped_not_merged(tmp_path):
    archive, brains = _legacy_archive(tmp_path / "archives")
    (archive / NEW.signature()).mkdir()
    (archive / NEW.signature() / "index.jsonl").write_text("")

    assert migrate_colour_outputs(tmp_path / "archives",
                                  tmp_path / "archives_legacy") == 0
    with np.load(archive / LEGACY.signature() / "vectors.npz") as z:
        np.testing.assert_array_equal(z["brains"], brains)


def test_the_migrated_archive_loads_every_entry(tmp_path):
    from services.archive import Archive
    from services.archive_io import ArchiveStore

    archive, _brains = _legacy_archive(tmp_path / "archives")
    migrate_colour_outputs(tmp_path / "archives", tmp_path / "archives_legacy")

    store = ArchiveStore(archive, NEW)
    loaded = Archive(store=store, layout=NEW, dim=4)
    got, dropped = loaded.load_from_store()
    assert (got, dropped) == (3, 0)
    assert loaded.brain_at(0).shape == (NEW.length,)
    loaded.close()


def test_a_legacy_directory_left_behind_is_not_loaded(tmp_path):
    """A skipped directory's genomes are the old width; letting its entries
    in would hand a colourless layout to the GPU the moment one is adopted."""
    from services.archive import Archive
    from services.archive_io import ArchiveStore

    archive, _brains = _legacy_archive(tmp_path / "archives")
    (archive / NEW.signature()).mkdir()
    (archive / NEW.signature() / "index.jsonl").write_text("")
    migrate_colour_outputs(tmp_path / "archives", tmp_path / "archives_legacy")

    store = ArchiveStore(archive, NEW)
    loaded = Archive(store=store, layout=NEW, dim=4)
    assert loaded.load_from_store() == (0, 0)
    assert LEGACY.signature() not in {loaded.layout_at(i)
                                      for i in range(len(loaded))}
    loaded.close()


def test_no_archives_folder_is_not_an_error(tmp_path):
    assert migrate_colour_outputs(tmp_path / "nope", tmp_path / "legacy") == 0
    assert not (tmp_path / "legacy").exists()
