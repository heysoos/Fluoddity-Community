"""An archive refuses to load into a scorer that speaks a different space.

A 512-d CLIP vector and a 768-d SigLIP vector describe the same tile
incompatibly, and at equal width nothing downstream would even notice - which
is why this is a guard rather than a shape check.
"""
from services.archive import Archive
from services.archive_io import ArchiveStore
from services.vision_models import DEFAULT_KEY


def test_the_archive_takes_its_width_from_its_encoder():
    assert Archive(encoder="siglip2-b16")._dim == 768
    assert Archive(encoder="clip-b32")._dim == 512


def test_an_explicit_dim_still_wins():
    """Many call sites build a narrow Archive for speed."""
    assert Archive(encoder="siglip2-b16", dim=16)._dim == 16


def test_the_default_encoder_is_the_original_one():
    assert Archive().encoder == DEFAULT_KEY
    assert Archive()._dim == 512


def test_loading_a_store_under_the_wrong_encoder_loads_nothing(tmp_path):
    s = ArchiveStore(tmp_path / "arc")
    s.save_encoder("siglip2-b16")
    a = Archive(store=s, encoder="clip-b32")
    assert a.load_from_store() == (0, 0)
    assert a.entries == []


def test_the_mismatch_names_both_sides(tmp_path):
    s = ArchiveStore(tmp_path / "arc")
    s.save_encoder("siglip2-b16")
    a = Archive(store=s, encoder="clip-b32")
    a.load_from_store()
    assert "siglip2-b16" in a.encoder_mismatch
    assert "clip-b32" in a.encoder_mismatch


def test_a_matching_store_loads_and_reports_no_mismatch(tmp_path):
    s = ArchiveStore(tmp_path / "arc")
    s.save_encoder("clip-b16")
    a = Archive(store=s, encoder="clip-b16")
    assert a.load_from_store() == (0, 0)
    assert a.encoder_mismatch == ""


def test_an_archive_with_no_pin_loads_under_the_default(tmp_path):
    """Every archive written before the choice existed."""
    s = ArchiveStore(tmp_path / "arc")
    a = Archive(store=s, encoder=DEFAULT_KEY)
    a.load_from_store()
    assert a.encoder_mismatch == ""


def test_a_storeless_archive_never_reports_a_mismatch():
    a = Archive(store=None, encoder="clip-l14")
    assert a.load_from_store() == (0, 0)
    assert a.encoder_mismatch == ""
