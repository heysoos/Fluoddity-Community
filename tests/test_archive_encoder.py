"""An archive name identifies one embedding space, permanently.

A stored vector is only comparable to others from the same encoder, so the
encoder is pinned when the archive is made and never changes. Every archive
written before the choice existed is clip-b32, which is what a missing file
means.
"""
from services.archive_io import ArchiveStore
from services.vision_models import DEFAULT_KEY


def _store(tmp_path, name="arc"):
    return ArchiveStore(tmp_path / name)


def test_an_archive_with_no_file_reads_as_the_original_encoder(tmp_path):
    assert _store(tmp_path).encoder == DEFAULT_KEY


def test_the_encoder_is_written_once(tmp_path):
    s = _store(tmp_path)
    assert s.save_encoder("siglip2-b16") is True
    assert s.encoder == "siglip2-b16"


def test_a_second_write_is_refused_and_changes_nothing(tmp_path):
    """A second write would reinterpret every entry already filed here."""
    s = _store(tmp_path)
    s.save_encoder("siglip2-b16")
    assert s.save_encoder("clip-l14") is False
    assert s.encoder == "siglip2-b16"


def test_rewriting_the_same_encoder_is_still_refused(tmp_path):
    """Idempotent in effect, but the file is never rewritten - the created
    timestamp is part of the record."""
    s = _store(tmp_path)
    s.save_encoder("clip-b16")
    stamp = s.encoder_path.read_text(encoding="utf-8")
    assert s.save_encoder("clip-b16") is False
    assert s.encoder_path.read_text(encoding="utf-8") == stamp


def test_an_unknown_encoder_is_never_written(tmp_path):
    s = _store(tmp_path)
    assert s.save_encoder("not-a-model") is False
    assert not s.encoder_path.exists()
    assert s.encoder == DEFAULT_KEY


def test_a_corrupt_file_reads_as_the_default_rather_than_raising(tmp_path):
    """A disk problem must never stop the search."""
    s = _store(tmp_path)
    s.encoder_path.write_text("{ this is not json", encoding="utf-8")
    assert s.encoder == DEFAULT_KEY


def test_a_file_naming_an_unknown_encoder_reads_as_the_default(tmp_path):
    """A key removed from the registry must not take the archive down with it."""
    s = _store(tmp_path)
    s.encoder_path.write_text('{"encoder": "gone-away"}', encoding="utf-8")
    assert s.encoder == DEFAULT_KEY


def test_the_encoder_lives_beside_goals_not_under_the_layout(tmp_path):
    """One archive holds every brain layout, and they share an encoder."""
    s = _store(tmp_path)
    s.save_encoder("clip-b16")
    assert s.encoder_path.parent == s.goals_path.parent
    assert s.encoder_path.parent != s.root


def test_two_layouts_of_one_archive_see_the_same_encoder(tmp_path):
    """Switching brain must not switch embedding space."""
    a = ArchiveStore(tmp_path / "arc", signature="fourier-n10")
    a.save_encoder("clip-b16")
    b = ArchiveStore(tmp_path / "arc", signature="gabor-n7")
    assert b.encoder == "clip-b16"


def test_two_archives_pin_independently(tmp_path):
    _store(tmp_path, "one").save_encoder("clip-b16")
    _store(tmp_path, "two").save_encoder("siglip2-b16")
    assert _store(tmp_path, "one").encoder == "clip-b16"
    assert _store(tmp_path, "two").encoder == "siglip2-b16"
