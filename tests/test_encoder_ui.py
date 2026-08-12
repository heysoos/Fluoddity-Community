"""The encoder pickers, and what they are allowed to change.

Explore's is locked once the archive holds anything - its vectors are in that
encoder's space. Auto's is free, because a prompt run stores nothing.
"""
import pytest

from services.vision_models import REGISTRY, get
from state.archive_state import PERSISTED_FIELDS, ArchiveState
from state.auto_tournament_state import AutoTournamentState


def test_the_archive_carries_an_encoder_and_persists_it():
    assert ArchiveState().encoder_key == "clip-b32"
    assert "encoder_key" in PERSISTED_FIELDS


def test_the_view_buffers_are_not_persisted():
    """Two thirds of ArchiveState is one-shots and view state; persisting a
    stale entry count would lock a fresh archive's encoder combo."""
    for name in ("archive_entry_count", "history_rows",
                 "request_history_reload"):
        assert name not in PERSISTED_FIELDS


def test_auto_carries_its_own_encoder_choice():
    assert AutoTournamentState().model_key == "clip-b32"


def test_a_fresh_archive_state_names_a_real_encoder():
    assert ArchiveState().encoder_key in REGISTRY


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_the_separation_track_reaches_every_encoders_default(key):
    """A track fixed at clip-b32's 0..0.05 puts a wider encoder's own default
    at the top of it, so half its useful range is unreachable."""
    m = get(key)
    assert m.separation_slider_max >= m.default_min_separation
    assert m.separation_slider_max > 0.0


def test_the_widest_encoder_needs_a_longer_track_than_the_narrowest():
    """If this ever collapses, the per-encoder track is dead weight."""
    widest = max(REGISTRY.values(), key=lambda m: m.default_min_separation)
    narrowest = min(REGISTRY.values(), key=lambda m: m.default_min_separation)
    assert widest.separation_slider_max > narrowest.separation_slider_max


def test_every_encoder_label_is_distinct():
    """The combo shows labels; two encoders sharing one would be unpickable."""
    labels = [m.label for m in REGISTRY.values()]
    assert len(set(labels)) == len(labels)
