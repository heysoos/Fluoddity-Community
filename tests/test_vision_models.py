"""The registry is the one home for what differs between encoders."""
import pytest

from services.vision_models import DEFAULT_KEY, REGISTRY, VisionModel, get


def test_the_default_is_the_encoder_every_existing_archive_used():
    assert DEFAULT_KEY == "clip-b32"
    assert DEFAULT_KEY in REGISTRY


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_entry_declares_its_fixed_facts(key):
    m = REGISTRY[key]
    assert m.key == key, "the dict key and the entry must agree"
    assert m.dim > 0 and m.px > 0 and m.context > 0
    assert len(m.mean) == 3 and len(m.std) == 3
    assert "vision_model_fp16.onnx" in m.files
    assert "text_model_fp16.onnx" in m.files
    assert "tokenizer.json" in m.files


def test_keys_are_stable_identifiers_not_paths():
    for key in REGISTRY:
        assert "/" not in key and "\\" not in key


def test_get_refuses_an_unknown_key_by_name():
    with pytest.raises(KeyError, match="nope"):
        get("nope")


def test_the_known_encoders_are_present():
    assert set(REGISTRY) == {"clip-b32", "clip-b16", "siglip2-b16", "clip-l14"}


def test_clip_b32_is_already_calibrated():
    """Its three values are the ones the app shipped with."""
    m = get("clip-b32")
    assert m.text_logit_scale == 100.0
    assert m.image_logit_scale == 30.0
    assert m.default_min_separation == 0.02


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_entry_is_calibrated(key):
    """A model shipping an uncalibrated scale would silently saturate the
    landscape - a wrong scale never raises. See CLAUDE.md."""
    assert REGISTRY[key].calibrated, f"{key} was never calibrated"


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_the_separation_slider_has_room_above_the_default(key):
    m = REGISTRY[key]
    assert m.separation_slider_max > m.default_min_separation


def test_clip_b32s_slider_range_is_unchanged():
    """The existing 0..0.05 track is what the generalisation has to
    reproduce."""
    assert get("clip-b32").separation_slider_max == pytest.approx(0.05)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_the_scales_are_in_the_range_the_measurement_produced(key):
    """A typo in a calibrated value is invisible at runtime."""
    m = REGISTRY[key]
    assert 10.0 <= m.image_logit_scale <= 50.0
    assert 90.0 <= m.text_logit_scale <= 250.0
    assert 0.005 <= m.default_min_separation <= 0.10


def test_the_text_scale_stays_well_above_the_image_scale():
    """They differ by more than 3x for a reason - image-image similarity sits
    in a different regime. If a calibration ever collapses them, the
    per-modality split is dead weight."""
    for m in REGISTRY.values():
        assert m.text_logit_scale / m.image_logit_scale > 3.0


def test_a_vision_model_is_frozen():
    """A key is written to disk; a mutable entry could drift from it."""
    with pytest.raises(Exception):
        get("clip-b32").dim = 1
