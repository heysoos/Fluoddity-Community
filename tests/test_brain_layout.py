import pytest

from services.brains import MAX_BRAIN_FLOATS, BrainLayout


def test_signature_is_modality_and_shape():
    assert BrainLayout("fourier", (10,), 80).signature() == "fourier-n10"
    assert BrainLayout("mlp", (16, 0), 148).signature() == "mlp-n16-a0"


def test_signature_distinguishes_layouts():
    a = BrainLayout("gabor", (12,), 168)
    b = BrainLayout("gabor", (8,), 112)
    assert a.signature() != b.signature()


def test_layout_rejects_over_budget():
    with pytest.raises(ValueError, match="exceeds MAX_BRAIN_FLOATS"):
        BrainLayout("fourier", (100,), 800)


def test_max_brain_floats_is_512():
    assert MAX_BRAIN_FLOATS == 512
