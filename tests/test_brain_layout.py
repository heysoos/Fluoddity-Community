import numpy as np
import pytest

from services.brains import MAX_BRAIN_FLOATS, BrainLayout, default_layout
from services.genome_spec import spec_for


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


def test_spec_for_layout_sizes_the_brain_block():
    layout = default_layout()
    spec = spec_for(layout)
    assert spec.dim == layout.length
    assert spec.signature() == "brain:80"


def test_spec_decode_keeps_the_fourier_centre_shape():
    """Fourier brains stay (N, 8) out of GenomeSpec.decode. Callers across the
    app index them by centre; handing back a flat vector fails far from here."""
    layout = default_layout()
    spec = spec_for(layout)
    parts = spec.decode(np.zeros(layout.length, dtype=np.float32))
    assert parts["brain"].shape == (layout.shape[0], 8)
