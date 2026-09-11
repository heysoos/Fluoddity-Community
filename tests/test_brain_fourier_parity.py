"""The new decode must reproduce the old one exactly, or every archived
genome silently becomes a different creature."""
import numpy as np

from services.brains import BrainLayout, with_colour_outputs
from services.brains.fourier import FourierModality

# The legacy implementation, frozen here on purpose. Task 3 rewrites
# genome_spec.decode to delegate to the modality, so importing it would make
# this test compare the modality against itself and silently stop testing.
LEGACY_FREQ_SCALE = 3.0
LEGACY_AMP_SCALE = 1.0


def legacy_decode(z):
    z = np.asarray(z, dtype=np.float32).reshape(20, 4)
    freq = LEGACY_FREQ_SCALE * np.tanh(z[:10])
    amp = LEGACY_AMP_SCALE * np.tanh(z[10:])
    return np.concatenate([freq, amp], axis=1).astype(np.float32)


def test_decode_matches_legacy_bit_for_bit():
    # Motion decode is unchanged; colour and audio are appended floats, so
    # comparing on the colourless, deaf layout isolates the legacy formula.
    rng = np.random.default_rng(0)
    m = FourierModality()
    layout = with_colour_outputs(m.layout_from_settings({}), 0)
    for _ in range(20):
        z = rng.normal(0, 1.5, 80).astype(np.float32)
        got = m.decode(z, layout)
        want = legacy_decode(z).reshape(-1)
        assert np.array_equal(got, want)


def test_default_layout_is_ten_centers_eighty_floats():
    # Every modality-built layout carries colour outputs unless told otherwise.
    layout = FourierModality().layout_from_settings({})
    assert layout == BrainLayout("fourier", (10,), 100, colour_outputs=2)
    assert layout.signature() == "fourier-n10+c2"


def test_encode_decode_round_trips():
    rng = np.random.default_rng(1)
    m = FourierModality()
    layout = m.layout_from_settings({})
    z = rng.normal(0, 0.5, layout.length).astype(np.float32)
    params = m.decode(z, layout)
    back, clamped = m.encode(params, layout)
    assert clamped == 0
    assert np.allclose(back, z, atol=1e-4)


def test_random_matches_the_hand_tuned_prior_scale():
    """random_genome biases frequencies low on purpose; the modality's prior
    must keep that. Measured mean |freq| of the legacy generator is 0.832."""
    m = FourierModality()
    layout = m.layout_from_settings({})
    rng = np.random.default_rng(7)
    freqs = []
    for _ in range(500):
        p = m.random(rng, layout).reshape(10, -1)
        freqs.append(np.abs(p[:, :4]))
    assert 0.75 < float(np.mean(freqs)) < 0.92


def test_larger_layout_scales_length():
    # 24 centres * (8 motion + 2 colour) floats, no audio inputs.
    m = FourierModality()
    assert m.layout_from_settings({"centers": 24}).length == 240
