"""A settings slider must actually change the brain.

Every non-count setting was declared in settings_schema() and then ignored:
BrainLayout carried only (modality, shape, length), so decode() could not see
them. The sliders moved and nothing happened - which is the worst kind of
broken, because the UI says otherwise.
"""
import numpy as np
import pytest

from services.brains import REGISTRY, BrainLayout, default_layout

# (modality, setting, a value that must change the decode)
SCALE_SETTINGS = [
    ("fourier", "freq_scale", 1.0),
    ("gabor", "freq_scale", 1.0),
    ("gabor", "envelope_width", 0.3),
    ("gabor", "phase_spread", 0.5),
    ("lenia", "mu_scale", 0.5),
    ("lenia", "sigma_max", 0.1),
]


@pytest.mark.parametrize("name,key,value", SCALE_SETTINGS)
def test_a_scale_setting_changes_the_decoded_brain(name, key, value):
    m = REGISTRY[name]
    base = m.layout_from_settings({})
    tuned = m.layout_from_settings({key: value})
    assert base.length == tuned.length, "a scale must not change the width"

    z = np.random.default_rng(0).normal(0, 0.6, base.length).astype(np.float32)
    assert not np.allclose(m.decode(z, base), m.decode(z, tuned), atol=1e-6), (
        f"{name}.{key} is declared in the UI but does nothing"
    )


@pytest.mark.parametrize("name,key,value", SCALE_SETTINGS)
def test_a_scale_setting_round_trips(name, key, value):
    """encode must invert decode UNDER THE SAME LAYOUT, or an archived brain
    re-enters the search as a different creature."""
    m = REGISTRY[name]
    layout = m.layout_from_settings({key: value})
    z = np.random.default_rng(1).normal(0, 0.4, layout.length).astype(np.float32)
    back, clamped = m.encode(m.decode(z, layout), layout)
    assert clamped == 0
    assert np.allclose(back.reshape(-1), z, atol=1e-3)


@pytest.mark.parametrize("name,key,value", SCALE_SETTINGS)
def test_a_scale_does_not_split_the_archive(name, key, value):
    """Scales change what a z MEANS but not how many floats it has, and the
    archive stores DECODED brains - so a creature already in it is unaffected.
    Splitting the archive on a slider move would strand every entry."""
    m = REGISTRY[name]
    base = m.layout_from_settings({})
    tuned = m.layout_from_settings({key: value})
    assert base.signature() == tuned.signature()
    assert base == tuned, "a scale change must not read as a layout change"


def test_a_count_setting_still_splits_the_archive():
    m = REGISTRY["gabor"]
    a = m.layout_from_settings({"filters": 12})
    b = m.layout_from_settings({"filters": 8})
    assert a != b and a.signature() != b.signature()


def test_the_default_layout_is_unchanged_by_all_of_this():
    """Fourier at its defaults must still decode bit-exactly, or every existing
    genome and archive shifts underneath the user."""
    m = REGISTRY["fourier"]
    layout = m.layout_from_settings({})
    assert layout == default_layout()
    z = np.random.default_rng(2).normal(0, 0.5, 80).astype(np.float32)
    got = m.decode(z, layout)
    want = np.concatenate(
        [3.0 * np.tanh(z.reshape(20, 4)[:10]),
         1.0 * np.tanh(z.reshape(20, 4)[10:])], axis=1).reshape(-1)
    assert np.array_equal(got, want.astype(np.float32))


def test_widths_stay_positive_under_every_reachable_scale():
    """The GPU divides by these. Envelope Width scales the sigma BAND rather
    than replacing it, so the floor cannot be scaled to zero."""
    for name, key in (("gabor", "envelope_width"), ("lenia", "sigma_max")):
        m = REGISTRY[name]
        s = next(x for x in m.settings_schema() if x.key == key)
        for v in (s.lo, s.default, s.hi):
            layout = m.layout_from_settings({key: v})
            idx = 12 if name == "gabor" else 9
            stride = 14 if name == "gabor" else 10
            for scale in (-50.0, 0.0, 50.0):
                p = m.decode(np.full(layout.length, scale, np.float32), layout)
                w = p.reshape(layout.shape[0], stride)[:, idx]
                assert np.all(w > 0.0), f"{name}.{key}={v} gave width {w.min()}"


def test_an_unknown_scale_key_falls_back():
    layout = BrainLayout("gabor", (4,), 56)
    assert layout.scale("no_such_key", 7.5) == 7.5
