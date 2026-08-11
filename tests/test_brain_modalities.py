"""Round-trip and range tests for every modality.

The round-trip is what enforces agreement between a modality's Python decode and
its GLSL parameter order - they are two hand-written halves of one layout, and
nothing else checks that they describe the same thing.
"""
import numpy as np
import pytest

from services.brains import MAX_BRAIN_FLOATS, REGISTRY


def _all_modalities():
    import services.brains.fourier  # noqa: F401
    import services.brains.gabor    # noqa: F401
    import services.brains.lenia    # noqa: F401
    import services.brains.mlp      # noqa: F401
    return sorted(REGISTRY.values(), key=lambda m: m.modality_id)


ALL = _all_modalities()


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_decode_length_matches_layout(m):
    layout = m.layout_from_settings({})
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.5, layout.length).astype(np.float32)
    assert m.decode(z, layout).reshape(-1).shape == (layout.length,)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_encode_decode_round_trips(m):
    layout = m.layout_from_settings({})
    rng = np.random.default_rng(1)
    z = rng.normal(0, 0.4, layout.length).astype(np.float32)
    params = m.decode(z, layout)
    back, clamped = m.encode(params, layout)
    assert clamped == 0
    assert np.allclose(back.reshape(-1), z, atol=1e-3)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_every_reachable_setting_fits_the_budget(m):
    for s in m.settings_schema():
        if s.kind != "int":
            continue
        layout = m.layout_from_settings({s.key: int(s.hi)})
        assert layout.length <= MAX_BRAIN_FLOATS, (
            f"{m.name} at {s.key}={int(s.hi)} needs {layout.length}"
        )


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_decode_is_finite_for_extreme_z(m):
    """The optimizer explores unbounded z; every one must decode to a usable
    brain rather than an inf that poisons the whole canvas."""
    layout = m.layout_from_settings({})
    for scale in (5.0, 50.0, -50.0):
        z = np.full(layout.length, scale, dtype=np.float32)
        assert np.all(np.isfinite(m.decode(z, layout)))


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_random_is_finite_and_the_right_length(m):
    layout = m.layout_from_settings({})
    p = m.random(np.random.default_rng(7), layout).reshape(-1)
    assert p.shape == (layout.length,)
    assert np.all(np.isfinite(p))


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_modality_ids_and_names_are_unique(m):
    ids = [x.modality_id for x in ALL]
    names = [x.name for x in ALL]
    assert len(set(ids)) == len(ids)
    assert len(set(names)) == len(names)


# --- width parameters must never reach zero: the GPU divides by them ---

@pytest.mark.parametrize("mod_name,stride,idx", [
    ("gabor", 14, 12),   # envelope sigma
    ("lenia", 10, 9),    # band sigma
])
def test_width_parameters_are_strictly_positive(mod_name, stride, idx):
    m = REGISTRY[mod_name]
    layout = m.layout_from_settings({})
    for scale in (-50.0, 0.0, 50.0):
        p = m.decode(np.full(layout.length, scale, np.float32), layout)
        widths = p.reshape(layout.shape[0], stride)[:, idx]
        assert np.all(widths > 0.0), f"{mod_name} width hit {widths.min()}"


def test_mlp_length_is_nine_h_plus_four():
    m = REGISTRY["mlp"]
    for h in (4, 16, 48):
        assert m.layout_from_settings({"hidden": h}).length == 9 * h + 4


def test_mlp_signature_includes_activation():
    """Two activations are different function families; a genome evolved under
    one means nothing under the other, so they must not share an archive."""
    m = REGISTRY["mlp"]
    a = m.layout_from_settings({"hidden": 16, "activation": 0})
    b = m.layout_from_settings({"hidden": 16, "activation": 1})
    assert a.signature() != b.signature()


def test_lenia_growth_is_negative_off_the_band():
    """The -1 offset is what makes Lenia Lenia: positive inside a narrow band of
    sensor values and negative everywhere else. Without it the response is a
    smooth blur and the membranes never form."""
    from services.brains.lenia import growth

    assert growth(0.0, 0.0, 0.1) == pytest.approx(1.0)
    assert growth(10.0, 0.0, 0.1) == pytest.approx(-1.0)
    assert growth(0.2, 0.0, 0.1) < 0.0
