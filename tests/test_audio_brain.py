"""Encode once, decode many - the brain follows the same never-write-the-base rule."""
import numpy as np
import pytest

from services import brains
from services.audio_brain import BrainModulator


def fourier_base(n=10, seed=3):
    m = brains.get("fourier")
    layout = m.layout_from_settings({"centers": n})
    params = m.random(np.random.default_rng(seed), layout)
    return m, layout, params


def test_with_no_base_set_there_is_nothing_to_modulate():
    assert BrainModulator().modulated({"freq_scale": 2.0}) is None


def test_unmodulated_scales_reproduce_the_base_brain():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    out = mod.modulated(mod.base_scales())
    assert np.allclose(out, params, atol=1e-3)


def test_the_base_params_array_is_never_written():
    m, layout, params = fourier_base()
    before = params.copy()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    mod.modulated({"freq_scale": 6.0})
    assert np.array_equal(params, before)


def test_raising_freq_scale_scales_exactly_the_frequency_floats():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    base = mod.base_scales()
    out = mod.modulated({**base, "freq_scale": base["freq_scale"] * 2.0})

    g_in = params.reshape(-1, 8)
    g_out = np.asarray(out).reshape(-1, 8)
    assert np.allclose(g_out[:, :4], g_in[:, :4] * 2.0, atol=1e-2)
    assert np.allclose(g_out[:, 4:], g_in[:, 4:], atol=1e-3)


def test_the_output_width_matches_the_layout():
    m, layout, params = fourier_base(n=7)
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    assert np.asarray(mod.modulated(mod.base_scales())).size == layout.length


def test_base_scales_are_the_layouts_own():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    assert mod.base_scales() == {k: pytest.approx(v) for k, v in layout.scales}


def test_a_missing_scale_key_falls_back_to_the_base():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    assert np.allclose(mod.modulated({}), params, atol=1e-3)


def test_setting_a_new_base_re_encodes():
    m, layout, first = fourier_base(seed=1)
    _, _, second = fourier_base(seed=2)
    mod = BrainModulator()
    mod.set_base(first, m, layout)
    mod.set_base(second, m, layout)
    assert np.allclose(mod.modulated(mod.base_scales()), second, atol=1e-3)


def test_a_wrong_width_base_is_refused_rather_than_reinterpreted():
    m, layout, _ = fourier_base(n=10)
    mod = BrainModulator()
    mod.set_base(np.zeros(7, dtype=np.float32), m, layout)
    assert mod.modulated({"freq_scale": 2.0}) is None


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia"])
def test_every_modality_with_scales_round_trips(name):
    m = brains.get(name)
    layout = m.layout_from_settings({})
    params = m.random(np.random.default_rng(0), layout)
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    out = mod.modulated(mod.base_scales())
    assert out is not None and np.asarray(out).size == layout.length


def test_decoding_repeatedly_does_not_drift():
    """The z is cached, so the hundredth frame matches the first."""
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    first = np.asarray(mod.modulated(mod.base_scales())).copy()
    for _ in range(100):
        mod.modulated({"freq_scale": 1.0 + np.random.random()})
    assert np.allclose(mod.modulated(mod.base_scales()), first, atol=1e-6)
