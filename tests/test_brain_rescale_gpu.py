"""Moving a scale slider must change the brain that is LOADED, not only the
next one the optimizer decodes.

The GPU holds decoded parameters. A scale only says what a z decodes TO, so
adopting a new layout on its own reaches nothing already uploaded - which is why
Band Center, Band Width and Freq Scale all appeared dead until a count change
forced a full rebuild. They were wired correctly and changed nothing visible,
which is the worst way for a setting to be broken.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.brains import REGISTRY, is_fallback  # noqa: E402
from utilities.gl_helpers import readback_rule  # noqa: E402

# (modality, setting, a value far from its default)
RESCALES = [
    ("fourier", "freq_scale", 1.0),
    ("gabor", "input_scale", 2.0),
    ("gabor", "envelope_width", 0.3),
    ("lenia", "mu_scale", 0.5),
    ("lenia", "sigma_max", 0.1),
]


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


@pytest.fixture(scope="module")
def sim(ctx):
    from sim import Sim

    return Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
               particle_density=0.05)


def _slot0(sim):
    """What is actually on the GPU, not what Python believes."""
    return np.asarray(readback_rule(sim.multi_load_rule_buffer, 0,
                                    sim.brain_layout),
                      dtype=np.float32).reshape(-1)


@pytest.mark.parametrize("name,key,value", RESCALES)
def test_a_scale_change_rewrites_the_live_brain(sim, name, key, value):
    m = REGISTRY[name]
    base = m.layout_from_settings({})
    sim.realloc_brain_buffers(base)
    sim.apply_rule(m.random(np.random.default_rng(0), base))
    before = _slot0(sim)

    sim.set_brain_scales(m.layout_from_settings({key: value}))
    after = _slot0(sim)
    assert not np.allclose(before, after, atol=1e-6), (
        f"{name}.{key} moved and the loaded brain did not")


@pytest.mark.parametrize("name,key,value", RESCALES)
def test_the_rescale_is_the_decode_of_the_same_z(sim, name, key, value):
    """Not any old change - the SAME creature expressed under the new scale.
    Anything else would silently swap the brain out from under the user."""
    m = REGISTRY[name]
    base = m.layout_from_settings({})
    tuned = m.layout_from_settings({key: value})
    sim.realloc_brain_buffers(base)

    z = np.random.default_rng(1).normal(0, 0.4, base.length).astype(np.float32)
    sim.apply_rule(m.decode(z, base))
    sim.set_brain_scales(tuned)
    assert np.allclose(_slot0(sim), m.decode(z, tuned), atol=1e-3)


def test_repeated_scale_moves_do_not_grind_the_brain_down(sim):
    """A drag calls this every frame. Re-encoding each time would clamp at the
    rails over and over and walk a parameter that leaves the range down to
    nothing; decoding from the STORED z cannot, so returning to a scale must
    return the same brain.
    """
    m = REGISTRY["gabor"]
    base = m.layout_from_settings({})
    sim.realloc_brain_buffers(base)
    z = np.random.default_rng(2).normal(0, 0.8, base.length).astype(np.float32)
    sim.apply_rule(m.decode(z, base))
    start = _slot0(sim)

    for v in (0.05, 0.02, 0.01, 0.02, 0.05, 0.2, 1.0, 4.0, 2.0, 0.5):
        sim.set_brain_scales(m.layout_from_settings({"input_scale": v}))
    sim.set_brain_scales(base)
    assert np.allclose(_slot0(sim), start, atol=1e-3), (
        "a round trip through the extremes did not come back")


def test_a_blank_fourier_brain_is_left_alone(sim):
    """It is not decoded from anything - the shader generates it - so
    re-encoding the zeros would manufacture a rule out of nowhere and the
    particles would stop using the fallback."""
    m = REGISTRY["fourier"]
    base = m.layout_from_settings({})
    sim.realloc_brain_buffers(base)
    sim.apply_rule(None)
    assert is_fallback(sim.slot0_params, base)

    sim.set_brain_scales(m.layout_from_settings({"freq_scale": 1.0}))
    assert not _slot0(sim).any(), "the blank brain was decoded into a real one"
    assert is_fallback(sim.slot0_params, sim.brain_layout)


def test_a_width_change_is_still_refused(sim):
    """The light path must not silently accept one - it would reinterpret the
    buffer at the wrong stride."""
    m = REGISTRY["gabor"]
    sim.realloc_brain_buffers(m.layout_from_settings({}))
    with pytest.raises(ValueError):
        sim.set_brain_scales(m.layout_from_settings({"filters": 24}))


def test_slot0_params_tracks_what_was_written(sim):
    """The Inspector reads this to decide whether to draw the generated rule,
    and reading it back off the GPU every frame would sync the pipeline."""
    m = REGISTRY["lenia"]
    layout = m.layout_from_settings({})
    sim.realloc_brain_buffers(layout)
    p = np.asarray(m.random(np.random.default_rng(3), layout), np.float32)
    sim.apply_rule(p)
    assert np.allclose(sim.slot0_params, p.reshape(-1), atol=1e-6)
    assert np.allclose(sim.slot0_params, _slot0(sim), atol=1e-6)
