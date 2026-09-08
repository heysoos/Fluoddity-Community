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

from services.brains import REGISTRY  # noqa: E402
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
    return np.asarray(readback_rule(sim.multi_load_rule_buffer,
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


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia", "mlp"])
def test_a_generated_brain_is_regenerated_not_re_encoded(sim, name):
    """Generated brains do not come from a z at all - random() draws them
    directly - so a scale change re-draws them under the new scales. What must
    never happen is the slot going blank or silent."""
    m = REGISTRY[name]
    base = m.layout_from_settings({})
    sim.realloc_brain_buffers(base)
    sim.apply_rule(None)
    assert sim.brain_per_cohort
    assert _slot0(sim).any(), f"{name}: no rule loaded left the slot empty"

    key = "freq_scale" if name == "fourier" else (
        "input_scale" if name == "gabor" else "mu_scale")
    if any(s.key == key for s in m.settings_schema()):
        sim.set_brain_scales(m.layout_from_settings({key: 1.0}))
        assert _slot0(sim).any(), f"{name}: rescaling emptied the slot"
        assert sim.brain_per_cohort, "the rescale dropped out of per-cohort mode"


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


# ---- the audio scale ------------------------------------------------------
#
# Audio Scale is the one scale a rig drags with the music on, and its zero is
# a promise: the deaf ancestor, bit for bit. Re-encoding the whole brain to
# get there clips every deaf float past its rail.

def _wide(name, scale):
    return REGISTRY[name].layout_from_settings(
        {"audio_inputs": 1, "audio_scale": scale})


def _split(layout, rule):
    from services.brains import audio_weight_index

    audio = audio_weight_index(layout)
    keep = np.setdiff1d(np.arange(layout.length), audio)
    return rule[keep], rule[audio]


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia", "mlp"])
def test_the_audio_scale_never_touches_a_deaf_float(sim, name):
    m = REGISTRY[name]
    one = _wide(name, 1.0)
    sim.realloc_brain_buffers(one)
    # Past the rails on purpose: a layer Scale or an old preset puts a brain
    # there, and a round trip through encode() would clip it.
    rule = np.asarray(m.random(np.random.default_rng(3), one),
                      np.float32).reshape(-1) * 3.0
    sim.apply_rule(rule)
    deaf0, ears0 = _split(one, _slot0(sim))
    np.testing.assert_array_equal(deaf0, _split(one, rule)[0])

    sim.set_brain_scales(_wide(name, 0.0))
    deaf, ears = _split(one, _slot0(sim))
    np.testing.assert_array_equal(deaf, deaf0)
    assert not ears.any(), "at zero the ears must be exactly zero"

    sim.set_brain_scales(_wide(name, 0.5))
    deaf, ears = _split(one, _slot0(sim))
    np.testing.assert_array_equal(deaf, deaf0)
    assert np.allclose(ears, ears0 * 0.5, atol=1e-6)

    sim.set_brain_scales(one)
    deaf, ears = _split(one, _slot0(sim))
    np.testing.assert_array_equal(deaf, deaf0)
    assert np.allclose(ears, ears0, atol=1e-6)


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia", "mlp"])
def test_a_brain_that_arrived_deaf_by_scale_hears_the_seed_when_raised(sim, name):
    """At scale zero there are no weights to scale up, so raising it draws
    the rig's - the same weights a transfer at that scale gives."""
    from services.brains.layout_moves import transfer_audio_inputs

    m = REGISTRY[name]
    zero, half = _wide(name, 0.0), _wide(name, 0.5)
    sim.realloc_brain_buffers(zero)
    sim.set_audio_seed(0.31)
    rule = np.asarray(m.random(np.random.default_rng(4), zero),
                      np.float32).reshape(-1)
    assert not _split(zero, rule)[1].any()
    sim.apply_rule(rule)

    sim.set_brain_scales(half)
    deaf, ears = _split(half, _slot0(sim))
    np.testing.assert_array_equal(deaf, _split(zero, rule)[0])
    want = transfer_audio_inputs(rule, zero, half, 0.31)
    assert np.allclose(ears, _split(half, want)[1], atol=1e-6)
    assert ears.any()
