"""The MLP layer stack, and the promise that a depth-1 net did not move.

Everything already on disk - genomes, archive directories, config stamps,
checkpoints - was written by the one-hidden-layer version. Depth 1 is therefore
not merely supported: it must be BIT-IDENTICAL, in packing, length, decode,
initial draw and signature string. That is what makes the rest of this feature
safe to reason about, so it is the first thing asserted here.

The GPU half - that the GLSL reads these floats where Python writes them - is
tests/test_brain_modalities_gpu.py, which compares a pure evaluation over a
fixed input grid. A trajectory diff cannot validate a shader change here.
"""
import numpy as np
import pytest

from services.brains import (MAX_BRAIN_FLOATS, layout_from_signature,
                             settings_of, unit_count)
from services.brains.mlp import (ACTIVATIONS, B_SCALE, MAX_DEPTH, MAX_WIDTH,
                                 SCRATCH_BUCKETS, W_SCALE, MLPModality,
                                 _weight_mask, layer_spans, scratch_width)

M = MLPModality()

STACKS = [
    [[16, 0]],
    [[48, 2]],
    [[8, 0], [8, 1]],
    [[8, 1], [7, 2], [5, 0]],
    [[6, 0], [6, 1], [6, 2], [6, 0], [6, 1], [6, 2], [6, 0], [6, 1]],
]


def ids(ls):
    return "x".join(f"{w}a{a}" for w, a in ls)


def layout(layers):
    return M.layout_from_settings({"layers": layers})


# ---- the compatibility invariant -------------------------------------------

@pytest.mark.parametrize("h", [4, 16, 32, 48])
@pytest.mark.parametrize("act", range(len(ACTIVATIONS)))
def test_depth_one_is_what_it_always_was(h, act):
    """Length 9h+4, the same field offsets, and the same signature string."""
    lay = layout([[h, act]])
    assert lay.shape == (h, act)
    assert lay.length == 9 * h + 4
    assert lay.signature() == f"mlp-n{h}-a{act}"

    hidden, out_w, out_b, n = layer_spans(lay.shape)
    assert n == 9 * h + 4
    assert hidden == [(0, 4 * h, 4, h)]     # W1 at 0, b1 at 4h, fan-in 4
    assert (out_w, out_b) == (5 * h, 9 * h)  # W_out at 5h, b_out at 9h


@pytest.mark.parametrize("h", [4, 16, 48])
def test_depth_one_decodes_exactly_as_it_used_to(h):
    """The pre-stack decode, written out by slice. A mask that disagrees by one
    float shifts every genome on disk."""
    lay = layout([[h, 0]])
    z = np.random.default_rng(3).normal(0, 0.6, lay.length).astype(np.float32)
    want = np.empty_like(z)
    want[: 4 * h] = W_SCALE * np.tanh(z[: 4 * h])              # W1
    want[4 * h: 5 * h] = B_SCALE * np.tanh(z[4 * h: 5 * h])    # b1
    want[5 * h: 9 * h] = W_SCALE * np.tanh(z[5 * h: 9 * h])    # W_out
    want[9 * h:] = B_SCALE * np.tanh(z[9 * h:])                # b_out
    assert np.array_equal(M.decode(z, lay), want.astype(np.float32))


def test_a_depth_one_draw_is_bit_identical():
    """random() fan-in normalises the hidden->hidden weights, and depth 1 has
    none - so its draw must be the plain one, float for float."""
    lay = layout([[16, 0]])
    got = M.random(np.random.default_rng(9), lay)
    z = np.random.default_rng(9).normal(0.0, 0.5, lay.length).astype(np.float32)
    assert np.array_equal(got, M.decode(z, lay))


def test_the_legacy_settings_still_name_a_stack():
    """8 user configs are stamped mlp-n16-a0 and carry hidden/activation. This
    acceptance IS the migration; there is no other."""
    assert M.layout_from_settings({"hidden": 24, "activation": 2}).shape == (24, 2)
    assert M.layout_from_settings({}).shape == (16, 0)
    # And `layers` wins when a file carries both.
    both = {"hidden": 24, "activation": 2, "layers": [[8, 1]]}
    assert M.layout_from_settings(both).shape == (8, 1)


# ---- the stack -------------------------------------------------------------

@pytest.mark.parametrize("layers", STACKS, ids=ids)
def test_the_length_is_the_sum_of_the_blocks(layers):
    lay = layout(layers)
    widths = [w for w, _ in layers]
    want = 4 * widths[0] + widths[0]
    for prev, w in zip(widths, widths[1:]):
        want += prev * w + w
    want += 4 * widths[-1] + 4
    assert lay.length == want <= MAX_BRAIN_FLOATS


@pytest.mark.parametrize("layers", STACKS, ids=ids)
def test_the_signature_round_trips(layers):
    lay = layout(layers)
    back = layout_from_signature(lay.signature())
    assert back is not None and back == lay
    # The scales ride along because a saved rule is re-encoded by dividing by
    # them; only `layers` is structural and so only it is in the signature.
    got = settings_of(lay)
    assert got.pop("layers") == layers
    assert got == {"w_scale": W_SCALE, "b_scale": B_SCALE}


@pytest.mark.parametrize("sig", ["mlp-n16-a1", "mlp-n16-a2"])
def test_a_non_default_activation_can_be_rebuilt(sig):
    """It could not be, and nothing said so: the parser took only `int`
    settings, dropped the activation, rebuilt with 0 and the round-trip check
    correctly refused. Every sin and gelu entry in an archive was therefore
    unadoptable and unpreviewable, and a same-WIDTH mismatch is invisible to
    apply_rule's width check."""
    lay = layout_from_signature(sig)
    assert lay is not None and lay.signature() == sig


@pytest.mark.parametrize("sig", ["mlp-n16.8-a0", "mlp-n16-a0.1", "mlp-n16",
                                 "mlp-n16-a3", "mlp-n99-a0", "mlp-nx-a0",
                                 "mlp-n16-a0-b0", "mlp-n-a0", "mlp-n16.-a0.1"])
def test_a_malformed_signature_is_refused(sig):
    """None, never a plausible layout of the wrong width - the one outcome
    worse than refusing."""
    assert layout_from_signature(sig) is None


@pytest.mark.parametrize("layers", STACKS, ids=ids)
def test_encode_inverts_decode(layers):
    lay = layout(layers)
    z = np.random.default_rng(1).normal(0, 0.4, lay.length).astype(np.float32)
    back, clamped = M.encode(M.decode(z, lay), lay)
    assert clamped == 0
    assert np.allclose(back, z, atol=1e-3)


@pytest.mark.parametrize("layers", STACKS, ids=ids)
def test_the_rails_are_flat_across_every_layer(layers):
    """decode is flat, so the full +/-W_SCALE range is reachable in EVERY layer.
    A fan-in gain folded in here would be a cap, not a normalisation."""
    lay = layout(layers)
    hidden, out_w, out_b, n = layer_spans(lay.shape)
    p = M.decode(np.full(n, 12.0, np.float32), lay)     # tanh saturated
    for w_off, b_off, _fan, _w in hidden:
        assert np.allclose(p[w_off:b_off], W_SCALE, atol=1e-5)
        assert np.allclose(p[b_off:b_off + _w], B_SCALE, atol=1e-5)
    assert np.allclose(p[out_w:out_b], W_SCALE, atol=1e-5)
    assert np.allclose(p[out_b:], B_SCALE, atol=1e-5)


@pytest.mark.parametrize("layers", STACKS, ids=ids)
def test_initialisation_normalises_only_the_hidden_to_hidden_weights(layers):
    """A hidden->hidden layer sums w_{l-1} terms rather than 4, so a plain draw
    would be born saturated. Every OTHER slice must be untouched, or a depth-1
    draw is no longer the one this modality has always produced."""
    lay = layout(layers)
    got = M.random(np.random.default_rng(21), lay)
    plain = M.decode(np.random.default_rng(21).normal(
        0.0, 0.5, lay.length).astype(np.float32), lay)

    hidden, out_w, out_b, n = layer_spans(lay.shape)
    scaled = np.zeros(n, dtype=bool)
    for w_off, b_off, _fan, _w in hidden[1:]:
        scaled[w_off:b_off] = True
    assert np.array_equal(got[~scaled], plain[~scaled])
    if scaled.any():
        assert not np.array_equal(got[scaled], plain[scaled])


def _final_activations(p, lay, x):
    hidden, _ow, _ob, _n = layer_spans(lay.shape)
    h = x
    for w_off, b_off, fan_in, w in hidden:
        h = np.tanh(h @ p[w_off:b_off].reshape(w, fan_in).T + p[b_off:b_off + w])
    return h


def test_the_normalisation_is_what_keeps_a_deep_draw_off_the_rails():
    """The point of it: a hidden->hidden layer sums w_{l-1} terms rather than 4,
    so the plain draw is born much closer to a sign function. Compared against
    the same z undoing the gain, rather than a threshold, so the test states the
    effect and not a number."""
    lay = layout([[8, 0]] * 6)
    x = np.random.default_rng(5).standard_normal((512, 4)).astype(np.float32) * 0.5
    plain = M.decode(np.random.default_rng(4).normal(
        0.0, 0.5, lay.length).astype(np.float32), lay)
    normalised = M.random(np.random.default_rng(4), lay)

    def rail_fraction(p):
        return float(np.mean(np.abs(_final_activations(p, lay, x)) > 0.99))

    assert rail_fraction(normalised) < rail_fraction(plain) / 2.0, (
        f"normalised {rail_fraction(normalised):.3f} vs "
        f"plain {rail_fraction(plain):.3f}"
    )


# ---- the decode scales -----------------------------------------------------

def test_the_defaults_decode_exactly_as_the_constants_always_did():
    """The scales are new; what they decode to at their defaults is not. A
    different answer here would rewrite every genome on disk."""
    lay = M.layout_from_settings({"layers": [[16, 0]]})
    z = np.random.default_rng(3).normal(0.0, 0.5, lay.length).astype(np.float32)
    want = np.where(_weight_mask(lay.shape), W_SCALE, B_SCALE) * np.tanh(z)
    assert np.allclose(M.decode(z, lay), want.astype(np.float32))


def test_the_weight_scale_scales_the_weights_and_leaves_the_biases():
    lay_a = M.layout_from_settings({"layers": [[16, 0]], "w_scale": 2.0})
    lay_b = M.layout_from_settings({"layers": [[16, 0]], "w_scale": 4.0})
    z = np.random.default_rng(4).normal(0.0, 0.5, lay_a.length).astype(np.float32)
    a, b = M.decode(z, lay_a), M.decode(z, lay_b)
    w = _weight_mask(lay_a.shape)
    assert np.allclose(b[w], 2.0 * a[w])
    assert np.allclose(b[~w], a[~w])


def test_encode_inverts_decode_under_a_non_default_scale():
    """The modulator encodes once and re-decodes every frame; an encode that
    ignored the scale would move the brain the moment audio touched it."""
    lay = M.layout_from_settings({"layers": [[16, 0]], "w_scale": 5.0,
                                  "b_scale": 3.0})
    z = np.random.default_rng(5).normal(0.0, 0.5, lay.length).astype(np.float32)
    back, _clamped = M.encode(M.decode(z, lay), lay)
    assert np.allclose(back, z, atol=1e-3)


def test_a_zero_bias_scale_does_not_divide_by_zero():
    lay = M.layout_from_settings({"layers": [[16, 0]], "b_scale": 0.0})
    z = np.zeros(lay.length, dtype=np.float32)
    assert np.all(np.isfinite(M.decode(z, lay)))
    assert np.all(np.isfinite(M.encode(M.decode(z, lay), lay)[0]))


def test_the_scales_do_not_change_the_signature_or_the_length():
    """They change what a z MEANS, not how many there are, so they must not
    split an archive or reset a search."""
    plain = M.layout_from_settings({"layers": [[16, 0]]})
    scaled = M.layout_from_settings({"layers": [[16, 0]], "w_scale": 7.0})
    assert scaled.signature() == plain.signature()
    assert scaled.length == plain.length
    assert scaled == plain


# ---- limits ----------------------------------------------------------------

def test_the_stack_is_clamped_rather_than_raising():
    """A config from a build with wider limits must not stop the app. Clamped,
    and then refused by the signature round trip - which is how a wrong-width
    layout is caught rather than loaded."""
    assert layout([[999, 0]]).shape == (MAX_WIDTH, 0)
    assert layout([[16, 99]]).shape == (16, len(ACTIVATIONS) - 1)
    assert len(layout([[4, 0]] * (MAX_DEPTH + 4)).shape) == 2 * MAX_DEPTH
    assert layout_from_signature("mlp-n999-a0") is None


def test_a_deep_layer_may_be_as_wide_as_a_lone_one():
    """Every layer reaches MAX_WIDTH. The old cap of 8 existed because ONE
    shader served every layout and its scratch arrays were sized for the widest
    stack anyone might build; the shader is compiled per layout now, so a wide
    stack costs only itself."""
    assert layout([[48, 0]]).shape == (48, 0)
    assert layout([[48, 0], [8, 0]]).shape == (48, 0, 8, 0)
    assert layout_from_signature("mlp-n48.8-a0.0") is not None


def test_the_float_budget_is_what_stops_a_stack_growing():
    """And it CLAMPS rather than raises, because the input may be a config from
    a build with different limits. One cap across the stack, so a layer already
    below it is left where it is."""
    assert layout([[48, 0], [48, 0]]).length <= MAX_BRAIN_FLOATS
    # [48, 48] is over budget, so both come down; [48, 2] is not, so neither moves.
    assert max(layout([[48, 0], [48, 0]]).shape[0::2]) < 48
    assert layout([[48, 0], [2, 0]]).shape == (48, 0, 2, 0)


def test_a_depth_one_stack_asks_for_the_smallest_scratch():
    """It never enters the deep path, and the arrays it does not use are
    exactly what used to tax it - the whole reason the cap existed."""
    assert scratch_width(layout([[48, 0]]).shape) == min(SCRATCH_BUCKETS)
    assert scratch_width(layout([[1, 0]]).shape) == min(SCRATCH_BUCKETS)


def test_a_deep_stack_asks_for_the_bucket_its_widest_layer_needs():
    assert scratch_width(layout([[8, 0], [8, 0]]).shape) == 8
    assert scratch_width(layout([[8, 0], [9, 0]]).shape) == 16
    assert scratch_width(layout([[2, 0], [2, 0]]).shape) == min(SCRATCH_BUCKETS)


def test_the_scratch_bucket_is_never_narrower_than_the_stack():
    """A shader narrower than the layout indexes past the end of an array."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        depth = int(rng.integers(1, MAX_DEPTH + 1))
        widths = rng.integers(1, MAX_WIDTH + 1, depth)
        shape = layout([[int(w), 0] for w in widths]).shape
        if len(shape) > 2:
            assert scratch_width(shape) >= max(shape[0::2])


def test_the_defines_are_what_the_program_cache_keys_on():
    """Every layout that needs no scratch must produce the SAME key, or the
    cache holds one identical program per layout the user ever touches."""
    from services.brains import default_layout, layout_defines

    flat = layout_defines(layout([[16, 0]]))
    assert flat == {"MAX_MLP_WIDTH": min(SCRATCH_BUCKETS)}
    assert layout_defines(layout([[48, 0]])) == flat, (
        "every depth-1 stack shares one program")
    # A Fourier brain pays for mlp.glsl's arrays too - one program holds all
    # four shaders - so it must ask for the floor rather than fall through to
    # the shader's own default, which has to cover the widest stack.
    assert layout_defines(default_layout()) == flat
    assert layout_defines(layout([[16, 0], [16, 0]])) != flat


def test_every_reachable_stack_fits_the_buffer():
    """Nothing the UI can build may exceed the stride the flat buffer is packed
    at, and nothing may RAISE on the way - a config written by another build
    has to open."""
    rng = np.random.default_rng(0)
    worst = 0
    for _ in range(400):
        depth = int(rng.integers(1, MAX_DEPTH + 1))
        widths = rng.integers(1, MAX_WIDTH + 1, depth)
        lay = layout([[int(w), 0] for w in widths])
        worst = max(worst, lay.length)
        assert lay.length <= MAX_BRAIN_FLOATS
    assert layout([[MAX_WIDTH, 0]] * MAX_DEPTH).length <= MAX_BRAIN_FLOATS
    assert worst > 512, "the budget raise is doing nothing"


@pytest.mark.parametrize("layers", STACKS, ids=ids)
def test_the_drawable_units_are_the_last_layers(layers):
    assert unit_count(layout(layers)) == layers[-1][0]
