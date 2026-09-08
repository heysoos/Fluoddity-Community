"""Audio inputs widen every modality's input from the four sensor taps.

The invariant these guard: at K=0 nothing changes - same signature, same
length, same decode - and at K>0 every non-audio float of a brain sits where
it did, so a brain with its channels at zero is its deaf ancestor. See
docs/superpowers/specs/2026-09-06-brain-audio-inputs-design.md.
"""
import numpy as np
import pytest

from services.brains import (MAX_AUDIO_INPUTS, REGISTRY, audio_weight_index,
                             audio_z_index, layout_from_signature, settings_of)
from services.brains.layout_moves import (LayoutBounds, candidate_moves,
                                          grow_inputs, shrink_inputs,
                                          transfer_audio_inputs)


def _all_modalities():
    import services.brains.fourier  # noqa: F401
    import services.brains.gabor    # noqa: F401
    import services.brains.lenia    # noqa: F401
    import services.brains.mlp      # noqa: F401
    return sorted(REGISTRY.values(), key=lambda m: m.modality_id)


ALL = _all_modalities()
K = 4

# The lengths and signatures every archive on disk was written under. A
# change here is a change to what those files mean.
HISTORICAL = {
    "fourier": ("fourier-n10", 80),
    "gabor": ("gabor-n12", 168),
    "lenia": ("lenia-n12", 120),
    "mlp": ("mlp-n16-a0", 148),
}


def _units(m, layout):
    """How many audio blocks a layout holds: one per unit, which for MLP is
    one per first-layer row."""
    return int(layout.shape[0])


# ---- K=0 is today ---------------------------------------------------------

@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_zero_inputs_is_the_historical_layout(m):
    sig, length = HISTORICAL[m.name]
    base = m.layout_from_settings({})
    explicit = m.layout_from_settings({"audio_inputs": 0})
    assert base == explicit
    assert base.audio_inputs == 0
    assert base.signature() == sig
    assert base.length == length
    assert "+a" not in base.signature()


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_zero_inputs_has_no_audio_weights(m):
    layout = m.layout_from_settings({})
    assert audio_weight_index(layout).size == 0


# ---- the setting and the signature ----------------------------------------

@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_every_modality_declares_the_shared_setting(m):
    keys = {s.key: s for s in m.settings_schema()}
    s = keys["audio_inputs"]
    assert s.kind == "int" and int(s.lo) == 0 and int(s.hi) == MAX_AUDIO_INPUTS
    assert keys["audio_scale"].kind == "float"


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_the_signature_carries_k_as_a_suffix(m):
    layout = m.layout_from_settings({"audio_inputs": K})
    assert layout.audio_inputs == K
    assert layout.signature() == HISTORICAL[m.name][0] + f"+a{K}"


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_the_signature_round_trips(m):
    layout = m.layout_from_settings({"audio_inputs": K})
    back = layout_from_signature(layout.signature())
    assert back == layout


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_settings_of_emits_k(m):
    layout = m.layout_from_settings({"audio_inputs": K})
    s = settings_of(layout)
    assert s["audio_inputs"] == K
    assert m.layout_from_settings(s) == layout


def test_a_malformed_suffix_is_refused():
    assert layout_from_signature("fourier-n10+ax") is None
    assert layout_from_signature("fourier-n10+a") is None


# ---- where the floats go --------------------------------------------------

@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_the_length_grows_by_k_per_unit(m):
    base = m.layout_from_settings({})
    wide = m.layout_from_settings({"audio_inputs": K})
    assert wide.length == base.length + K * _units(m, wide)
    idx = audio_weight_index(wide)
    assert idx.shape == (K * _units(m, wide),)
    assert len(set(idx.tolist())) == idx.size
    assert idx.min() >= 0 and idx.max() < wide.length


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_the_non_audio_floats_decode_exactly_as_before(m):
    """Drop the audio coordinates from a wide z and decode the rest under the
    K=0 layout: every non-audio float must match, so the ancestor is what the
    brain computes with its channels at zero."""
    base = m.layout_from_settings({})
    wide = m.layout_from_settings({"audio_inputs": K})
    rng = np.random.default_rng(3)
    z_wide = rng.normal(0, 0.5, wide.length).astype(np.float32)
    z_keep = np.setdiff1d(np.arange(wide.length), audio_z_index(wide))
    keep = np.setdiff1d(np.arange(wide.length), audio_weight_index(wide))
    z_base = z_wide[z_keep]
    assert z_base.shape == (base.length,)
    p_wide = m.decode(z_wide, wide).reshape(-1)
    p_base = m.decode(z_base, base).reshape(-1)
    np.testing.assert_array_equal(p_wide[keep], p_base)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_wide_encode_decode_round_trips(m):
    wide = m.layout_from_settings({"audio_inputs": K})
    rng = np.random.default_rng(5)
    z = rng.normal(0, 0.4, wide.length).astype(np.float32)
    back, clamped = m.encode(m.decode(z, wide), wide)
    assert clamped == 0
    assert np.allclose(back.reshape(-1), z, atol=1e-3)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_audio_scale_moves_only_the_audio_weights(m):
    wide = m.layout_from_settings({"audio_inputs": K})
    quiet = m.layout_from_settings({"audio_inputs": K, "audio_scale": 0.0})
    z = np.random.default_rng(9).normal(0, 0.5, wide.length).astype(np.float32)
    p, q = m.decode(z, wide).reshape(-1), m.decode(z, quiet).reshape(-1)
    audio = audio_weight_index(wide)
    keep = np.setdiff1d(np.arange(wide.length), audio)
    np.testing.assert_array_equal(p[keep], q[keep])
    assert np.all(q[audio] == 0.0)
    assert np.any(p[audio] != 0.0)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_random_fills_the_audio_weights(m):
    wide = m.layout_from_settings({"audio_inputs": K})
    p = np.asarray(m.random(np.random.default_rng(2), wide)).reshape(-1)
    assert p.shape == (wide.length,)
    assert np.all(np.isfinite(p))
    assert np.any(p[audio_weight_index(wide)] != 0.0)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_the_budget_holds_at_the_widest_input(m):
    for s in m.settings_schema():
        if s.kind != "int" or s.key == "audio_inputs":
            continue
        m.layout_from_settings({s.key: int(s.hi),
                                "audio_inputs": MAX_AUDIO_INPUTS})


# ---- growing and shrinking ------------------------------------------------

@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_grow_and_shrink_are_inverses(m):
    base = m.layout_from_settings({"freq_scale": 2.0} if m.name == "fourier"
                                  else {})
    wide = grow_inputs(base, K)
    assert wide.audio_inputs == K
    assert wide.signature() == base.signature() + f"+a{K}"
    assert tuple(wide.scales) == tuple(base.scales)
    assert shrink_inputs(wide, 0) == base


def test_grow_is_bounded():
    base = REGISTRY["fourier"].layout_from_settings({})
    with pytest.raises(ValueError):
        grow_inputs(base, MAX_AUDIO_INPUTS + 1)
    with pytest.raises(ValueError):
        grow_inputs(base, 0)
    with pytest.raises(ValueError):
        shrink_inputs(grow_inputs(base, 2), 3)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_transfer_keeps_every_non_audio_float_and_fills_the_rest(m):
    base = m.layout_from_settings({})
    wide = grow_inputs(base, K)
    rng = np.random.default_rng(11)
    parent = np.asarray(m.random(rng, base), dtype=np.float32).reshape(-1)
    child = transfer_audio_inputs(parent, base, wide, 0.1)
    assert child.shape == (wide.length,)
    audio = audio_weight_index(wide)
    keep = np.setdiff1d(np.arange(wide.length), audio)
    np.testing.assert_array_equal(child[keep], parent)
    assert np.all(np.isfinite(child[audio]))
    assert np.any(child[audio] != 0.0)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_transfer_is_seeded(m):
    base = m.layout_from_settings({})
    wide = grow_inputs(base, K)
    parent = np.asarray(m.random(np.random.default_rng(0), base)).reshape(-1)
    a = transfer_audio_inputs(parent, base, wide, 0.7)
    b = transfer_audio_inputs(parent, base, wide, 0.7)
    c = transfer_audio_inputs(parent, base, wide, 0.8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_shrinking_drops_the_audio_weights_and_nothing_else(m):
    base = m.layout_from_settings({})
    wide = grow_inputs(base, K)
    parent = np.asarray(m.random(np.random.default_rng(4), wide)).reshape(-1)
    back = transfer_audio_inputs(parent, wide, base, 0.5)
    keep = np.setdiff1d(np.arange(wide.length), audio_weight_index(wide))
    np.testing.assert_array_equal(back, parent[keep])


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_rerolling_from_a_wide_brain_keeps_its_deaf_half(m):
    """Reroll re-runs the transfer FROM THE CURRENT RULE, so the deaf brain
    underneath never moves."""
    wide = grow_inputs(m.layout_from_settings({}), K)
    current = np.asarray(m.random(np.random.default_rng(4), wide)).reshape(-1)
    rerolled = transfer_audio_inputs(current, wide, wide, 0.55)
    audio = audio_weight_index(wide)
    keep = np.setdiff1d(np.arange(wide.length), audio)
    np.testing.assert_array_equal(rerolled[keep], current[keep])
    assert not np.array_equal(rerolled[audio], current[audio])


def test_transfer_refuses_a_layout_that_differs_in_more_than_k():
    m = REGISTRY["fourier"]
    a = m.layout_from_settings({"centers": 10})
    b = m.layout_from_settings({"centers": 12, "audio_inputs": K})
    with pytest.raises(ValueError):
        transfer_audio_inputs(np.zeros(a.length, np.float32), a, b, 0.0)


# ---- the search leaves K alone -------------------------------------------

@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_the_layout_search_never_proposes_a_change_of_k(m):
    wide = grow_inputs(m.layout_from_settings({}), K)
    moves = candidate_moves(wide, LayoutBounds())
    assert moves, "a widened layout must still have its ordinary moves"
    for mv in moves:
        assert mv.child.audio_inputs == K, mv.operator


# ---- the (N, stride) presentation ------------------------------------------

def _wide_fourier(k):
    from services.brains.layout_moves import grow_inputs
    return grow_inputs(REGISTRY["fourier"].layout_from_settings({}), k)


def test_a_wide_fourier_brain_presents_as_n_by_stride():
    """A Fourier centre is 8 + K floats, and every caller that reshapes by
    centre has to agree - the tournament reseeds through present()."""
    from services.genome_spec import present, random_genome_for

    wide = _wide_fourier(1)
    g = random_genome_for(np.random.default_rng(0), wide)
    assert g.shape == (wide.shape[0], 9)
    assert g.size == wide.length
    z = present(np.zeros(wide.length, np.float32), wide)
    assert z.shape == (wide.shape[0], 9)


def test_the_tournament_survives_adding_an_input_to_a_fourier_brain():
    from services.tournament_service import TournamentService

    base = REGISTRY["fourier"].layout_from_settings({})
    svc = TournamentService(grid=2, layout=base)
    svc.init_population()
    wide = _wide_fourier(1)
    svc.set_layout(wide)
    assert all(np.asarray(g).size == wide.length for g in svc.population)


def test_readback_of_a_wide_fourier_brain_keeps_its_centres():
    from utilities.gl_helpers import readback_rule

    class _Buf:
        def __init__(self, data):
            self._d = data.tobytes()

        def read(self, size, offset=0):
            return self._d[offset:offset + size]

    wide = _wide_fourier(2)
    rule = np.arange(wide.length, dtype=np.float32)
    out = readback_rule(_Buf(rule), wide)
    assert out.shape == (wide.shape[0], 10)
    assert np.array_equal(out.reshape(-1), rule)


# ---- the draw itself keeps the invariant -----------------------------------

@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_a_generated_wide_brain_is_the_deaf_draw_with_ears(m):
    """"No rule loaded" draws one brain per cohort from rule_seed under the
    LIVE layout, so the draw has to keep the deaf brain's floats where the
    deaf layout would have put them."""
    from services.brains import brain_rng

    base = m.layout_from_settings({})
    wide = grow_inputs(base, 3)
    deaf = np.asarray(m.random(brain_rng(0.3), base), np.float32).reshape(-1)
    heard = np.asarray(m.random(brain_rng(0.3), wide), np.float32).reshape(-1)
    keep = np.setdiff1d(np.arange(wide.length), audio_weight_index(wide))
    np.testing.assert_array_equal(heard[keep], deaf)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_each_channels_weights_come_from_the_seed_and_its_own_number(m):
    """Column k is a function of (seed, k, brain): adding a channel never
    moves the ones before it, and one step to K equals K steps of one."""
    base = m.layout_from_settings({})
    deaf = np.asarray(m.random(np.random.default_rng(2), base),
                      np.float32).reshape(-1)
    one_l, two_l, three_l = (grow_inputs(base, k) for k in (1, 2, 3))
    one = transfer_audio_inputs(deaf, base, one_l, 0.42)
    two_step = transfer_audio_inputs(one, one_l, two_l, 0.42)
    two = transfer_audio_inputs(deaf, base, two_l, 0.42)
    np.testing.assert_array_equal(two_step, two)
    three = transfer_audio_inputs(deaf, base, three_l, 0.42)
    c2 = two[audio_weight_index(two_l)].reshape(-1, 2)
    c3 = three[audio_weight_index(three_l)].reshape(-1, 3)
    np.testing.assert_array_equal(c3[:, :2], c2)
    assert not np.array_equal(c3[:, 2], c3[:, 1])


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_every_generated_cohort_brain_is_its_deaf_self_with_ears(m):
    """One stream draws all the cohorts, so the ears must come AFTER every
    deaf draw or cohort 1's brain moves when cohort 0 grows ears."""
    from services.brains import generated_brains

    base = m.layout_from_settings({})
    wide = grow_inputs(base, 2)
    deaf = generated_brains(base, 0.37, 5)
    heard = generated_brains(wide, 0.37, 5, audio_seed=0.5)
    keep = np.setdiff1d(np.arange(wide.length), audio_weight_index(wide))
    for d, h in zip(deaf, heard):
        np.testing.assert_array_equal(h[keep], d)
    # The ears are the rig's: the same on every cohort, and the seed's.
    ears = [h[audio_weight_index(wide)] for h in heard]
    assert all(np.array_equal(e, ears[0]) for e in ears)
    other = generated_brains(wide, 0.37, 1, audio_seed=0.6)[0]
    assert not np.array_equal(other[audio_weight_index(wide)], ears[0])


def _one_more_unit(m, base):
    """`base` with one more unit, whatever the modality calls a unit."""
    from services.brains import settings_of

    s = dict(settings_of(base))
    if m.name == "mlp":
        layers = [list(x) for x in s["layers"]]
        layers[0][0] += 1
        s["layers"] = layers
    else:
        from services.brains.layout_moves import _structural_int

        key = _structural_int(m).key
        s[key] = int(s[key]) + 1
    return m.layout_from_settings(s)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_a_units_weight_for_a_channel_does_not_depend_on_the_unit_count(m):
    """The seed names a weight for (channel, unit): a brain one unit larger
    gains ONE new weight per channel and keeps every other, so a rig moved
    between two presets of nearly the same size hears nearly the same."""
    base = _one_more_unit(m, m.layout_from_settings({}))
    small, big = m.layout_from_settings({}), base
    n = int(small.shape[0])
    assert int(big.shape[0]) == n + 1
    s_wide, b_wide = grow_inputs(small, 2), grow_inputs(big, 2)
    s = transfer_audio_inputs(np.zeros(small.length, np.float32), small,
                              s_wide, 0.42)
    b = transfer_audio_inputs(np.zeros(big.length, np.float32), big,
                              b_wide, 0.42)
    s_cols = s[audio_weight_index(s_wide)].reshape(n, 2)
    b_cols = b[audio_weight_index(b_wide)].reshape(n + 1, 2)
    np.testing.assert_array_equal(b_cols[:n], s_cols)
    assert b_cols[n].any()
