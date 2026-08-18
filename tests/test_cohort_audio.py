"""The cohort mask: a selection over the normalised cohort axis."""
import numpy as np
import pytest

from services import cohort_audio as ca
from services.cohort_tiling import MAX_COHORTS


def test_the_mask_is_as_wide_as_the_cohort_cap():
    assert ca.MASK_SLOTS == MAX_COHORTS


def test_a_fresh_mask_covers_every_cohort():
    m = ca.full_mask()
    assert m.shape == (ca.MASK_SLOTS,)
    assert m.dtype == np.bool_
    assert ca.is_full(m)
    assert not ca.is_empty(m)
    assert all(ca.covers(m, c, 64) for c in range(64))


def test_painting_a_cell_at_a_coarse_count_fills_its_whole_span():
    m = np.zeros(ca.MASK_SLOTS, dtype=bool)
    ca.paint(m, 0, 8, True)
    lo, hi = ca.paint_span(0, 8)
    assert (lo, hi) == (0, 18)
    assert m[lo:hi].all()
    assert not m[hi:].any()


def test_every_cell_paints_at_least_one_slot():
    for n in (1, 7, 63, 64, 100, 143, 144):
        for cell in range(n):
            lo, hi = ca.paint_span(cell, n)
            assert hi > lo, (n, cell)
            assert 0 <= lo < ca.MASK_SLOTS
            assert hi <= ca.MASK_SLOTS


def test_a_mask_holds_its_proportions_when_the_cohort_count_changes():
    """The whole point: the first half stays the first half."""
    m = np.zeros(ca.MASK_SLOTS, dtype=bool)
    for cell in range(32):
        ca.paint(m, cell, 64, True)

    for n in (8, 16, 64, 100, 144):
        lit = ca.cells_lit(m, n)
        assert lit[: n // 2].all(), n
        assert not lit[n // 2 :].any(), n


def test_cells_lit_is_one_entry_per_live_cohort():
    m = ca.full_mask()
    assert ca.cells_lit(m, 7).shape == (7,)
    assert ca.cells_lit(m, 144).shape == (144,)


def test_an_empty_mask_covers_nothing():
    m = np.zeros(ca.MASK_SLOTS, dtype=bool)
    assert ca.is_empty(m)
    assert not any(ca.covers(m, c, 64) for c in range(64))


def test_slot_of_never_leaves_the_mask():
    for n in (1, 3, 64, 144):
        for c in range(n):
            assert 0 <= ca.slot_of(c, n) < ca.MASK_SLOTS


def test_the_param_rows_are_the_ones_the_shader_can_reach():
    assert ca.COHORT_AUDIO_PARAMS == (
        "SENSOR_GAIN", "SENSOR_ANGLE", "SENSOR_DISTANCE", "MUTATION_SCALE",
        "GLOBAL_FORCE_MULT", "DRAG", "AXIAL_FORCE", "LATERAL_FORCE",
        "STRAFE_POWER", "HAZARD_RATE",
    )


def test_the_trail_and_the_clock_are_not_maskable():
    """canvas.frag has no cohort, and TIME_SCALE is one float for the canvas."""
    for name in ("TRAIL_PERSISTENCE", "TRAIL_DIFFUSION", "TIME_SCALE", "V_MAX"):
        assert name not in ca.COHORT_AUDIO_PARAMS


def test_every_maskable_param_is_a_real_modulation_target():
    from services.audio_mapping import physics_targets
    from state.sim_state import SimState

    keys = {t.key for t in physics_targets(SimState())}
    assert set(ca.COHORT_AUDIO_PARAMS) <= keys


def test_a_new_mapping_drives_every_cohort():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    assert m.cohorts.shape == (ca.MASK_SLOTS,)
    assert ca.is_full(m.cohorts)


def test_two_mappings_do_not_share_one_mask():
    from services.audio_mapping import Mapping
    a = Mapping(signal="bass", target="SENSOR_GAIN")
    b = Mapping(signal="hi", target="SENSOR_GAIN")
    a.cohorts[:10] = False
    assert ca.is_full(b.cohorts)


def test_an_unmasked_mapping_writes_nothing_to_the_rig():
    from services.audio_mapping import Mapping
    from state.audio_in_state import _mapping_to_dict
    d = _mapping_to_dict(Mapping(signal="bass", target="SENSOR_GAIN"))
    assert "cohorts" not in d


def test_a_mask_round_trips_through_the_rig():
    from services.audio_mapping import Mapping
    from state.audio_in_state import _mapping_from_dict, _mapping_to_dict
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    for cell in range(32):
        ca.paint(m.cohorts, cell, 64, False)

    back = _mapping_from_dict(_mapping_to_dict(m))
    assert back is not None
    assert np.array_equal(back.cohorts, m.cohorts)


def test_a_rig_written_before_this_feature_loads_as_all_cohorts():
    from state.audio_in_state import _mapping_from_dict
    m = _mapping_from_dict({"signal": "bass", "target": "SENSOR_GAIN",
                            "mode": "add", "depth": 0.5})
    assert m is not None
    assert ca.is_full(m.cohorts)


def test_a_malformed_mask_falls_back_to_all_cohorts():
    """One bad row must not lose the rig."""
    from state.audio_in_state import _mapping_from_dict
    for bad in ("nonsense", [1, 2, 3], [None] * ca.MASK_SLOTS, {}):
        m = _mapping_from_dict({"signal": "bass", "target": "SENSOR_GAIN",
                                "cohorts": bad})
        assert m is not None, bad
        assert ca.is_full(m.cohorts), bad


def test_a_mask_edit_is_a_rig_change():
    """_save_last_rig diffs by value, so an edit has to be visible there."""
    from services.audio_mapping import Mapping
    from state.audio_in_state import AudioInState, to_dict
    s = AudioInState()
    s.mappings = [Mapping(signal="bass", target="SENSOR_GAIN")]
    before = to_dict(s)
    ca.paint(s.mappings[0].cohorts, 0, 64, False)
    assert to_dict(s) != before


def _targets():
    from services.audio_mapping import physics_targets
    from state.sim_state import SimState
    return physics_targets(SimState())


def _apply(arr, row, cohort, base):
    """Exactly what cohort_audio() in the shader does.

    COHORT-indexed, and clamped to the bounds the row carries at MASK_SLOTS.
    """
    g, o = arr[row, cohort]
    lo, hi = arr[row, ca.MASK_SLOTS]
    return min(float(hi), max(float(lo), float(g) * base + float(o)))


def test_nothing_masked_is_reported_as_nothing_masked():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    arr, active = ca.build_arrays(
        [m], _targets(), {"bass": 1.0}, {}, {}, 1.0, 1 / 60.0, set(), 64)
    assert active is False


def test_a_masked_row_is_reported_as_masked():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    ca.paint(m.cohorts, 0, 64, False)
    _, active = ca.build_arrays(
        [m], _targets(), {"bass": 1.0}, {}, {}, 1.0, 1 / 60.0, set(), 64)
    assert active is True


def test_a_masked_row_moves_only_the_cohorts_it_names():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", mode="add", depth=0.5)
    m.cohorts[:] = False
    for cell in range(32):
        ca.paint(m.cohorts, cell, 64, True)

    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 64)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    assert _apply(arr, row, 0, 2.0) > 2.0
    assert _apply(arr, row, 63, 2.0) == pytest.approx(2.0)


def test_the_two_band_split_the_feature_exists_for():
    """bass ADDS over the first half, hi SUBTRACTS over the second."""
    from services.audio_mapping import Mapping
    lo = Mapping(signal="bass", target="SENSOR_GAIN", mode="add", depth=0.5)
    hi = Mapping(signal="hi", target="SENSOR_GAIN", mode="subtract", depth=0.5)
    lo.cohorts[:] = False
    hi.cohorts[:] = False
    for cell in range(32):
        ca.paint(lo.cohorts, cell, 64, True)
    for cell in range(32, 64):
        ca.paint(hi.cohorts, cell, 64, True)

    arr, _ = ca.build_arrays([lo, hi], _targets(), {"bass": 1.0, "hi": 1.0},
                             {}, {}, 1.0, 1 / 60.0, set(), 64)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    assert _apply(arr, row, 0, 2.0) > 2.0
    assert _apply(arr, row, 63, 2.0) < 2.0


def test_a_disabled_row_contributes_nothing():
    from services.audio_mapping import Mapping
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5, enabled=False)
    ca.paint(m.cohorts, 0, 64, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 64)
    assert _apply(arr, row, 40, 2.0) == pytest.approx(2.0)


def test_a_deaf_target_contributes_nothing():
    """A swept or muted parameter is refused here as it is in modulate()."""
    from services.audio_mapping import Mapping
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5)
    ca.paint(m.cohorts, 0, 64, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, {"SENSOR_GAIN"}, 64)
    assert _apply(arr, row, 40, 2.0) == pytest.approx(2.0)


def test_an_empty_mask_is_idle_rather_than_everything():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5)
    m.cohorts[:] = False
    arr, active = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                                  1 / 60.0, set(), 64)
    assert active is True
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    for c in (0, 31, 63):
        assert _apply(arr, row, c, 2.0) == pytest.approx(2.0)


@pytest.mark.parametrize("mode", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("strength", [0.0, 0.4, 1.0])
def test_a_covered_cohort_gets_exactly_what_modulate_would_give_it(mode, strength):
    """The property that keeps the two paths from drifting apart.

    The mask has to be painted somewhere or build_arrays returns identity and
    the comparison is vacuous - so cohort 63 is dropped and cohort 5, which is
    still covered, is the one measured.
    """
    from services.audio_mapping import Mapping, modulate
    from state.sim_state import SimState

    targets = _targets()
    base = float(getattr(SimState(), "SENSOR_GAIN"))
    signals = {"bass": 0.7}
    m = Mapping(signal="bass", target="SENSOR_GAIN", mode=mode, depth=0.6,
                gain=1.3)
    ca.paint(m.cohorts, 63, 64, False)
    assert ca.covers(m.cohorts, 5, 64)

    want = modulate({"SENSOR_GAIN": base}, targets, [m], signals, {},
                    {"SENSOR_GAIN": strength}, 1.0, 1 / 60.0, set(),
                    apply_shapers=False)["SENSOR_GAIN"]

    arr, active = ca.build_arrays([m], targets, signals, {},
                                  {"SENSOR_GAIN": strength}, 1.0, 1 / 60.0,
                                  set(), 64, apply_shapers=False)
    assert active is True
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    assert _apply(arr, row, 5, base) == pytest.approx(want, rel=1e-5)


def test_cohorts_past_the_live_count_are_identity():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=1.0)
    ca.paint(m.cohorts, 0, 8, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 8)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    tail = arr[row, 8:ca.MASK_SLOTS]
    assert np.allclose(tail[:, 0], 1.0)
    assert np.allclose(tail[:, 1], 0.0)


def test_a_row_nothing_masked_carries_open_bounds():
    """Or a masked row elsewhere would start clipping this one's sweep."""
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5)
    ca.paint(m.cohorts, 0, 64, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 64)
    untouched = ca.COHORT_AUDIO_PARAMS.index("DRAG")
    lo, hi = arr[untouched, ca.MASK_SLOTS]
    assert lo < -1e29 and hi > 1e29
    assert _apply(arr, untouched, 5, 12345.0) == pytest.approx(12345.0)


def test_a_masked_row_carries_the_bounds_modulate_clamps_to():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5)
    ca.paint(m.cohorts, 0, 64, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 64)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    target = next(t for t in _targets() if t.key == "SENSOR_GAIN")
    lo, hi = arr[row, ca.MASK_SLOTS]
    assert float(lo) == pytest.approx(
        target.hard_lo if target.hard_lo is not None else target.lo)
    assert float(hi) == pytest.approx(
        target.hard_hi if target.hard_hi is not None else target.hi)
