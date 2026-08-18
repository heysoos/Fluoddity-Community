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
