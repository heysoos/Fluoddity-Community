"""Each shaper against a step, an impulse and silence."""
import numpy as np
import pytest

from services.audio_shapers import SHAPER_KINDS, ShaperParams, ShaperState

DT = 1.0 / 60.0


def drive(state, params, values, dt=DT):
    return [state.apply(v, dt, params) for v in values]


def test_the_six_kinds_are_named():
    assert SHAPER_KINDS == ("none", "smooth", "gate", "envelope", "lfo",
                            "sample_hold")


def test_none_is_the_identity():
    s, p = ShaperState(), ShaperParams(kind="none")
    assert drive(s, p, [0.0, 0.3, 1.0]) == [0.0, 0.3, 1.0]


def test_smooth_rises_toward_a_step_without_reaching_it_at_once():
    s = ShaperState()
    p = ShaperParams(kind="smooth", attack=0.1, release=0.5)
    out = drive(s, p, [1.0] * 30)
    assert out[0] < 0.5
    assert out[-1] > 0.9
    assert out == sorted(out)


def test_smooth_releases_more_slowly_than_it_attacks():
    p = ShaperParams(kind="smooth", attack=0.01, release=1.0)
    up = ShaperState()
    for _ in range(10):
        risen = up.apply(1.0, DT, p)
    fell = up.apply(0.0, DT, p)
    assert risen > 0.9
    assert fell > 0.5          # release is slow, so one step barely moves it


def test_gate_is_zero_or_one_only():
    s = ShaperState()
    p = ShaperParams(kind="gate", threshold=0.5, hold=0.0)
    out = drive(s, p, [0.1, 0.9, 0.2, 0.7])
    assert set(out) <= {0.0, 1.0}


def test_gate_holds_open_for_its_hold_time():
    s = ShaperState()
    p = ShaperParams(kind="gate", threshold=0.5, hold=0.1)
    s.apply(0.9, DT, p)                       # opens
    held = [s.apply(0.0, DT, p) for _ in range(3)]
    assert all(v == 1.0 for v in held)        # 3 * 1/60 < 0.1s
    for _ in range(10):
        last = s.apply(0.0, DT, p)
    assert last == 0.0


def test_envelope_fires_on_a_crossing_and_decays_without_one():
    s = ShaperState()
    p = ShaperParams(kind="envelope", threshold=0.5, attack=0.01, release=0.3)
    s.apply(0.9, DT, p)
    peak = max(s.apply(0.0, DT, p) for _ in range(3))
    tail = [s.apply(0.0, DT, p) for _ in range(30)]
    assert peak > 0.5
    assert tail[-1] < 0.2


def test_envelope_does_not_retrigger_while_the_signal_stays_high():
    s = ShaperState()
    p = ShaperParams(kind="envelope", threshold=0.5, attack=0.001, release=0.2)
    out = drive(s, p, [0.9] * 40)
    assert out[-1] < out[3]      # decayed; a retrigger would keep it pinned


def test_lfo_oscillates_and_a_louder_signal_makes_it_faster():
    p = ShaperParams(kind="lfo", rate_min=1.0, rate_max=20.0, wave="sine")
    slow = drive(ShaperState(), p, [0.0] * 60)
    fast = drive(ShaperState(), p, [1.0] * 60)

    def crossings(seq):
        c = np.array(seq) - 0.5
        return int(np.count_nonzero(np.diff(np.sign(c)) != 0))

    assert crossings(fast) > crossings(slow)
    assert all(0.0 <= v <= 1.0 for v in slow + fast)


def test_lfo_runs_even_when_the_signal_is_silent():
    """rate_min is a floor, so the oscillator never stops dead."""
    out = drive(ShaperState(), ShaperParams(kind="lfo", rate_min=4.0), [0.0] * 60)
    assert max(out) - min(out) > 0.5


def test_sample_hold_latches_until_the_next_crossing():
    s = ShaperState()
    p = ShaperParams(kind="sample_hold", threshold=0.5)
    s.apply(0.8, DT, p)                        # crossing: latch 0.8
    assert s.apply(0.1, DT, p) == pytest.approx(0.8)
    assert s.apply(0.2, DT, p) == pytest.approx(0.8)
    s.apply(0.9, DT, p)                        # crossing again: latch 0.9
    assert s.apply(0.0, DT, p) == pytest.approx(0.9)


def test_every_shaper_survives_silence_and_stays_in_range():
    for kind in SHAPER_KINDS:
        s, p = ShaperState(), ShaperParams(kind=kind)
        out = drive(s, p, [0.0] * 120)
        assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in out), kind


def test_an_unknown_kind_falls_through_to_identity():
    """A rig file from a newer build must not crash this one."""
    s, p = ShaperState(), ShaperParams(kind="does_not_exist")
    assert s.apply(0.42, DT, p) == pytest.approx(0.42)


def test_reset_clears_carried_state():
    s = ShaperState()
    p = ShaperParams(kind="smooth", attack=0.5, release=0.5)
    for _ in range(30):
        s.apply(1.0, DT, p)
    s.reset()
    assert s.apply(1.0, DT, p) < 0.5
