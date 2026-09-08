"""Each shaper against a step, an impulse and silence."""
import numpy as np
import pytest

from services.audio_shapers import SHAPER_KINDS, ShaperParams, ShaperState

DT = 1.0 / 60.0


def drive(state, params, values, dt=DT):
    return [state.apply(v, dt, params) for v in values]


def test_the_six_kinds_are_named():
    assert SHAPER_KINDS == ("none", "smooth", "gate", "envelope", "phase",
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


def test_the_band_drives_the_phase_RATE_not_the_phase_itself():
    """An INTEGRATOR: a steady note keeps the wave moving rather than parking
    it. This is the whole difference from a plain waveshaper."""
    p = ShaperParams(kind="phase", rate=4.0)
    out = drive(ShaperState(), p, [0.5] * 120)
    assert max(out) - min(out) > 0.9, "a held note should keep it cycling"


def test_the_phase_only_ever_goes_forward():
    """dphase = band, and the band cannot be negative, so it lurches forward
    and never rewinds however the music moves."""
    s = ShaperState()
    p = ShaperParams(kind="phase", rate=1.0, wave="ramp")
    rng = np.random.default_rng(0)
    seen = []
    for x in rng.random(200):
        s.apply(float(x), DT, p)
        seen.append(s._phase)
    # A ramp wraps at 1.0; unwrap before checking monotonicity.
    unwrapped = np.unwrap(np.asarray(seen) * 2 * np.pi) / (2 * np.pi)
    assert np.all(np.diff(unwrapped) >= -1e-9)


@pytest.mark.parametrize("wave", ("sine", "triangle", "ramp"))
def test_a_silent_band_FREEZES_the_wave_where_it_stands(wave):
    """Not back to zero - frozen. The complaint was MOTION with the music off,
    and a stopped integrator is motionless wherever it happens to be."""
    s = ShaperState()
    p = ShaperParams(kind="phase", rate=6.0, wave=wave)
    drive(s, p, [0.8] * 20)                    # music runs the phase along
    held = drive(s, p, [0.0] * 120)            # and then stops
    assert len(set(held)) == 1, "the wave kept moving on silence"


def test_a_never_driven_phase_shaper_contributes_nothing():
    """Phase starts at zero and every wave starts at zero, so a rig that has
    heard nothing yet leaves the parameter on the user's own value."""
    for wave in ("sine", "triangle", "ramp"):
        p = ShaperParams(kind="phase", rate=6.0, wave=wave)
        assert drive(ShaperState(), p, [0.0] * 60) == [0.0] * 60, wave


def test_rate_is_cycles_per_second_at_a_full_scale_band():
    s = ShaperState()
    p = ShaperParams(kind="phase", rate=3.0)
    for _ in range(60):                        # one second at full scale
        s.apply(1.0, 1 / 60.0, p)
    assert s._phase == pytest.approx(0.0, abs=1e-6)   # 3.0 cycles, wrapped
    s2 = ShaperState()
    for _ in range(30):                        # half a second
        s2.apply(1.0, 1 / 60.0, p)
    assert s2._phase == pytest.approx(0.5, abs=1e-6)  # 1.5 cycles


def test_half_a_band_advances_half_as_far():
    def phase_after(x):
        s = ShaperState()
        p = ShaperParams(kind="phase", rate=1.0)
        for _ in range(30):
            s.apply(x, 1 / 60.0, p)
        return s._phase

    assert phase_after(0.5) == pytest.approx(phase_after(1.0) / 2, abs=1e-6)


def test_the_phase_advance_depends_on_elapsed_time_not_frame_count():
    """Or a slow render would advance the wave less per second of music than a
    fast one, and a recording would not match what was on screen."""
    coarse = ShaperState()
    fine = ShaperState()
    p = ShaperParams(kind="phase", rate=2.0)
    for _ in range(10):
        coarse.apply(0.7, 1 / 30.0, p)
    for _ in range(20):
        fine.apply(0.7, 1 / 60.0, p)
    assert coarse._phase == pytest.approx(fine._phase, abs=1e-6)


def test_no_shaper_moves_on_its_own_when_the_band_is_silent():
    """The rule the free-running LFO broke. NOT 'silence gives zero' - a
    stopped integrator holds a non-zero value quite legitimately - but that
    silence generates no MOTION. Derived from SHAPER_KINDS, so a kind added
    later cannot skip it."""
    for kind in SHAPER_KINDS:
        s = ShaperState()
        p = ShaperParams(kind=kind)
        drive(s, p, [0.9] * 30)                # give it something to remember
        settled = drive(s, p, [0.0] * 600)     # long enough for any release
        tail = settled[-120:]
        # A tolerance, not equality: `smooth` converges on its target rather
        # than arriving, so it is still creeping by ~1e-18. The oscillation
        # this rules out swung the whole 0..1.
        assert max(tail) - min(tail) < 1e-9, kind


def test_no_shaper_moves_on_its_own_when_the_signal_is_only_being_HELD():
    """The same rule, and the hole the first version left in it.

    A held signal is not a silent one: `centroid` parks at whatever the last
    music was, so the test above - which drives an input of exactly zero -
    walked straight past a `phase` integrator swinging the full range with the
    track paused. `live=False` is how a shaper is told the difference.
    """
    for kind in SHAPER_KINDS:
        s = ShaperState()
        p = ShaperParams(kind=kind)
        for _ in range(30):
            s.apply(0.9, DT, p)                # something to remember
        settled = [s.apply(0.73, DT, p, False) for _ in range(600)]
        tail = settled[-120:]
        assert max(tail) - min(tail) < 1e-9, kind


def test_a_held_signal_still_reaches_the_shapers_at_its_real_value():
    """Holding must stop MOTION, not silence the signal - the whole point is
    that the parameter stays where the music left it."""
    s = ShaperState()
    p = ShaperParams(kind="none")
    assert s.apply(0.73, DT, p, False) == pytest.approx(0.73)


def test_a_live_signal_is_the_default():
    """Every caller that predates holding must behave exactly as it did."""
    live, told = ShaperState(), ShaperState()
    p = ShaperParams(kind="phase", rate=2.0)
    for _ in range(30):
        a = live.apply(0.7, DT, p)
        b = told.apply(0.7, DT, p, True)
    assert a == pytest.approx(b)
    assert live._phase > 0.0


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


def test_abs_is_on_by_default_and_keeps_the_raised_wave():
    """Every rig tuned before the box existed was tuned on 0..1."""
    assert ShaperParams().abs is True
    out = drive(ShaperState(), ShaperParams(kind="phase", rate=3.0), [0.7] * 90)
    assert min(out) >= 0.0 and max(out) <= 1.0 and max(out) > 0.9


@pytest.mark.parametrize("wave", ("sine", "triangle", "ramp"))
def test_a_signed_phase_swings_both_ways_and_still_starts_at_zero(wave):
    p = ShaperParams(kind="phase", rate=3.0, wave=wave, abs=False)
    assert drive(ShaperState(), p, [0.0] * 30) == [0.0] * 30
    out = drive(ShaperState(), p, [0.8] * 240)
    assert min(out) < -0.9 and max(out) > 0.9
    assert all(-1.0 <= v <= 1.0 for v in out)
