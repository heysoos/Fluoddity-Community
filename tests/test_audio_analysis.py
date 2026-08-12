"""The band chain: a tone must land in its own band and nowhere else."""
import numpy as np
import pytest

from services.audio_analysis import (FFT_SIZE, SIGNAL_NAMES, Analyzer,
                                     analysis_matrix)

SR = 48000


def tone(hz, n=FFT_SIZE, sr=SR, amp=0.5):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_signal_names_are_the_five_the_ui_shows():
    assert SIGNAL_NAMES == ("bass", "mid", "presence", "hi", "volume")


def test_the_matrix_has_one_row_per_mel_bin_plus_one_per_band():
    m = analysis_matrix(SR, FFT_SIZE, n_mel=40)
    assert m.shape == (44, FFT_SIZE // 2 + 1)


def test_every_band_row_sums_to_one_so_it_is_a_mean():
    m = analysis_matrix(SR, FFT_SIZE, n_mel=40)
    for row in m[40:]:
        assert row.sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("hz,expected", [
    (80, "bass"), (900, "mid"), (3000, "presence"), (9000, "hi"),
])
def test_a_tone_lands_in_its_own_band(hz, expected):
    a = Analyzer(SR, auto_gain=False)
    snap = a.process(tone(hz))
    bands = {k: v for k, v in snap.signals.items() if k != "volume"}
    assert max(bands, key=bands.get) == expected


def test_silence_gives_zeros_and_no_nan():
    a = Analyzer(SR, auto_gain=False)
    snap = a.process(np.zeros(FFT_SIZE, dtype=np.float32))
    assert all(v == 0.0 for v in snap.signals.values())
    assert np.all(np.isfinite(snap.mel))


def test_dc_and_clipping_produce_no_nan():
    a = Analyzer(SR, auto_gain=True)
    for block in (np.ones(FFT_SIZE, dtype=np.float32),
                  np.full(FFT_SIZE, -1.0, dtype=np.float32),
                  np.zeros(FFT_SIZE, dtype=np.float32)):
        snap = a.process(block)
        assert all(np.isfinite(v) for v in snap.signals.values())


def test_band_edges_follow_the_device_rate_not_a_hardcoded_one():
    """A tone at 900 Hz is 'mid' at both rates, which only holds if the bin
    lookup uses the rate it was given."""
    for sr in (44100, 48000, 96000):
        a = Analyzer(sr, auto_gain=False)
        snap = a.process(tone(900, sr=sr))
        bands = {k: v for k, v in snap.signals.items() if k != "volume"}
        assert max(bands, key=bands.get) == "mid"


def test_auto_gain_lifts_a_quiet_signal_toward_full_scale():
    loud, quiet = Analyzer(SR), Analyzer(SR)
    for _ in range(200):
        hi = loud.process(tone(80, amp=0.5)).signals["bass"]
        lo = quiet.process(tone(80, amp=0.005)).signals["bass"]
    assert lo > 0.5 * hi


def test_auto_gain_off_leaves_a_quiet_signal_quiet():
    loud, quiet = Analyzer(SR, auto_gain=False), Analyzer(SR, auto_gain=False)
    for _ in range(200):
        hi = loud.process(tone(80, amp=0.5)).signals["bass"]
        lo = quiet.process(tone(80, amp=0.005)).signals["bass"]
    assert lo < 0.1 * hi


def test_signals_stay_inside_zero_to_one():
    a = Analyzer(SR)
    rng = np.random.default_rng(0)
    for _ in range(50):
        snap = a.process(rng.standard_normal(FFT_SIZE).astype(np.float32))
        assert all(0.0 <= v <= 1.0 for v in snap.signals.values())


def test_seq_increments_so_a_reader_can_tell_snapshots_apart():
    a = Analyzer(SR)
    first = a.process(tone(80)).seq
    assert a.process(tone(80)).seq == first + 1
