"""The band chain: a tone must land in its own band and nowhere else."""
import numpy as np
import pytest

from services.audio_analysis import (FFT_SIZE, HOP, SIGNAL_NAMES, Analyzer,
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


def test_auto_gain_off_leaves_a_quiet_signal_quieter():
    """The spectrum is read in decibels, so a hundredfold drop in amplitude is
    a fixed step down the scale rather than a hundredfold drop in the reading.
    """
    loud, quiet = Analyzer(SR, auto_gain=False), Analyzer(SR, auto_gain=False)
    for _ in range(200):
        hi = loud.process(tone(80, amp=0.5)).signals["bass"]
        lo = quiet.process(tone(80, amp=0.005)).signals["bass"]
    assert lo < hi - 0.05


# --- what stops the traces vibrating -----------------------------------------

def test_the_mel_rows_average_so_a_flat_spectrum_draws_flat():
    """Unnormalised triangles grow with the band's width, which tilts the
    display upward and reads as a ramp whatever is playing."""
    m = analysis_matrix(SR, FFT_SIZE, n_mel=40)
    for row in m[:40]:
        assert row.sum() == pytest.approx(1.0, abs=1e-5)


def test_a_flat_spectrum_is_not_a_ramp():
    m = analysis_matrix(SR, FFT_SIZE, n_mel=40)
    mel = m[:40] @ np.ones(FFT_SIZE // 2 + 1, dtype=np.float32)
    assert mel.max() / max(mel.min(), 1e-9) < 1.05


def test_smoothing_makes_a_band_move_far_less_per_block_than_its_input():
    """A per-block sample of band energy is what the trace was drawing."""
    rng = np.random.default_rng(0)
    a = Analyzer(SR, auto_gain=False)
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    values = []
    for _ in range(120):
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = rng.standard_normal(HOP).astype(np.float32) * 0.1
        values.append(a.process(tail).signals["bass"])
    steps = np.abs(np.diff(np.asarray(values[40:])))
    assert steps.mean() < 0.02


def _hat_through(analyzer, blocks=90, onset=30):
    """One sharp bright transient; returns the `hi` band per block."""
    rng = np.random.default_rng(0)
    n = HOP * (blocks - onset)
    t = np.arange(n) / SR
    hit = (rng.standard_normal(n) * np.exp(-t * 90.0) * 0.6).astype(np.float32)
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    seen = []
    for i in range(blocks):
        chunk = np.zeros(HOP, dtype=np.float32)
        if i >= onset:
            j = (i - onset) * HOP
            chunk = hit[j:j + HOP]
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = chunk
        seen.append(analyzer.process(tail).signals["hi"])
    return np.asarray(seen)


def test_a_transient_keeps_most_of_its_height():
    """A hi-hat is over in a few milliseconds. Smoothing its RISE reported a
    fraction of it and the highs lost their snap."""
    a = Analyzer(SR, auto_gain=False)
    assert _hat_through(a).max() > 0.6


def test_the_rise_is_not_smoothed_but_the_fall_is():
    """The two directions solve different problems: falling slowly is what
    stops a steady band vibrating, rising slowly only costs the transient."""
    from services.audio_analysis import ATTACK_SECONDS, SMOOTHING_SECONDS
    assert ATTACK_SECONDS < SMOOTHING_SECONDS

    a = Analyzer(SR, auto_gain=False)
    seen = _hat_through(a)
    peak = int(np.argmax(seen))
    rise = peak - int(np.argmax(seen > 0.05 * seen.max()))
    fall = int(np.argmax(seen[peak:] < 0.5 * seen.max()))
    assert fall > rise, f"fall {fall} blocks is not slower than rise {rise}"


def test_the_window_is_wide_and_the_hop_is_what_sets_time_resolution():
    """Narrowing the window to chase the transient costs frequency resolution
    where there is least to spare: the bass band is about ten bins wide."""
    from services.audio_analysis import BAND_EDGES_HZ
    bins = SR / FFT_SIZE
    _n, lo, hi = BAND_EDGES_HZ[0]
    assert (hi - lo) / bins >= 9.0
    assert HOP <= FFT_SIZE // 2, "a transient must be analysed more than once"


def test_volume_follows_loudness_rather_than_the_unlit_bins():
    """Averaging the spectrum measures the bins nothing is playing in, which
    outnumber the ones a bass hit lights by a hundred to one."""
    a = Analyzer(SR, auto_gain=False)
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    seen = []
    for i in range(160):
        t = (np.arange(HOP) + i * HOP) / SR
        env = np.exp(-8.0 * (t % 0.5))
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = (0.7 * env *
                                 np.sin(2 * np.pi * 60 * t)).astype(np.float32)
        seen.append(a.process(tail).signals["volume"])
    swing = max(seen[40:]) - min(seen[40:])
    assert swing > 0.2


def test_a_step_in_level_still_arrives_within_a_few_blocks():
    """Smoothing that outran the eye would read as the panel lagging."""
    a = Analyzer(SR, auto_gain=False)
    for _ in range(60):
        a.process(np.zeros(FFT_SIZE, dtype=np.float32))
    for i in range(1, 20):
        if a.process(tone(80, amp=0.5)).signals["bass"] > 0.2:
            break
    assert i < 15


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
