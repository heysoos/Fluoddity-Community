"""The band chain: a tone must land in its own band and nowhere else."""
import numpy as np
import pytest

from services.audio_analysis import (FFT_SIZE, HOP, SIGNAL_NAMES, Analyzer,
                                     band_bins, mel_matrix)

SR = 48000


def tone(hz, n=FFT_SIZE, sr=SR, amp=0.5):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_signal_names_are_the_ones_the_ui_shows():
    assert SIGNAL_NAMES == ("bass", "mid", "presence", "hi", "volume",
                            "centroid")


def test_the_bands_are_the_spectral_ones_only():
    """`volume` and `centroid` are signals but not bands: neither is one
    band's bins reduced to a number."""
    from services.audio_analysis import BAND_EDGES_HZ, BAND_NAMES
    assert BAND_NAMES == tuple(n for n, _lo, _hi in BAND_EDGES_HZ)


def test_the_matrix_is_the_display_bars_and_nothing_else():
    """Bands left it deliberately: a mean over one of these rows is the one
    measure that cannot rest at zero."""
    m = mel_matrix(SR, FFT_SIZE, n_mel=40)
    assert m.shape == (40, FFT_SIZE // 2 + 1)


def test_every_band_covers_the_bins_its_edges_name():
    bins = band_bins(SR, FFT_SIZE)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SR)
    for name, lo_hz, hi_hz in [("bass", 20.0, 250.0), ("hi", 6000.0, 20000.0)]:
        lo, hi = bins[name]
        assert freqs[lo] >= lo_hz and freqs[hi - 1] < hi_hz
        assert freqs[lo - 1] < lo_hz


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
    m = mel_matrix(SR, FFT_SIZE, n_mel=40)
    for row in m:
        assert row.sum() == pytest.approx(1.0, abs=1e-5)


def test_a_flat_spectrum_is_not_a_ramp():
    m = mel_matrix(SR, FFT_SIZE, n_mel=40)
    mel = m @ np.ones(FFT_SIZE // 2 + 1, dtype=np.float32)
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


# --- how a band is measured --------------------------------------------------

def test_power_is_the_default_measure():
    """A band must be able to rest at zero, and averaging per-bin decibels
    cannot: the hundreds of bins carrying nothing set the result."""
    from services.audio_analysis import default_band
    assert default_band("bass")["measure"] == "power"


def _quiet_band(measure, amp=0.003):
    """What `hi` reads for a faint broadband signal - a quiet passage."""
    from services.audio_analysis import MEASURE_WINDOWS
    rng = np.random.default_rng(0)
    a = Analyzer(SR, auto_gain=False, release=0.0, bands={
        "hi": dict(zip(("floor", "ceiling"), MEASURE_WINDOWS[measure]),
                   measure=measure)})
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(30):
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = (rng.standard_normal(HOP) * amp).astype(
            np.float32)
        value = a.process(tail).signals["hi"]
    return value


def test_a_quiet_passage_reads_zero_under_power_and_did_not_under_mean_db():
    """The complaint this measure exists for: every band reporting a healthy
    signal with almost nothing playing."""
    assert _quiet_band("power") == 0.0
    assert _quiet_band("mean_db") > 0.1


def test_a_sparse_band_is_not_outvoted_by_its_empty_bins():
    """`hi` spans about 600 bins and a cymbal lights a handful. Averaging each
    bin's level hides it; summing the band's energy does not."""
    from services.audio_analysis import MEASURE_WINDOWS

    def read(measure):
        a = Analyzer(SR, auto_gain=False, release=0.0, bands={
            "hi": dict(zip(("floor", "ceiling"), MEASURE_WINDOWS[measure]),
                       measure=measure)})
        return a.process(tone(9000, amp=0.5)).signals["hi"]

    assert read("power") > 0.8
    assert read("mean_db") < 0.2


@pytest.mark.parametrize("measure", ["power", "rms", "peak", "mean_db"])
def test_every_measure_is_silent_on_silence_and_rises_with_level(measure):
    from services.audio_analysis import MEASURE_WINDOWS

    def read(amp):
        a = Analyzer(SR, auto_gain=False, release=0.0, bands={
            "bass": dict(zip(("floor", "ceiling"), MEASURE_WINDOWS[measure]),
                         measure=measure)})
        block = (np.zeros(FFT_SIZE, dtype=np.float32) if amp == 0
                 else tone(80, amp=amp))
        return a.process(block).signals["bass"]

    assert read(0) == 0.0
    assert 0.0 <= read(0.01) < read(0.5) <= 1.0


def test_the_floor_is_what_decides_where_zero_is():
    """Calibration, not a running average: the same sound always reads the
    same, which is exactly what a moving floor could not promise."""
    quiet = tone(80, amp=0.02)
    low = Analyzer(SR, auto_gain=False, release=0.0, bands={
        "bass": {"measure": "power", "floor": -60.0, "ceiling": -5.0}})
    high = Analyzer(SR, auto_gain=False, release=0.0, bands={
        "bass": {"measure": "power", "floor": -30.0, "ceiling": -5.0}})
    assert low.process(quiet).signals["bass"] > 0.2
    assert high.process(quiet).signals["bass"] == 0.0


def test_an_unknown_measure_falls_back_rather_than_raising():
    from services.audio_analysis import effective_bands
    bands = effective_bands({"bass": {"measure": "wavelets"}})
    assert bands["bass"]["measure"] == "power"


def test_volume_keeps_its_own_measure():
    """It is the block's loudness in the time domain; a spectral measure has
    no meaning for it."""
    from services.audio_analysis import VOLUME_MEASURE, effective_bands
    bands = effective_bands({"volume": {"measure": "peak"}})
    assert bands["volume"]["measure"] == VOLUME_MEASURE


def test_a_rig_that_stored_nothing_gets_the_defaults():
    from services.audio_analysis import SIGNAL_NAMES, effective_bands
    bands = effective_bands({})
    assert set(bands) == set(SIGNAL_NAMES)
    assert all("floor" in b and "ceiling" in b for b in bands.values())


def test_a_stored_window_overrides_only_what_it_names():
    from services.audio_analysis import MEASURE_WINDOWS, effective_bands
    bands = effective_bands({"mid": {"floor": -42.0}})
    assert bands["mid"]["floor"] == -42.0
    assert bands["mid"]["ceiling"] == MEASURE_WINDOWS["power"][1]


# --- flux: what is NEW, not what is there ------------------------------------

def _flux_analyzer():
    return Analyzer(SR, auto_gain=False, release=0.0, bands={
        "bass": {"measure": "flux", "floor": -35.0, "ceiling": 0.0}})


def _run(analyzer, blocks, amp):
    """Feed `blocks` of an 80 Hz tone; -> the `bass` reading per block."""
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    seen = []
    for i in range(blocks):
        t = (np.arange(HOP) + i * HOP) / SR
        chunk = (amp * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = chunk
        seen.append(analyzer.process(tail).signals["bass"])
    return seen


def test_flux_ignores_a_sustained_note_however_loud():
    """A held bass note keeps `power` open for its whole length; flux is
    supposed to fire on the attack and then get out of the way."""
    seen = _run(_flux_analyzer(), 30, 0.5)
    assert max(seen[-8:]) < 0.05


def test_flux_fires_when_the_sound_arrives():
    a = _flux_analyzer()
    _run(a, 8, 0.0)
    assert max(_run(a, 6, 0.5)) > 0.3


def test_the_first_block_is_not_treated_as_one_big_onset():
    """Everything is new against an empty buffer, so without a guard every
    flux mapping would fire the instant capture starts."""
    a = _flux_analyzer()
    assert a.process(tone(80, amp=0.5)).signals["bass"] == 0.0


def test_flux_is_silent_on_silence():
    assert max(_run(_flux_analyzer(), 10, 0.0)) == 0.0


# --- centroid: brightness, and not loudness ----------------------------------

def _centroid(hz, amp=0.3, blocks=4):
    a = Analyzer(SR, auto_gain=False, release=0.0)
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    for i in range(blocks):
        t = (np.arange(HOP) + i * HOP) / SR
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = (amp * np.sin(2 * np.pi * hz * t)
                                 ).astype(np.float32)
    return a.process(tail).signals["centroid"]


def test_a_brighter_sound_reads_higher():
    assert _centroid(300) < _centroid(1500) < _centroid(7000)


def test_centroid_ignores_level():
    """The property no band has: a quiet bright break and a loud dark one are
    indistinguishable to every loudness signal there is."""
    loud, quiet = _centroid(4000, amp=0.5), _centroid(4000, amp=0.02)
    assert loud == pytest.approx(quiet, abs=0.02)


def test_centroid_is_read_on_a_log_axis():
    """An octave is the same distance anywhere: 400->800 must move it as far
    as 2000->4000, or the whole control lives in the top of the range."""
    low = _centroid(800) - _centroid(400)
    high = _centroid(4000) - _centroid(2000)
    assert low == pytest.approx(high, abs=0.03)


def test_centroid_holds_through_silence_rather_than_going_dark():
    """A ratio over silence is not a dark sound, it is no sound - and a signal
    that dived to zero every gap would drag its target with it."""
    a = Analyzer(SR, auto_gain=False, release=0.0)
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    for i in range(6):
        t = (np.arange(HOP) + i * HOP) / SR
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = (0.4 * np.sin(2 * np.pi * 6000 * t)
                                 ).astype(np.float32)
        bright = a.process(tail).signals["centroid"]
    assert bright > 0.5

    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(10):
        held = a.process(silence).signals["centroid"]
    assert held == pytest.approx(bright, abs=1e-6)


def test_centroid_starts_at_zero_so_a_rig_that_has_heard_nothing_is_still():
    a = Analyzer(SR, auto_gain=False, release=0.0)
    assert a.process(np.zeros(FFT_SIZE, dtype=np.float32)
                     ).signals["centroid"] == 0.0


def test_auto_gain_leaves_the_centroid_alone():
    """Dividing a ratio by its own running peak means nothing, and would put
    every bright moment at 1.0."""
    gained = Analyzer(SR, auto_gain=True, release=0.0)
    plain = Analyzer(SR, auto_gain=False, release=0.0)
    block = tone(2000, amp=0.4)
    for _ in range(20):
        a = gained.process(block).signals["centroid"]
        b = plain.process(block).signals["centroid"]
    assert a == pytest.approx(b)


def test_the_centroid_window_is_in_hertz_not_decibels():
    from services.audio_analysis import (FIXED_MEASURES, MEASURE_UNITS,
                                         MEASURE_WINDOWS)
    measure = FIXED_MEASURES["centroid"]
    assert MEASURE_UNITS[measure] == "Hz"
    assert MEASURE_WINDOWS[measure] == (200.0, 8000.0)


def test_the_centroid_window_moves_where_the_range_sits():
    a = Analyzer(SR, auto_gain=False, release=0.0, bands={
        "centroid": {"measure": "centroid_hz", "floor": 2000.0,
                     "ceiling": 16000.0}})
    tail = np.zeros(FFT_SIZE, dtype=np.float32)
    for i in range(4):
        t = (np.arange(HOP) + i * HOP) / SR
        tail = np.roll(tail, -HOP)
        tail[FFT_SIZE - HOP:] = (0.3 * np.sin(2 * np.pi * 500 * t)
                                 ).astype(np.float32)
    assert a.process(tail).signals["centroid"] == 0.0


# --- the release is the user's, and the rise is nobody's ---------------------

def _fall(analyzer, blocks=6):
    """Loud, then silence: what `bass` reads on each block afterwards."""
    analyzer.process(tone(80, amp=0.5))
    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    return [analyzer.process(silence).signals["bass"] for _ in range(blocks)]


def test_a_zero_release_hands_the_shapers_the_raw_measurement():
    """The point of the shapers is to do the smoothing; the analyser must be
    able to stay out of it entirely."""
    assert _fall(Analyzer(SR, auto_gain=False, release=0.0))[0] == 0.0


def test_a_release_makes_the_fall_take_several_blocks():
    seen = _fall(Analyzer(SR, auto_gain=False, release=0.075))
    assert seen[0] > 0.3
    assert seen[-1] < seen[0]


def test_the_rise_is_never_smoothed_whatever_the_release():
    slow = Analyzer(SR, auto_gain=False, release=1.0)
    fast = Analyzer(SR, auto_gain=False, release=0.0)
    loud = tone(80, amp=0.5)
    assert slow.process(loud).signals["bass"] == pytest.approx(
        fast.process(loud).signals["bass"])


def test_the_release_can_be_retuned_while_blocks_arrive():
    a = Analyzer(SR, auto_gain=False, release=0.075)
    a.set_release(0.0)
    assert _fall(a)[0] == 0.0


def test_retuning_the_bands_takes_effect_on_the_next_block():
    a = Analyzer(SR, auto_gain=False, release=0.0)
    before = a.process(tone(80, amp=0.02)).signals["bass"]
    a.set_bands({"bass": {"measure": "power", "floor": -30.0,
                          "ceiling": -5.0}})
    assert before > 0.0
    assert a.process(tone(80, amp=0.02)).signals["bass"] == 0.0
