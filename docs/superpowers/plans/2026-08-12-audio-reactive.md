# Audio-Reactive Modulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live audio drives Fluoddity's physics and brain parameters, with a panel showing what every band is doing to every parameter.

**Architecture:** A capture thread owned by `PyAudioWPatch` runs the whole DSP chain and publishes one immutable `SignalSnapshot`. The frame loop reads the newest snapshot, computes modulated values through a pure mapping function, and hands `sim.apply_state()` a *copy* — the user's sliders are never written. The brain follows the same seam: encode `z` once, re-decode each frame with modulated scales.

**Tech Stack:** Python 3.12, NumPy, imgui_bundle (imgui + implot), PyAudioWPatch (new, optional), pytest.

**Spec:** [docs/superpowers/specs/2026-08-12-audio-reactive-design.md](../specs/2026-08-12-audio-reactive-design.md)

**Branch:** `worktree-audio-reactive`, off `integration` at `f2272d0` — the
variable-depth MLP stack, so `brains.STRUCTURAL_KINDS` is available as Task 3
assumes.

## Coexisting with the other branches in flight

The window module is named `audio_reactive_window.py`, NOT `audio_window.py`.
The unmerged sonification branch already ships `ui/audio_window.py` with a class
called `AudioWindowMixin`, a `render_audio_window()` method and a
`tests/test_audio_window_render.py` — all four names identical to what this plan
originally asked for. The two features point opposite ways (sim to MIDI out
there, audio in to physics here), so both will land, and one mixin would shadow
the other in the MRO with nothing raising. The spec caught this for the state
module and chose `audio_in_state.py`; it missed the window. `audio_in_state.py`
keeps its spec name — nothing collides with it.

These files are edited by more than one branch in flight and will conflict
textually at merge. Keep each addition to its own line, appended rather than
interleaved: `main.py`, `ui/core.py`, `state/__init__.py`, `state/ui_state.py`,
`requirements.txt`, `Fluoddity.spec`, `docs/testing_checklist.md`.

## Global Constraints

- **`PyAudioWPatch` must be imported lazily**, inside functions, never at module scope and never at startup. With it absent the feature disables itself and every other part of the app runs unchanged. This is the same rule `onnxruntime` and `cmaes` follow.
- **`sim.py` is user-owned.** Do not restructure it. This feature does not modify it at all.
- **The user's slider values are never written.** `UI.get_state()` returns the live `UIState`; modulation goes into a `dataclasses.replace()` copy handed to `sim.apply_state()`.
- **Analysis never runs on the frame loop.** It runs in the capture callback. The frame loop reads the newest snapshot and never blocks.
- **Traces use `imgui.plot_lines` over a `float32` numpy array** — never `add_line` per segment, never a Python list. ImPlot only where series overlay in one canvas.
- **Ring buffers stay `float32` numpy** end to end.
- **Structural brain settings are never audio targets.** Use `services.brains.STRUCTURAL_KINDS`, not a hardcoded kind list — it is `("int", "choice", "layers")` today and may grow.
- **Naming:** SimState fields and shader uniforms `ALL_CAPS_UNDERSCORE`; UI labels Title Case with spaces; private UI state `_snake_case`.
- **Comments state the rule, never the evidence.** No measured percentages, timings, or dates in comments, docstrings or tooltips. A tooltip is ONE sentence naming what the control does.
- **Windows platform** — forward slashes or `os.path`; `rm` not `del` in bash.
- **Run tests with the venv interpreter:** `.venv/Scripts/python.exe -m pytest tests/<file> -v`. Bare `python` is 3.10 and has no pytest.
- **Never `git add -A`.** Stage the exact files each step names; the working tree carries unrelated in-progress work.

---

## File Structure

| File | Responsibility |
|---|---|
| `services/audio_analysis.py` | Pure: FFT → mel rows + band rows in one matmul, auto-gain, `SignalSnapshot` |
| `services/audio_shapers.py` | Pure: the six shapers and their per-mapping state |
| `services/audio_mapping.py` | Pure: target registry, the modulation maths, sweep-deaf detection |
| `services/audio_brain.py` | Pure: encode-once / decode-many for brain scale modulation |
| `services/audio_capture.py` | The only module that performs IO: device list, stream, capture thread |
| `state/audio_in_state.py` | Mappings, strengths, live snapshot, persistence allowlist |
| `ui/audio_reactive_window.py` | Source row, spectrum, band traces, matrix, drawer |
| `ui/slider_widgets.py` | *(modify)* in-track swing display, `Audio…` context item |
| `main.py` | *(modify)* the apply seam, Auto/Explore suppression, cleanup step |

---

### Task 1: Band analysis

**Files:**
- Create: `services/audio_analysis.py`
- Test: `tests/test_audio_analysis.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SIGNAL_NAMES: tuple[str, ...]`; `FFT_SIZE: int`; `analysis_matrix(sample_rate: float, fft_size: int, n_mel: int) -> np.ndarray` of shape `(n_mel + 4, fft_size // 2 + 1)`; `SignalSnapshot` frozen dataclass with fields `signals: dict[str, float]`, `mel: np.ndarray`, `seq: int`; `Analyzer(sample_rate: float, fft_size: int = FFT_SIZE, n_mel: int = 40, auto_gain: bool = True)` with `.process(block: np.ndarray) -> SignalSnapshot`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_analysis.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.audio_analysis'`

- [ ] **Step 3: Write the implementation**

Create `services/audio_analysis.py`:

```python
"""Turn a block of samples into named signals.

Pure numpy, no IO. Runs on the capture thread, so nothing here may touch GL,
ImGui or any state the frame loop owns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FFT_SIZE = 2048
N_MEL = 40

SIGNAL_NAMES: tuple[str, ...] = ("bass", "mid", "presence", "hi", "volume")

# The four spectral bands, in Hz. `volume` is RMS and has no band.
BAND_EDGES_HZ: tuple[tuple[str, float, float], ...] = (
    ("bass", 20.0, 250.0),
    ("mid", 250.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("hi", 6000.0, 20000.0),
)

# A band's running peak decays by this factor per block, so auto-gain follows a
# track down as well as up.
_PEAK_DECAY = 0.9995
# Below this the peak is treated as silence rather than divided by.
_PEAK_FLOOR = 1e-6


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def analysis_matrix(sample_rate: float, fft_size: int = FFT_SIZE,
                    n_mel: int = N_MEL) -> np.ndarray:
    """Rows 0..n_mel-1 are a mel filterbank; the last four average one band each.

    Both live in one matrix so a single matmul yields the display spectrum and
    the band signals together.
    """
    n_bins = fft_size // 2 + 1
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    out = np.zeros((n_mel + len(BAND_EDGES_HZ), n_bins), dtype=np.float32)

    nyquist = sample_rate / 2.0
    edges = _mel_to_hz(np.linspace(_hz_to_mel(20.0),
                                   _hz_to_mel(min(16000.0, nyquist)),
                                   n_mel + 2))
    for i in range(n_mel):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs >= lo) & (freqs < ctr)
        falling = (freqs >= ctr) & (freqs < hi)
        if ctr > lo:
            out[i, rising] = (freqs[rising] - lo) / (ctr - lo)
        if hi > ctr:
            out[i, falling] = (hi - freqs[falling]) / (hi - ctr)

    for j, (_name, lo, hi) in enumerate(BAND_EDGES_HZ):
        sel = (freqs >= lo) & (freqs < min(hi, nyquist))
        count = int(np.count_nonzero(sel))
        if count:
            out[n_mel + j, sel] = 1.0 / count
    return out


@dataclass(frozen=True)
class SignalSnapshot:
    """One analysis result. Immutable so the frame loop may read it without a
    lock while the capture thread builds the next one."""
    signals: dict[str, float]
    mel: np.ndarray
    seq: int


class Analyzer:
    """Stateful across blocks: auto-gain peaks and the sequence counter."""

    def __init__(self, sample_rate: float, fft_size: int = FFT_SIZE,
                 n_mel: int = N_MEL, auto_gain: bool = True) -> None:
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.n_mel = int(n_mel)
        self.auto_gain = bool(auto_gain)
        self._m = analysis_matrix(sample_rate, fft_size, n_mel)
        self._window = np.hanning(fft_size).astype(np.float32)
        self._peaks = np.full(len(BAND_EDGES_HZ) + 1, _PEAK_FLOOR,
                              dtype=np.float32)
        self._seq = 0

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """raw is [bass, mid, presence, hi, volume]."""
        if not self.auto_gain:
            return np.clip(raw, 0.0, 1.0)
        self._peaks *= _PEAK_DECAY
        np.maximum(self._peaks, raw, out=self._peaks)
        return np.clip(raw / np.maximum(self._peaks, _PEAK_FLOOR), 0.0, 1.0)

    def process(self, block: np.ndarray) -> SignalSnapshot:
        b = np.asarray(block, dtype=np.float32)
        if b.size != self.fft_size:
            b = np.resize(b, self.fft_size)
        mag = np.abs(np.fft.rfft(b * self._window)).astype(np.float32)

        rows = self._m @ mag
        mel = rows[:self.n_mel]
        raw = np.empty(len(BAND_EDGES_HZ) + 1, dtype=np.float32)
        raw[:len(BAND_EDGES_HZ)] = rows[self.n_mel:]
        raw[-1] = np.sqrt(np.mean(b * b))

        # A DC or clipped block can still produce a non-finite magnitude on some
        # inputs; scrub here so nothing downstream has to.
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        norm = self._normalise(raw)

        self._seq += 1
        return SignalSnapshot(
            signals={n: float(v) for n, v in zip(SIGNAL_NAMES, norm)},
            mel=np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0),
            seq=self._seq,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_analysis.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add services/audio_analysis.py tests/test_audio_analysis.py
git commit -m "feat: a block of samples becomes five named signals

Mel rows and band rows share one matrix, so a single matmul yields both the
display spectrum and the signals. Band edges resolve against the rate the
device reports rather than an assumed 44100."
```

---

### Task 2: Shapers

**Files:**
- Create: `services/audio_shapers.py`
- Test: `tests/test_audio_shapers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SHAPER_KINDS: tuple[str, ...]`; `ShaperParams` dataclass with fields `kind, attack, release, threshold, hold, rate_min, rate_max, wave`; `ShaperState()` with `.apply(x: float, dt: float, p: ShaperParams) -> float` and `.reset()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_shapers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_shapers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.audio_shapers'`

- [ ] **Step 3: Write the implementation**

Create `services/audio_shapers.py`:

```python
"""Per-mapping shaping of a signal before the modulation maths.

Every shaper takes a value in [0,1] and returns one in [0,1]. State is per
mapping, because the same band commonly drives one target directly and another
through an oscillator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SHAPER_KINDS: tuple[str, ...] = ("none", "smooth", "gate", "envelope", "lfo",
                                 "sample_hold")


@dataclass
class ShaperParams:
    kind: str = "none"
    attack: float = 0.02        # seconds to rise
    release: float = 0.25       # seconds to fall
    threshold: float = 0.5      # gate / envelope / sample-hold trigger level
    hold: float = 0.05          # seconds a gate stays open after falling below
    rate_min: float = 0.5       # LFO Hz at signal 0
    rate_max: float = 12.0      # LFO Hz at signal 1
    wave: str = "sine"          # "sine" | "triangle" | "ramp"


def _coeff(seconds: float, dt: float) -> float:
    """One-pole coefficient reaching ~63% of a step in `seconds`."""
    if seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / seconds)


def _wave(phase: float, wave: str) -> float:
    if wave == "triangle":
        return 1.0 - abs(2.0 * (phase % 1.0) - 1.0)
    if wave == "ramp":
        return phase % 1.0
    return 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)


class ShaperState:
    """Carried between blocks for one mapping."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._value = 0.0
        self._phase = 0.0
        self._hold_left = 0.0
        self._open = False
        self._above = False
        self._latched = 0.0

    def apply(self, x: float, dt: float, p: ShaperParams) -> float:
        x = 0.0 if x != x else min(1.0, max(0.0, float(x)))
        dt = max(1e-6, float(dt))
        kind = p.kind

        if kind == "smooth":
            k = _coeff(p.attack if x > self._value else p.release, dt)
            self._value += (x - self._value) * k
            return min(1.0, max(0.0, self._value))

        if kind == "gate":
            if x >= p.threshold:
                self._open = True
                self._hold_left = p.hold
            elif self._open:
                self._hold_left -= dt
                if self._hold_left <= 0.0:
                    self._open = False
            return 1.0 if self._open else 0.0

        if kind == "envelope":
            crossed = x >= p.threshold and not self._above
            self._above = x >= p.threshold
            if crossed:
                self._value = 1.0
            else:
                self._value -= self._value * _coeff(p.release, dt)
            return min(1.0, max(0.0, self._value))

        if kind == "lfo":
            rate = p.rate_min + (p.rate_max - p.rate_min) * x
            self._phase = (self._phase + rate * dt) % 1.0
            return min(1.0, max(0.0, _wave(self._phase, p.wave)))

        if kind == "sample_hold":
            crossed = x >= p.threshold and not self._above
            self._above = x >= p.threshold
            if crossed:
                self._latched = x
            return self._latched

        # "none", and anything a newer build wrote that this one does not know.
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_shapers.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add services/audio_shapers.py tests/test_audio_shapers.py
git commit -m "feat: shape a signal before it modulates anything

Six shapers, state per mapping rather than per signal, because one band
commonly drives a target directly and another through an oscillator. An
unknown kind falls through to identity so a newer build's rig still opens."
```

---

### Task 3: Targets and the modulation maths

**Files:**
- Create: `services/audio_mapping.py`
- Test: `tests/test_audio_mapping.py`

**Interfaces:**
- Consumes: `services.audio_shapers.ShaperParams`, `ShaperState`.
- Produces: `TargetDef` frozen dataclass with `key, label, group, lo, hi, hard_lo, hard_hi`; `Mapping` dataclass with `signal, target, mode, depth, gain, shaper, enabled`; `MODES: tuple[str, ...]`; `physics_targets(sim_state) -> list[TargetDef]`; `brain_targets(modality, layout) -> list[TargetDef]`; `deaf_targets(sim_state) -> set[str]`; `modulate(bases, targets, mappings, signals, states, strengths, global_strength, dt, deaf) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_mapping.py`:

```python
"""The modulation maths, and which parameters may be modulated at all."""
import pytest

from services.audio_mapping import (MODES, Mapping, TargetDef, brain_targets,
                                    deaf_targets, modulate, physics_targets)
from services.audio_shapers import ShaperParams, ShaperState
from state.sim_state import SimState

DT = 1 / 60


def one(target="SENSOR_GAIN", **kw):
    kw.setdefault("signal", "bass")
    kw.setdefault("shaper", ShaperParams())
    return Mapping(target=target, **kw)


def gain_target(lo=0.0, hi=10.0, hard_lo=None, hard_hi=None):
    return TargetDef("SENSOR_GAIN", "Sensor Gain", "physics", lo, hi,
                     hard_lo, hard_hi)


def run(mappings, targets, signals, bases, **kw):
    states = {id(m): ShaperState() for m in mappings}
    return modulate(bases, targets, mappings, signals, states,
                    kw.get("strengths", {}), kw.get("global_strength", 1.0),
                    DT, kw.get("deaf", set()))


# --- targets -----------------------------------------------------------------

def test_modes_are_the_three_boids_had():
    assert MODES == ("add", "subtract", "multiply")


def test_physics_targets_come_from_the_registry_not_a_second_list():
    from ui.physics_params import PHYSICS_PARAMS
    keys = {t.key for t in physics_targets(SimState())}
    registry = {p.name for p in PHYSICS_PARAMS if not p.off_at_max}
    assert keys == registry


def test_a_parameter_that_switches_off_at_max_is_never_a_target():
    """V Max's top of track means Off; modulating it would make the readout lie."""
    assert "V_MAX" not in {t.key for t in physics_targets(SimState())}


def test_a_target_carries_the_users_current_slider_range():
    st = SimState()
    st.slider_ranges["Sensor Gain"] = [1.0, 4.0, 0.0, 10.0]
    t = next(t for t in physics_targets(st) if t.key == "SENSOR_GAIN")
    assert (t.lo, t.hi) == (1.0, 4.0)


def test_brain_targets_are_the_float_settings_only():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    keys = [t.key for t in brain_targets(m, layout)]
    assert keys == ["freq_scale", "low_freq_bias"]


def test_structural_brain_settings_are_never_targets():
    from services import brains
    for name in ("fourier", "gabor", "lenia", "mlp"):
        m = brains.get(name)
        layout = m.layout_from_settings({})
        schema = {s.key: s for s in m.settings_schema()}
        for t in brain_targets(m, layout):
            assert schema[t.key].kind not in brains.STRUCTURAL_KINDS


def test_a_modality_with_no_float_settings_offers_no_rows():
    from services import brains
    m = brains.get("mlp")
    assert brain_targets(m, m.layout_from_settings({})) == []


# --- deafness ----------------------------------------------------------------

def test_a_swept_parameter_is_reported_deaf():
    st = SimState()
    st.x_sweeps["SENSOR_GAIN"] = 1.0
    assert "SENSOR_GAIN" in deaf_targets(st)


def test_every_sweep_axis_counts():
    for axis in ("x_sweeps", "y_sweeps", "cohort_sweeps"):
        st = SimState()
        getattr(st, axis)["DRAG"] = -1.0
        assert "DRAG" in deaf_targets(st)


def test_nothing_is_deaf_by_default():
    assert deaf_targets(SimState()) == set()


def test_a_deaf_target_is_returned_at_its_base_value():
    out = run([one()], [gain_target()], {"bass": 1.0}, {"SENSOR_GAIN": 0.116},
              deaf={"SENSOR_GAIN"})
    assert out["SENSOR_GAIN"] == pytest.approx(0.116)


# --- the maths ---------------------------------------------------------------

def test_add_moves_by_depth_times_the_range():
    out = run([one(mode="add", depth=0.5)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 1.0})
    assert out["SENSOR_GAIN"] == pytest.approx(6.0)


def test_subtract_moves_the_other_way():
    out = run([one(mode="subtract", depth=0.2)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 5.0})
    assert out["SENSOR_GAIN"] == pytest.approx(3.0)


def test_multiply_scales_by_one_plus_depth():
    out = run([one(mode="multiply", depth=0.5)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 4.0})
    assert out["SENSOR_GAIN"] == pytest.approx(6.0)


def test_additive_resolves_before_multiplicative():
    """add then multiply: (1 + 1) * 1.5 = 3, not 1 + 1*1.5 = 2.5."""
    ms = [one(mode="add", depth=0.1), one(signal="mid", mode="multiply", depth=0.5)]
    out = run(ms, [gain_target(0.0, 10.0)], {"bass": 1.0, "mid": 1.0},
              {"SENSOR_GAIN": 1.0})
    assert out["SENSOR_GAIN"] == pytest.approx(3.0)


def test_gain_scales_the_signal_before_depth():
    out = run([one(mode="add", depth=0.5, gain=2.0)], [gain_target(0.0, 10.0)],
              {"bass": 0.25}, {"SENSOR_GAIN": 0.0})
    assert out["SENSOR_GAIN"] == pytest.approx(2.5)


def test_a_signal_amplified_past_one_is_clamped_before_use():
    a = run([one(mode="add", depth=1.0, gain=4.0)], [gain_target(0.0, 10.0)],
            {"bass": 1.0}, {"SENSOR_GAIN": 0.0})
    b = run([one(mode="add", depth=1.0, gain=1.0)], [gain_target(0.0, 10.0)],
            {"bass": 1.0}, {"SENSOR_GAIN": 0.0})
    assert a["SENSOR_GAIN"] == pytest.approx(b["SENSOR_GAIN"])


def test_strength_scales_the_whole_delta_not_one_term():
    ms = [one(mode="add", depth=0.2), one(signal="mid", mode="add", depth=0.2)]
    full = run(ms, [gain_target(0.0, 10.0)], {"bass": 1.0, "mid": 1.0},
               {"SENSOR_GAIN": 0.0})
    half = run(ms, [gain_target(0.0, 10.0)], {"bass": 1.0, "mid": 1.0},
               {"SENSOR_GAIN": 0.0}, global_strength=0.5)
    assert half["SENSOR_GAIN"] == pytest.approx(full["SENSOR_GAIN"] / 2)


def test_per_target_and_global_strength_multiply():
    out = run([one(mode="add", depth=1.0)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 0.0},
              strengths={"SENSOR_GAIN": 0.5}, global_strength=0.5)
    assert out["SENSOR_GAIN"] == pytest.approx(2.5)


def test_zero_strength_returns_the_base_exactly():
    out = run([one(mode="add", depth=1.0)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 0.3}, global_strength=0.0)
    assert out["SENSOR_GAIN"] == pytest.approx(0.3)


def test_a_disabled_mapping_contributes_nothing():
    out = run([one(mode="add", depth=1.0, enabled=False)],
              [gain_target(0.0, 10.0)], {"bass": 1.0}, {"SENSOR_GAIN": 0.3})
    assert out["SENSOR_GAIN"] == pytest.approx(0.3)


def test_an_unknown_signal_name_contributes_nothing():
    out = run([one(signal="nope", mode="add", depth=1.0)],
              [gain_target(0.0, 10.0)], {"bass": 1.0}, {"SENSOR_GAIN": 0.3})
    assert out["SENSOR_GAIN"] == pytest.approx(0.3)


def test_a_mapping_for_an_unknown_target_is_ignored():
    out = run([one(target="NOT_A_PARAM", mode="add", depth=1.0)],
              [gain_target()], {"bass": 1.0}, {"SENSOR_GAIN": 0.3})
    assert out == {"SENSOR_GAIN": pytest.approx(0.3)}


# --- clamping ----------------------------------------------------------------

def test_the_slider_range_clamps_when_there_is_no_hard_limit():
    out = run([one(mode="add", depth=1.0)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 8.0})
    assert out["SENSOR_GAIN"] == pytest.approx(10.0)


def test_a_hard_limit_wins_over_the_slider_range():
    t = gain_target(0.0, 10.0, hard_lo=0.0, hard_hi=2.0)
    out = run([one(mode="add", depth=1.0)], [t], {"bass": 1.0},
              {"SENSOR_GAIN": 1.0})
    assert out["SENSOR_GAIN"] == pytest.approx(2.0)


def test_a_bipolar_target_clamps_at_its_lower_rail():
    t = TargetDef("DRAG", "Drag", "physics", -1.0, 1.0, -1.0, 1.0)
    out = run([one(target="DRAG", mode="subtract", depth=1.0)], [t],
              {"bass": 1.0}, {"DRAG": 0.5})
    assert out["DRAG"] == pytest.approx(-1.0)


# --- the rule the whole design rests on --------------------------------------

def test_the_bases_dict_is_never_mutated():
    bases = {"SENSOR_GAIN": 0.116}
    run([one(mode="add", depth=1.0)], [gain_target()], {"bass": 1.0}, bases)
    assert bases == {"SENSOR_GAIN": 0.116}


def test_an_unbound_target_is_absent_from_the_result():
    """Only modulated targets are returned, so the caller copies nothing else."""
    targets = [gain_target(), TargetDef("DRAG", "Drag", "physics", -1, 1, None, None)]
    out = run([one()], targets, {"bass": 0.5}, {"SENSOR_GAIN": 1.0, "DRAG": 0.5})
    assert set(out) == {"SENSOR_GAIN"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_mapping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.audio_mapping'`

- [ ] **Step 3: Write the implementation**

Create `services/audio_mapping.py`:

```python
"""What may be modulated, and by how much.

Pure. Takes a dict of signal values and returns a dict of modulated values; it
cannot tell an FFT band from a shaper output from anything added later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from services.audio_shapers import ShaperParams, ShaperState

MODES: tuple[str, ...] = ("add", "subtract", "multiply")


@dataclass(frozen=True)
class TargetDef:
    """One modulatable parameter. Physics and brain rows share this shape."""
    key: str
    label: str
    group: str                 # "physics" | "brain"
    lo: float                  # modulation scale, from the live slider range
    hi: float
    hard_lo: float | None      # clamp, when the parameter declares one
    hard_hi: float | None


@dataclass
class Mapping:
    signal: str
    target: str
    mode: str = "add"
    depth: float = 0.5
    gain: float = 1.0
    shaper: ShaperParams = field(default_factory=ShaperParams)
    enabled: bool = True


def physics_targets(sim_state) -> list[TargetDef]:
    """Every physics slider except those whose maximum switches them off."""
    from ui.physics_params import PHYSICS_PARAMS

    out = []
    for p in PHYSICS_PARAMS:
        if p.off_at_max:
            continue
        rng = sim_state.slider_ranges.get(p.label)
        lo, hi = (rng[0], rng[1]) if rng else (p.default_min, p.default_max)
        out.append(TargetDef(p.name, p.label, "physics", float(lo), float(hi),
                             p.hard_min, p.hard_max))
    return out


def brain_targets(modality, layout) -> list[TargetDef]:
    """The active modality's decode scales.

    Structural settings change the parameter count, so modulating one would
    reshape the brain buffer every frame.
    """
    from services import brains

    out = []
    for s in modality.settings_schema():
        if s.kind in brains.STRUCTURAL_KINDS:
            continue
        out.append(TargetDef(s.key, s.label, "brain", float(s.lo), float(s.hi),
                             float(s.lo), float(s.hi)))
    return out


def deaf_targets(sim_state) -> set[str]:
    """Parameters a sweep has made unreachable.

    calculate_setting() returns slider_value only when every sweep is zero, so
    a swept parameter cannot be modulated at all.
    """
    deaf = set()
    for axis in ("x_sweeps", "y_sweeps", "cohort_sweeps"):
        for name, value in getattr(sim_state, axis, {}).items():
            if value:
                deaf.add(name)
    return deaf


def modulate(bases: dict[str, float], targets, mappings, signals,
             states: dict[int, ShaperState], strengths: dict[str, float],
             global_strength: float, dt: float,
             deaf: set[str]) -> dict[str, float]:
    """Modulated values for the targets that have an enabled mapping.

    `bases` is read and never written. Targets with no mapping, and targets a
    sweep has made deaf, are absent from the result.
    """
    by_target = {t.key: t for t in targets}
    adds: dict[str, list] = {}
    muls: dict[str, list] = {}

    for m in mappings:
        if not m.enabled or m.target not in by_target or m.target in deaf:
            continue
        if m.signal not in signals:
            continue
        (muls if m.mode == "multiply" else adds).setdefault(m.target, []).append(m)

    out: dict[str, float] = {}
    for key in set(adds) | set(muls):
        t = by_target[key]
        base = float(bases.get(key, 0.0))
        v = base
        span = t.hi - t.lo

        for m in adds.get(key, ()):
            state = states.setdefault(id(m), ShaperState())
            s = min(1.0, max(0.0, signals[m.signal] * m.gain))
            s = state.apply(s, dt, m.shaper)
            sign = -1.0 if m.mode == "subtract" else 1.0
            v += sign * s * m.depth * span

        for m in muls.get(key, ()):
            state = states.setdefault(id(m), ShaperState())
            s = min(1.0, max(0.0, signals[m.signal] * m.gain))
            s = state.apply(s, dt, m.shaper)
            v *= 1.0 + s * m.depth

        v = base + (v - base) * strengths.get(key, 1.0) * global_strength

        lo = t.hard_lo if t.hard_lo is not None else t.lo
        hi = t.hard_hi if t.hard_hi is not None else t.hi
        out[key] = min(hi, max(lo, v))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_mapping.py -v`
Expected: PASS, 27 tests

- [ ] **Step 5: Commit**

```bash
git add services/audio_mapping.py tests/test_audio_mapping.py
git commit -m "feat: which parameters audio may move, and by how much

Targets derive from PHYSICS_PARAMS and from a modality's own settings_schema,
so there is no second list to drift. A parameter whose maximum switches it off
is never a target, and a swept parameter is reported deaf rather than silently
ignoring its mappings.

The bases dict is read and never written - the modulated values go into a copy
the sim receives, so the user's sliders keep what the user set."
```

---

### Task 4: Brain scale modulation

**Files:**
- Create: `services/audio_brain.py`
- Test: `tests/test_audio_brain.py`

**Interfaces:**
- Consumes: `services.brains`.
- Produces: `BrainModulator()` with `.set_base(params, modality, layout) -> None`, `.modulated(scales: dict[str, float]) -> np.ndarray | None`, `.base_scales() -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_brain.py`:

```python
"""Encode once, decode many - the brain follows the same never-write-the-base rule."""
import numpy as np
import pytest

from services import brains
from services.audio_brain import BrainModulator


def fourier_base(n=10, seed=3):
    m = brains.get("fourier")
    layout = m.layout_from_settings({"centers": n})
    params = m.random(np.random.default_rng(seed), layout)
    return m, layout, params


def test_with_no_base_set_there_is_nothing_to_modulate():
    assert BrainModulator().modulated({"freq_scale": 2.0}) is None


def test_unmodulated_scales_reproduce_the_base_brain():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    out = mod.modulated(mod.base_scales())
    assert np.allclose(out, params, atol=1e-3)


def test_the_base_params_array_is_never_written():
    m, layout, params = fourier_base()
    before = params.copy()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    mod.modulated({"freq_scale": 6.0})
    assert np.array_equal(params, before)


def test_raising_freq_scale_scales_exactly_the_frequency_floats():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    base = mod.base_scales()
    out = mod.modulated({**base, "freq_scale": base["freq_scale"] * 2.0})

    g_in = params.reshape(-1, 8)
    g_out = np.asarray(out).reshape(-1, 8)
    assert np.allclose(g_out[:, :4], g_in[:, :4] * 2.0, atol=1e-2)
    assert np.allclose(g_out[:, 4:], g_in[:, 4:], atol=1e-3)


def test_the_output_width_matches_the_layout():
    m, layout, params = fourier_base(n=7)
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    assert np.asarray(mod.modulated(mod.base_scales())).size == layout.length


def test_base_scales_are_the_layouts_own():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    assert mod.base_scales() == {k: pytest.approx(v) for k, v in layout.scales}


def test_a_missing_scale_key_falls_back_to_the_base():
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    assert np.allclose(mod.modulated({}), params, atol=1e-3)


def test_setting_a_new_base_re_encodes():
    m, layout, first = fourier_base(seed=1)
    _, _, second = fourier_base(seed=2)
    mod = BrainModulator()
    mod.set_base(first, m, layout)
    mod.set_base(second, m, layout)
    assert np.allclose(mod.modulated(mod.base_scales()), second, atol=1e-3)


def test_a_wrong_width_base_is_refused_rather_than_reinterpreted():
    m, layout, _ = fourier_base(n=10)
    mod = BrainModulator()
    mod.set_base(np.zeros(7, dtype=np.float32), m, layout)
    assert mod.modulated({"freq_scale": 2.0}) is None


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia"])
def test_every_modality_with_scales_round_trips(name):
    m = brains.get(name)
    layout = m.layout_from_settings({})
    params = m.random(np.random.default_rng(0), layout)
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    out = mod.modulated(mod.base_scales())
    assert out is not None and np.asarray(out).size == layout.length


def test_decoding_repeatedly_does_not_drift():
    """The z is cached, so the hundredth frame matches the first."""
    m, layout, params = fourier_base()
    mod = BrainModulator()
    mod.set_base(params, m, layout)
    first = np.asarray(mod.modulated(mod.base_scales())).copy()
    for _ in range(100):
        mod.modulated({"freq_scale": 1.0 + np.random.random()})
    assert np.allclose(mod.modulated(mod.base_scales()), first, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_brain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.audio_brain'`

- [ ] **Step 3: Write the implementation**

Create `services/audio_brain.py`:

```python
"""Audio modulation of a brain's decode scales.

A scale change is a re-decode. Every modality already has a correct
encode()/decode() pair, so nothing here is modality-specific.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout


class BrainModulator:
    """Holds the search vector for the current brain.

    encode() clips at the rails and is lossy at the extremes, so the vector is
    computed once when the base brain changes and re-decoded every frame after.
    Re-encoding per frame would drift.
    """

    def __init__(self) -> None:
        self._z = None
        self._modality = None
        self._layout: BrainLayout | None = None

    def clear(self) -> None:
        self._z = None
        self._modality = None
        self._layout = None

    def set_base(self, params, modality, layout: BrainLayout) -> None:
        """Adopt a decoded brain. A wrong-width array is refused."""
        self.clear()
        if params is None:
            return
        arr = np.asarray(params, dtype=np.float32).reshape(-1)
        if arr.size != layout.length:
            return
        try:
            z, _clamped = modality.encode(arr, layout)
        except Exception:
            return
        self._z = np.asarray(z, dtype=np.float32)
        self._modality = modality
        self._layout = layout

    def base_scales(self) -> dict[str, float]:
        if self._layout is None:
            return {}
        return {k: float(v) for k, v in self._layout.scales}

    def modulated(self, scales: dict[str, float]):
        """The brain decoded under `scales`, or None if there is no base.

        Keys absent from `scales` keep the base layout's value.
        """
        if self._z is None or self._layout is None:
            return None
        merged = self.base_scales()
        for k, v in (scales or {}).items():
            if k in merged:
                merged[k] = float(v)
        layout = BrainLayout(self._layout.modality, self._layout.shape,
                             self._layout.length,
                             scales=tuple(merged.items()))
        try:
            return self._modality.decode(self._z, layout)
        except Exception:
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_brain.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add services/audio_brain.py tests/test_audio_brain.py
git commit -m "feat: audio moves a brain's decode scales

Encode the search vector once when the brain changes, re-decode each frame
under the modulated scales. Encoding per frame would drift, because encode()
clips at the rails.

Nothing here knows which modality it holds - every one of them already has a
correct encode/decode pair, so a modality that gains a float setting works
without a change."
```

---

### Task 5: Audio state

**Files:**
- Create: `state/audio_in_state.py`
- Modify: `state/ui_state.py`, `state/__init__.py`
- Test: `tests/test_audio_in_state.py`

**Interfaces:**
- Consumes: `services.audio_mapping.Mapping`, `services.audio_shapers.ShaperParams`.
- Produces: `AudioInState` dataclass; `PERSISTED_FIELDS: tuple[str, ...]`; `to_dict(state) -> dict`; `apply_dict(state, data) -> None`. `UIState` gains field `audio: AudioInState`.

> **Named `audio_in_state.py`, not `audio_state.py`** — the sonification work uses that name for MIDI out. One application should not have two things called audio state.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_in_state.py`:

```python
"""What a rig remembers, and what it must refuse to remember."""
import pytest

from services.audio_mapping import Mapping
from services.audio_shapers import ShaperParams
from state.audio_in_state import (PERSISTED_FIELDS, AudioInState, apply_dict,
                                  to_dict)


def test_it_hangs_off_ui_state():
    from state import UIState
    assert isinstance(UIState().audio, AudioInState)


def test_a_fresh_rig_is_empty_and_off():
    st = AudioInState()
    assert st.mappings == [] and st.enabled is False


def test_enabled_is_never_persisted():
    """Launching the app must not start capturing audio on its own."""
    assert "enabled" not in PERSISTED_FIELDS


def test_one_shot_commands_are_never_persisted():
    for name in ("request_start", "request_stop", "open_target"):
        assert name not in PERSISTED_FIELDS


def test_a_round_trip_preserves_a_mapping():
    st = AudioInState()
    st.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN",
                               mode="multiply", depth=0.25, gain=1.5,
                               shaper=ShaperParams(kind="lfo", rate_max=9.0)))
    st.global_strength = 1.5
    st.strengths["SENSOR_GAIN"] = 0.5

    fresh = AudioInState()
    apply_dict(fresh, to_dict(st))

    m = fresh.mappings[0]
    assert (m.signal, m.target, m.mode) == ("bass", "SENSOR_GAIN", "multiply")
    assert m.depth == pytest.approx(0.25) and m.gain == pytest.approx(1.5)
    assert m.shaper.kind == "lfo" and m.shaper.rate_max == pytest.approx(9.0)
    assert fresh.global_strength == pytest.approx(1.5)
    assert fresh.strengths["SENSOR_GAIN"] == pytest.approx(0.5)


def test_brain_mappings_round_trip_per_modality():
    st = AudioInState()
    st.brain_mappings["lenia"] = [Mapping(signal="hi", target="mu_scale")]
    fresh = AudioInState()
    apply_dict(fresh, to_dict(st))
    assert fresh.brain_mappings["lenia"][0].target == "mu_scale"
    assert "gabor" not in fresh.brain_mappings


def test_an_empty_dict_keeps_what_is_already_there():
    st = AudioInState()
    st.global_strength = 1.7
    apply_dict(st, {})
    assert st.global_strength == pytest.approx(1.7)


def test_a_rig_written_by_a_newer_build_still_opens():
    st = AudioInState()
    apply_dict(st, {"global_strength": 1.2, "some_future_field": 99})
    assert st.global_strength == pytest.approx(1.2)


def test_a_malformed_mapping_is_dropped_rather_than_crashing():
    st = AudioInState()
    apply_dict(st, {"mappings": [
        {"signal": "bass", "target": "DRAG", "mode": "add"},
        {"signal": "bass"},                       # no target
        "not a dict",
        {"signal": "bass", "target": "DRAG", "mode": "sideways"},
    ]})
    assert len(st.mappings) == 1
    assert st.mappings[0].target == "DRAG"


def test_strengths_are_clamped_to_the_slider_range():
    st = AudioInState()
    apply_dict(st, {"global_strength": 99.0, "strengths": {"DRAG": -4.0}})
    assert st.global_strength == pytest.approx(2.0)
    assert st.strengths["DRAG"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_in_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'state.audio_in_state'`

- [ ] **Step 3: Write the implementation**

Create `state/audio_in_state.py`:

```python
"""UI state for audio-driven modulation.

Named audio_in_state so it cannot be confused with the MIDI-out side. The
persisted set is an explicit allowlist: most of this dataclass is one-shot
commands and view buffers, and persisting one would replay a command on load.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from services.audio_mapping import MODES, Mapping
from services.audio_shapers import SHAPER_KINDS, ShaperParams

# `enabled` is deliberately absent: opening the app must never start capturing.
PERSISTED_FIELDS: tuple[str, ...] = (
    "mappings", "brain_mappings", "strengths", "global_strength",
    "auto_gain", "device_name",
)

_STRENGTH_MAX = 2.0


@dataclass
class AudioInState:
    enabled: bool = False               # is capture running
    show_window: bool = False           # is the panel open

    device_name: str = ""               # last chosen input, by name
    auto_gain: bool = True

    # Physics mappings, and brain mappings keyed by modality name so a rig
    # built for one brain is waiting when you switch back to it.
    mappings: list[Mapping] = field(default_factory=list)
    brain_mappings: dict[str, list[Mapping]] = field(default_factory=dict)

    strengths: dict[str, float] = field(default_factory=dict)
    global_strength: float = 1.0

    # One-shot commands, read and cleared by the orchestrator.
    request_start: bool = False
    request_stop: bool = False
    # Set by the physics slider's context menu; opens the panel on this row.
    open_target: str = ""

    # Live view state, written by the orchestrator for the panel to draw.
    status: str = "idle"                # "idle" | "active" | "error"
    last_error: str = ""


def _mapping_to_dict(m: Mapping) -> dict:
    return {
        "signal": m.signal, "target": m.target, "mode": m.mode,
        "depth": float(m.depth), "gain": float(m.gain),
        "enabled": bool(m.enabled),
        "shaper": {
            "kind": m.shaper.kind, "attack": m.shaper.attack,
            "release": m.shaper.release, "threshold": m.shaper.threshold,
            "hold": m.shaper.hold, "rate_min": m.shaper.rate_min,
            "rate_max": m.shaper.rate_max, "wave": m.shaper.wave,
        },
    }


def _mapping_from_dict(d) -> Mapping | None:
    """None for anything malformed, so one bad row cannot lose the rig."""
    if not isinstance(d, dict):
        return None
    signal, target = d.get("signal"), d.get("target")
    if not isinstance(signal, str) or not isinstance(target, str):
        return None
    mode = d.get("mode", "add")
    if mode not in MODES:
        return None
    sd = d.get("shaper") or {}
    shaper = ShaperParams()
    if isinstance(sd, dict):
        kind = sd.get("kind", "none")
        shaper = replace(
            shaper,
            kind=kind if kind in SHAPER_KINDS else "none",
            **{k: float(sd[k]) for k in
               ("attack", "release", "threshold", "hold", "rate_min", "rate_max")
               if isinstance(sd.get(k), (int, float))},
        )
        if isinstance(sd.get("wave"), str):
            shaper.wave = sd["wave"]
    try:
        return Mapping(
            signal=signal, target=target, mode=mode,
            depth=float(d.get("depth", 0.5)), gain=float(d.get("gain", 1.0)),
            shaper=shaper, enabled=bool(d.get("enabled", True)),
        )
    except (TypeError, ValueError):
        return None


def to_dict(state: AudioInState) -> dict:
    return {
        "mappings": [_mapping_to_dict(m) for m in state.mappings],
        "brain_mappings": {k: [_mapping_to_dict(m) for m in v]
                           for k, v in state.brain_mappings.items()},
        "strengths": {k: float(v) for k, v in state.strengths.items()},
        "global_strength": float(state.global_strength),
        "auto_gain": bool(state.auto_gain),
        "device_name": str(state.device_name),
    }


def _clamp_strength(v) -> float:
    return min(_STRENGTH_MAX, max(0.0, float(v)))


def apply_dict(state: AudioInState, data: dict) -> None:
    """Apply a stored rig. A missing key keeps the current value."""
    if not isinstance(data, dict):
        return

    if isinstance(data.get("mappings"), list):
        state.mappings = [m for m in
                          (_mapping_from_dict(d) for d in data["mappings"])
                          if m is not None]

    if isinstance(data.get("brain_mappings"), dict):
        state.brain_mappings = {
            k: [m for m in (_mapping_from_dict(d) for d in v) if m is not None]
            for k, v in data["brain_mappings"].items() if isinstance(v, list)
        }

    if isinstance(data.get("strengths"), dict):
        state.strengths = {k: _clamp_strength(v)
                           for k, v in data["strengths"].items()
                           if isinstance(v, (int, float))}

    if isinstance(data.get("global_strength"), (int, float)):
        state.global_strength = _clamp_strength(data["global_strength"])
    if isinstance(data.get("auto_gain"), bool):
        state.auto_gain = data["auto_gain"]
    if isinstance(data.get("device_name"), str):
        state.device_name = data["device_name"]
```

- [ ] **Step 4: Wire it into UIState**

In `state/ui_state.py`, add the import beside the others:

```python
from .audio_in_state import AudioInState
```

and the field after `brain`:

```python
    brain: BrainState = field(default_factory=BrainState)
    audio: AudioInState = field(default_factory=AudioInState)
```

In `state/__init__.py`, add the import and extend `__all__`:

```python
from .audio_in_state import AudioInState
```

```python
__all__ = ['SimState', 'CameraState', 'RecordingState', 'UIState', 'PreferencesState', 'save_preferences', 'load_preferences', 'MultiLoadState', 'TournamentState', 'AutoTournamentState', 'ArchiveState', 'BrainState', 'AudioInState']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_in_state.py -v`
Expected: PASS, 10 tests

- [ ] **Step 6: Commit**

```bash
git add state/audio_in_state.py state/ui_state.py state/__init__.py tests/test_audio_in_state.py
git commit -m "feat: an audio rig, and the allowlist of what it remembers

enabled is not on the list, so launching the app never starts capturing on its
own. Neither are the one-shot commands, which would otherwise replay on load.

A malformed mapping is dropped rather than raising: one bad row must not cost
the whole rig."
```

---

### Task 6: Capture

**Files:**
- Create: `services/audio_capture.py`
- Modify: `requirements.txt`, `Fluoddity.spec:74-92`
- Test: `tests/test_audio_capture.py`

**Interfaces:**
- Consumes: `services.audio_analysis.Analyzer`, `SignalSnapshot`.
- Produces: `is_available() -> bool`; `list_devices() -> list[dict]` each with keys `index: int`, `name: str`, `loopback: bool`, `rate: float`; `AudioCapture()` with `.start(device_index: int | None, auto_gain: bool) -> bool`, `.stop() -> None`, `.snapshot() -> SignalSnapshot | None`, `.status: str`, `.last_error: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_capture.py`:

```python
"""Capture is IO, so the tests cover only what can be checked without a device:
availability, graceful absence, and that nothing imports at module scope."""
import ast
from pathlib import Path

import pytest

from services import audio_capture

SRC = Path(__file__).resolve().parent.parent / "services" / "audio_capture.py"


def test_pyaudiowpatch_is_never_imported_at_module_scope():
    """CLAUDE.md: optional dependencies must stay out of the startup path."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in tree.body:                       # module level only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            mod = getattr(node, "module", "") or ""
            assert "pyaudio" not in mod.lower()
            assert not any("pyaudio" in n.lower() for n in names)


def test_is_available_answers_without_raising():
    assert isinstance(audio_capture.is_available(), bool)


def test_listing_devices_never_raises():
    devices = audio_capture.list_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert set(d) >= {"index", "name", "loopback", "rate"}


def test_a_fresh_capture_is_idle_with_no_snapshot():
    cap = audio_capture.AudioCapture()
    assert cap.status == "idle"
    assert cap.snapshot() is None


def test_stopping_a_capture_that_never_started_is_harmless():
    audio_capture.AudioCapture().stop()


def test_starting_without_the_package_reports_an_error_rather_than_raising(monkeypatch):
    monkeypatch.setattr(audio_capture, "is_available", lambda: False)
    cap = audio_capture.AudioCapture()
    assert cap.start(None, auto_gain=True) is False
    assert cap.status == "error"
    assert cap.last_error


def test_a_bad_device_index_reports_an_error_rather_than_raising():
    cap = audio_capture.AudioCapture()
    if not audio_capture.is_available():
        pytest.skip("PyAudioWPatch not installed")
    assert cap.start(999999, auto_gain=True) is False
    assert cap.status == "error"
    cap.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_capture.py -v`
Expected: FAIL — `ImportError: cannot import name 'audio_capture' from 'services'`

- [ ] **Step 3: Write the implementation**

Create `services/audio_capture.py`:

```python
"""The only module here that performs IO.

Analysis runs on the capture callback thread and publishes an immutable
snapshot; the frame loop reads the newest one and never blocks. A dropped audio
buffer therefore costs a stale signal rather than a stalled frame.
"""
from __future__ import annotations

import threading

import numpy as np

from services.audio_analysis import FFT_SIZE, Analyzer, SignalSnapshot

_BLOCK = 1024        # frames per callback; the analyser keeps its own history


def _pyaudio():
    """Imported here, never at module scope - it is an optional dependency."""
    import pyaudiowpatch
    return pyaudiowpatch


def is_available() -> bool:
    try:
        _pyaudio()
        return True
    except Exception:
        return False


def list_devices() -> list[dict]:
    """Inputs and loopbacks, or an empty list if the package is absent."""
    if not is_available():
        return []
    pa_mod = _pyaudio()
    pa = None
    out: list[dict] = []
    try:
        pa = pa_mod.PyAudio()
        seen = set()
        try:
            for info in pa.get_loopback_device_info_generator():
                out.append({"index": int(info["index"]),
                            "name": str(info["name"]),
                            "loopback": True,
                            "rate": float(info["defaultSampleRate"])})
                seen.add(int(info["index"]))
        except Exception:
            pass
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if i in seen or int(info.get("maxInputChannels", 0)) < 1:
                continue
            out.append({"index": i, "name": str(info["name"]),
                        "loopback": False,
                        "rate": float(info["defaultSampleRate"])})
    except Exception:
        return out
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
    return out


class AudioCapture:
    """Owns the stream and the analyser. Never raises into the frame loop."""

    def __init__(self) -> None:
        self.status = "idle"
        self.last_error = ""
        self.sample_rate = 0.0
        self._pa = None
        self._stream = None
        self._analyzer: Analyzer | None = None
        self._snapshot: SignalSnapshot | None = None
        self._lock = threading.Lock()
        self._tail = np.zeros(FFT_SIZE, dtype=np.float32)

    def snapshot(self) -> SignalSnapshot | None:
        with self._lock:
            return self._snapshot

    def _on_block(self, in_data, _frame_count, _time_info, _status):
        pa_mod = _pyaudio()
        try:
            samples = np.frombuffer(in_data, dtype=np.float32)
            if self._channels > 1:
                samples = samples.reshape(-1, self._channels).mean(axis=1)
            # Keep one FFT window of history so a small callback still analyses
            # a full block.
            n = min(samples.size, FFT_SIZE)
            self._tail = np.roll(self._tail, -n)
            self._tail[FFT_SIZE - n:] = samples[-n:]
            snap = self._analyzer.process(self._tail)
            with self._lock:
                self._snapshot = snap
        except Exception as exc:            # never kill the audio thread
            self.last_error = repr(exc)
        return (None, pa_mod.paContinue)

    def start(self, device_index: int | None, auto_gain: bool = True) -> bool:
        self.stop()
        if not is_available():
            self.status = "error"
            self.last_error = ("PyAudioWPatch is not installed, so audio input "
                               "is unavailable.")
            return False
        pa_mod = _pyaudio()
        try:
            self._pa = pa_mod.PyAudio()
            if device_index is None:
                info = self._pa.get_default_input_device_info()
                device_index = int(info["index"])
            else:
                info = self._pa.get_device_info_by_index(int(device_index))

            self.sample_rate = float(info["defaultSampleRate"])
            self._channels = max(1, int(info.get("maxInputChannels", 1)))
            self._analyzer = Analyzer(self.sample_rate, auto_gain=auto_gain)
            self._tail = np.zeros(FFT_SIZE, dtype=np.float32)

            self._stream = self._pa.open(
                format=pa_mod.paFloat32,
                channels=self._channels,
                rate=int(self.sample_rate),
                input=True,
                input_device_index=int(device_index),
                frames_per_buffer=_BLOCK,
                stream_callback=self._on_block,
            )
            self._stream.start_stream()
            self.status = "active"
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc) or repr(exc)
            self.stop()
            self.status = "error"
            return False

    def stop(self) -> None:
        for close in (self._close_stream, self._close_pa):
            try:
                close()
            except Exception:
                pass
        self._stream = None
        self._pa = None
        self._analyzer = None
        with self._lock:
            self._snapshot = None
        self.status = "idle"

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()

    def _close_pa(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
```

- [ ] **Step 4: Declare the dependency**

Append to `requirements.txt`:

```
PyAudioWPatch>=0.2.12.8
```

In `Fluoddity.spec`, add to the `hiddenimports` list (after `'imgui_bundle.python_backends.glfw_backend',`):

```python
    'pyaudiowpatch',
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_capture.py -v`
Expected: PASS, 7 tests (the last may skip if PyAudioWPatch is absent)

- [ ] **Step 6: Verify the app still starts without the package**

Run: `.venv/Scripts/python.exe -c "import main; print('startup import OK')"`
Expected: `startup import OK` — proving nothing pulled `pyaudiowpatch` into the import path.

- [ ] **Step 7: Commit**

```bash
git add services/audio_capture.py tests/test_audio_capture.py requirements.txt Fluoddity.spec
git commit -m "feat: capture the microphone or whatever the machine is playing

PyAudioWPatch exposes every output device as a loopback input, so system audio
needs no virtual cable. Imported inside functions only - with it absent the
feature disables itself and the rest of the app is unchanged.

Analysis runs on the callback thread and publishes an immutable snapshot, so a
dropped buffer costs a stale signal rather than a stalled frame."
```

---

### Task 7: The orchestrator seam

**Files:**
- Modify: `main.py` (`__init__`, `orchestrate_frame` around line 683, `cleanup` around line 1195)
- Create: `services/audio_runtime.py`
- Test: `tests/test_audio_runtime.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `AudioRuntime()` with `.update(ui_state, dt, brain_layout, current_rule) -> tuple[SimState, np.ndarray | None]`, `.close() -> None`, `.capture` attribute.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_runtime.py`:

```python
"""The seam: what the sim receives, and what the user's sliders keep."""
import numpy as np
import pytest

from services.audio_mapping import Mapping
from services.audio_runtime import AudioRuntime
from state import UIState


class FakeSnapshot:
    def __init__(self, **signals):
        self.signals = signals
        self.mel = np.zeros(40, dtype=np.float32)
        self.seq = 1


def rig(**kw):
    st = UIState()
    st.audio.enabled = True
    st.audio.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN",
                                     mode="add", depth=0.5))
    st.sim.SENSOR_GAIN = 1.0
    for k, v in kw.items():
        setattr(st.audio, k, v)
    return st


def runtime_with(signals):
    rt = AudioRuntime()
    rt.capture._snapshot = FakeSnapshot(**signals)
    return rt


def test_disabled_returns_the_state_object_untouched():
    rt, st = AudioRuntime(), rig()
    st.audio.enabled = False
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_with_no_snapshot_the_state_is_returned_untouched():
    rt, st = AudioRuntime(), rig()
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_the_sim_receives_a_modulated_copy():
    rt, st = runtime_with({"bass": 1.0}), rig()
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is not st.sim
    assert out.SENSOR_GAIN > 1.0


def test_the_users_slider_value_is_never_written():
    rt, st = runtime_with({"bass": 1.0}), rig()
    rt.update(st, 1 / 60, None, None)
    assert st.sim.SENSOR_GAIN == pytest.approx(1.0)


def test_the_copy_shares_nothing_the_caller_will_mutate():
    """A shallow copy shares the sweep dicts; writing one must not reach back."""
    rt, st = runtime_with({"bass": 1.0}), rig()
    out, _ = rt.update(st, 1 / 60, None, None)
    assert out.slider_ranges is st.sim.slider_ranges


def test_auto_mode_suppresses_modulation():
    """CLIP ranks tiles against each other; audio would move the physics
    mid-comparison."""
    rt, st = runtime_with({"bass": 1.0}), rig()
    st.auto_tournament.enabled = True
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_explore_mode_suppresses_modulation():
    rt, st = runtime_with({"bass": 1.0}), rig()
    st.archive.enabled = True
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_a_swept_target_is_left_alone():
    rt, st = runtime_with({"bass": 1.0}), rig()
    st.sim.x_sweeps["SENSOR_GAIN"] = 1.0
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out.SENSOR_GAIN == pytest.approx(1.0)


def test_a_brain_mapping_produces_a_modulated_brain():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    params = m.random(np.random.default_rng(0), layout)

    rt, st = runtime_with({"bass": 1.0}), rig()
    st.brain.modality = "fourier"
    st.audio.brain_mappings["fourier"] = [
        Mapping(signal="bass", target="freq_scale", mode="add", depth=0.5)
    ]
    _sim, brain = rt.update(st, 1 / 60, layout, params)
    assert brain is not None
    assert not np.allclose(brain, params)


def test_no_brain_mapping_means_no_brain_write():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    params = m.random(np.random.default_rng(0), layout)
    rt, st = runtime_with({"bass": 1.0}), rig()
    _sim, brain = rt.update(st, 1 / 60, layout, params)
    assert brain is None


def test_a_new_base_brain_is_re_encoded_rather_than_stacked():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    first = m.random(np.random.default_rng(1), layout)
    second = m.random(np.random.default_rng(2), layout)

    rt, st = runtime_with({"bass": 0.0}), rig()
    st.brain.modality = "fourier"
    st.audio.brain_mappings["fourier"] = [
        Mapping(signal="bass", target="freq_scale", mode="add", depth=0.5)
    ]
    rt.update(st, 1 / 60, layout, first)
    _sim, brain = rt.update(st, 1 / 60, layout, second)
    assert np.allclose(brain, second, atol=1e-2)


def test_close_is_safe_to_call_twice():
    rt = AudioRuntime()
    rt.close()
    rt.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.audio_runtime'`

- [ ] **Step 3: Write the implementation**

Create `services/audio_runtime.py`:

```python
"""Holds the capture, the shaper states and the brain vector between frames.

The orchestrator calls update() once per frame and hands what it returns to the
sim. Nothing here writes the user's state.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from services.audio_brain import BrainModulator
from services.audio_capture import AudioCapture
from services.audio_mapping import (brain_targets, deaf_targets, modulate,
                                    physics_targets)


class AudioRuntime:
    def __init__(self) -> None:
        self.capture = AudioCapture()
        self._brain = BrainModulator()
        self._states: dict[int, object] = {}
        self._base_brain_id = None

    def close(self) -> None:
        self.capture.stop()
        self._brain.clear()

    def _sync_capture(self, ast) -> None:
        if ast.request_start:
            ast.request_start = False
            index = None
            for d in __import__("services.audio_capture", fromlist=["x"]).list_devices():
                if d["name"] == ast.device_name:
                    index = d["index"]
                    break
            ast.enabled = self.capture.start(index, ast.auto_gain)
        if ast.request_stop:
            ast.request_stop = False
            self.capture.stop()
            ast.enabled = False
        ast.status = self.capture.status
        ast.last_error = self.capture.last_error

    def update(self, ui_state, dt: float, brain_layout, current_rule):
        """Returns (sim_state_for_the_sim, modulated_brain_or_None).

        The first is `ui_state.sim` itself when nothing is modulated, and a
        copy otherwise - so the caller never has to know which.
        """
        ast = ui_state.audio
        self._sync_capture(ast)

        # Auto and Explore rank tiles against each other. Modulating physics
        # mid-comparison would move what is being compared.
        if ui_state.auto_tournament.enabled or ui_state.archive.enabled:
            return ui_state.sim, None
        if not ast.enabled:
            return ui_state.sim, None

        snap = self.capture.snapshot()
        if snap is None:
            return ui_state.sim, None
        signals = snap.signals

        sim_out = ui_state.sim
        p_targets = physics_targets(ui_state.sim)
        deaf = deaf_targets(ui_state.sim)
        bases = {t.key: float(getattr(ui_state.sim, t.key, 0.0))
                 for t in p_targets}
        moved = modulate(bases, p_targets, ast.mappings, signals, self._states,
                         ast.strengths, ast.global_strength, dt, deaf)
        if moved:
            sim_out = replace(ui_state.sim, **moved)

        brain_out = self._update_brain(ui_state, ast, signals, dt,
                                       brain_layout, current_rule)
        return sim_out, brain_out

    def _update_brain(self, ui_state, ast, signals, dt, layout, current_rule):
        if layout is None or current_rule is None:
            return None
        mappings = ast.brain_mappings.get(ui_state.brain.modality, ())
        if not mappings:
            return None

        from services import brains
        modality = brains.get(ui_state.brain.modality)

        # Re-encode only when the base brain actually changes; encode() clips
        # at the rails, so a round trip per frame would drift.
        arr = np.asarray(current_rule, dtype=np.float32).reshape(-1)
        ident = (id(current_rule), arr.size, float(arr[:8].sum()) if arr.size else 0.0)
        if ident != self._base_brain_id:
            self._brain.set_base(arr, modality, layout)
            self._base_brain_id = ident

        targets = brain_targets(modality, layout)
        if not targets:
            return None
        bases = self._brain.base_scales()
        moved = modulate(bases, targets, mappings, signals, self._states,
                         ast.strengths, ast.global_strength, dt, set())
        if not moved:
            return None
        return self._brain.modulated({**bases, **moved})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_runtime.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Wire it into the orchestrator**

In `main.py`, in `App.__init__`, beside the other services:

```python
        from services.audio_runtime import AudioRuntime
        self.audio_runtime = AudioRuntime()
```

In `orchestrate_frame`, replace the single line `self.sim.apply_state(ui_state.sim)` (around line 683) with:

```python
        # Audio modulates a COPY. ui_state.sim keeps what the user set, so the
        # sliders do not drift and Save writes slider values rather than
        # whatever the music was doing at that instant.
        _audio_sim, _audio_brain = self.audio_runtime.update(
            ui_state, dt, self.sim.brain_layout,
            self.rule_manager.get_current_rule())
        self.sim.apply_state(_audio_sim)
        if _audio_brain is not None:
            self.sim.apply_rule(_audio_brain)
```

In `cleanup`, add a guarded step immediately before `self._step("advanced drawing", ...)`:

```python
        self._step("audio", self.audio_runtime.close)
```

- [ ] **Step 6: Verify the app still imports and the suite is green**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 7: Commit**

```bash
git add main.py services/audio_runtime.py tests/test_audio_runtime.py
git commit -m "feat: audio reaches the sim without touching the sliders

get_state() returns the live UIState, so modulating it in place would make the
sliders drift and would write momentary values into saved configs. The sim gets
a dataclasses.replace copy instead.

Auto and Explore suppress modulation entirely: they rank tiles against each
other, and audio would move the physics mid-comparison."
```

---

### Task 8: The panel

**Files:**
- Create: `ui/audio_reactive_window.py`
- Modify: `ui/core.py:19-56` (mixin), `ui/core.py:703` (render call), `ui/menu_bar.py`
- Test: `tests/test_audio_reactive_window_render.py`

**Interfaces:**
- Consumes: `services.audio_mapping`, `state.audio_in_state`.
- Produces: `AudioReactiveWindowMixin` with `render_audio_reactive_window()`; `SIGNAL_COLORS: dict[str, tuple]`; `TraceRing(length: int)` with `.push(v: float)` and `.values -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_reactive_window_render.py`:

```python
"""Render smoke, ID collisions, and the drawing rules the panel must follow."""
import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "ui" / "audio_reactive_window.py"


def test_the_mixin_is_part_of_UI():
    from ui import UI
    from ui.audio_reactive_window import AudioReactiveWindowMixin
    assert issubclass(UI, AudioReactiveWindowMixin)


def test_every_signal_has_a_colour():
    from services.audio_analysis import SIGNAL_NAMES
    from ui.audio_reactive_window import SIGNAL_COLORS
    assert set(SIGNAL_COLORS) == set(SIGNAL_NAMES)


def test_the_ring_is_float32_numpy_all_the_way_to_the_widget():
    from ui.audio_reactive_window import TraceRing
    r = TraceRing(64)
    r.push(0.5)
    assert isinstance(r.values, np.ndarray)
    assert r.values.dtype == np.float32
    assert r.values.size == 64


def test_the_ring_keeps_the_newest_sample_last():
    from ui.audio_reactive_window import TraceRing
    r = TraceRing(4)
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        r.push(v)
    assert r.values[-1] == pytest.approx(0.5)
    assert r.values[0] == pytest.approx(0.2)


def test_traces_never_use_add_line():
    """add_line per segment costs twenty times what plot_lines does."""
    src = SRC.read_text(encoding="utf-8")
    assert "add_line" not in src


def test_traces_go_through_plot_lines():
    assert "plot_lines" in SRC.read_text(encoding="utf-8")


def test_no_visible_label_is_used_twice():
    """Two visible items hashing to one ImGui ID puts a modal over the app and
    stops one of them responding to the mouse."""
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    labels: dict[str, int] = {}
    id_makers = {"button", "collapsing_header", "combo", "checkbox",
                 "slider_float", "begin_tab_item", "selectable"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in id_makers or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if "##" in first.value or not first.value:
                continue
            labels[first.value] = labels.get(first.value, 0) + 1
    dupes = {k: v for k, v in labels.items() if v > 1}
    assert not dupes, f"duplicate ImGui ids: {dupes}"


def test_the_window_renders_headless():
    from imgui_bundle import imgui

    from ui.audio_reactive_window import AudioReactiveWindowMixin
    from state import UIState

    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value

    class Host(AudioReactiveWindowMixin):
        def __init__(self):
            self.state = UIState()
            self.state.audio.show_window = True
            self.audio_runtime = None

        def _delayed_tooltip(self, text):
            pass

    host = Host()
    imgui.new_frame()
    host.render_audio_reactive_window()
    imgui.end_frame()
    imgui.render()


def test_a_closed_window_renders_nothing_and_does_not_raise():
    from imgui_bundle import imgui

    from ui.audio_reactive_window import AudioReactiveWindowMixin
    from state import UIState

    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value

    class Host(AudioReactiveWindowMixin):
        def __init__(self):
            self.state = UIState()
            self.state.audio.show_window = False
            self.audio_runtime = None

        def _delayed_tooltip(self, text):
            pass

    imgui.new_frame()
    Host().render_audio_reactive_window()
    imgui.end_frame()
    imgui.render()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_reactive_window_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.audio_reactive_window'`

- [ ] **Step 3: Write the implementation**

Create `ui/audio_reactive_window.py`:

```python
"""The Audio Reactive panel: source, spectrum, band traces, mapping matrix.

Traces go through imgui.plot_lines over a float32 array - one crossing into C++
per trace rather than one per segment.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services import audio_capture
from services.audio_analysis import SIGNAL_NAMES
from services.audio_mapping import (MODES, Mapping, brain_targets,
                                    deaf_targets, physics_targets)
from ui import layout

SIGNAL_COLORS: dict[str, tuple] = {
    "bass": (0.88, 0.31, 0.38, 1.0),
    "mid": (0.88, 0.56, 0.13, 1.0),
    "presence": (0.50, 0.82, 0.38, 1.0),
    "hi": (0.25, 0.63, 0.88, 1.0),
    "volume": (0.71, 0.55, 0.94, 1.0),
}

SIGNAL_ABBR: dict[str, str] = {
    "bass": "B", "mid": "M", "presence": "P", "hi": "H", "volume": "V",
}

TRACE_LEN = 180          # capped near the trace's width in pixels


class TraceRing:
    """A rolling window kept as float32 so it can go straight to plot_lines."""

    def __init__(self, length: int = TRACE_LEN) -> None:
        self._buf = np.zeros(int(length), dtype=np.float32)

    def push(self, value: float) -> None:
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = np.float32(value)

    @property
    def values(self) -> np.ndarray:
        return self._buf


class AudioReactiveWindowMixin:
    """Combined into UI via multiple inheritance."""

    def _audio_rings(self) -> dict:
        if not hasattr(self, "_audio_trace_rings"):
            self._audio_trace_rings = {n: TraceRing() for n in SIGNAL_NAMES}
        return self._audio_trace_rings

    def _audio_devices(self, refresh: bool = False) -> list:
        if refresh or not hasattr(self, "_audio_device_cache"):
            self._audio_device_cache = audio_capture.list_devices()
        return self._audio_device_cache

    def render_audio_reactive_window(self):
        ast = self.state.audio
        if not ast.show_window:
            return

        imgui.set_next_window_size(imgui.ImVec2(420, 640),
                                   imgui.Cond_.first_use_ever)
        layout.constrain_panel()
        _expanded, opened = imgui.begin("Audio Reactive", True)
        if not opened:
            ast.show_window = False
            imgui.end()
            return

        # push_settings_width pairs with imgui.pop_item_width, not a layout call.
        layout.push_settings_width()
        self._render_audio_source(ast)
        imgui.separator()
        self._render_audio_signals(ast)
        imgui.separator()
        self._render_audio_matrix(ast)
        imgui.pop_item_width()
        imgui.end()

    # --- source ---------------------------------------------------------

    def _render_audio_source(self, ast):
        if not audio_capture.is_available():
            imgui.text_wrapped(
                "Audio input needs PyAudioWPatch, which is not installed.")
            return

        devices = self._audio_devices()
        names = [f"{'loopback' if d['loopback'] else 'input'}: {d['name']}"
                 for d in devices]
        current = next((i for i, d in enumerate(devices)
                        if d["name"] == ast.device_name), 0)
        if names:
            changed, idx = imgui.combo("Device", current, names)
            if changed:
                ast.device_name = devices[idx]["name"]
        self._delayed_tooltip("Which input the bands are read from.")

        if ast.enabled:
            if imgui.button("Stop##audio_source"):
                ast.request_stop = True
        else:
            if imgui.button("Start##audio_source"):
                ast.request_start = True
        imgui.same_line()
        if imgui.button("Rescan##audio_source"):
            self._audio_devices(refresh=True)
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(0.5, 0.8, 0.4, 1.0) if ast.status == "active"
            else imgui.ImVec4(0.8, 0.35, 0.35, 1.0) if ast.status == "error"
            else imgui.ImVec4(0.5, 0.5, 0.55, 1.0),
            ast.status)

        if ast.status == "error" and ast.last_error:
            imgui.text_wrapped(ast.last_error)

        changed, value = imgui.checkbox("Auto Gain", ast.auto_gain)
        if changed:
            ast.auto_gain = value
        self._delayed_tooltip("Normalises each band against its recent peak.")

        changed, value = imgui.slider_float("Strength", ast.global_strength,
                                            0.0, 2.0, "%.2f")
        if changed:
            ast.global_strength = value
        self._delayed_tooltip("Scales every mapping at once.")

    # --- spectrum and band traces ---------------------------------------

    def _render_audio_signals(self, ast):
        runtime = getattr(self, "audio_runtime", None)
        snap = runtime.capture.snapshot() if runtime is not None else None
        rings = self._audio_rings()
        if snap is not None:
            for name in SIGNAL_NAMES:
                rings[name].push(snap.signals.get(name, 0.0))
            imgui.plot_lines("##audio_spectrum",
                             np.asarray(snap.mel, dtype=np.float32),
                             graph_size=imgui.ImVec2(0, 46))

        width = max(40.0, imgui.get_content_region_avail().x / len(SIGNAL_NAMES) - 4)
        for i, name in enumerate(SIGNAL_NAMES):
            if i:
                imgui.same_line()
            imgui.begin_group()
            imgui.push_style_color(imgui.Col_.plot_lines,
                                   imgui.ImVec4(*SIGNAL_COLORS[name]))
            imgui.plot_lines(f"##audio_trace_{name}", rings[name].values,
                             scale_min=0.0, scale_max=1.0,
                             graph_size=imgui.ImVec2(width, 26))
            imgui.pop_style_color()
            imgui.text_colored(imgui.ImVec4(*SIGNAL_COLORS[name]), name[:4])
            imgui.end_group()

    # --- the matrix -----------------------------------------------------

    def _mapping_list(self, ast, group: str) -> list:
        if group == "physics":
            return ast.mappings
        return ast.brain_mappings.setdefault(self.state.brain.modality, [])

    def _render_audio_matrix(self, ast):
        deaf = deaf_targets(self.state.sim)
        groups = [("physics", "Physics", physics_targets(self.state.sim))]

        from services import brains
        modality = brains.get(self.state.brain.modality)
        b_layout = getattr(getattr(self, "sim", None), "brain_layout", None)
        if b_layout is None:
            b_layout = modality.layout_from_settings(self.state.brain.settings)
        groups.append(("brain", "Brain", brain_targets(modality, b_layout)))

        for group, title, targets in groups:
            if not imgui.collapsing_header(f"{title}##audio_group"):
                continue
            if not targets:
                imgui.text_disabled("This brain has no scales to modulate.")
                continue
            mappings = self._mapping_list(ast, group)
            for target in targets:
                self._render_audio_row(ast, group, mappings, target,
                                       target.key in deaf)

    def _render_audio_row(self, ast, group, mappings, target, is_deaf):
        bound = [m for m in mappings if m.target == target.key]
        imgui.push_id(f"{group}:{target.key}")

        if bound:
            imgui.text(target.label)
        else:
            imgui.text_disabled(target.label)
        if is_deaf:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(0.85, 0.65, 0.25, 1.0), "swept")
            self._delayed_tooltip(
                "A swept parameter ignores its slider value, so audio cannot "
                "move it. Zero the sweeps to use this row.")

        for signal in SIGNAL_NAMES:
            imgui.same_line()
            existing = next((m for m in bound if m.signal == signal), None)
            colour = imgui.ImVec4(*SIGNAL_COLORS[signal])
            if existing is None:
                colour = imgui.ImVec4(colour.x, colour.y, colour.z, 0.30)
            imgui.push_style_color(imgui.Col_.button, colour)
            if imgui.button(f"{SIGNAL_ABBR[signal]}##{signal}"):
                if existing is None:
                    mappings.append(Mapping(signal=signal, target=target.key))
                else:
                    mappings.remove(existing)
            imgui.pop_style_color()

        for m in bound:
            self._render_audio_mapping(ast, mappings, m)
        imgui.pop_id()

    def _render_audio_mapping(self, ast, mappings, m):
        imgui.push_id(m.signal)
        imgui.indent()
        imgui.text_colored(imgui.ImVec4(*SIGNAL_COLORS[m.signal]), m.signal)

        imgui.same_line()
        if imgui.button(m.mode):
            m.mode = MODES[(MODES.index(m.mode) + 1) % len(MODES)]
        self._delayed_tooltip("Cycles add, subtract and multiply.")

        changed, value = imgui.slider_float("Depth", m.depth, 0.0, 1.0, "%.2f")
        if changed:
            m.depth = value
        changed, value = imgui.slider_float("Gain", m.gain, 0.0, 4.0, "%.2f")
        if changed:
            m.gain = value

        imgui.unindent()
        imgui.pop_id()
```

- [ ] **Step 4: Register the mixin and the menu item**

In `ui/core.py`, add the import beside the other window mixins (after line 32):

```python
from .audio_reactive_window import AudioReactiveWindowMixin
```

Add it to the `UI` base list, after `BrainWindowMixin,`:

```python
    AudioReactiveWindowMixin,
```

Beside `self.render_brain_window()` (around line 703), add:

```python
            self.render_audio_reactive_window()
```

In `ui/menu_bar.py`, inside the Extras menu, beside the Archive Browser item, add:

```python
            if imgui.menu_item("Audio Reactive", "", False)[0]:
                self.state.audio.show_window = True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_reactive_window_render.py -v`
Expected: PASS, 9 tests

- [ ] **Step 6: Verify no label width regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_label_widths.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ui/audio_reactive_window.py ui/core.py ui/menu_bar.py tests/test_audio_reactive_window_render.py
git commit -m "feat: the panel that shows what each band is doing

Traces go through plot_lines over a float32 array, one crossing into C++ per
trace rather than one per segment. A swept row is badged, because a sweep makes
a parameter ignore its slider value and the mapping would otherwise appear
broken."
```

---

### Task 9: In-track modulation display

**Files:**
- Modify: `ui/slider_widgets.py:23-115` (`slider_float_with_range_menu`), `ui/slider_widgets.py:120-181` (context menu)
- Test: `tests/test_slider_audio_display.py`

**Interfaces:**
- Consumes: `state.audio_in_state.AudioInState`.
- Produces: `swing_fraction(base, lo, hi, reach) -> tuple[float, float]` returning `(start, width)` as fractions of the track.

- [ ] **Step 1: Write the failing test**

Create `tests/test_slider_audio_display.py`:

```python
"""The hatched swing and the base tick, in track fractions."""
import pytest

from ui.slider_widgets import swing_fraction


def test_a_swing_upward_starts_at_the_base():
    start, width = swing_fraction(base=2.0, lo=0.0, hi=10.0, reach=5.0)
    assert start == pytest.approx(0.2)
    assert width == pytest.approx(0.3)


def test_a_swing_downward_ends_at_the_base():
    start, width = swing_fraction(base=5.0, lo=0.0, hi=10.0, reach=2.0)
    assert start == pytest.approx(0.2)
    assert width == pytest.approx(0.3)


def test_a_swing_past_the_top_is_clipped_to_the_track():
    start, width = swing_fraction(base=8.0, lo=0.0, hi=10.0, reach=40.0)
    assert start == pytest.approx(0.8)
    assert start + width == pytest.approx(1.0)


def test_a_swing_past_the_bottom_is_clipped_to_the_track():
    start, width = swing_fraction(base=1.0, lo=0.0, hi=10.0, reach=-40.0)
    assert start == pytest.approx(0.0)
    assert start + width == pytest.approx(0.1)


def test_a_bipolar_range_places_zero_in_the_middle():
    start, _width = swing_fraction(base=0.0, lo=-1.0, hi=1.0, reach=0.0)
    assert start == pytest.approx(0.5)


def test_no_swing_has_zero_width():
    _start, width = swing_fraction(base=3.0, lo=0.0, hi=10.0, reach=3.0)
    assert width == pytest.approx(0.0)


def test_a_degenerate_range_does_not_divide_by_zero():
    start, width = swing_fraction(base=1.0, lo=1.0, hi=1.0, reach=5.0)
    assert start == pytest.approx(0.0) and width == pytest.approx(0.0)


def test_the_slider_never_uses_add_line_for_the_swing():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "ui" / "slider_widgets.py").read_text(encoding="utf-8")
    assert "add_line(" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slider_audio_display.py -v`
Expected: FAIL — `ImportError: cannot import name 'swing_fraction'`

- [ ] **Step 3: Write the helper**

At module level in `ui/slider_widgets.py`, above the mixin class:

```python
def swing_fraction(base: float, lo: float, hi: float,
                   reach: float) -> tuple[float, float]:
    """Where the modulation's reachable span sits on the track, as (start, width).

    Both are fractions in [0,1], clipped to the track, so the hatching can never
    be drawn outside the slider it belongs to.
    """
    span = hi - lo
    if span <= 0.0:
        return 0.0, 0.0
    a = min(1.0, max(0.0, (base - lo) / span))
    b = min(1.0, max(0.0, (reach - lo) / span))
    start, end = (a, b) if a <= b else (b, a)
    return start, end - start
```

- [ ] **Step 4: Draw the swing, the base tick and the live handle**

Inside `slider_float_with_range_menu`, immediately after the `imgui.slider_float(...)` call that draws the track, add:

```python
        audio = getattr(self.state, "audio", None)
        overlay = getattr(self, "audio_overlays", {}).get(param_name) if audio else None
        if overlay is not None:
            p0 = imgui.get_item_rect_min()
            p1 = imgui.get_item_rect_max()
            dl = imgui.get_window_draw_list()
            w = p1.x - p0.x
            start, width = swing_fraction(overlay["base"], overlay["lo"],
                                          overlay["hi"], overlay["reach"])
            colour = overlay["color"]
            if width > 0.0:
                dl.add_rect_filled(
                    imgui.ImVec2(p0.x + w * start, p0.y),
                    imgui.ImVec2(p0.x + w * (start + width), p1.y),
                    imgui.get_color_u32(imgui.ImVec4(colour[0], colour[1],
                                                     colour[2], 0.18)))
            base_x = p0.x + w * swing_fraction(overlay["base"], overlay["lo"],
                                               overlay["hi"],
                                               overlay["base"])[0]
            dl.add_rect_filled(
                imgui.ImVec2(base_x, p0.y + 1), imgui.ImVec2(base_x + 1.0, p1.y - 1),
                imgui.get_color_u32(imgui.ImVec4(0.72, 0.72, 0.77, 0.9)))
            live_x = p0.x + w * swing_fraction(overlay["live"], overlay["lo"],
                                               overlay["hi"],
                                               overlay["live"])[0]
            dl.add_rect_filled(
                imgui.ImVec2(live_x - 1.0, p0.y), imgui.ImVec2(live_x + 2.0, p1.y),
                imgui.get_color_u32(imgui.ImVec4(*colour)))
```

At the end of the context-menu block in the same file, immediately before `imgui.end_popup()`, add:

```python
            imgui.separator()
            if imgui.button(f"Audio...##{slider_name}"):
                self.state.audio.show_window = True
                self.state.audio.open_target = self._label_to_param_name(
                    slider_name, for_jitter=True) or ""
                imgui.close_current_popup()
```

- [ ] **Step 5: Populate the overlays each frame**

In `main.py`, immediately after the `self.audio_runtime.update(...)` call added in Task 7:

```python
        # The panel draws the modulation inside each slider's own track.
        self.ui.audio_overlays = self.audio_runtime.overlays(
            ui_state, _audio_sim)
```

In `services/audio_runtime.py`, add:

```python
    def overlays(self, ui_state, modulated_sim) -> dict:
        """Per-target drawing data for the physics sliders.

        `reach` is where a full-scale signal would land, so the hatching shows
        the modulation's size rather than its current value.
        """
        ast = ui_state.audio
        if not ast.enabled or modulated_sim is ui_state.sim:
            return {}
        from ui.audio_reactive_window import SIGNAL_COLORS

        deaf = deaf_targets(ui_state.sim)
        targets = {t.key: t for t in physics_targets(ui_state.sim)}
        signals = {n: 1.0 for n in SIGNAL_COLORS}
        bases = {k: float(getattr(ui_state.sim, k, 0.0)) for k in targets}
        full = modulate(bases, list(targets.values()), ast.mappings, signals,
                        dict(), ast.strengths, ast.global_strength, 1 / 60.0,
                        deaf)

        out = {}
        for key, target in targets.items():
            bound = [m for m in ast.mappings
                     if m.target == key and m.enabled and key not in deaf]
            if not bound:
                continue
            out[key] = {
                "lo": target.lo, "hi": target.hi,
                "base": bases[key],
                "live": float(getattr(modulated_sim, key, bases[key])),
                "reach": full.get(key, bases[key]),
                "color": SIGNAL_COLORS[bound[0].signal],
            }
        return out
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slider_audio_display.py tests/test_label_widths.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ui/slider_widgets.py services/audio_runtime.py main.py tests/test_slider_audio_display.py
git commit -m "feat: a bound slider shows its modulation in its own track

Hatching for the span a full-scale signal could reach, a pale tick for the
base value the user set, a bright handle for the live one. An unbound slider
is unchanged - no empty affordance.

The context menu gains one item, which opens the panel on this row, so binding
still happens in exactly one place."
```

---

### Task 10: Persistence

**Files:**
- Create: `services/audio_rig_io.py`
- Modify: `main.py` (`__init__`, `cleanup`)
- Test: `tests/test_audio_rig_io.py`

**Interfaces:**
- Consumes: `state.audio_in_state.to_dict`, `apply_dict`.
- Produces: `rig_path() -> Path`; `save_rig(state, path=None) -> bool`; `load_rig(state, path=None) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_rig_io.py`:

```python
"""One global rig file. Not attached to a physics config."""
import json

import pytest

from services.audio_mapping import Mapping
from services.audio_rig_io import load_rig, rig_path, save_rig
from state.audio_in_state import AudioInState


def test_the_rig_lives_beside_the_other_user_files():
    assert rig_path().name == "audio_rig.json"


def test_a_saved_rig_loads_back(tmp_path):
    path = tmp_path / "audio_rig.json"
    st = AudioInState()
    st.mappings.append(Mapping(signal="mid", target="DRAG", mode="multiply"))
    st.global_strength = 1.25
    assert save_rig(st, path) is True

    fresh = AudioInState()
    assert load_rig(fresh, path) is True
    assert fresh.mappings[0].target == "DRAG"
    assert fresh.global_strength == pytest.approx(1.25)


def test_loading_a_missing_file_is_not_an_error(tmp_path):
    st = AudioInState()
    assert load_rig(st, tmp_path / "nothing.json") is False
    assert st.mappings == []


def test_a_corrupt_file_does_not_raise(tmp_path):
    path = tmp_path / "audio_rig.json"
    path.write_text("{not json", encoding="utf-8")
    st = AudioInState()
    assert load_rig(st, path) is False


def test_enabled_is_not_written_to_disk(tmp_path):
    path = tmp_path / "audio_rig.json"
    st = AudioInState()
    st.enabled = True
    save_rig(st, path)
    assert "enabled" not in json.loads(path.read_text(encoding="utf-8"))


def test_loading_never_starts_capture(tmp_path):
    path = tmp_path / "audio_rig.json"
    save_rig(AudioInState(), path)
    st = AudioInState()
    load_rig(st, path)
    assert st.enabled is False


def test_saving_into_a_missing_directory_creates_it(tmp_path):
    path = tmp_path / "deep" / "deeper" / "audio_rig.json"
    assert save_rig(AudioInState(), path) is True
    assert path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_rig_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.audio_rig_io'`

- [ ] **Step 3: Write the implementation**

Create `services/audio_rig_io.py`:

```python
"""Reading and writing the audio rig.

One rig, global to the application. Not written into physics configs: preset
hopping is the workflow, and a rig that vanished on every load would be
unusable.
"""
from __future__ import annotations

import json
from pathlib import Path

from state.audio_in_state import apply_dict, to_dict


def rig_path() -> Path:
    return Path.home() / "Documents" / "Fluoddity" / "audio_rig.json"


def save_rig(state, path: Path | None = None) -> bool:
    target = Path(path) if path is not None else rig_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(to_dict(state), indent=2),
                          encoding="utf-8")
        return True
    except Exception:
        return False


def load_rig(state, path: Path | None = None) -> bool:
    """False for a missing or unreadable rig, leaving `state` as it was."""
    target = Path(path) if path is not None else rig_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    apply_dict(state, data)
    return True
```

- [ ] **Step 4: Wire load and save into the app**

In `main.py`, in `App.__init__`, immediately after `self.audio_runtime = AudioRuntime()`:

```python
        from services.audio_rig_io import load_rig
        load_rig(self.ui.state.audio)
```

In `cleanup`, immediately after the `save preferences` step:

```python
            from services.audio_rig_io import save_rig
            self._step("save audio rig", save_rig, ui_state.audio)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audio_rig_io.py -v`
Expected: PASS, 7 tests

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 7: Commit**

```bash
git add services/audio_rig_io.py main.py tests/test_audio_rig_io.py
git commit -m "feat: the rig survives a restart, but never auto-starts capture

One global file rather than one per physics config: preset hopping is the
workflow, and a rig that vanished on every load would be unusable.

enabled is outside the persisted set, so launching Fluoddity does not open the
microphone."
```

---

### Task 11: Manual verification pass

**Files:**
- Modify: `docs/testing_checklist.md`

- [ ] **Step 1: Add the audio section**

Append to `docs/testing_checklist.md`:

```markdown
## Audio Reactive

Needs sound playing on the machine, or a microphone.

- [ ] Extras > Audio Reactive opens the panel; it is closed on a fresh launch.
- [ ] The device list contains both `input:` and `loopback:` entries.
- [ ] Picking a loopback device and pressing Start makes the spectrum move to
      music playing in another application.
- [ ] Stop halts the traces; the status reads `idle`.
- [ ] Binding bass to Sensor Gain visibly changes the simulation on a beat.
- [ ] The Sensor Gain slider shows hatching, a pale base tick, and a handle that
      rides with the music. Its printed value is the live one.
- [ ] Dragging that slider moves the base tick; the hatching follows it.
- [ ] File > Save writes the slider value, not the momentary modulated one.
- [ ] Right-click on a bound slider > Audio... opens the panel.
- [ ] Setting an X sweep on a bound parameter badges the row `swept`, and the
      parameter stops responding to audio.
- [ ] Switching brain modality changes the Brain rows; switching back restores
      the mappings that were there.
- [ ] Enabling Auto (CLIP) stops audio modulating anything; disabling restores it.
- [ ] Quitting and relaunching restores the mappings, with capture stopped.
- [ ] With the panel closed, the frame rate matches a run with the feature never
      enabled.
```

- [ ] **Step 2: Commit**

```bash
git add docs/testing_checklist.md
git commit -m "docs: the manual pass for audio input

Covers what unit tests cannot: that a loopback device actually captures another
application's sound, and that a swept parameter visibly stops responding."
```

---

## Self-Review

**Spec coverage.** Architecture → Tasks 1–7. Signals and auto-gain → Task 1. Mappings, maths, shapers → Tasks 2–3. Physics and brain targets, `V_MAX` exclusion, `encode`-once → Tasks 3–4. Both traps → Task 3 (`deaf_targets`) and Task 7 (Auto/Explore suppression). Performance rules → enforced by tests in Tasks 8–9 (`add_line` is asserted absent). UI → Tasks 8–9. Persistence → Tasks 5 and 10. Error handling → Task 6, plus the cleanup steps in Tasks 7 and 10. Testing → every task.

**Deviation from the spec, deliberate.** The spec says brain targets are settings with `kind == "float"`. The code defines `STRUCTURAL_KINDS = ("int", "choice", "layers")` — a fourth kind exists. Task 3 filters on `kind not in brains.STRUCTURAL_KINDS`, which is the same result today and stays correct when the MLP respec lands `layers`.

**Type consistency.** `Mapping`, `TargetDef`, `ShaperParams`, `ShaperState`, `SignalSnapshot`, `AudioInState` are defined once and referenced with the same field names throughout. `modulate()` keeps one signature across Tasks 3, 7 and 9. `swing_fraction()` returns `(start, width)` in both its definition and its two call sites.
