# Audio-Reactive Modulation — Design

**Date:** 2026-08-12
**Status:** Design approved, pending implementation plan
**Branch context:** `integration` (brain modalities merged)

## Goal

Let live audio drive Fluoddity's physics and brain parameters, so a simulation
responds to music the way the boids sketch in the `website` project did — with a
panel that shows what every band is doing to every parameter, rather than a black
box with a "reactivity" slider.

Ported in spirit from `website/src/components/simulations/boids/boids-audio.ts`
and its `panel/audio-tab.ts`. The mapping semantics are largely that file's; the
UI is that file's matrix-and-drawer; everything about the signal path, the
targets and the injection seam is new, because Fluoddity is not a browser.

## Decisions

| Decision | Choice |
|---|---|
| Capture | `PyAudioWPatch` — microphone and WASAPI loopback |
| Analysis | NumPy in-house — `rfft`, mel filterbank, band energy, RMS, spectral flux |
| Signals | Five fixed bands: bass, mid, presence, hi, volume |
| Shapers | Per-mapping: none, smooth, gate, envelope, LFO, sample & hold |
| Mapping UI | Matrix of targets × bands, with an expanding drawer (boids' shape) |
| Slider UI | Live modulated value drawn in the physics slider's own track |
| Physics targets | `PHYSICS_PARAMS`, minus `V_MAX` |
| Brain targets | Every `kind="float"` entry in the active modality's `settings_schema()` |
| Brain mechanism | Encode once, decode per frame with modulated scales |
| Brain mapping storage | Per modality |
| Persistence | One global rig file; not saved into physics configs |
| Auto-gain | On by default, per band |
| Trace drawing | `imgui.plot_lines` over numpy; ImPlot only where series overlay |

## Why NumPy and not a C++ library

The obvious move is to import an optimised realtime MIR library. Every candidate
was checked on this machine (Python 3.12, Windows) rather than by reputation, and
none survives:

| Library | Licence | Py 3.12 / Windows | Outcome |
|---|---|---|---|
| GIST | GPL-3.0 | source only | Right feature set, wrong licence |
| aubio | GPL-3.0 | no wheel | Wrong licence *and* no binary |
| essentia | AGPL-3.0 | no distribution | `No matching distribution found` |
| madmom | BSD-ish | build fails | Needs Cython and a compiler per install |
| vamphost | BSD host | wheel exists | Useful plugins (QM) are GPL and ship separately |
| audioFlux | MIT code | wheel + DLL | Links `libfftw3f-3.dll`; **FFTW is GPL-2+** |
| numpy | BSD | already a dependency | Chosen |

**Fluoddity is MIT** (`LICENSE`). GPL is viral through linking, so any of the
first four inside a distributed build forces the whole application — shaders
included — to GPL-3.0. The realtime-MIR category is uniformly GPL because it grew
out of academic music research; audioFlux looked like the exception until its
import table turned up FFTW.

The licence argument turned out not to be the deciding one. For this workload the
C library is **slower**:

| Per 2048-sample block at 48 kHz | mean | p99 |
|---|---|---|
| NumPy: `rfft` + 40 mel bins + 4 band energies + RMS + spectral flux | **0.056 ms** | 0.159 ms |
| audioFlux: mel spectrogram + five spectral descriptors | 0.203 ms | — |

One small block per frame is a trivial amount of work, so the time goes to call
overhead and array marshalling rather than to the FFT, which is microseconds
either way. Optimised C wins on batch throughput — a whole song at once — and
cannot win here. `np.fft` is pocketfft, the same lineage as the libraries above.

What NumPy genuinely cannot provide is **tempo/downbeat tracking** (a research
problem; nothing permissive delivers it) and **stem separation** (Demucs-class,
seconds per chunk, on the GPU the particles already saturate). Both are out of
scope. The signal layer is shaped so an ONNX beat model could later publish a
signal without the mapping layer changing, since `onnxruntime-directml` is
already a dependency.

## Architecture

```
PyAudioWPatch callback thread            main frame loop
─────────────────────────────            ───────────────
 services/audio_capture.py     ← the only module that performs IO
   mic | WASAPI loopback → ring buffer
        │
 services/audio_analysis.py    ← pure numpy: rfft, mel, bands, RMS, flux
 services/audio_shapers.py     ← pure: smooth, gate, envelope, LFO, S&H
        │
        └─ publishes an immutable SignalSnapshot ─┐
                                                  ▼
                              state/audio_in_state.py   mappings + latest snapshot
                                                  │
                              services/audio_mapping.py ← pure: snapshot + base → modulated
                                                  │
                              main.py  sim.apply_state(modulated copy)
                                       sim.apply_rule(modulated brain)
                                                  │
                              ui/audio_window.py   matrix, drawer, traces
```

Four properties the design rests on.

**Analysis never runs on the frame loop.** It runs inside the capture callback
and publishes one immutable snapshot; the frame loop reads the most recent one
and never waits. An audio hiccup therefore cannot drop a frame. This is the
discipline `AutoTournamentService` already uses for CLIP precompute.

**The base value is never written.** `UI.get_state()` returns the *live*
`UIState`, not a copy — so modulating `ui_state.sim` in place would make the
sliders visibly drift and would write momentary values into saved configs.
`main.py` instead hands `sim.apply_state()` a `dataclasses.replace()` copy.
`ui_state.sim` remains exactly what the user set, which is also what Save writes.

**The brain goes through the same seam.** `sim.apply_rule()` sets 80 floats of
uniform, so per-frame modulation is free. Audio modulates a *copy* of whatever
`rule_manager` currently holds and never pushes onto the rule stack — `pop_rule()`
returns `(None, None)` when the popped rule was the only one, and the archive
browser's live preview already rides that stack.

**A signal is a signal.** `audio_mapping` consumes `dict[str, float]` and cannot
tell an FFT band from a shaper output from a future model output.

`state/audio_in_state.py` deliberately avoids the name `audio_state.py`, which
the sonification work uses for MIDI **out**. One application should not have two
things called audio state.

## Signals

Five, fixed, each normalised to `[0, 1]`:

| Signal | Definition |
|---|---|
| `bass` | mean magnitude, 20–250 Hz |
| `mid` | 250–2000 Hz |
| `presence` | 2000–6000 Hz |
| `hi` | 6000–20000 Hz |
| `volume` | RMS of the full spectrum |

FFT size 2048, Hann window, **one analysis per capture callback** — not per
rendered frame. The two rates are independent by design: the frame loop reads
whatever snapshot is most recent and never blocks, so a dropped audio buffer
costs a stale signal rather than a stalled frame, and a slow frame costs no audio
at all. Band edges are Hz and resolve to bin indices at the stream's real sample
rate, read from the device rather than assumed.

**Auto-gain is per band and on by default.** A slowly-decaying running peak
normalises each band, because raw band energy tracks system volume — without it
every mapping needs re-tuning whenever the track changes. It is a toggle, since a
deliberately quiet passage should be allowed to read as quiet.

The matrix stays five columns wide. A user-definable signal graph was considered
and dropped: it existed to accommodate OSC channels from an external analyser, and
that route was rejected. The extension point survives in the data model.

## Mappings

A mapping is `(signal, target, mode, depth, gain, shaper, enabled)`, with
`mode ∈ {add, subtract, multiply}`. Several mappings may share a target.

Per target, per frame:

```
base = the user's slider value                     # never written
v    = base
for m in add/subtract mappings:  v += ±shape(s)·depth·(range_max − range_min)
for m in multiply mappings:      v *= 1 + shape(s)·depth
v = base + (v − base)·strength[target]·global_strength
v = clamp(v, hard_min ?? range_min, hard_max ?? range_max)
```

Additive terms resolve before multiplicative ones, as in boids. `strength` is
per-target and global, each `0–2`, and scales the whole delta rather than any one
term — so turning a target down cannot change the balance between its bands.

The modulation **scale** comes from the current slider range, so widening a
range widens the swing; that is what makes the in-track display honest. The
**clamp** uses `hard_min`/`hard_max` where a `PhysicsParamDef` declares them and
the slider range otherwise, so the live handle can never leave its own track.

### Shapers

Applied to the `[0,1]` signal before the mapping maths. Each is pure and tested
against a synthetic input.

| Shaper | Behaviour |
|---|---|
| `none` | identity |
| `smooth` | asymmetric envelope follower, separate attack and release |
| `gate` | threshold with hold time; output is 0 or 1 |
| `envelope` | signal crossing a threshold triggers an attack/release envelope |
| `lfo` | the signal drives the *rate* of a sine/triangle/ramp oscillator |
| `sample & hold` | latch the value on each gate crossing |

`lfo` is the "drive a periodic function from the power in a bin" case, and is the
reason shapers belong per-mapping rather than per-signal: the same band commonly
drives one target directly and another through an oscillator.

## Targets

### Physics

Every entry in `ui/physics_params.PHYSICS_PARAMS`, taking `label`, `default_min`,
`default_max`, `hard_min`, `hard_max` from the registry — no second list.

**`V_MAX` is excluded.** The top of its track means *off*, and the shader skips
the clamp entirely at `vraw >= vm.max_value`. Audio nudging it below max would
silently switch the speed cap on while the readout still said "Off", so the
readout would lie. It renders in the matrix, greyed, with that as its tooltip.

`MUTATION_SCALE` is bindable in normal mode. Tournament mode already overrides it
at the uniform, so a mapping there is inert by the same mechanism that makes a
user sweep inert.

### Brain

Rows come from the active modality's `settings_schema()`, filtered to
`kind == "float"`. `Setting(key, label, kind, lo, hi, default)` carries the same
information `PhysicsParamDef` does, so physics and brain rows share one code path.

- `kind == "float"` → a decode **scale**; changes what a `z` means, not how many
  floats it has. Modulatable.
- `kind in ("int", "choice")` → **structural**; changes `layout.length` and the
  checkpoint signature. Never a target — modulating `centers` or `hidden` would
  reshape the parameter buffer every frame.

Today that yields `freq_scale`, `low_freq_bias` (fourier); `input_scale`,
`freq_cycles`, `envelope_width`, `phase_spread` (gabor); `w_scale`, `mu_scale`,
`sigma_max` (lenia); and nothing for mlp, whose schema declares only an int and a
choice. Nothing special-cases mlp; it simply has no float to offer yet.

**Mechanism.** A scale change is a re-decode, and every modality already has
correct `encode()`/`decode()`, so the audio feature needs no modality-specific
knowledge:

- **Encode once**, when the base brain changes: `z = encode(base_params, base_layout)`.
- **Decode every frame** with the modulated scales substituted into the layout.

Encoding per frame instead would drift, because `encode()` clips at the rails and
is lossy at the extremes; encoding once and decoding many avoids the round trip.
Cost is one decode over at most `MAX_BRAIN_FLOATS` (512) floats per frame.

**Mappings are stored per modality.** Switching modality shows that modality's own
set; switching back restores the rig. Keyed by `Setting.key`, so a modality that
renames a key orphans its mappings — the same failure as switching modality, and
acceptable while a schema is still being designed.

## Two traps

**A sweep makes a target deaf, silently.** `calculate_setting()` returns
`slider_value` only when every sweep is zero, and otherwise computes from the
sweeps and ignores it. A target with a non-zero X/Y/cohort sweep therefore cannot
be modulated at all, with no error anywhere. The matrix row shows a warning badge
and the drawer states it. Modulating the sweep endpoints instead was considered
and rejected as too clever to predict.

**Audio must be suppressed while Auto or Explore is running.**
`write_tournament_physics` falls back to `self._state` for any parameter a tile
does not override, so modulation would reach tournament tiles for free — which is
desirable in manual tournament mode and fatal in Auto, where CLIP ranks tiles
against each other and audio would vary the physics mid-comparison. Suppressed at
the seam that already forces `plain_colour`, leaving mappings untouched.

## Performance

The panel costs more than the analysis, and the drawing technique dominates both.
Measured at 11 traces × 400 points, one frame:

| | ms | % of 16.7 ms |
|---|---|---|
| `add_line` per segment | 2.5–3.4 | 15–20% |
| `add_polyline`, cached `ImVec2` list | 0.866 | 5.2% |
| `implot.plot_line(numpy)` | 0.183 | 1.1% |
| **`imgui.plot_lines(numpy)` + per-trace colour** | **0.122** | **0.73%** |

ImGui renders 4400 segments in microseconds; the 2.5 ms was 4400 crossings
through pybind11 at roughly 0.56 µs each. The fix is not to draw less but to stop
making the calls — one numpy buffer crosses once.

Rules that follow:

1. Band traces and matrix sparklines use `imgui.plot_lines`, with the band colour
   pushed via `push_style_color(Col_.plot_lines, …)` around each call.
2. The ∑ tab overlays three series in one canvas, which `plot_lines` cannot do.
   **ImPlot there, and only there.**
3. The in-track swing is draw-list rectangles, about four calls per bound slider.
4. Ring buffers stay `float32` numpy all the way to the widget — never a Python
   list.
5. Point count is capped at the trace's width in pixels. This is a legibility
   rule, not a speed one: 400 points into a 200 px trace is mush.

**Whole feature: ~0.18 ms/frame with the panel open, about 1%. Zero with it
closed** — no capture thread, no analysis, no traces.

`ui/auto_tournament_window._render_series` uses the `add_line` path. Not urgent,
since it draws one panel, but it is the same fix.

## UI

Two surfaces, agreed against mockups in `.superpowers/brainstorm/`.

**`ui/audio_window.py` — the binding surface.** Source pills (Microphone / System
Audio) with a device combo and a status dot; a mel-scaled spectrum; a rolling
trace per band; a global Strength slider. Below, the mapping matrix: targets as
rows grouped into Physics and Brain, the five bands as columns, a filled dot where
a mapping exists and a hollow one where it does not. Clicking a dot toggles a
mapping; clicking a row name expands a drawer with one tab per bound band plus a
**∑ total** tab showing live value, base, delta, a range bar, stacked traces and
per-band contributions. Unbound rows are dimmed and draw no sparkline, so the
panel reads as empty until it is filled.

**`ui/physics_window.py` — the sliders.** A bound slider draws the modulation's
reachable span as a hatched band inside its own track, the base value as a pale
tick, and the live modulated value as a bright handle with the number to match.
Coloured dots to the right name the bands driving it. **An unbound slider is
pixel-identical to today** — no empty affordance. The existing right-click menu
gains one item, `Audio…`, which opens the panel with that row's drawer already
expanded; binding still happens in exactly one place.

Power-scaled parameters (`is_power_scaled`) map value to track position through
their own curve, so the handle and the hatching land where the eye expects.

Every settings panel pushes `layout.push_settings_width()`, and no visible widget
label may collide — a duplicate ImGui ID silently kills one of the two.

## Persistence

One rig, global to the application, at `Documents/Fluoddity/audio_rig.json`.

Mappings are **not** written into physics configs. Preset-hopping is the core
workflow, and a rig that vanished on every load would be unusable; the inverse —
a preset carrying its own rig — is a plausible later feature and is explicitly out
of scope here.

Persisted through an explicit allowlist, following `PERSISTED_FIELDS` in
`state/archive_state.py`: mappings, per-target strengths, global strength, band
edges, auto-gain flags, selected device name.

**`enabled` is deliberately not persisted.** Launching Fluoddity must never start
capturing audio on its own. This matches the archive's rule that opening the app
does not resume a search, and here it is also a privacy property.

## Error handling

- **`PyAudioWPatch` absent** — the feature disables itself with a message and the
  rest of the application runs unchanged. Imported lazily, inside functions, never
  at module scope, exactly as `onnxruntime` and `cmaes` are.
- **No device, or permission refused** — status goes to error, the message is
  shown in the panel, mappings are untouched.
- **Device disappears mid-stream** — treated as the above; the panel offers to
  restart. Never raises into the frame loop.
- **Non-finite analysis output** — scrubbed at the snapshot boundary, so no NaN
  can reach a uniform. Silence, DC and a clipped input are all ordinary inputs
  here, and each can produce a zero denominator in normalisation.
- **Shutdown** — the stream is closed from `cleanup()`, as its own guarded
  `_step()`, so a failure there cannot skip the archive flush below it.
- **Frozen build** — `PyAudioWPatch` needs a `hiddenimports` entry in
  `Fluoddity.spec`.

## Testing

Pure modules carry the tests; the sim itself stays a manual check.

| File | Covers |
|---|---|
| `tests/test_audio_analysis.py` | a synthetic tone lands in the right band at several sample rates; silence gives zeros; DC and clipping give no NaN; band edges resolve from the device rate, not a hardcoded 44100 |
| `tests/test_audio_shapers.py` | each shaper against step, impulse and silence; the LFO's rate tracks its driving signal; attack and release are genuinely asymmetric |
| `tests/test_audio_mapping.py` | the base value is never mutated; additive resolves before multiplicative; strength scales the delta and not the terms; clamping respects `hard_min`/`hard_max` before slider range; `V_MAX` is not offered as a target; a target with a live sweep is reported as deaf |
| `tests/test_audio_brain.py` | rows come from `settings_schema()` and exclude int/choice; a modality with no float settings yields no rows; encode-once/decode-many round trip; modulating `freq_scale` scales exactly the frequency floats |
| `tests/test_audio_window_render.py` | headless render smoke; no duplicate ImGui IDs; every label fits `WIDEST_LABEL` |

Capture is IO and is not unit-tested; it is exercised by the manual pass in
`docs/testing_checklist.md`, which gains an audio section.

## Coordination — the MLP respec

An MLP re-specification is in progress under a separate orchestrator. The
contract between it and this feature is one rule:

> Any `Setting` declared with `kind="float"` in a modality's `settings_schema()`
> automatically becomes an audio-modulatable target. `lo`/`hi` are the modulation
> range, `label` is the row name, `key` is the storage key. The audio feature
> reaches the brain only through `encode()`/`decode()`, so no modality-specific
> code is required on either side.

Two asks of that work:

1. **`lo`/`hi` must be the genuinely usable range.** The swing is
   `depth·(hi − lo)` and the in-track display is drawn against it, so a
   placeholder range makes every mapping on that row feel wrong.
2. **Treat MLP audio mappings as disposable while `key` names churn.** They are
   stored per modality keyed by `key`; a rename orphans them silently.

No change to this design is needed when MLP gains float settings — the rows appear
on their own.

## Out of scope

- Tempo, beat and downbeat tracking. Revisit as an ONNX model publishing a signal.
- Stem / instrument separation. Not realtime on a GPU already running the sim.
- Harmonic–percussive separation. Cheap enough to add later as a signal; not v1.
- OSC or any external analyser input. Rejected in favour of working standalone.
- A user-definable signal graph. Its motivation left with OSC.
- Audio rigs attached to physics configs.
- Modulating camera, drawing brush or appearance parameters.
- Modulating structural brain settings (`centers`, `hidden`, `activation`).
