# Phase diagrams of a preset

Sweep two physics parameters over a grid, run each cell to a fixed step count,
and reduce each run to a row of scalars. One pixel per run. The first subject is
the user config `fish soup` over (`AXIAL_FORCE`, `LATERAL_FORCE`).

## The problem

Nothing in the app answers "what does this preset do *around* where it sits".
The tournament grid comes closest — 64 isolated small worlds, each with its own
physics — but it is built for a search that scores on CLIP and keeps what it
likes, not for a dense controlled sweep, and 8x8 is not a phase diagram.

The prior art is Lenia's survival plots: a scalar per (parameter, parameter)
cell, usually "how long did the pattern stay between dead and saturated". That
measure does not transcribe directly. Lenia's mass is a free variable with two
failure modes; Fluoddity's particle count is fixed, so mass is not free. What
*is* free is how the trail concentrates — the same particles piled into a few
texels or spread across all of them — and that carries the same two failure
modes.

## Shape

Three pieces, split so the part that has to be correct needs no GPU:

- **`services/phase_metrics.py`** — pure functions over numpy arrays.
  `(positions, velocities, rho) -> dict[str, float]`. No GL, no `sim` import,
  no state. Unit-testable against synthetic fields of known answer.
- **`tools/phase_diagram.py`** — the GL harness and sweep loop. Standalone
  moderngl context, `ConfigSaver`, argparse, `python -m tools.phase_diagram`;
  the structure of `tools/measure_speed.py`, which is the existing precedent for
  stepping a preset headlessly with no Camera and no CLIP.
- **`tools/phase_view.py`** — `.npz`/memmap in, PNG out. Any single feature as a
  colormap, or three mapped to RGB.

The sweep is the expensive part and the picture is free, so the picture is never
baked into the sweep. Re-render as often as you like without re-running.

## Why `world_size` is the resolution knob and `particle_density` is not

Every length in `entity_update.glsl` is scaled by `1/sqrt(WORLD_SIZE)` — sensor
distance (`:658`), force and strafe (`:700-701`), particle size (`:501`), the
V_MAX clamp (`:739`) — and sensor gain by `sqrt(WORLD_SIZE)` (`:689`). The
canvas side goes as `sqrt(ws)` and the entity count as `ws`. So
particles-per-texel *and* the pattern's length scale in texels are both
invariant: a small world is a smaller window onto the same phase, not different
physics.

| `world_size` | canvas | particles |
|---|---|---|
| 0.40 (app default) | 647² | 240k |
| 0.10 | 323² | 60k |
| 0.05 | 228² | 30k |
| 0.025 | 161² | 15k |

`particle_density` is the trap next door. It is the cheaper knob and it looks
equivalent, but it changes trail intensity, which feeds back through the sensors
into speed — CLAUDE.md records one preset running 28x faster at density 1.0 than
at 0.25. It moves the phase diagram. It must stay at whatever the run declares
and never be used to buy speed.

What a small world does cost is finite-size effects: fewer characteristic
lengths across, and under wrap a small torus can lock a wavelength or stabilise
something a large world would break up. `fish soup` is `boundary_conditions: 2`
(wrap), so that is the case in play. This is measured, not assumed — see the
pilot.

## Experimental controls

Every cell must differ from its neighbours in exactly one thing. Three seeds are
in play and two of them are not the axes:

- **`rule_seed` held at the preset's value** (0.3796 for `fish soup`). It drives
  the per-particle mutation field, so letting it vary sweeps brain identity
  underneath the force axes.
- **`sim.reset_seed` held fixed.** Same initial particle field in every cell.
- **`num_cohorts` and `MUTATION_SCALE` kept at the preset's values** (6 and
  0.328). They are constant across cells, so they are part of the system rather
  than a confound. The consequence is that a cell is an *ecosystem* of six
  brains under per-particle mutation, not one creature, and its variance is
  correspondingly higher. That is deliberate: the subject is the preset as it is
  actually watched.
- **`parameter_sweeps_enabled` must be off.** An x/y sweep makes a parameter
  vary *within* a cell by position, which is the confound
  `capture_health.sweeping_parameters()` exists to warn about. `fish soup` has
  it off already; the tool asserts rather than trusts.

The sim does not reproduce itself run to run — particles splat additively into a
shared texture, which races — so cells are noisy whatever is held fixed. Two
things handle it:

- **Neighbouring cells are replicates.** A real phase boundary is spatially
  coherent across the grid; speckle is not.
- **A noise-floor strip.** ~32 extra runs with EVERYTHING identical — both
  seeds included — differing only by the intrinsic race, stored beside the grid.
  Both seeds are held fixed across the grid too, so the race is the grid's only
  noise source and reproducing it is the only way to measure the right number. A
  strip that varied `reset_seed` would measure initial-condition sensitivity,
  which is a different (also interesting, out of scope) quantity. Without the
  strip there is no way to tell which contrast in the image is signal. Required,
  not optional, and cheap against a grid of thousands.

Twin-run Lyapunov divergence is deliberately not attempted: the intrinsic noise
floor is the thing it would measure.

## The features

Twelve scalars, numpy only. scipy is not a dependency and adding one for this is
a separate decision, so nothing here may need a KD-tree.

From the entity buffer (`pos`, `vel`; the struct is pos:2, vel:2, size:1,
pad:3, color:4):

| | |
|---|---|
| `speed_p50`, `speed_p90` | percentiles of \|v\|. Frozen vs. active |
| `polar_order` | ‖mean v̂‖. The Vicsek order parameter — coherent stream 1, disordered 0 |
| `particle_pr` | participation ratio of a coarse 2-D histogram of positions. Clustering without a KD-tree |

From the canvas, ρ = ‖canvas.xy‖:

| | |
|---|---|
| `participation_ratio` | `(Σρ)² / (N·Σρ²)`. Blob → 0, uniform → 1. The Lenia-mass analogue: parameter-free, no threshold |
| `coverage` | fraction of texels above an ABSOLUTE threshold on ρ, not a per-cell fraction of max. A per-cell normalisation would make the number incomparable between the cells the diagram exists to compare. The value comes from the pilot and is written into the sidecar |
| `structure` | `capture_health.structure()` called as `structure(rho[..., None])`. It takes `(n, H, W, C)` and immediately averages over the last axis, so a single-channel float field needs no adaptation and no new code. Best \|autocorrelation\| over lags (1,2,3,4,6,8,12,16); 1 coherent, 0 white noise. Scores a *frozen* field 1.0, so it only means something beside a motion term |
| `spec_peak_k` | peak of the azimuthally-averaged 2-D power spectrum — the characteristic length scale. Varies smoothly within a phase and jumps between them |
| `spec_entropy` | normalised Shannon entropy of the same radial spectrum. Periodic vs. broadband |
| `field_order` | ρ-weighted ‖mean v̂‖ over texels. The field-side companion to `polar_order` |

From the probe series:

| | |
|---|---|
| `change_rate` | mean over the last window of `‖ρ_t − ρ_{t−K}‖₁ / ‖ρ_t‖₁`. `descriptor.liveness()` without the CLIP bill |
| `alive_steps` | first probe where `participation_ratio` leaves `[pr_lo, pr_hi]`, else the step budget. The Lenia transcription. The bracket is not guessable in advance — it comes from the pilot's observed range and is written into the sidecar, so a diagram always carries the definition that produced it |

`descriptor.liveness()` itself is deliberately excluded: it needs CLIP, and at
thousands of cells the ONNX session cost dominates everything else here.

## Two time resolutions

Cheap scalars are probed every 50 steps — 100 probes x 12 floats is nothing —
and full fields are stored at only 5 times. The dense series is what
`change_rate` and `alive_steps` read; the sparse full snapshots are what
features invented later will read.

The 5 snapshot times are **log-spaced** (~100 / 400 / 1200 / 2600 / 5000).
Transients are fast early; linear spacing spends four of five snapshots on the
part that has already settled.

## Storage

Two independent knobs, because the two costs differ by three orders of
magnitude.

**Features and the probe series** are ~5 KB per cell — 20 MB at 64², 80 MB at
128², 320 MB at 256². Grid resolution is therefore effectively free, and the
picture is always at full grid resolution.

**Raw snapshots** are ~1.15 MiB per cell at `world_size` 0.05 (canvas fp16 two
channels, 203 KiB; a 4096-particle subsample of pos+vel at fp16, 32 KiB; times
5). Stored on a nested sub-lattice at stride `--raw-stride`, not on every cell:

| raw on | cells | total |
|---|---|---|
| every cell of 64² | 4096 | 4.7 GiB |
| every cell of 128² | 16384 | 19 GiB |
| every 4th cell of 128² | 1024 | 1.2 GiB |
| every 8th cell of 128² | 256 | 295 MiB |

Default `--grid 128 --raw-stride 4 --snapshots 5` at `world_size` 0.05: about
1.3 GiB total.

What the sub-lattice costs is that a feature invented later is computable at 32²
rather than 128². That is still enough to decide whether it is interesting, at
which point the sweep is re-run with it added. Raw storage was never going to
cover every future question, only cheapen the common ones.

Three details:

- **Particles are subsampled to a FIXED index set**, so the same particles are
  tracked across cells and across time. 4096 of 30k gives ~1.6% error on a
  population statistic, well under the run-to-run noise.
- **fp16 storage needs a safety net.** CLAUDE.md records fp16 failing as a
  canvas *format* because density reaches 47.19 where fp16 spacing is 0.031
  against a particle's 0.01 deposit. Storage is not accumulation, so it is a
  milder problem, but the bright regions still lose fine gradation. The twelve
  features are therefore computed **from the fp32 readback at capture time** and
  stored alongside. fp16 then limits only features that do not exist yet, and a
  future feature that turns out to be precision-sensitive will disagree with its
  stored fp32 counterpart rather than being silently wrong.
- **One memmapped `.npy` per array, not one file per cell.** Canvas is
  `(R, R, S, H, W, 2)` fp16, particles `(R, R, S, n_sub, 4)` fp16, both written
  in place. Per-cell files make the later exploration slow; a memmap slices an
  arbitrary transect through the diagram instantly. The features, probe series,
  axis values and the whole preset config go in a sidecar `.npz`.

## Testing

`tests/test_phase_metrics.py` drives the pure functions with synthetic fields of
known answer, no GPU:

- a single Gaussian blob → `participation_ratio` ≈ 0
- uniform noise → `participation_ratio` ≈ 1, `structure` ≈ 0, `spec_entropy` ≈ 1
- a stripe lattice at a constructed wavelength → `structure` ≈ 1, `spec_peak_k`
  at that wavenumber
- two identical frames → `change_rate` == 0
- a rigidly translating field → `change_rate` > 0 while `structure` is unchanged,
  which is the case that separates motion from disorder

`structure()` is called, not reimplemented or adapted, so its existing coverage
carries entirely and there is nothing new to test about it.

The GL harness gets a smoke test only — one 2x2 grid at a tiny world size and
step count, asserting the output arrays have the declared shapes and no NaNs.
Correctness of the sim is not this tool's business.

## The pilot, which runs before any grid

A 5x5 coarse sweep at four world sizes (0.20 / 0.10 / 0.05 / 0.025), keeping the
full probe series. It answers three things that cannot be answered on paper:

1. **Seconds per run**, which is what turns "high resolution" into a number.
2. **Where the features stop moving with world size** — the smallest world that
   can be trusted, and whether 0.05 is honest for this preset.
3. **Whether 5000 steps is right.** The probe series shows whether the features
   have plateaued or have not.

The full grid is not sized until the pilot has answered all three.

## Deliberately not in scope

- **The tournament grid as the execution route.** It batches 64 isolated worlds
  per process and would be faster per cell, but tile size shrinks with the grid
  (647/8 ≈ 80px at the default world size), which caps the pattern scale that
  fits. Sequential full-canvas runs at a small `world_size` get the same speedup
  without the tile-edge effects, and `world_size` shrinks the world without
  shrinking it *relative to the pattern*.
- **x/y parameter sweeps as a free phase diagram.** One run appears to give the
  whole plane, but particles migrate across the gradient, so a location's state
  is not that parameter's steady state.
- **A UI.** This is a `tools/` script, like every other measurement in this
  repo, and its output is a file.
- **Sweeping anything outside `_TOURNAMENT_PHYSICS_ORDER`'s eleven.** The axes
  are chosen by SimState field name; trails, boundary mode and cohorts stay
  where the preset put them.
