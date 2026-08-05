# CLIP-Guided Automatic Tournament — Execution Log

Deviations from `2026-08-04-clip-guided-tournament.md` and their reasons.

## Task 1 — Dependencies and model fetch

**Done as planned.** `requirements.txt`, `.gitignore`, `tools/fetch_clip_onnx.py`,
4 tests passing. All three assets downloaded (176 MB / 127 MB / 2 MB).

`DmlExecutionProvider` is available on this machine, so the DirectML choice works
as intended — no CUDA toolkit, no PyTorch.

### Finding 1: the split exports are correct — no fallback needed

Spec §3.1 required verifying that `vision_model` / `text_model` emit *projected*
embeddings rather than `pooler_output`. They do:

```
vision: in 'pixel_values'          -> out 'image_embeds'
text:   in 'input_ids' (int64)     -> out 'text_embeds'
```

The combined `model_fp16.onnx` fallback is not needed. The text model takes
**only** `input_ids` — there is no `attention_mask` input.

### Finding 2: fp16 models take float32 input — plan was wrong

`pixel_values` is declared `tensor(float)`, not `tensor(float16)`. The fp16
export has fp16 *weights* but fp32 I/O.

The plan's `preprocess()` cast the batch to `np.float16`, which would have been
rejected or silently reinterpreted. **Deviation:** `preprocess()` takes a `dtype`
argument, and `CLIPScorer` passes the dtype declared by the session's own input
descriptor. That is correct for both this export and any future fp16-I/O one, so
the fix is not merely "use float32".

### Finding 3: ORT_ENABLE_ALL crashes on CPU — must pin ORT_ENABLE_BASIC

Loading the fp16 vision model with default session options fails:

```
FAIL : Exception during initialization: graph_utils.cc:30
GetIndexFromName ... Attempting to get index by a name which does not exist:
InsertedPrecisionFreeCast_/vision_model/encoder/layers.6/layer_norm1/Constant_output_0
for node: /vision_model/pre_layrnorm/Mul/SimplifiedLayerNormFusion/
```

Measured across providers and optimization levels:

| Provider | ORT_ENABLE_ALL | ORT_ENABLE_BASIC |
|---|---|---|
| DirectML | OK | OK |
| CPU | **FAIL** | OK |

This matters because spec §11.1 requires the CPU fallback to work when DirectML
is unavailable — with default options that path would crash rather than degrade.

**Deviation:** `CLIPScorer` sets
`SessionOptions.graph_optimization_level = ORT_ENABLE_BASIC` on both sessions.
Applied unconditionally rather than only on CPU, so the two providers run the
same graph and cannot diverge in behaviour.

## Task 2 — CLIPScorer

Implemented with the three Task 1 findings folded in. 11 unit tests + 4
gpu-marked real-model tests passing, including an explicit CPU-fallback test.

**Deviation:** `preprocess()` gained a `dtype` parameter (default float32),
driven by the session's declared input type, per Finding 2.

## Task 3 — CLIP signal gate

**GATE PASSED: 5/6 prompts** above the std 0.05 bar, on 32 real tournament
tiles captured headlessly across two generations.

| prompt | std | range |
|---|---|---|
| glowing coral | 0.190 | 0.008 – 0.741 |
| a spider web | 0.182 | 0.044 – 0.753 |
| a dense city map | 0.146 | 0.027 – 0.587 |
| tree branches | 0.083 | 0.004 – 0.299 |
| a swirling galaxy | 0.060 | 0.027 – 0.231 |
| flowing water | 0.039 | 0.004 – 0.146 (flat) |

Rankings inspected visually: coral → pink polyp-like clusters, tree branches →
radiating frond structures, spider web → a cyan reticulated network with real
junctions. All defensible. Losers were uniformly dense noise, consistently.

### Finding 4: a dead canvas was a degenerate attractor

The visual inspection caught what the std numbers hid — "flowing water" was won
by a near-black tile. Testing a pure black image directly, with the spec's
original six distractors:

| prompt | black scores | black outranks |
|---|---|---|
| flowing water | 0.366 | **31 of 32 real tiles** |
| tree branches | 0.352 | 30 of 32 |
| a spider web | 0.297 | 12 of 32 |

CMA-ES would have driven straight to an empty simulation. `"a blank image"` and
`"a solid color"` did not catch it.

**Deviation:** `DEFAULT_DISTRACTORS` gains three black-specific entries —
`"a black image"`, `"an empty black background"`, `"a dark empty scene"`.
Measured effect:

| | before | after |
|---|---|---|
| black, worst case | outranks 31/32 | outranks 6/32 |
| black on flowing water | 0.366 | 0.029 |
| coral std (real signal) | 0.189 | 0.194 |
| spider web std | 0.177 | 0.161 |

The degenerate attractor is removed at essentially no cost to real signal.
`tests/test_clip_real_model.py::test_black_canvas_is_not_a_degenerate_attractor`
pins this.

"flowing water" drops below the bar as a result. That is the honest outcome:
it was only passing because black was gaming it. The substrate does not produce
water-like imagery, and the prompt is a poor fit rather than the scorer being
broken.

## Task 4 — Variable grid, cohort tiling, mutation suppression fix

Done as planned. `TILES = 16` is now a constructor argument; the UI grid, click
mapping and shader uniform all follow `TournamentService.grid`. 16 new cohort
tiling tests, 3 new service tests, 67 passing overall.

Confirmed the spec's claim that no GLSL change is needed: a 6x6 grid renders 36
independent, seam-isolated tiles with only the Python side changed.

**Deviation (verification method):** plan Step 9 said to launch `main.py` and
check by hand. Ran the headless harness instead and inspected the rendered grid
directly at N=4 and N=6, which is stronger evidence than a GUI glance and is
reproducible. Non-black coverage 0.967 (N=4) and 0.994 (N=6).

The `sim.py` change is the additive one the plan permitted: `apply_tournament`
gained a `mutation` parameter, and the suppression block now also zeroes
`x_sweep` / `y_sweep` / `cohort_sweep` / `jitter`.

## Tasks 5-12 — pure-Python units

GenomeSpec, optimizers, genome_io, TileCapture, RunLogger, checkpoints, the
state machine, state and CommandHandler wiring. All as planned.

**Test bugs found and fixed (implementation was correct in both cases):**

- `test_higher_fitness_is_better` asserted Random Search improves over 40
  generations. It does not learn, and in 80-D its best-of-640 equals its
  best-of-16 - which is exactly why it is the control. Split into a direct
  sign-convention test covering all four optimizers, plus an improvement test
  for the three that learn.
- `test_real_optimizer_state_survives_a_roundtrip` captured the reference
  `ask()` before snapshotting state, but `ask()` advances the RNG, so the
  checkpoint was one batch ahead.

**Real bug found by a test:** `np.savez` appends `.npz` to any path lacking it,
so writing to `ck.npz.tmp` produced `ck.npz.tmp.npz` and the atomic rename found
nothing. Fixed by writing through a file handle.

**Deviation:** `_CMAFamily.state_dict` captures every numpy array and scalar
attribute rather than an allowlist of the ones believed mutable. Constants
restore harmlessly and nothing can be silently missed. Verified against the real
`cmaes` internals by roundtrip tests.

## Tasks 13-14 — orchestrator and UI

### Deviation 1: speedmult already exists

The plan called for looping the physics step. `SimulationRunner` already honours
`ui_state.preferences.speedmult`, so Auto mode simply sets it - fewer moving
parts and it reuses the existing motion-blur-aware step loop.

### Deviation 2: capture is a blit, not a re-render

`FrameAssembler.assemble_frame` binds its own accumulation FBO, so it cannot be
redirected to the square capture target without modifying it.

Instead `services/capture_blit.py` blits the already-assembled view into the
square FBO, deriving the source rectangle from `camera.tex_to_screen()`. Using
the camera's own transform means the crop is exact at any window size, zoom or
pan, rather than assuming a square window or a centred fit. Verified visually:
16 tiles land on exact 224px boundaries in full display colour.

### Ordering bug found by the end-to-end run

`_handle_auto_tournament` clears the one-shot flags, and it runs inside
`process_commands` (step 2). With `_ensure_auto_service()` at step 5.1.6 the
service did not exist on the first frame, so the handler returned early and
consumed `start_requested` before anything could act on it - the loop sat in
`idle` for 40,000 frames. Service creation now happens at step 1.5, before
command processing.

### End-to-end result (12 generations, "glowing coral")

```
fit_best  0.202 0.368 0.287 0.315 0.293 0.299 0.245 0.418 0.352 0.361 0.407 0.445
fit_mean  0.066 0.116 0.064 0.100 0.107 0.152 0.099 0.117 0.162 0.131 0.140 0.149
sigma     0.477 0.462 0.450 0.441 0.435 0.433 0.434 0.434 0.437 0.438 0.439 0.438
```

144 frames for 12 generations. Best fitness roughly doubles, mean rises, sigma
contracts. The rendered grid at generation 12 contains recognisably coral-like
morphologies. log.jsonl has one line per generation and the gen-10 checkpoint
autosaved.

**Harness note:** the Manual tab sets `auto_tournament.enabled = False` every
frame it renders, which is the spec'd "switching to Manual pauses Auto"
behaviour. Headless cannot select a tab, so the harness re-asserts the flag each
frame. Not an app bug.

## Tasks 15-16 — verification

**Verified end to end, headlessly:**

| check | result |
|---|---|
| 12 generations, "glowing coral" | fit_best 0.20 -> 0.45, sigma 0.477 -> 0.438 |
| capture correctness | 16 tiles on exact 224px boundaries, full display colour |
| cohorts driven to k x tiles | grid4/k2=32, grid4/k9=144, grid2/k7=28 - all exact |
| checkpoint save -> resume | gen 4 -> 4, prompt preserved, best fitness identical |
| genome export | loads through the normal ConfigSaver, 80/80 coefficients |
| suite | 178 passed (CPU), 9 passed (GPU) |

### Bug found by the grid-sweep verification

`cmaes` fixes its population at construction and asserts on it in `tell()`. A
grid change through any route that did not also reset the optimizer crashed the
app with `AssertionError: Must tell popsize-length solutions`. The
CommandHandler path did reset, but the service was trusting its callers.

`_begin_generation` now compares `optimizer.popsize` against the live tile count
and rebuilds on mismatch. Pinned by
`test_grid_change_without_reset_rebuilds_instead_of_crashing`.

### Not verified

Interactive GUI behaviour - tab switching, tooltips, sliders, the sparkline and
the "Open run folder" button - was never exercised by a human. Everything behind
those widgets is covered headlessly, but the widget wiring itself is not.

## Post-review round: first run in the real app

Three defects, all in the widget layer that the "Not verified" note above called
out. Fixed in `62724db`.

1. **No confirmation when setting the goal.** The typed buffer and the applied
   prompt were rendered identically, so there was no way to tell whether CLIP
   was scoring what the box showed. The input is now tinted while they differ.
2. **Reset left the old fitness curve on the plot.** `reset()` cleared the
   optimizer but not the logger, which is what the plot reads.
3. **Sigma was logged but never drawn.** Now overlaid, separately normalised.

### The widget layer is now testable

ImGui needs only `io.display_size` and a frame - no window, no GL renderer - so
`tests/test_tournament_window_render.py` renders both tabs for real. Two gotchas
worth keeping:

- ImGui 1.92 builds its font atlas lazily; without
  `io.backend_flags |= BackendFlags_.renderer_has_textures` NewFrame asserts.
- A window emits **no geometry on the frame it is created** (it is auto-sizing),
  so vertex-count assertions must render twice or they depend on test order.

An unbalanced begin/end trips an assert inside `EndFrame`, so a test that
completes has proved the stack balances.

## world_size is a workload multiplier, not a coordinate rescale

Reported as "0.4 -> 4.0 lags a LOT more". Not a bug - measured and expected.

`world_size` scales *area*, holding particle density and feature size constant:
`entity_count = 600000 * ws` and the canvas is `1024 * sqrt(ws)` per side, so
both particles and texels grow linearly in `ws`. The shader deliberately
compensates the physics - particle size, sensor distance and forces all carry a
`1/SQRT_WORLD_SIZE` factor - so you get *more world*, not a *bigger* world.

Measured on an RTX 3080 Laptop, full `sim.update()`:

| ws | particles | canvas | VRAM | ms/step | vs 0.4 |
|---|---|---|---|---|---|
| 0.4 | 240,000 | 647² | 108 MB | 0.64 | 1.00x |
| 1.0 | 600,000 | 1024² | 271 MB | 1.36 | 2.15x |
| 2.0 | 1,200,000 | 1448² | 542 MB | 2.22 | 3.49x |
| 4.0 | 2,400,000 | 2048² | 1085 MB | 4.43 | 6.97x |

10x the work costs 6.97x the time - sublinear, so nothing pathological. The
apparent severity comes from the multiplier: cost is per *step*, and the app
runs `speedmult` steps per frame.

The one disproportionate number is `rule_buffer`, at 320 B/particle: 768 MB of
the 1085 MB at ws=4.0. Harmless on an 8 GB card, but on a smaller GPU it is the
term that would spill to system memory and turn this linear curve into a cliff.
