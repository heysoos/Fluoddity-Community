# CLIP-Guided Automatic Tournament — Design

**Date:** 2026-08-04
**Status:** Approved (revision 2, pending spec review)
**Scope:** Add a second, *automatic* tournament mode alongside the existing manual one. A
text prompt the user types at any time is turned into a fitness score by CLIP, and an
evolution strategy (CMA-ES by default, selectable) drives the tile population toward it.
The user can steer or veto at any point.

**Builds on:** `2026-08-03-interactive-tournament-design.md`. The tiled canvas, hard tile
isolation, genome representation, rule packing and GPU upload are all reused unchanged.

**Revision 2 adds:** variable grid size (§4), an explicit CLIP input pipeline contract (§7),
run logging (§8), and checkpoint save/load (§9).

## 1. Goal & non-goals

**Goal:** Replace the human selector with a CLIP-derived fitness signal, without giving up
human control. The user types a goal ("glowing coral", "a dense city map"), presses Start,
and watches the grid converge — retyping the prompt or clicking tiles at any moment.

**Design constraint (from the user, verbatim):** the evolutionary machinery must never be
the bottleneck. The simulation should remain the dominant cost. See §10 for the budget that
makes this true by construction rather than by tuning.

**Non-goals (YAGNI):**
- No evolution of the 12 physics sliders in v1. The genome stays the 80-float Fourier
  brain. The encoding is structured so physics can be added later (§5.2), but that work is
  explicitly out of scope here.
- No non-square grids. Grid is always N×N, and Auto mode forces a 1:1 canvas (§4.2).
- No decoupling of population size from tile count — popsize *is* N² (§4.1).
- No 3D (Fluoddity3D) support.
- No CLIP fine-tuning, no aesthetic-predictor model, no diffusion anything.
- No open-endedness / novelty objective in v1 (considered and deferred — see §14).
- No MAP-Elites / illumination archive.
- No change to manual tournament behaviour. Manual mode must work identically after this
  change, including with the CLIP dependencies absent.

## 2. User-facing behavior

The existing Tournament window gains a mode selector at the top: **Manual** | **Auto (CLIP)**.
Manual is the current behaviour, unchanged and the default.

In Auto mode:

- A **Goal** text field. Editing it takes effect on Enter (or on a "Set" button). Changing
  the prompt does **not** restart the search — the optimizer keeps its learned covariance
  and simply starts climbing a new landscape. This is deliberate: it means "keep the search
  shape, change the target".
- **Algorithm** dropdown: `CMA-ES` (default), `Sep-CMA-ES`, `GA`, `Random Search`.
- **Grid** slider: N ∈ [2, 8], default 4. Population size is N² and is shown next to the
  slider as a read-only readout (§4.1), along with the effective per-tile source resolution
  (§4.3) so the user can see when tiles are being upscaled for CLIP.
- **Start / Pause / Reset** buttons. Reset discards the optimizer state and reseeds.
- Sliders: **Steps per generation** (default 300, range 50–2000), **Sim steps per frame**
  (default 10, range 1–50), **Snapshots per generation** (default 4, range 1–8),
  **Initial sigma** (default 0.5, range 0.05–1.5).
- The existing N×N button grid stays. Each button now also shows that tile's fitness for
  the last completed generation. Clicking a tile still selects it; in Auto mode a selected
  tile is **force-injected as an elite** into the next generation (§6.4) and the selection
  is then cleared. This is the "steer and veto" the user asked for.
- A **metrics panel**: generation number, best-ever score, a sparkline of best and mean
  fitness per generation, and a scrollable table of the last 50 generations (§8).
- **Save checkpoint** / **Load checkpoint** buttons, plus an **Autosave every N gens**
  control (§9.1).
- **Save best genome** / **Save this tile** / **Load genome as starting point** buttons.
  These write and read ordinary Fluoddity config files, so an evolved creature can be opened
  in the normal single-simulation view at full resolution (§9.2).

If the CLIP model files or `onnxruntime` are missing, the Auto tab renders a single
explanatory line and a **Download CLIP model (~330 MB)** button (§11.1). Manual mode is
unaffected in every failure case.

## 3. New dependencies

Added to `requirements.txt`:

```
onnxruntime-directml>=1.20   # DirectX 12 GPU inference; no CUDA toolkit, no PyTorch
tokenizers>=0.20             # CLIP BPE tokenizer (Rust, ~3 MB)
cmaes>=0.11                  # CyberAgentAILab CMA-ES; pure numpy, ~50 KB
```

Logging and checkpointing add no dependencies — they use stdlib `json` and `numpy.savez`.
Image resampling for augmented crops uses **Pillow**, which is already a dependency.

Rationale for `onnxruntime-directml` over PyTorch: PyTorch with CUDA wheels is ~3 GB and
introduces a CUDA-version coupling that would be a persistent support burden in the
PyInstaller build. DirectML runs on any DX12 device, which on this machine means the
RTX 3080 and degrades gracefully to the Intel UHD. Total added weight is ~250 MB of
runtime plus ~330 MB of model weights.

All three imports are **lazy and optional**. `services/clip_scorer.py` and
`services/optimizers.py` must not be imported at app startup; they are imported the first
time Auto mode is enabled, inside a `try/except ImportError` that disables Auto mode with a
message rather than crashing.

### 3.1 Model assets

Downloaded once by `tools/fetch_clip_onnx.py` into `models/clip-vit-b32/` (gitignored):

| File | Source | Size |
|---|---|---|
| `vision_model_fp16.onnx` | `Xenova/clip-vit-base-patch32` @ `onnx/` | ~176 MB |
| `text_model_fp16.onnx` | same | ~127 MB |
| `tokenizer.json` | same, repo root | ~2 MB |

Fetched by direct HTTPS from `https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/<path>`
using stdlib `urllib` — no `huggingface_hub` dependency, no auth required (verified
2026-08-04: all four URLs return 200).

**Verification step required during implementation:** confirm the split `vision_model` /
`text_model` exports emit the *projected* `image_embeds` / `text_embeds` tensors and not
raw `pooler_output`. If they emit `pooler_output`, fall back to the combined
`onnx/model_fp16.onnx` (~300 MB), which exposes `image_embeds` and `text_embeds` directly.
The scorer interface does not change either way; only the session setup does.

## 4. Grid, population size, and canvas aspect

### 4.1 Population size is the tile count

`popsize = N²`, where N is the grid slider. There is no separate population control. The
slider range N ∈ [2, 8] gives popsize ∈ {4, 9, 16, 25, 36, 49, 64}.

`TOURNAMENT_GRID` is **already a shader uniform**, so variable N requires no GLSL change.
What must change is the Python side, which currently hardcodes `TILES = 16` in
`services/tournament_service.py`. That constant becomes a constructor argument, and the
UI grid in `ui/tournament_window.py` (including `ui_row_order`, which already takes a
`grid` argument) becomes N-driven.

**Guidance shown in the UI:** the theoretically-motivated CMA-ES population for an 80-D
problem is `4 + 3·ln(80) ≈ 17`, so N=4 (16) is close to ideal and is the default. Below
N=3 (popsize 9) CMA-ES has too few samples to estimate an 80×80 covariance and will
degrade badly; the UI shows an inline warning at N=2 rather than forbidding it, since
Random Search and GA remain perfectly usable there.

**Changing N mid-run resets the optimizer.** `cmaes.CMA` fixes its population size at
construction, and the covariance estimate is meaningless at a different sample count. The
UI states this next to the slider, and the change is applied at the next generation
boundary, never mid-rollout.

### 4.2 Auto mode forces a 1:1 canvas

CLIP consumes square 224×224 images. `tournament_tile_box()` divides the canvas by a scalar
grid, so every tile inherits the canvas aspect ratio — at 16:9 each tile would be 1.78:1,
and fitting that to a square requires either distortion, black-bar padding, or discarding
content. All three degrade the signal.

Therefore: **enabling Auto mode forces the canvas aspect to 1:1**, and disabling it restores
the user's previous ratio. The UI shows a one-line note explaining this. Tiles are then
always square, and the CLIP crop is exact with no resampling, padding or cropping.

Implementation: the aspect change goes through the existing world-resize path, which already
calls `TournamentService.mark_dirty()` to reupload genomes after buffer reallocation. The
force happens on the Auto-mode *enable edge*, before any run starts, so it never interrupts
a rollout.

### 4.3 Source resolution and upscaling

The capture FBO is `N·224` square regardless of N, so CLIP always receives exact 224×224
crops. But the *underlying* simulation canvas is fixed (1024² by default), so the real
information per tile is `1024/N` pixels:

| N | popsize | source px/tile | vs. CLIP's 224 |
|---|---|---|---|
| 2 | 4 | 512 | downscaled 2.3× |
| 3 | 9 | 341 | downscaled 1.5× |
| 4 | 16 | 256 | downscaled 1.14× — near-ideal |
| 5 | 25 | 205 | upscaled 1.09× |
| 6 | 36 | 171 | upscaled 1.31× |
| 8 | 64 | 128 | upscaled 1.75× — visibly soft |

Above N=4 CLIP is being fed upscaled, increasingly blurry tiles, which flattens the fitness
landscape. The UI displays the source px/tile next to the grid slider and hints at raising
the canvas resolution when it drops below 224. This is surfaced, not enforced — the user
decides the trade.

## 5. Genome encoding

### 5.1 Search space vs. genome space

The optimizer searches an unbounded normalized space `z ∈ ℝ^80`. Decoding to a real genome
applies a bounded squash, matching the ranges `services/genome.py` already produces:

```
freq = 3.0 * tanh(z[0:40].reshape(10, 4))
amp  = 1.0 * tanh(z[40:80].reshape(10, 4))
genome = concat([freq, amp], axis=1)   # (10, 8) float32, as today
```

A squash rather than clipping is used so the optimizer cannot wander to `freq = 50` and so
that no repair bias is introduced at the boundary (clipping makes many distinct `z` map to
the same genome, which distorts CMA-ES's covariance estimate).

`sigma0` defaults to 0.5. During implementation this must be **calibrated** so that the
initial population's decoded distribution visually matches `genome.random_genome()` — i.e.
generation 0 of Auto mode should look about as varied as generation 0 of Manual mode.
Calibration is a one-off measurement, not a runtime feature.

### 5.2 Forward compatibility with physics parameters

Encoding lives behind a `GenomeSpec` — an ordered list of named blocks:

```python
GenomeSpec(blocks=[Block(name="brain", size=80, decode=brain_decode)])
```

`spec.dim`, `spec.decode(z) -> dict[str, np.ndarray]`, `spec.encode(parts) -> z`, and
`spec.signature() -> str` (a stable string of block names and sizes, used to reject
incompatible checkpoints — see §9). Adding physics later means appending
`Block(name="physics", size=12, decode=physics_decode)`; the optimizer, scorer, logger and
loop are all dimension-agnostic and need no changes. The remaining work at that point is
per-tile physics uniforms in the shader, which is why it is out of scope now.

## 6. Architecture

Nine new units. Each has one purpose, a small interface, and can be understood and tested
without the others.

### 6.1 `services/clip_scorer.py` — `CLIPScorer`

```python
class CLIPScorer:
    def __init__(self, model_dir: str, providers: list[str] | None = None) -> None
    def set_prompt(self, text: str, distractors: list[str] | None = None) -> None
    def score(self, images: np.ndarray) -> np.ndarray   # uint8 (B,224,224,3) -> float32 (B,)
    @property
    def available(self) -> bool
```

Depends only on `onnxruntime`, `tokenizers`, `numpy` and `PIL`. Knows nothing about
tournaments, tiles, or OpenGL.

**Fitness formulation.** `score` returns the softmax probability of the target prompt
against a fixed distractor set, which gives a full 0–1 dynamic range and is substantially
harder to exploit than raw cosine similarity (whose useful band is roughly 0.15–0.35):

```
logits[b, j] = logit_scale * (img_emb[b] · txt_emb[j])       # both L2-normalized
score[b]     = softmax(logits[b])[0]                          # index 0 = target prompt
```

Default distractor set (target prompt is always index 0):

```
"a blank image", "random noise", "a blurry photograph",
"a screenshot of text", "a solid color", "an abstract texture"
```

`"an abstract texture"` is included deliberately: without it, nearly every trail pattern
scores highly on nearly every prompt, because "abstract texture" is the honest description
of most of the search space.

**Augmentation.** Each tile image is scored as the mean over 3 views — the full 224×224
frame plus two random sub-crops of 160–224 px resampled to 224 with PIL bilinear. Averaging
over random views is the standard defense against a search process finding adversarial
texture that satisfies a single fixed view.

Text embeddings are computed once per `set_prompt` and cached, so retyping the goal costs
one ~5 ms text-encoder pass.

### 6.2 `services/optimizers.py` — `Optimizer` protocol + 4 implementations

```python
class Optimizer(Protocol):
    name: str
    def ask(self, n: int) -> np.ndarray          # (n, dim) float32
    def tell(self, z: np.ndarray, fitness: np.ndarray) -> None   # higher fitness = better
    def best(self) -> tuple[np.ndarray, float]   # (z, fitness)
    def state_dict(self) -> dict                 # JSON/npz-serializable, for checkpoints
    def load_state_dict(self, d: dict) -> None
```

`tell` takes **fitness where higher is better**; implementations negate internally as
needed (the `cmaes` library minimizes). This keeps the sign convention in one place instead
of leaking it into the loop.

`state_dict` returns explicit named numpy arrays and scalars — **not** a pickle. Pickling
`cmaes.CMA` would couple checkpoint files to the installed library version and to Python's
pickle protocol; an explicit dict of `{mean, C, sigma, p_sigma, p_c, generation, rng_state}`
is versionable and inspectable. See §9.

- `CMAESOptimizer` — wraps `cmaes.CMA`. Full covariance. At dim 80 the eigendecomposition
  is ~1 ms, so the separable variant buys nothing yet.
- `SepCMAESOptimizer` — wraps `cmaes.SepCMA`. Present for when the dimension grows.
- `GAOptimizer` — reuses `services/genome.py`'s `mutate`/`crossover` operators, applied in
  `z` space. This is manual mode's operator driven by CLIP instead of by a human, which
  makes it a genuinely useful comparison rather than filler.
- `RandomSearchOptimizer` — uniform resampling. The control. If CMA-ES cannot beat this,
  the fitness signal is not real, and that is worth being able to check on demand.

Pure numpy. No OpenGL, no CLIP, no app state. Fully unit-testable.

### 6.3 `services/tile_capture.py` — `TileCapture`

```python
class TileCapture:
    def __init__(self, ctx, grid: int, tile_px: int = 224) -> None
    def capture(self, render_fn) -> np.ndarray   # -> uint8 (grid*grid, tile_px, tile_px, 3)
    def resize(self, grid: int) -> None          # reallocate on grid change
    def release(self) -> None
```

Owns an offscreen framebuffer of exactly `grid·tile_px` square, calls `render_fn(fbo)` to
draw the assembled tournament view into it via the normal `frame_assembly` path, reads back
once, and splits the result into exact `tile_px` crops with no resampling. The full contract
for that readback is §7 — it is specified separately because it is the part most likely to
be silently wrong.

Rendering through `frame_assembly` means **CLIP judges what the user judges**, including
colormap, exposure and view mode. The consequence — that changing the view changes the
fitness landscape — is accepted and documented in the UI with a one-line hint.

The only GL-dependent unit.

### 6.4 `services/auto_tournament_service.py` — `AutoTournamentService`

The generation state machine. Holds an `Optimizer` and a `CLIPScorer` (both injected), a
`RunLogger`, the rollout counters, and the score history. Contains no GL calls and no ImGui.

States: `IDLE → ASK → ROLLOUT → SCORE → TELL → ASK …`, with `PAUSED` reachable from any
state.

```
ASK      z = optimizer.ask(popsize)
         population = [spec.decode(z_i) for z_i in z]
         → orchestrator writes rules and resets the sim with gen_seed
ROLLOUT  each app frame: step the sim `sim_steps_per_frame` times.
         at evenly spaced step counts (e.g. 75/150/225/300 for 300 steps, 4 snapshots):
             → orchestrator calls TileCapture.capture(); frames are buffered
SCORE    fitness = mean over snapshots of clip.score(all buffered crops)   # (popsize,)
TELL     optimizer.tell(z, fitness); logger.log_generation(...)
         inject any user-selected tiles as elites, then clear the selection
         autosave a checkpoint every N generations
```

The rollout is deliberately **spread across real application frames** — the service returns
an action each frame and never loops internally. This is the mechanism by which the app
stays responsive; it is not an optimization to be added later.

**Seeding.** All tiles share one seed within a generation, so comparisons between genomes
are fair. The seed changes every generation, so the population does not overfit to one
lucky initial scatter. `gen_seed = base_seed + generation`.

*Implementation note:* it is not yet confirmed that the existing reset path accepts a seed.
Particle scatter in `entity_update.glsl` derives from `get_particle_rule_seed()`, so the
likely mechanism is driving that seed uniform per generation rather than adding a parameter
to `sim.reset()` — `sim.py` is user-owned and must not be restructured. The exact hook is
to be confirmed in the first implementation task that touches the loop; if no per-generation
seed control is reachable without modifying `sim.py`, fall back to a fixed seed and record
the deviation, since fixed-seed is an accepted (if slightly overfit-prone) option.

**Elite injection.** User-selected tiles are already members of the current generation's
population, so injection means **overwriting their fitness immediately before `tell`**:
each selected tile's score is set to `max(fitness) + 1e-3 * rank`, forcing them to occupy
the top ranks and therefore to dominate the next mean/covariance update. This expresses
"I like this one" to CMA-ES without special-casing the optimizer interface. The selection
is cleared after `tell`.

Auto mode reuses `TournamentService.selected` — the same set the manual picker writes to —
so the N×N button grid, its click handling, and the canvas-click-to-tile mapping are shared
between both modes with no duplication.

### 6.5 `services/run_logger.py` — `RunLogger`

Per-generation metrics to disk and to the UI. See §8.

### 6.6 `services/run_checkpoint.py` — `save_checkpoint` / `load_checkpoint`

Pure functions over a state dict. See §9.

### 6.7 `services/genome_io.py` — `export_genome` / `import_genome`

Converts between `z` vectors and `PhysicsConfig` files. See §9.2.

### 6.8 `state/auto_tournament_state.py` — `AutoTournamentState`

A dataclass following the existing convention: plain fields for UI state (`enabled`,
`prompt`, `algorithm`, `grid`, `steps_per_gen`, `sim_steps_per_frame`, `snapshots_per_gen`,
`sigma0`, `autosave_every`, `running`) plus one-shot request flags (`start_requested`,
`pause_requested`, `reset_requested`, `prompt_changed`, `grid_changed`,
`save_checkpoint_requested`, `load_checkpoint_path`, `save_best_requested`). One-shots are
cleared by the consuming side in `CommandHandler`, matching the correction already made for
`TournamentState`.

### 6.9 `ui/auto_tournament_window.py` — `AutoTournamentWindowMixin`

Follows the existing mixin pattern; mixed into `UI` in `ui/core.py` and rendered from
inside the existing tournament window under the Auto tab. Passive: renders widgets, exposes
state, runs no logic.

## 7. The CLIP input pipeline — exact contract

This section exists because a rendered-to-scored image pipeline has several failure modes
that produce plausible-looking but wrong data, and one of them (vertical tile ordering) has
**already caused a user-visible bug** in manual mode. Every step below is normative.

### 7.1 Framebuffer format — clamping

The capture FBO texture is created as `ctx.texture((N*224, N*224), 4, dtype='f1')`
(GL_RGBA8), **not** `f4`.

Rationale: `frame_assembly.frag` can emit values above 1.0 after exposure and gain. A
`dtype='f4'` float attachment stores those unclamped, so a readback would contain values
the display never shows, and the uint8 conversion would then have to invent a clamping
policy. An 8-bit unsigned-normalized attachment makes the GPU clamp to [0,1] on write,
exactly as it does for the visible framebuffer. CLIP therefore sees precisely what the user
sees, blown-out highlights included. This is the specific answer to "make sure it's not
overflowing".

### 7.2 Readback — row alignment

`fbo.read(components=3, alignment=1)`.

`alignment=1` is mandatory. OpenGL's default `GL_PACK_ALIGNMENT` is 4, which pads every row
up to a 4-byte boundary. For a 3-channel row of width `N*224` the byte count happens to be
divisible by 4 for every N in range, so the default would work *by coincidence* — and would
silently corrupt the image the moment `tile_px` or the channel count changed. Set it
explicitly.

Expected buffer length is asserted: `len(buf) == (N*224)**2 * 3`.

### 7.3 Orientation — the flip

`fbo.read()` returns rows **bottom-up**: byte row 0 is the bottom row of the image.
Meanwhile `tournament_home_tile()` numbers tile 0 as **bottom-left**. These two conventions
are easy to accidentally cancel out or accidentally double-apply.

The normative sequence is:

```python
img = np.frombuffer(buf, np.uint8).reshape(N*224, N*224, 3)[::-1]   # -> top-down
# img[0] is now the TOP row, standard image convention.
# Tile (tx, ty), ty measured from the BOTTOM, occupies:
row0 = (N - 1 - ty) * 224
crop = img[row0:row0 + 224, tx*224:(tx+1)*224]
tile_index = ty * N + tx        # matches tournament_home_tile()
```

Flipping once at the buffer level and inverting `ty` in the index formula keeps this in one
place. The vertical flip matters to CLIP beyond bookkeeping: prompts with orientation
semantics ("a tree", "a city skyline", "a waterfall") score differently on an upside-down
image.

This is pinned by a GPU test that renders `N²` known distinct colours and asserts each crop
comes back with the right colour at the right tile index (§12).

### 7.4 Preprocessing — dtype, layout, contiguity

```python
x = crops.astype(np.float32) / 255.0                     # (B,224,224,3)
x = (x - MEAN) / STD                                     # CLIP constants, per channel
x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))        # -> NCHW
x = x.astype(np.float16)                                 # fp16 models
```

Constants from `preprocessor_config.json`: mean `(0.4815, 0.4578, 0.4082)`,
std `(0.2686, 0.2613, 0.2758)`.

`np.ascontiguousarray` after the transpose is required, not cosmetic: a non-contiguous
array either forces a silent copy inside ONNX Runtime or errors outright depending on the
execution provider.

No resize is performed on the primary view — the crop is already exactly 224×224. Resizing
occurs only for the two augmented sub-crops (§6.1).

### 7.5 Inference batching — memory

At N=8 the batch is `64 tiles × 4 snapshots × 3 views = 768` images. As fp16 NCHW that is
`768 × 3 × 224 × 224 × 2 B ≈ 231 MB` for the input tensor alone, before activations. That
is the real overflow risk at large grid sizes.

Vision inference is therefore **chunked at a maximum of 64 images per `session.run` call**,
with results concatenated. 64 images ≈ 19 MB of input, which is comfortable on any DX12
device including the Intel UHD fallback.

### 7.6 Sanity checks

After each capture, cheaply validated and reported once per run rather than per frame:

- `crops.max() > 0` — catches an unbound or never-drawn FBO, which would otherwise produce
  a perfectly flat fitness landscape that looks like "CLIP has no signal".
- Mean luminance outside `[2, 253]` triggers a single UI warning that the capture is
  effectively black or blown out, since fitness is meaningless in either case.

## 8. Run logging

`services/run_logger.py` — `RunLogger`.

A run is created on Start and lives in `runs/<run_id>/` where `run_id` is
`YYYYmmdd-HHMMSS`. `runs/` is gitignored.

```
runs/20260804-143022/
  config.json         settings at run start
  log.jsonl           one JSON object per generation, appended and flushed
  frames/gen_000010.png   best tile image, every `frame_every` generations (default 10, 0=off)
  checkpoint_gen000010.npz
  latest.npz          copy of the most recent checkpoint
```

One `log.jsonl` line per generation:

```json
{"gen": 42, "t_wall": 1785880000.1, "t_gen_s": 0.44, "prompt": "glowing coral",
 "algorithm": "CMA-ES", "sigma": 0.31, "popsize": 16, "grid": 4, "seed": 1042,
 "fit_best": 0.71, "fit_mean": 0.38, "fit_median": 0.36, "fit_min": 0.09,
 "fit_std": 0.17, "best_tile": 6, "steps_per_gen": 300, "snapshots": 4,
 "canvas_px": 1024, "elites_injected": 0}
```

JSONL rather than CSV because the schema will grow (physics blocks, novelty terms), and
appending a field must not break older files or readers. Flushed every generation, so a
crash loses at most one line.

`fit_*` are **fitness, higher is better** — not loss. The UI labels the axis "fitness" and
the sparkline is drawn ascending, so there is no sign ambiguity anywhere.

In-UI, the same data drives a sparkline of `fit_best` and `fit_mean` over generations, and
a scrollable table of the last 50. An **Open run folder** button reveals the directory.

The periodic best-tile PNG is deliberate: a run folder full of `gen_000010.png`,
`gen_000020.png`, … is directly assemblable into a timelapse of the evolution, which is a
natural artifact for a generative art tool and costs one image write per 10 generations.

## 9. Checkpoints and genome export

Two distinct save formats, because they answer two different questions.

A **checkpoint** (§9.1) is *"resume this exact search"* — it carries optimizer state, RNG,
history and settings, and is inseparable from its grid size.

A **genome export** (§9.2) is *"keep this creature"* — it is just the 80 floats plus
provenance, carries no optimizer state at all, and can be loaded anywhere: as a new search's
starting point with different settings, or into the normal single-simulation view at full
resolution.

### 9.1 Checkpoints (full run state)

`services/run_checkpoint.py` — `save_checkpoint(path, state)` and `load_checkpoint(path)`,
pure functions over a dict. Format is a single `.npz`, with non-array metadata stored as a
JSON string under a `meta` key.

**Contents:**

| Key | Purpose |
|---|---|
| `format_version` | int; a mismatch fails loudly rather than misreading |
| `genome_spec_signature` | e.g. `"brain:80"`; refuses a brain-only checkpoint in a brain+physics build |
| `generation` | resume point |
| `optimizer_name` | must match the selected algorithm, or the UI switches to it |
| `optimizer_state` | from `Optimizer.state_dict()` — explicit arrays, never a pickle |
| `numpy_rng_state` | so a resumed run is a continuation, not a fresh stochastic branch |
| `base_seed` | per-generation seeding |
| `prompt`, `distractors` | the fitness landscape |
| `settings` | grid, steps_per_gen, snapshots, sigma0, autosave_every |
| `history` | per-generation `fit_best` / `fit_mean` arrays, so the sparkline survives a reload |
| `best_z`, `best_fitness` | best-ever individual |

Explicitly **not** included: CLIP weights, canvas pixels, particle state. A checkpoint is a
few hundred KB — the CMA covariance at dim 80 is an 80×80 float64 matrix, ~51 KB.

**Writes are atomic:** write to `<name>.npz.tmp`, then `os.replace`. A crash mid-write must
never leave a corrupt checkpoint where a good one was.

**Autosave** every N generations (default 10, 0 = off), on Pause, on Auto-mode disable, and
on clean app exit. Each autosave also refreshes `latest.npz` (a copy, not a symlink —
symlinks require elevation on Windows).

**Load** presents a file picker over `runs/*/*.npz`. On load:

- The grid is forced to the checkpoint's value. `cmaes.CMA` fixes popsize at construction,
  so a checkpoint and a grid size are inseparable. Attempting to load into a different grid
  changes the grid rather than silently corrupting the search.
- The prompt, algorithm and settings are restored into the UI so what is on screen matches
  what is running.
- The run continues into a **new** `runs/<new_id>/` directory whose `config.json` records
  `resumed_from`. The original run's log is never appended to by a second process, which
  keeps every log file a single coherent timeline.

### 9.2 Genome export/import (model only, no optimizer state)

**Export uses the existing config format.** `services/config_saver.py`'s `PhysicsConfig`
already carries a `rule: np.ndarray (10, 8)` field alongside every physics slider, and
already has `save_to_file`, `load_from_file` and clipboard encode/decode. An evolved genome
is therefore exported as an **ordinary Fluoddity config file** — the evolved rule combined
with the physics slider values it was evolved under.

This is deliberately not a new file format. The consequence is that an exported genome:

- loads into the **normal single-simulation view at any canvas resolution** through the
  existing config browser, with no new load path — this is the "train small, then run it
  big" workflow;
- is copy-pasteable through the existing clipboard config string;
- can be dropped into a **manual** tournament tile;
- is readable by every existing tool that understands configs.

**Export triggers:** a **Save best genome** button (best-ever individual), a **Save this
tile** action on any tile in the grid, and an optional autosave of the best genome every N
generations into `runs/<run_id>/genomes/gen_XXXXXX.json`.

**Import as a search starting point.** Loading a genome into Auto mode sets the CMA-ES
initial mean `x0` and nothing else — sigma, algorithm, grid and popsize are all taken from
the current UI, not from the file. This is the "load the model, not the optimizer" case:
you can take a creature evolved at N=4 with CMA-ES and continue it at N=6 with Sep-CMA-ES
and a larger sigma, which a checkpoint deliberately cannot do.

Import requires inverting the §5.1 squash:

```python
EPS = 1e-4
z_freq = np.arctanh(np.clip(freq / 3.0, -1 + EPS, 1 - EPS))
z_amp  = np.arctanh(np.clip(amp,         -1 + EPS, 1 - EPS))
```

The clamp is necessary because a config file may hold values outside the squash's range —
from the legacy rule generator, from hand-editing, or from a future wider range. The
clamp is **lossy at the extremes**: a coefficient at or beyond the boundary comes back as
`atanh(1 - 1e-4) ≈ 4.95` rather than infinity. This is correct behaviour (the search must
start at a finite point) but means an imported genome is not always a bit-exact roundtrip.
Import therefore logs how many of the 80 coefficients were clamped, so a badly
out-of-range file is visible rather than silent.

**Resolution caveat, recorded in the export.** The simulation is only approximately
scale-invariant: forces are divided by `SQRT_WORLD_SIZE` and particle size scales with it,
but sensor geometry is in entity units, so the number of texels between a particle and its
sensors changes with canvas resolution. A genome evolved on 256 px tiles will not look
identical at 2048². The export therefore records `evolved_at_canvas_px` and
`evolved_at_tile_px` in the config's metadata, and loading it at a substantially different
resolution shows a one-line note rather than silently behaving differently. This is
information, not a restriction — running a genome at a resolution it was not evolved at is
a legitimate and interesting thing to do.

## 10. Data flow per frame (Auto mode running)

Inserted into `App.orchestrate_frame()` after the existing tournament handling:

```
1. ui.get_state()
2. CommandHandler.process_commands()      # handles auto one-shots, clears them
3. action = auto_service.update()
4. match action:
     WRITE_RULES → sim.write_tournament_rules(service.pack_rule_bytes()); sim.reset(seed)
     STEP        → SimulationRunner steps N times instead of 1
     CAPTURE     → auto_service.submit_frames(tile_capture.capture(render_fn))
     SCORE       → auto_service.score_and_tell()      # ~15 ms, once per generation
     NONE        → normal single step
5. render camera view, then UI
```

`SCORE` blocks for roughly 15 ms once every ~30 frames — about 3% of a generation, i.e. one
slightly long frame every half second. Moving CLIP inference to a worker thread is a known,
straightforward follow-up if this proves visible in practice, but it is not in v1: it would
add cross-thread ONNX session and buffer-lifetime concerns for a 3% win.

### 10.1 Performance budget

Measured baseline: 716 sim steps/s on the RTX 3080 with tournament mode on. At the default
N=4, 300 steps, 4 snapshots:

| Stage | Cost per generation | Share |
|---|---|---|
| CMA-ES `ask` + `tell`, dim 80, popsize 16 | <1 ms | 0.2% |
| 4 × FBO readback, 896×896 uint8 RGB (2.4 MB each) | ~4 ms | 1% |
| CLIP ViT-B/32 fp16, 192 images in 3 chunks | ~15 ms | 3% |
| Logging (one JSONL line, flushed) | <1 ms | 0.2% |
| **300 simulation steps** | **~420 ms** | **95%** |
| **Total** | **~440 ms** | |

≈2.3 generations/second. The simulation outruns the entire evolutionary apparatus by
roughly 20×, which satisfies the constraint with a wide margin even if the CLIP estimate is
off by 5×.

Scaling to N=8 quadruples the CLIP work to ~60 ms and the readback to ~16 ms while the
simulation cost is unchanged (the same particle budget is redistributed across more tiles),
giving ~500 ms/generation — still ~85% simulation. The constraint holds across the whole
grid range.

## 11. Failure handling

### 11.1 Missing or broken dependencies

| Condition | Behaviour |
|---|---|
| `onnxruntime` / `tokenizers` / `cmaes` not importable | Auto tab shows the missing package and a `pip install` line. Manual mode unaffected. |
| Model files absent from `models/clip-vit-b32/` | Auto tab shows a **Download CLIP model (~330 MB)** button; fetch runs on a worker thread with a progress bar; UI stays responsive. |
| DirectML provider unavailable | Fall back to `CPUExecutionProvider` with a visible warning that scoring will be slow. |
| Download interrupted | Write to `<name>.part` and rename on completion, so a partial file is never mistaken for a valid model. |

### 11.2 Runtime

| Condition | Behaviour |
|---|---|
| Empty prompt | Start button disabled. |
| Canvas or world resized mid-rollout | Abort the current generation and restart it. A partial rollout is not a valid fitness sample, and the existing `mark_dirty()` path already reuploads genomes. |
| Grid slider changed | Applied at the next generation boundary; resets the optimizer (§4.1) and reallocates the capture FBO. Never applied mid-rollout. |
| Manual mode toggled on mid-run | Auto pauses and autosaves; optimizer state is retained so resuming continues rather than restarting. |
| `NaN`/`inf` fitness | Replaced with the generation minimum before `tell`, and the occurrence is logged. |
| Shader hot-reload (`V`) mid-rollout | Abort and restart the generation, same as resize. |
| Checkpoint fails to load (version/signature mismatch) | Refuse with a specific message naming the mismatch; the current run is left untouched. |
| Imported genome has out-of-range coefficients | Clamped per §9.2 and the clamp count is reported in the UI, rather than producing `inf` in `x0` and silently breaking the search. |
| `runs/` not writable | Logging and autosave disabled with one warning; the run continues. Evolution must not be blocked by a disk problem. |

## 12. Testing

Following the existing convention — a real `pytest` suite plus headless GPU harnesses in
the scratchpad. No GPU required except where noted.

**Unit (no GPU):**
- `test_optimizers.py` — each optimizer converges on an 80-D sphere function; CMA-ES beats
  `RandomSearch` on 80-D Rastrigin within a fixed budget; `tell` sign convention is correct
  (higher fitness wins) for all four; `ask` returns the requested shape and dtype;
  `state_dict`/`load_state_dict` roundtrips such that a restored optimizer produces the
  *identical* next `ask` as the original.
- `test_genome_spec.py` — `encode`/`decode` roundtrip within tolerance; decoded genomes are
  inside the documented bounds for extreme `z`; `spec.dim` matches block sizes;
  `signature()` is stable.
- `test_clip_scorer.py` — against a stubbed ONNX session: softmax normalization is correct,
  target is index 0, prompt embeddings are cached across `score` calls, augmented views are
  averaged, uint8→normalized preprocessing matches the reference constants, NCHW output is
  C-contiguous, and batches over 64 are chunked.
- `test_auto_tournament_service.py` — full state machine against a fake capture and fake
  scorer: state sequence is correct, snapshots land on the expected steps for several
  `(steps_per_gen, snapshots)` combinations, seeds are shared within a generation and differ
  across generations, elite injection ranks selected tiles first, resize aborts and
  restarts, pause/resume preserves optimizer state, NaN fitness is replaced.
- `test_auto_tournament_state.py` — one-shot flags default false and are cleared by the
  consumer.
- `test_run_logger.py` — JSONL is one valid object per line and is flushed per generation;
  a mid-run schema addition does not break reading earlier lines; an unwritable directory
  degrades to a warning instead of raising.
- `test_run_checkpoint.py` — full roundtrip restores optimizer, RNG, history and settings;
  a `format_version` mismatch raises a specific error; a `genome_spec_signature` mismatch
  raises a specific error; writes are atomic (a simulated crash mid-write leaves the prior
  checkpoint intact).
- `test_genome_io.py` — an exported genome is a valid `PhysicsConfig` that `load_from_file`
  reads back with an identical rule; `z → genome → z` roundtrips within tolerance for
  in-range values; out-of-range coefficients are clamped rather than producing `inf`, and
  the clamp count is reported; importing sets only `x0` and leaves sigma, algorithm and
  grid untouched.
- `test_tile_geometry.py` — the §7.3 index formula, tested purely arithmetically for
  N ∈ [2,8]: every tile index in `[0, N²)` maps to a unique non-overlapping crop rectangle,
  and `tile_index = ty*N + tx` agrees with `tournament_home_tile`.

**GPU-marked (`@pytest.mark.gpu`, skipped by default):**
- `test_tile_capture.py` — render N² known distinct colours, assert the split returns them
  in `tournament_home_tile` order (tile 0 = bottom-left), for N ∈ {2, 4, 8}. This is the
  test that pins §7.3, the readback alignment, and the f1 clamping in one place.
- `test_clip_real_model.py` — a synthetic red square scores higher on `"a red square"` than
  on `"a blue circle"`, and higher than a blank image does on either.

**Manual verification:** extend `docs/testing_checklist.md` with the Auto-mode path,
including a save → quit → load → resume cycle.

## 13. Implementation risk and the first task

**The one unvalidated assumption is that CLIP has usable signal on 256 px slime-mold
trails.** It is entirely possible that CLIP perceives every tile as "abstract texture",
producing a flat landscape that no optimizer can climb — in which case every layer built on
top of it would be debugged in vain.

Therefore **Task 1 of the implementation plan is a standalone kill-switch check**, done
before any loop code exists:

1. Capture ~20 visually varied tiles from the existing manual tournament as PNGs.
2. Score them offline against ~6 prompts using only `CLIPScorer`.
3. Confirm (a) the ranking is defensible to a human eye, and (b) the score spread across
   tiles is wide enough to optimize against — as a concrete bar, the standard deviation of
   scores across the 20 tiles should exceed 0.05 for at least half the prompts.

If this fails, the remedy is to adjust the *eval image* — colormap, contrast, zoom level,
possibly raising the canvas resolution — and re-test, before building the loop.

## 14. Deferred (considered, deliberately not in v1)

- **Novelty / open-endedness objective** (ASAL's second mechanism): penalize consecutive
  snapshots whose CLIP embeddings are near-identical, so frozen patterns score worse.
  Deferred because it adds a weight to tune before we know the base objective works.
- **Non-square grids** (`TOURNAMENT_GRID` as a vec2), which would allow near-square tiles on
  a 16:9 canvas and remove the §4.2 aspect forcing. Real GLSL work across three shaders;
  revisit if forcing 1:1 proves annoying in practice.
- **Population size decoupled from tile count** via multi-batch generations. Useful if the
  fitness turns out noisy enough to want popsize ≫ 16 on a legible grid.
- **Threaded CLIP inference** — 3% win, real complexity cost. See §10.
- **Physics parameters in the genome** — see §5.2.
- **Multiple CLIP backbones via a dropdown** — trivial once the scorer interface exists,
  but not useful until ViT-B/32 is shown to work.
- **MAP-Elites illumination archive** — a different product, not an extension of this one.

## 15. Files touched

**New:**
```
services/clip_scorer.py
services/optimizers.py
services/genome_spec.py
services/tile_capture.py
services/auto_tournament_service.py
services/run_logger.py
services/run_checkpoint.py
services/genome_io.py
state/auto_tournament_state.py
ui/auto_tournament_window.py
tools/fetch_clip_onnx.py
tests/test_optimizers.py
tests/test_genome_spec.py
tests/test_clip_scorer.py
tests/test_auto_tournament_service.py
tests/test_auto_tournament_state.py
tests/test_run_logger.py
tests/test_run_checkpoint.py
tests/test_genome_io.py
tests/test_tile_geometry.py
tests/test_tile_capture.py           (gpu-marked)
tests/test_clip_real_model.py        (gpu-marked)
```

**Modified:**
```
requirements.txt              new optional deps
.gitignore                    models/, runs/
main.py                       construct service, wire into orchestrate_frame
command_handler.py            handle + clear auto one-shot flags
ui/core.py                    mix in AutoTournamentWindowMixin
ui/tournament_window.py       Manual | Auto tab selector; N-driven button grid
services/tournament_service.py  TILES 16 -> constructor arg
state/ui_state.py             auto_tournament field
state/__init__.py             export
services/__init__.py          export (lazy — must not import onnxruntime at startup)
docs/testing_checklist.md     Auto-mode manual checks
```

**Unchanged, deliberately:** `sim.py` (user-owned), all `shaders/*`.
