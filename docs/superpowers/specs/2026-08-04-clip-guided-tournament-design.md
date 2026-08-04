# CLIP-Guided Automatic Tournament — Design

**Date:** 2026-08-04
**Status:** Approved (pending spec review)
**Scope:** Add a second, *automatic* tournament mode alongside the existing manual one. A
text prompt the user types at any time is turned into a fitness score by CLIP, and an
evolution strategy (CMA-ES by default, selectable) drives the 16-tile population toward it.
The user can steer or veto at any point.

**Builds on:** `2026-08-03-interactive-tournament-design.md`. The tiled 4×4 canvas, hard
tile isolation, genome representation, rule packing and GPU upload are all reused unchanged.

## 1. Goal & non-goals

**Goal:** Replace the human selector with a CLIP-derived fitness signal, without giving up
human control. The user types a goal ("glowing coral", "a dense city map"), presses Start,
and watches the grid converge — retyping the prompt or clicking tiles at any moment.

**Design constraint (from the user, verbatim):** the evolutionary machinery must never be
the bottleneck. The simulation should remain the dominant cost. See §7 for the budget that
makes this true by construction rather than by tuning.

**Non-goals (YAGNI):**
- No evolution of the 12 physics sliders in v1. The genome stays the 80-float Fourier
  brain. The encoding is structured so physics can be added later (§4.2), but that work is
  explicitly out of scope here.
- No 3D (Fluoddity3D) support.
- No CLIP fine-tuning, no aesthetic-predictor model, no diffusion anything.
- No open-endedness / novelty objective in v1 (considered and deferred — see §11).
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
- **Start / Pause / Reset** buttons. Reset discards the optimizer state and reseeds.
- Sliders: **Steps per generation** (default 300, range 50–2000), **Sim steps per frame**
  (default 10, range 1–50), **Snapshots per generation** (default 4, range 1–8),
  **Initial sigma** (default 0.5, range 0.05–1.5).
- The existing 4×4 button grid stays. Each button now also shows that tile's fitness for
  the last completed generation. Clicking a tile still selects it; in Auto mode a selected
  tile is **force-injected as an elite** into the next generation (§5.4) and the selection
  is then cleared. This is the "steer and veto" the user asked for.
- A readout: current generation number, best-ever score, and a small sparkline of best
  score per generation.
- A **Save best genome** button, reusing the existing tournament save path.

If the CLIP model files or `onnxruntime` are missing, the Auto tab renders a single
explanatory line and a **Download CLIP model (~330 MB)** button (§8.1). Manual mode is
unaffected in every failure case.

## 3. New dependencies

Added to `requirements.txt`:

```
onnxruntime-directml>=1.20   # DirectX 12 GPU inference; no CUDA toolkit, no PyTorch
tokenizers>=0.20             # CLIP BPE tokenizer (Rust, ~3 MB)
cmaes>=0.11                  # CyberAgentAILab CMA-ES; pure numpy, ~50 KB
```

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

## 4. Genome encoding

### 4.1 Search space vs. genome space

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

### 4.2 Forward compatibility with physics parameters

Encoding lives behind a `GenomeSpec` — an ordered list of named blocks:

```python
GenomeSpec(blocks=[Block(name="brain", size=80, decode=brain_decode)])
```

`spec.dim`, `spec.decode(z) -> dict[str, np.ndarray]`, `spec.encode(parts) -> z`. Adding
physics later means appending `Block(name="physics", size=12, decode=physics_decode)`; the
optimizer, scorer and loop are all dimension-agnostic and need no changes. The remaining
work at that point is per-tile physics uniforms in the shader, which is why it is out of
scope now.

## 5. Architecture

Five new units. Each has one purpose, a small interface, and can be understood and tested
without the others.

### 5.1 `services/clip_scorer.py` — `CLIPScorer`

```python
class CLIPScorer:
    def __init__(self, model_dir: str, providers: list[str] | None = None) -> None
    def set_prompt(self, text: str, distractors: list[str] | None = None) -> None
    def score(self, images: np.ndarray) -> np.ndarray   # uint8 (B,224,224,3) -> float32 (B,)
    @property
    def available(self) -> bool
```

Depends only on `onnxruntime`, `tokenizers` and `numpy`. Knows nothing about tournaments,
tiles, or OpenGL.

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
frame plus two random 70–100% crops resampled to 224. Averaging over random views is the
standard defense against a search process finding adversarial texture that satisfies a
single fixed view. This makes the effective vision batch `16 × snapshots × 3`.

Text embeddings are computed once per `set_prompt` and cached, so retyping the goal costs
one ~5 ms text-encoder pass.

Preprocessing follows CLIP's `preprocessor_config.json`: uint8 → float, scale to [0,1],
normalize by mean `(0.4815, 0.4578, 0.4082)` / std `(0.2686, 0.2613, 0.2758)`, NCHW,
cast to fp16 for the fp16 models.

### 5.2 `services/optimizers.py` — `Optimizer` protocol + 4 implementations

```python
class Optimizer(Protocol):
    name: str
    def ask(self, n: int) -> np.ndarray          # (n, dim) float32
    def tell(self, z: np.ndarray, fitness: np.ndarray) -> None   # higher fitness = better
    def best(self) -> tuple[np.ndarray, float]   # (z, fitness)
```

`tell` takes **fitness where higher is better**; implementations negate internally as
needed (the `cmaes` library minimizes). This keeps the sign convention in one place instead
of leaking it into the loop.

- `CMAESOptimizer` — wraps `cmaes.CMA`. Full covariance. At dim 80 the eigendecomposition
  is ~1 ms, so the separable variant buys nothing yet.
- `SepCMAESOptimizer` — wraps `cmaes.SepCMA`. Present for when the dimension grows.
- `GAOptimizer` — reuses `services/genome.py`'s `mutate`/`crossover` operators, applied in
  `z` space. This is manual mode's operator driven by CLIP instead of by a human, which
  makes it a genuinely useful comparison rather than filler.
- `RandomSearchOptimizer` — uniform resampling. The control. If CMA-ES cannot beat this,
  the fitness signal is not real, and that is worth being able to check on demand.

Pure numpy. No OpenGL, no CLIP, no app state. Fully unit-testable.

### 5.3 `services/tile_capture.py` — `TileCapture`

```python
class TileCapture:
    def __init__(self, ctx, grid: int = 4, tile_px: int = 224) -> None
    def capture(self, render_fn) -> np.ndarray   # -> uint8 (grid*grid, tile_px, tile_px, 3)
    def release(self) -> None
```

Owns an offscreen framebuffer of exactly `grid*tile_px` square (896×896 for a 4×4 grid),
calls `render_fn(fbo)` to draw the assembled tournament view into it via the normal
`frame_assembly` path, reads back once with `fbo.read(components=3)`, and splits the result
into 16 exact 224×224 crops with no resampling. Tile index ordering must match
`tournament_home_tile()` — tile 0 is bottom-left — and this is asserted by a test, because
that exact ordering has already been the source of one user-visible bug.

Rendering through `frame_assembly` means **CLIP judges what the user judges**, including
colormap, exposure and view mode. The consequence — that changing the view changes the
fitness landscape — is accepted and documented in the UI with a one-line hint.

The only GL-dependent unit.

### 5.4 `services/auto_tournament_service.py` — `AutoTournamentService`

The generation state machine. Holds an `Optimizer` and a `CLIPScorer` (both injected), the
rollout counters, and the score history. Contains no GL calls and no ImGui.

States: `IDLE → ASK → ROLLOUT → SCORE → TELL → ASK …`, with `PAUSED` reachable from any
state.

```
ASK      z = optimizer.ask(16)
         population = [spec.decode(z_i) for z_i in z]
         → orchestrator writes rules and resets the sim with gen_seed
ROLLOUT  each app frame: step the sim `sim_steps_per_frame` times.
         at steps {75, 150, 225, 300} (evenly spaced by snapshot count):
             → orchestrator calls TileCapture.capture(); frames are buffered
SCORE    fitness = mean over snapshots of clip.score(all buffered crops)   # (16,)
TELL     optimizer.tell(z, fitness); history.append(fitness.max())
         inject any user-selected tiles as elites, then clear the selection
```

The rollout is deliberately **spread across real application frames** — the service returns
an action each frame and never loops internally. This is the mechanism by which the app
stays responsive; it is not an optimization to be added later.

**Seeding.** All 16 tiles share one seed within a generation, so comparisons between
genomes are fair. The seed changes every generation, so the population does not overfit to
one lucky initial scatter. `gen_seed = base_seed + generation`.

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
so the 4×4 button grid, its click handling, and the canvas-click-to-tile mapping are shared
between both modes with no duplication.

### 5.5 `state/auto_tournament_state.py` — `AutoTournamentState`

A dataclass following the existing convention: plain fields for UI state (`enabled`,
`prompt`, `algorithm`, `steps_per_gen`, `sim_steps_per_frame`, `snapshots_per_gen`,
`sigma0`, `running`) plus one-shot request flags (`start_requested`, `pause_requested`,
`reset_requested`, `prompt_changed`, `save_best_requested`). One-shots are cleared by the
consuming side in `CommandHandler`, matching the correction already made for
`TournamentState`.

### 5.6 `ui/auto_tournament_window.py` — `AutoTournamentWindowMixin`

Follows the existing mixin pattern; mixed into `UI` in `ui/core.py` and rendered from
inside the existing tournament window under the Auto tab. Passive: renders widgets, exposes
state, runs no logic.

## 6. Data flow per frame (Auto mode running)

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

## 7. Performance budget

Measured baseline: 716 sim steps/s on the RTX 3080 with tournament mode on.

| Stage | Cost per generation | Share |
|---|---|---|
| CMA-ES `ask` + `tell`, dim 80, popsize 16 | <1 ms | 0.2% |
| 4 × FBO readback, 896×896 uint8 RGB (2.4 MB each) | ~4 ms | 1% |
| CLIP ViT-B/32 fp16, effective batch 192 | ~15 ms | 3% |
| **300 simulation steps** | **~420 ms** | **96%** |
| **Total** | **~440 ms** | |

≈2.3 generations/second, ≈135 generations/minute. The simulation outruns the entire
evolutionary apparatus by roughly 20×, which satisfies the constraint with a wide margin
even if the CLIP estimate is off by 5×.

## 8. Failure handling

### 8.1 Missing or broken dependencies

| Condition | Behaviour |
|---|---|
| `onnxruntime` / `tokenizers` / `cmaes` not importable | Auto tab shows the missing package and a `pip install` line. Manual mode unaffected. |
| Model files absent from `models/clip-vit-b32/` | Auto tab shows a **Download CLIP model (~330 MB)** button; fetch runs on a worker thread with a progress bar; UI stays responsive. |
| DirectML provider unavailable | Fall back to `CPUExecutionProvider` with a visible warning that scoring will be slow. |
| Download interrupted | Write to `<name>.part` and rename on completion, so a partial file is never mistaken for a valid model. |

### 8.2 Runtime

| Condition | Behaviour |
|---|---|
| Empty prompt | Start button disabled. |
| Canvas or world resized mid-rollout | Abort the current generation and restart it. A partial rollout is not a valid fitness sample, and the existing `mark_dirty()` path already reuploads genomes. |
| Manual mode toggled on mid-run | Auto pauses; optimizer state is retained so resuming continues rather than restarting. |
| `NaN`/`inf` fitness | Replaced with the generation minimum before `tell`. |
| Shader hot-reload (`V`) mid-rollout | Abort and restart the generation, same as resize. |

## 9. Testing

Following the existing convention — a real `pytest` suite plus headless GPU harnesses in
the scratchpad. No GPU required except where noted.

**Unit (no GPU):**
- `test_optimizers.py` — each optimizer converges on an 80-D sphere function; CMA-ES beats
  `RandomSearch` on 80-D Rastrigin within a fixed budget; `tell` sign convention is correct
  (higher fitness wins) for all four; `ask` returns the requested shape and dtype.
- `test_genome_spec.py` — `encode`/`decode` roundtrip within tolerance; decoded genomes are
  inside the documented bounds for extreme `z`; `spec.dim` matches block sizes.
- `test_clip_scorer.py` — against a stubbed ONNX session: softmax normalization is correct,
  target is index 0, prompt embeddings are cached across `score` calls, augmented views are
  averaged, uint8→normalized preprocessing matches the reference constants.
- `test_auto_tournament_service.py` — full state machine against a fake capture and fake
  scorer: state sequence is correct, snapshots land on the expected steps, seeds are shared
  within a generation and differ across generations, elite injection ranks selected tiles
  first, resize aborts and restarts, pause/resume preserves optimizer state.
- `test_auto_tournament_state.py` — one-shot flags default false and are cleared by the
  consumer.

**GPU-marked (`@pytest.mark.gpu`, skipped by default):**
- `test_tile_capture.py` — render 16 known distinct colours, assert the split returns them
  in `tournament_home_tile` order (tile 0 = bottom-left).
- `test_clip_real_model.py` — a synthetic red square scores higher on `"a red square"` than
  on `"a blue circle"`, and higher than a blank image does on either.

**Manual verification:** extend `docs/testing_checklist.md` with the Auto-mode path.

## 10. Implementation risk and the first task

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
possibly rendering tiles at higher resolution — and re-test, before building the loop.

## 11. Deferred (considered, deliberately not in v1)

- **Novelty / open-endedness objective** (ASAL's second mechanism): penalize consecutive
  snapshots whose CLIP embeddings are near-identical, so frozen patterns score worse.
  Deferred because it adds a weight to tune before we know the base objective works.
- **Threaded CLIP inference** — 3% win, real complexity cost. See §6.
- **Physics parameters in the genome** — see §4.2.
- **Multiple CLIP backbones via a dropdown** — trivial once the scorer interface exists,
  but not useful until ViT-B/32 is shown to work.
- **MAP-Elites illumination archive** — a different product, not an extension of this one.

## 12. Files touched

**New:**
```
services/clip_scorer.py
services/optimizers.py
services/genome_spec.py
services/tile_capture.py
services/auto_tournament_service.py
state/auto_tournament_state.py
ui/auto_tournament_window.py
tools/fetch_clip_onnx.py
tests/test_optimizers.py
tests/test_genome_spec.py
tests/test_clip_scorer.py
tests/test_auto_tournament_service.py
tests/test_auto_tournament_state.py
tests/test_tile_capture.py          (gpu-marked)
tests/test_clip_real_model.py       (gpu-marked)
```

**Modified:**
```
requirements.txt          new optional deps
.gitignore                models/
main.py                   construct service, wire into orchestrate_frame
command_handler.py        handle + clear auto one-shot flags
ui/core.py                mix in AutoTournamentWindowMixin
ui/tournament_window.py   Manual | Auto tab selector
state/ui_state.py         auto_tournament field
state/__init__.py         export
services/__init__.py      export (lazy — must not import onnxruntime at startup)
docs/testing_checklist.md Auto-mode manual checks
```

**Unchanged, deliberately:** `sim.py` (user-owned), all `shaders/*`.
