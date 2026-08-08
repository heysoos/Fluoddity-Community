# IMGEP Novelty Search over a Persistent CLIP Archive — Design

**Date:** 2026-08-07
**Status:** Approved (pending spec review)
**Scope:** A third tournament mode, **Explore (IMGEP)**, that replaces the single fixed CLIP
prompt with an intrinsically-motivated goal exploration process over a growing, persistent
archive of patterns. Novelty is measured in CLIP image-embedding space. Exploration alternates
between undirected novelty-driven **expansion** and goal-directed **expedition**, following
Expedition & Expansion (E&E, arXiv:2509.03863). Temporal novelty follows ASAL
(arXiv:2412.17799, Eq. 3).

**Builds on:** `2026-08-04-clip-guided-tournament-design.md`. The tiled canvas, tile isolation,
genome encoding, per-tile physics, offscreen capture, CLIP scorer, optimizers, run logging and
checkpointing are all reused. §14 of that document deferred exactly this work; this is it.

**Explicitly closes:** the "Novelty / open-endedness objective" and "MAP-Elites illumination
archive" items in that document's Deferred list — the first as the liveness gate (§4.2), the
second as the archive's eviction rule (§5.4) and 2-D map (§8.3), rather than as a MAP-Elites
grid.

---

## 1. Goal and non-goals

**Goal.** Turn the auto tournament from "climb toward one prompt" into "map the space". The
user starts a run and walks away; the archive grows outward, and when they come back they have
a browsable library of distinct patterns, each exportable as an ordinary Fluoddity config.

**Why this substrate is unusually well suited.** E&E's loop is serial: one θ, one rollout, one
embedding. Fluoddity already runs N² independent rollouts per generation, on hard-isolated
tiles, with per-tile brains *and* per-tile physics. One generation is therefore a batch of 4–64
IMGEP samples at no extra cost. The existing CMA-ES loop is already E&E's expedition machinery;
what is genuinely new is the archive, the novelty-proportional parent sampler, and the goal
sources.

**Non-goals (YAGNI):**

- **No VLM.** E&E uses o4-mini to author linguistic goals from archive thumbnails. Here the
  human authors them (§7.2). Fluoddity has never required a network or an API key and will not
  start now.
- **No new runtime dependencies.** PCA is a 512×512 eigendecomposition in numpy; no
  `umap-learn`, no `scikit-learn`, no approximate-nearest-neighbour index. Brute-force k-NN over
  20k×512 is ~5 ms (§9).
- **No change to Manual or Auto (CLIP) mode.** Both must behave identically after this change,
  including with the CLIP dependencies absent. The existing auto-mode test files are the
  regression proof (§3.3).
- **No changes to `sim.py` or to any shader.** Every mechanism this design needs already exists
  on the GPU side.
- **No MAP-Elites grid.** The archive is unstructured; the 2-D projection is a *view*, not the
  data structure. A behaviour grid would require choosing and freezing axes before we know what
  the space looks like.
- **No multi-objective search.** Novelty is the objective; quality enters only as an admission
  gate, never as a weighted term.

---

## 2. User-facing behaviour

The Tournament window gains a third tab: **Manual | Auto (CLIP) | Explore (IMGEP)**.

In Explore mode:

- **Start / Pause / Reset.** No prompt is required — the search runs goal-free (expansion) until
  the first expedition fires.
- A **Goals** list: an editable list of text goals, each with an enable checkbox. Add, remove,
  reorder. Persisted across sessions. May be left empty, in which case every expedition uses a
  latent goal (§7.1).
- Exploration sliders (§7.4 for defaults and ranges): `σ_expand`, novelty exponent `α`,
  neighbours `k`, `seed_n`, expansion generations between expeditions `K`, `expedition_gens`,
  `expedition_sigma`, `latent goal share`, `liveness_min`, `target admission rate`.
- The shared rollout controls — grid, steps/generation, snapshots, sim steps/frame, physics
  search, per-tile mutation — are the same widgets Auto mode uses, rendered from one shared
  helper (§8.1).
- A **status line**: current regime (bootstrap / expansion / expedition), the active goal in
  words, generations remaining in the expedition, archive size, admission rate, the current
  adaptive novelty threshold, and the estimated wall-clock time of one full expansion+expedition
  cycle.
- **Left-click a tile** pins it to the archive, bypassing every gate (§5.3).
  **Right-click a tile** → "Chase this": starts an expedition immediately with that tile's own
  descriptor as the goal (§7.3).
- An **Archive** window (§8.2, §8.3): thumbnail gallery and 2-D semantic map, with per-entry
  Export / Seed / Pin / Delete.

If CLIP or its dependencies are unavailable, the Explore tab degrades exactly as the Auto tab
does today — one explanatory line and a download button. Manual mode is unaffected in every
failure case.

---

## 3. Architecture: extracting a search driver

### 3.1 The split

`AutoTournamentService` currently does two separable jobs. The first — a rollout state machine
spread across real application frames (`WRITE_RULES → STEP → CAPTURE×S → SCORE`), with
`abort_generation` on resize/shader-reload/grid-change — is *identical* for IMGEP and is
hard-won. The second — owning a CMA-ES, softmax scoring against distractors, elite injection —
is entirely prompt-specific and wrong for IMGEP, where expansion needs no optimizer at all and
expedition needs a *fresh* CMA-ES per goal.

The service keeps the first job and delegates the second to an injected driver.

```python
# services/search_driver.py
class SearchDriver(Protocol):
    name: str

    def set_spec(self, spec) -> None:
        """The genome layout changed (physics search toggled). Invalidates any
        optimizer state; must not raise."""

    def ask(self, n: int) -> np.ndarray:
        """(n, spec.dim) float32 — one search vector per tile."""

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray]) -> np.ndarray:
        """snapshots is the service's crop buffer verbatim: a list of S arrays
        of shape (n, 224, 224, 3) uint8, in capture order.
        Returns (n,) float32 — a per-tile score for the UI and the log only.
        The driver does whatever learning it does internally."""

    def status(self) -> dict:
        """Free-form, for the UI status line and the JSONL record."""

    def checkpoint_state(self) -> dict: ...
    def restore(self, d: dict) -> None: ...
```

`tell` receives raw crops rather than scores, which is what lets `PromptDriver` call
`scorer.score()` and `ImgepDriver` call `scorer.embed()`. The consequence is that
`AutoTournamentService` no longer imports or knows about CLIP at all.

### 3.2 What stays in the service

Genome decode stays in the service: it owns `spec`, `_resolve_spec()` and `physics_origin`,
calls `spec.decode(z)` on the driver's output, and writes `tournament.population` and
`tile_physics`. This is what keeps `main.py` unchanged — `svc.tile_physics` and `svc.gen_seed`
keep meaning exactly what they mean today. When the spec changes the service calls
`driver.set_spec(spec)`.

Also unchanged in the service: `snapshot_steps`, `_buffer`, `submit_frames`,
`abort_generation`, seeding (`gen_seed = base_seed + generation`), `RunLogger` integration, and
the `Phase`/`Action` enums.

### 3.3 Regression proof

`PromptDriver` is a mechanical extraction, not a rewrite. These files must pass **unmodified**:

```
tests/test_auto_tournament_service.py
tests/test_auto_tournament_state.py
tests/test_optimizers.py
tests/test_run_checkpoint.py
tests/test_run_logger.py
tests/test_genome_spec.py
tests/test_clip_scorer.py
tests/test_auto_tournament_plot.py
```

If any of them needs editing to pass, the refactor changed behaviour and is wrong. The one
permitted exception is *additive* — new tests appended to `test_clip_scorer.py` for `embed()`
and `embed_text()`.

### 3.4 The CLIPScorer extension

This is the whole "extend the CLIP comparison" step, and it is small:

```python
def embed(self, images: np.ndarray, n_views: int | None = None) -> np.ndarray:
    """uint8 (B,224,224,3) -> float32 (B, 512), L2-normalised.
    n_views=1 (the default here) skips augmentation entirely."""

def embed_text(self, prompts: list[str]) -> np.ndarray:
    """-> float32 (P, 512), L2-normalised. Does NOT touch the cached
    self._text_emb used by score()."""
```

`embed()` is the existing `_embed_images` made public with a view count. `embed_text()` is the
existing text path factored out of `set_prompt`. `score()` is then rewritten on top of `embed()`
and **must produce identical numbers** — pinned by a new test that recomputes the softmax from
`embed()` and asserts equality with `score()` to float32 tolerance.

Augmentation (`n_views=3`) exists as an anti-adversarial defence for a *directed* objective. It
is not wanted for novelty: three random sub-crops of one tile would be three archive-descriptor
candidates for one behaviour. Explore mode uses `n_views=1` throughout, which is also why it is
cheaper per generation than prompt mode (§9).

---

## 4. The behaviour descriptor

Per tile, per generation, the service already buffers S snapshots. Embed all S at `n_views=1` to
get `e_0 … e_{S-1}`, each unit-norm.

### 4.1 Descriptor

```
b = normalise(mean_s e_s)
```

The trajectory centroid, not the final frame. E&E uses the final state; ASAL uses per-timestep
embeddings. The centroid is chosen here because Fluoddity patterns are *watched in motion* — a
single frame discards what the pattern does, which is most of what distinguishes one creature
from another in this substrate. It is also insensitive to the exact stopping step, which matters
because `steps_per_gen` is user-adjustable.

**Known failure mode, accepted:** a pattern that oscillates between two dissimilar states gets a
centroid resembling neither, and may land next to an unrelated pattern in the archive. The
liveness scalar (§4.2) is stored alongside precisely so this case is *visible* rather than
silent, and the gallery can sort by it.

### 4.2 Liveness — ASAL Eq. 3

```
L = 1 - mean_{s>0} max_{s'<s} <e_s, e_s'>
```

ASAL's open-endedness objective is `argmin_θ E_T[max_{T'<T} <img_T, img_T'>]`; `L` is one minus
that quantity, evaluated at S-snapshot resolution, so that higher is better and the sign
convention matches everything else in the codebase. A frozen pattern has all `e_s` identical and
scores `L ≈ 0`. A dead canvas scores `L ≈ 0`.

This costs nothing: the snapshots and their embeddings are already computed for the descriptor.

**Snapshot count.** Explore mode defaults `snapshots_per_gen` to **6** rather than Auto mode's 4.
At 2000 steps per generation that places snapshots ~333 steps apart, which is enough for a slow
pattern to visibly change; 6 snapshots is 96 CLIP images at N=4, ~7 ms against ~2800 ms of
simulation. Same slider, different default.

---

## 5. The archive

A **single, gated, persistent** archive. One admission policy, one storage location, one
browser.

### 5.1 Novelty

```
NOV(b) = (1/k) * sum_{j in kNN(b)} (1 - <b, b_j>)
```

with `k = 10` and parent-sampling exponent `α = 4` — E&E's tuned values. All vectors are
L2-normalised, so cosine similarity is a dot product and the whole k-NN step is one matmul.

**Neighbours are drawn from `archive ∪ rejects_ring`.** The rejects ring holds the 2048 most
recently *rejected* descriptors (fp16, ~2 MB, never persisted, cleared on Reset). This is the
classic Lehman–Stanley formulation — novelty against archive ∪ current population — and it is
the mitigation for the known cost of a gated archive: without it, novelty is measured against a
filtered history, so the search has no memory of the regions it just rejected and will
re-explore them. With it, a region that was rejected 20 generations ago still suppresses novelty
there.

### 5.2 Admission gate

A tile is admitted when **all** hold:

| Gate | Test | Default |
|---|---|---|
| Viable | `capture_health.is_viable_tile(crop)` and `np.isfinite(b).all()` | — |
| Alive | `L >= liveness_min` | 0.02 (calibrate, §12.2) |
| Novel | `NOV(b) >= threshold` | adaptive, §5.2.1 |

`is_viable_tile(crop) -> bool` is a new per-tile function in `services/capture_health.py`,
alongside the existing batch-level `check_capture`. Same thresholds: non-empty, `max > 0`, mean
luminance in `[2, 253]`.

Order matters for cost: viability is a couple of numpy reductions on the crop, liveness is
already computed, and the k-NN matmul runs once for the whole batch — so the batch is embedded
once, then filtered.

#### 5.2.1 The novelty threshold is adaptive

A fixed novelty threshold either floods the archive in a rich region or starves it in a barren
one, and which of those happens depends on the preset the user loaded. Instead, track the
admission rate over a sliding window of the last 100 *candidates* and nudge the threshold toward
a target rate:

```
if rate > target: threshold *= 1.05
if rate < target: threshold *= 0.95
threshold = clip(threshold, 1e-4, 1.0)
```

`target` defaults to 0.15. This is Lehman & Stanley's dynamic-threshold rule. It makes archive
growth predictable regardless of region richness, and it is the second mitigation for the gated
archive: in a barren region the threshold falls until *something* gets in, so the search never
goes blind.

#### 5.2.2 Bootstrap admits unconditionally

While `len(archive) < seed_n`, the **novelty gate is off** — only viability and liveness apply.

This is not a shortcut. With the novelty gate active from a cold start at a 15% target, reaching
`seed_n = 256` would take ~107 generations, which at 2000 steps is five minutes before expansion
begins. E&E's 1000 seed configurations are likewise added unconditionally. Unconditional seeding
also gives the adaptive threshold a real distribution of novelty values to calibrate against
instead of a cold start.

### 5.3 Human pins

Left-clicking a tile in Explore mode **pins** it: admitted regardless of all three gates, marked
`pinned=True`, and never pruned. This reuses `TournamentService.selected` — the same set manual
mode's picker and Auto mode's elite injection already write to — so the on-canvas
click-to-tile mapping and the N×N button grid are shared with no duplication. The set is cleared
after each generation, exactly as it is today.

### 5.4 Capacity and eviction

Default capacity 20,000 entries. Over capacity, evict the non-pinned entry with the **lowest
stored novelty** — which is ASAL's illumination criterion (remove the least-novel, measured by
distance to nearest neighbours) applied as an eviction rule.

This is O(N) and effectively free *because* §6.4 keeps stored novelty continuously refreshed
against the full archive. A one-off sampled nearest-neighbour pass at eviction time would be the
obvious alternative and is the wrong choice: 512 candidates against 20k entries in 512 dimensions
is ~5 GFLOP, roughly half a second, and eviction fires on *every* admission once the cap is
reached.

Pinned entries are never evicted. If the archive is entirely pinned and at capacity, admission
stops and the UI says so.

### 5.5 What an entry stores — and the physics-origin trap

**Store the decoded phenotype, not `z`.** This is a correctness requirement, not a preference.

With physics search enabled, `z` is defined *relative to* `physics_origin` — the preset loaded
at the time (`services/physics_genome.py`: `value = origin + span·tanh(z)`). The same `z`
archived under *Defaulternate* and re-run under *HungryHungryHippos* is a different creature.
An archive of `z` vectors would silently mean different things depending on what happened to be
loaded, which is the same class of bug that made absolute-range physics encoding fail before.

Storing decoded values makes the archive origin-independent, and any entry can seed a run
started from any preset via `encode_physics(values, current_origin)`.

| Field | Type | Bytes |
|---|---|---|
| `brain` | float32 (10, 8) | 320 |
| `physics` | float32 (8,) — absolute values | 32 |
| `embedding` | float16 (512,) | 1024 |
| **total** | | **~1.4 KB** |

Plus one JSONL line of metadata and one ~6 KB thumbnail.

**Spec-signature handling.** Embeddings are spec-agnostic; genomes are not. An entry archived
under `brain:80` still contributes to novelty in a `brain:80,physics:8` run — behaviour space is
universal — it simply has no physics block. When used as a parent in a physics-enabled run its
physics genes start at `z = 0`, which decodes to the current origin exactly, so this is
well-defined rather than an error. The converse (a `brain:80,physics:8` entry in a `brain:80`
run) drops the physics block.

### 5.6 On disk

`Documents/Fluoddity/archive/` (new `utilities.paths.get_archive_dir()`, created by
`initialize_user_data()`):

```
archive/
  index.jsonl          one JSON object per entry, appended per generation
  vectors.npz          embeddings (N,512) f16, brains (N,10,8) f32,
                       physics (N,8) f32, ids (N,) int64, format_version
  goals.json           the user's text goal list
  thumbs/000123.jpg    160x160 JPEG q85, ~6 KB
```

One `index.jsonl` line:

```json
{"id": 123, "ts": 1785880000.1, "run_id": "20260807-143022", "source": "expedition",
 "goal": "coral reef", "novelty": 0.41, "liveness": 0.37, "pinned": false,
 "spec": "brain:80,physics:8", "gen": 42, "tile": 6, "thumb": "000123.jpg"}
```

`source` is one of `bootstrap | expansion | expedition | pin`.

**Write strategy**, mirroring the two patterns already in the codebase: `index.jsonl` appends
and flushes per generation (like `RunLogger`); `vectors.npz` is rewritten atomically via
tmp + `os.replace` (like `run_checkpoint.save_checkpoint`) every 200 admissions and on quit.
Thumbnails are loose files so the gallery can lazy-load them and the OS file browser stays
useful.

**Recovery.** `index.jsonl` is the authority for *which* entries exist; `vectors.npz` supplies
their arrays. On load, entries present in one but not the other are dropped with a counted
warning. This means a crash between the last npz flush and quit costs at most the trailing
entries, never the whole archive.

---

## 6. The IMGEP driver

`services/imgep_driver.py`. Holds the archive, the rejects ring, the goal queue, and — only
during an expedition — one CMA-ES.

### 6.1 `ask(n)`

Three regimes, checked in order:

| Regime | Condition | Behaviour |
|---|---|---|
| Bootstrap | `len(archive) < seed_n` | `z ~ N(x0, sigma0)` — today's random init, unchanged. `x0` is zero unless a genome was loaded via the existing "Load genome as starting point" |
| Expedition | an expedition is active | `optimizer.ask(n)` |
| Expansion | otherwise | per-tile independent parent sample + mutate |

Expansion, per tile independently:

1. Sample a parent from the archive with `p ∝ NOV^α`. Novelty values are recomputed lazily —
   see §6.4.
2. Re-encode its phenotype to `z` under the **current** `physics_origin`
   (`genome_spec.encode` for the brain, `physics_genome.encode_physics` for the physics block).
3. `z += σ_expand · N(0, I)`.

No optimizer, no covariance, no shared state between tiles. Each tile is an independent mutant
of an independently-drawn parent — the population is a sample, not a distribution being fitted.

### 6.2 `tell(z, snapshots)`

1. `E = [scorer.embed(s, n_views=1) for s in snapshots]` → S arrays of (n, 512).
2. Per tile: descriptor `b_i` (§4.1), liveness `L_i` (§4.2).
3. One k-NN matmul for the whole batch against `archive ∪ rejects_ring` → `NOV_i`.
4. Apply the admission gate (§5.2); pins (§5.3) bypass it. Admitted entries get a thumbnail
   written and a JSONL line appended; rejected descriptors go into the rejects ring.
5. Update the adaptive threshold (§5.2.1).
6. If an expedition is active: `optimizer.tell(z, [<b_i, g>])` — higher is better, matching the
   `Optimizer` protocol's existing convention, so nothing negates anywhere. Decrement the
   expedition counter; end the expedition at zero.
7. Return the display score: `<b_i, g>` during an expedition, `NOV_i` otherwise. The UI labels
   the axis accordingly, so a mid-run regime switch is never mistaken for a fitness collapse.

**Deliberate deviation from E&E:** every expedition generation's tiles go through the admission
gate, not just the endpoint. E&E archives only the final optimised solution because its
expeditions are a serial inner loop with a per-sample cost; here the embeddings are already
computed and the path toward a goal is itself territory worth keeping.

### 6.3 Expedition lifecycle

Triggered after `K` consecutive expansion generations (default 25), or immediately by "Chase
this" (§7.3).

1. Draw a goal (§7) → unit vector `g`.
2. `x0` = the archive entry maximising `<b, g>`, re-encoded under the current origin.
3. `make_optimizer(algorithm, spec.dim, popsize, expedition_sigma, seed, x0)` — a **fresh**
   optimizer per expedition. Nothing is carried across goals; a covariance learned climbing
   toward "coral reef" is not informative about "lightning".
4. Run for `expedition_gens` generations, then return to expansion.

The optimizer is discarded at the end. `expedition_sigma` defaults to 0.1 (E&E's value), which
is deliberately much smaller than `sigma0 = 0.5` — an expedition is a local refinement from an
already-relevant seed, not a fresh search.

### 6.4 Novelty bookkeeping

Archive novelty values go stale as the archive grows. Recomputing all N every generation is
20k×20k and out of the question; a large periodic batch refresh would produce a visible frame
hitch. Instead the refresh runs **continuously and in small slices**:

- Each entry stores its novelty at admission, computed against the full archive ∪ rejects ring.
- Every generation, `refresh_per_gen` entries (default 64) are re-scored **against the full
  archive**, round-robin by id. That is 64 × 20k × 512 ≈ 0.65 GFLOP, ~15 ms — 0.5% of a
  generation, with no hitch.
- At 20k entries every entry is refreshed once per ~312 generations (~15 minutes at defaults),
  so by the time the capacity cap is reached after ~6.5 hours every entry has been refreshed
  dozens of times.

Refreshing against the *full* archive rather than a subsample is deliberate: a subsampled
neighbour set inflates k-NN distances, so subsampled and full-archive novelty values are on
different scales and could not be compared against each other for parent sampling or eviction.

Parent sampling and eviction both use these possibly-stale values. This is acceptable because
sampling is stochastic and `NOV^α` only needs to be approximately right; a stale-high entry gets
sampled a few extra times before its next refresh. It is called out here so it is not mistaken
for a bug when a parent with high stored novelty produces unremarkable children.

---

## 7. Goals

Two sources, both fully offline. Which one an expedition draws from is decided by the
`latent goal share` slider (default 0.5).

### 7.1 Latent goals

Sample an archive entry with `p ∝ NOV^α`; let `c = normalise(mean of archive embeddings)`.

```
g = normalise(b + β·(b - c))          # β default 0.5
```

"Keep going in the direction that already looks unlike everything else." Entirely within image
space, so CLIP's modality gap is irrelevant, and no phrase can propose something the substrate
has no vocabulary for.

The centroid `c` is recomputed on the same schedule as the novelty refresh (§6.4).

### 7.2 Text goals — the human is the goal generator

E&E's expeditions are steered by o4-mini, which reads 25 archive thumbnails and writes a 5–15
word description of a hypothetical pattern. Here the user writes that list, and the app cycles
through it.

The list lives in `archive/goals.json` and is edited in the UI: add, remove, reorder, and an
enable checkbox per entry. It is embedded once with `embed_text()` and cached, re-embedded only
when the list changes.

**Ordering is round-robin over the enabled entries by default.** This is both what was asked for
and the technically safe choice. CLIP's modality gap means different phrases have different
baseline affinities to *any* image, so "which goal is the archive least able to match" is not
comparable across phrases without normalisation.

Within a single expedition the goal is fixed, so the gap is a constant offset on every tile's
score and CMA-ES's ranking is completely unaffected. That is why raw `<b, g>` is a perfectly good
expedition fitness even though it is a poor cross-goal comparison.

An optional **least-matched-first** ordering is offered. It must z-score each goal's
archive-max cosine against that goal's own distribution over the archive, not compare raw
cosines:

```
score(goal) = -(max_i <b_i, g> - mean_i <b_i, g>) / std_i(<b_i, g>)
```

If every goal is disabled or the list is empty, all expeditions are latent.

### 7.3 "Chase this"

Right-clicking a tile starts an expedition immediately with that tile's own descriptor as `g`,
pre-empting the cadence. It is the direct analogue of manual mode's "more like that one",
expressed as a goal instead of as a selection, and it costs three lines given §6.3.

The tile's descriptor is already computed for that generation, so no extra CLIP work is needed.

### 7.4 Defaults

| Setting | Default | Range | Note |
|---|---|---|---|
| `steps_per_gen` | 2000 | 50–2000 | Explore-mode default; existing slider |
| `snapshots_per_gen` | 6 | 1–8 | §4.2 |
| `seed_n` | 256 | 64–2048 | E&E uses 1000 |
| `sigma0` | 0.5 | 0.05–1.5 | bootstrap scatter only; the existing Auto-mode slider |
| `σ_expand` | 0.15 | 0.01–1.0 | calibrate, §12.2 |
| `α` (novelty exponent) | 4 | 0–8 | E&E's tuned value; 0 = uniform (the NS ablation) |
| `k` (neighbours) | 10 | 1–50 | E&E's value |
| `K` (expansion gens between expeditions) | 25 | 0–500 | 0 disables expeditions entirely |
| `expedition_gens` | 50 | 5–400 | E&E uses 350; that is ~16 min at 2000 steps |
| `expedition_sigma` | 0.1 | 0.01–1.0 | E&E's value |
| `latent goal share` | 0.5 | 0–1 | |
| `liveness_min` | 0.02 | 0–0.5 | calibrate, §12.2 |
| `target admission rate` | 0.15 | 0.01–1.0 | |
| `archive capacity` | 20000 | 1000–100000 | |
| `refresh_per_gen` | 64 | 0–512 | §6.4; 0 disables refresh |

`α = 0` with `K = 0` reduces the whole thing to random archive-parent mutation — E&E's *Random
GA* baseline. `α = 4` with `K = 0` is E&E's *Novelty Search* ablation. Both are reachable from
the UI without a code path, which makes the controls their own experiment harness.

**Wall-clock at defaults** (N=4, 2000 steps, ~2.8 s/generation): bootstrap 16 generations ≈ 45 s;
one cycle = 25 expansion generations (70 s) + one 50-generation expedition (140 s) ≈ 3.5 min.
The UI shows this estimate live, because `expedition_gens = 350` would be 16 minutes on a single
goal.

---

## 8. UI

### 8.1 The Explore tab

A third tab in the existing Tournament window. The rollout controls both auto modes share —
grid, steps/generation, snapshots, sim steps/frame, physics search, per-tile mutation — are
factored out of `ui/auto_tournament_window.py` into one `_render_rollout_controls(ats)` that both
tabs call. Explore-specific controls are the §7.4 sliders and the goal-list editor.

The tab follows the existing mixin pattern (`ui/archive_window.py` →
`ArchiveWindowMixin`, mixed into `UI` in `ui/core.py`) and stays passive: it renders widgets and
sets one-shot flags, and runs no logic.

### 8.2 The Archive window — gallery

A separate ImGui window, toggled from Extras or from the Explore tab. It needs real estate and
does not belong inside the tournament window.

- **Stats bar** — size, admission rate, current adaptive threshold, current goal in words,
  regime, growth sparkline.
- **Thumbnail grid** — sortable by novelty / recency / liveness; filterable to pinned-only or by
  source. GL textures are lazy-loaded behind a 256-entry LRU cache; there is precedent for
  ImGui-side GL textures in `ui/history_window.py` and `utilities/imgui_gltexture_demo.py`.
- **Per-entry actions** — Export as config (the existing `services/genome_io.export_genome`,
  with the archive metadata added to its provenance block), Seed a run from here, Pin/unpin,
  Delete.

### 8.3 The Archive window — 2-D map

A second tab in the same window: a scatter plot of the archive projected to 2-D.

Projection is PCA via the eigendecomposition of the 512×512 covariance of the embeddings —
~10 ms, refit every 500 admissions rather than per frame. Points are coloured by `source`; the
current goal is drawn as a distinct marker so you can see where an expedition is aiming relative
to the cloud's frontier. Hover shows the thumbnail; click selects the entry in the gallery.

**Each refit must sign-align its components to the previous ones** — flip component `i` if
`<v_i_new, v_i_old> < 0`. Eigenvectors have arbitrary sign, so without this the map mirrors
itself at every refit and reads as a bug.

The projection is a *view*. Nothing in the search depends on it; novelty is always computed in
the full 512-d space.

---

## 9. Performance budget

Measured baseline: 716 sim steps/s with tournament mode on. At N=4, 2000 steps, 6 snapshots:

| Stage | Cost per generation | Share |
|---|---|---|
| CLIP embed, 96 images at `n_views=1` | ~7 ms | 0.25% |
| k-NN novelty, 20k × 512 against 16 × 512 | ~5 ms | 0.18% |
| Novelty refresh, 64 entries against 20k (§6.4) | ~15 ms | 0.53% |
| Admission + ~2.4 JPEG thumbnails | ~2 ms | 0.07% |
| PCA refit, amortised over 500 admissions | <1 ms | — |
| `index.jsonl` append + flush | <1 ms | — |
| **2000 simulation steps** | **~2800 ms** | **98.9%** |

Total non-simulation cost is ~30 ms per generation, and the two dominant terms both scale
linearly with archive size — at the 100k capacity ceiling they would be ~25 ms and ~75 ms, still
under 4% of a generation.

Explore mode is *cheaper* per generation than prompt mode, because `n_views=1` is a third of the
CLIP work. The original design's constraint — the simulation must dominate — holds with far more
margin than Auto mode has today.

**Archive growth at defaults:** 15% of 16 tiles ≈ 2.4 admissions per 2.8 s ≈ 51/minute. The 20k
cap is reached after roughly 6.5 hours of continuous running.

**Memory:** 20k × 1.4 KB ≈ 28 MB of vectors resident, plus a 2 MB rejects ring, plus at most 256
live GL thumbnail textures. Thumbnails on disk: 20k × 6 KB ≈ 120 MB.

---

## 10. Failure handling

| Condition | Behaviour |
|---|---|
| `onnxruntime` / `tokenizers` / `cmaes` absent | Explore tab shows the missing package, as the Auto tab does. Manual mode unaffected. |
| Archive directory unwritable | In-memory archive, one warning, run continues (matches `RunLogger`'s policy) |
| `vectors.npz` corrupt or `format_version` mismatched | Refuse to load, start empty, **move the bad file to `vectors.npz.bad-<ts>`** — never overwrite it |
| `index.jsonl` and `vectors.npz` disagree | Keep the intersection, warn with the dropped count |
| Entry's spec signature ≠ current | Embedding still counts for novelty; missing genome blocks default to `z = 0` (§5.5) |
| Grid changed mid-run | Existing `abort_generation`; any active expedition ends and its optimizer is discarded (`cmaes.CMA` fixes popsize at construction) |
| Physics search toggled mid-run | `driver.set_spec()` ends any expedition and discards its optimizer. The archive is unaffected — phenotypes, not `z`, are stored |
| Canvas/world resized, or shader hot-reload | Existing `abort_generation` path; the partial generation contributes nothing to the archive |
| Goal list empty or all disabled | Every expedition is latent |
| Admission rate 0 for 20 consecutive generations | UI warning naming the likely cause: a dead preset, a broken capture, or `liveness_min` set too high |
| Non-finite descriptor | Rejected, counted, surfaced in the stats bar |
| Thumbnail write fails | Entry still admitted; the gallery shows a placeholder |
| Archive at capacity and entirely pinned | Admission stops; the UI says so rather than silently discarding |
| Archive empty when an expedition is requested | Fall back to expansion and log it; an expedition needs a seed |

---

## 11. Testing

Following the existing convention: a real `pytest` suite, GPU-marked tests skipped by default.

**Unit, no GPU:**

- `test_search_driver.py` — both drivers satisfy the protocol; `PromptDriver` reproduces
  pre-refactor scores against a recorded fixture; `set_spec` invalidates optimizer state without
  raising.
- `test_descriptor.py` — the centroid is unit-norm; liveness is 0 for identical snapshots and
  ~1 for orthogonal ones; the ASAL Eq. 3 formula is pinned against a hand-computed 3-snapshot
  case.
- `test_novelty.py` — k-NN novelty matches a brute-force reference; the rejects ring
  participates in the neighbour set; `NOV^α` sampling matches the expected distribution over many
  draws (chi-square).
- `test_archive.py` — each of the three gates fires independently; bootstrap bypasses the
  novelty gate and only the novelty gate; the adaptive threshold converges to the target rate on
  synthetic streams of both rich and barren candidates; pins bypass every gate and survive
  pruning; eviction drops the most redundant entry; capacity is respected; an all-pinned archive
  at capacity stops admitting rather than discarding.
- `test_archive_io.py` — full roundtrip; an interrupted `vectors.npz` write leaves the prior
  file intact; a corrupt file is moved aside and not overwritten; an index/vectors mismatch keeps
  the intersection and reports the count; a `format_version` mismatch raises a specific error.
- `test_imgep_driver.py` — bootstrap → expansion → expedition transitions fire at the right
  counts; goals round-robin over enabled entries only; an empty list gives all-latent; a latent
  goal is unit-norm and strictly further from the centroid than its source; an expedition seeds
  at the archive argmax and builds a fresh optimizer per goal; "Chase this" pre-empts the
  cadence; `K = 0` never expeditions; `α = 0` samples parents uniformly.
- `test_physics_origin_roundtrip.py` — a phenotype archived under origin A and re-encoded under
  origin B decodes to identical absolute physics values; a `brain:80` entry used in a
  `brain:80,physics:8` run gets `z_physics = 0` and therefore reproduces the current origin
  exactly.
- `test_clip_scorer.py` (**additive only**) — `embed()` returns L2-normalised rows;
  `n_views=1` skips augmentation; a softmax recomputed from `embed()` equals `score()` to
  float32 tolerance; `embed_text()` does not disturb the cached `set_prompt` embedding.
- `test_archive_projection.py` — PCA components are orthonormal; a refit on appended data
  sign-aligns to the previous components; the projection is deterministic for fixed input.
- `test_capture_health.py` (**additive**) — `is_viable_tile` agrees with `check_capture` on
  single-tile batches.

**GPU-marked (`@pytest.mark.gpu`):**

- `test_archive_thumbnails.py` — a thumbnail written from a known-colour crop reads back the
  same colours at 160×160.

**Manual:** extend `docs/testing_checklist.md` with the Explore-mode path, including quit →
relaunch → archive reloads with the same size and the same thumbnails.

---

## 12. Implementation risk

### 12.1 The one unvalidated assumption

Auto mode already proved CLIP has usable *directed* signal on these tiles (the Task 3 gate in
the previous design). What is unproven is that CLIP embeddings **spread** these patterns enough
for k-NN novelty to have a gradient. A representation in which every Fluoddity tile is
"abstract texture" would give a directed prompt enough to climb while still packing every
descriptor into a tiny cone, and novelty search would then be sampling noise.

**Task 1 of the implementation plan is a standalone check, before any loop code exists:**

1. Take ~200 tiles from existing runs' `frames/` directories plus fresh captures across several
   presets.
2. Embed them and compute the distribution of pairwise cosine distances.
3. **Bar:** the mean pairwise distance must exceed 0.15 and the distribution must be visibly
   multi-modal or at least broad (std > 0.05). Additionally, k=10 novelty must correlate with
   human judgement on a hand-labelled set of 20 obvious outliers.

If this fails, the remedy is the same as before — adjust the eval image (colormap, exposure,
zoom) — or fall back to concatenating CLIP with a cheap hand-built descriptor. That fallback is
**not** designed here; it would be its own spec.

#### 12.1.1 Result, measured 2026-08-07 — PASSED

Measured by `tools/capture_presets.py` (131 presets, 2000 steps each, headless through the
production `FrameAssembler → TileCapture → CaptureBlit` path) then `tools/clip_spread_check.py`:

| Sample | n | mean | std | p50 | verdict |
|---|---|---|---|---|---|
| All presets | 131 | 0.1924 | 0.0841 | 0.1786 | PASS |
| **Viable presets only** | **97** | **0.1586** | **0.0627** | **0.1500** | **PASS** |

**The viable-only row is the one that matters** — it is what the admission gate will actually
admit. Removing the 34 near-black presets *lowers* the spread rather than raising it, because a
black image is highly distinctive in CLIP space and was inflating the mean. The real margin is
therefore ~6% over the bar, not ~28%.

Consequence for §5.2.1: novelty operates in a compressed range, so the adaptive threshold is
load-bearing rather than a convenience. A fixed threshold picked from the all-presets number
would have been ~20% too high.

**Rejected evidence.** `runs/*/frames/` was measured first and scored 0.1382 (FAIL). That sample
was invalid twice over: 481 of its 635 images came from one converged run whose best tile had
stopped changing (mean pairwise 0.043 among themselves, 57% of all pairs), and every run in it
predated the NaN-isolation and cohort-colouring fixes. `spread_report` now returns
`duplicate_dominated` to make the first failure mode self-announcing.

Also recorded: **34 of 131 presets (26%) render near-black** at 2000 steps under default world
settings. That is independent justification for the viability gate in §5.2.

### 12.2 Calibration, not features

Three constants are stated as defaults but must be *measured* during implementation, exactly as
`sigma0` was:

- `σ_expand = 0.15` — the decoded spread of a mutated population should be visibly related to,
  but narrower than, a fresh random population.
- `liveness_min = 0.02` — measure `L` on a deliberately frozen tile and on a known-lively one;
  the threshold goes between them, nearer the frozen end.
- The `NOV` scale, which sets the sensible range of the threshold slider.

Each is a one-off measurement recorded in the implementation log, not a runtime feature.

### 12.3 The refactor is the risk to the existing feature

Extracting `PromptDriver` touches working, shipped code. It is sequenced **first**, alone, with
the §3.3 test files as the gate, and lands as its own commit before any archive code exists. If
the extraction cannot be made to pass those tests unmodified, it is reverted and the design falls
back to a mode flag inside `AutoTournamentService`.

---

## 13. Files touched

**New:**

```
services/search_driver.py            SearchDriver protocol
services/prompt_driver.py            extracted from AutoTournamentService
services/imgep_driver.py             bootstrap / expansion / expedition
services/archive.py                  admission, capacity, pins, novelty bookkeeping
services/archive_io.py               index.jsonl + vectors.npz + thumbs, atomic writes
services/novelty.py                  kNN novelty, rejects ring, NOV^alpha sampling
services/descriptor.py               trajectory centroid + ASAL liveness
services/goal_source.py              latent goals, text goal list, round-robin
services/archive_projection.py       PCA with sign alignment
state/archive_state.py               ArchiveState + one-shot flags
ui/archive_window.py                 ArchiveWindowMixin: Explore tab, gallery, map
tests/test_search_driver.py
tests/test_descriptor.py
tests/test_novelty.py
tests/test_archive.py
tests/test_archive_io.py
tests/test_imgep_driver.py
tests/test_physics_origin_roundtrip.py
tests/test_archive_projection.py
tests/test_archive_thumbnails.py     (gpu-marked)
```

**Modified:**

```
services/clip_scorer.py              + embed(), embed_text(); score() rebuilt on embed()
services/auto_tournament_service.py  driver extraction; no CLIP, no optimizer
services/capture_health.py           + is_viable_tile()
tests/test_capture_health.py         + is_viable_tile coverage (additive)
services/genome_io.py                archive provenance in the export metadata block
command_handler.py                   Explore one-shot flags
main.py                              one mode branch; construct ImgepDriver lazily
ui/core.py                           mix in ArchiveWindowMixin
ui/tournament_window.py              third tab
ui/auto_tournament_window.py         factor out _render_rollout_controls()
state/ui_state.py                    archive field
state/__init__.py                    export
services/__init__.py                 export (lazy — must not import onnxruntime at startup)
utilities/paths.py                   get_archive_dir(); create in initialize_user_data()
docs/testing_checklist.md            Explore-mode manual checks
.gitignore                           nothing new — the archive lives in Documents
```

**Unchanged, deliberately:** `sim.py`, every file in `shaders/`, and the behaviour of Manual and
Auto (CLIP) modes.

---

## 14. Implementation stages

This is a large spec, so it is sequenced into five stages that each end somewhere runnable. The
implementation plan expands these; they are recorded here so the decomposition is part of the
design rather than an afterthought.

| Stage | Content | Done when |
|---|---|---|
| 0 | The §12.1 embedding-spread check. No product code. | The bar in §12.1 is met, or the design is revisited |
| 1 | `SearchDriver` + `PromptDriver` extraction; `CLIPScorer.embed`/`embed_text`; `is_viable_tile` | The §3.3 test files pass **unmodified**; Auto mode behaves identically by hand |
| 2 | `descriptor.py`, `novelty.py`, `archive.py`, `archive_io.py` — headless, no UI | Archive round-trips to disk; all gate/eviction/IO tests pass |
| 3 | `imgep_driver.py`, `goal_source.py`, Explore tab with sliders and goal list | A run bootstraps, expands, expeditions, and grows the archive on disk unattended |
| 4 | `archive_window.py` gallery, `archive_projection.py` map, export/seed/pin/delete | The archive is browsable and harvestable in-app |

Stages 0 and 1 are the risk. Stage 1 in particular touches working, shipped code and lands as
its own commit with nothing else in it (§12.3).

## 15. Deferred

- **VLM-authored goals.** The `goal_source.py` interface is written so a third source slots in
  beside latent and text without touching the driver. Adding it means an API key, a network
  dependency, a worker thread and a fallback path — a separate decision, not a refinement of
  this one.
- **A learned or fine-tuned descriptor.** DINOv2 alongside CLIP would give the paper's
  second-embedding-space diversity check, which is the honest way to tell genuine behavioural
  diversity from CLIP-specific overfitting. It is a second ~300 MB model download and is not
  needed until the first is shown to work.
- **Approximate nearest neighbours.** Brute force is ~5 ms at 20k and ~25 ms at 100k. An index
  earns its dependency only past that.
- **Cross-run archive comparison / diversity metrics.** E&E's evaluation methodology — mean
  pairwise distance in CLIP *and* DINO, genealogical attribution of expedition descendants —
  belongs in an offline analysis script over `index.jsonl`, not in the app.
- **Threaded CLIP inference.** 0.25% of a generation. Not worth the cross-thread ONNX session
  concerns.
- **Archive-conditioned initial states.** Currently every generation reshuffles the initial
  scatter. Archiving the seed alongside the genome would make entries exactly reproducible; it is
  a genuine improvement and a genuine scope increase.
