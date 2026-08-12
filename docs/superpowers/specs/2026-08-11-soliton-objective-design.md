# Soliton objective: rewarding spatially localized patterns

**Date:** 2026-08-11
**Status:** approved design
**Scope:** two new statistics in `services/capture_health.py`, a stored archive
column for each, a calibration tool, and a fourth expedition goal kind. The
archive's admission gate is **out of scope** — nothing here changes what is
admitted, only what an expedition climbs and how the browser sorts.

## What is being asked for

A "soliton", in the Lenia literature a **spatially localized pattern (SLP)** and
in Game of Life an "animal" or "spaceship": a bounded structure that keeps its
identity against an empty background, whether or not it travels.

The prior art is the direct ancestor of Explore mode. Reinke, Etcheverry &
Oudeyer (ICLR 2020) ran an IMGEP over Lenia and filtered for SLPs with a
hand-coded classifier — activation mass finite, non-dead, non-diverging, and not
touching the border after recentring on the torus. Leniabreeder (Faldor & Cully,
ALIFE 2024) covers the same ground with AURORA's learned descriptors. Neither
uses a text prompt, and neither should: this is a cheap image statistic, not
something CLIP has vocabulary for.

**Decided during design:** motion is optional. A stationary structure that keeps
its shape counts equally with a glider. The objective is *localized and
persistent*, not *localized and travelling*.

## Two constraints the substrate imposes

### Bloom is in every picture the optimizer sees

`CaptureView.draw_grid` blooms each tile individually, and the chain reaches
30–60px on a 224px tile (see CLAUDE.md). A compact core therefore arrives as a
core plus a halo a quarter of the tile wide, and `bloom_enabled` is a user
preference the search does not control.

Two consequences carried through the design: the measure must not be a raw
second moment (bloom's halo inflates it for exactly the compact objects wanted),
and the calibration tool must report the statistic bloom-on and bloom-off so the
sensitivity is a measured number rather than an assumption.

### "Localized and persistent" has a degenerate maximum

One small dim blob that does not move maximises both halves. `is_viable_tile`
only rejects black and blown-out captures, so that tile survives the gate. This
is the same shape of trap that `capture_health.structure` exists to close, and
like that one it is answered inside the measure rather than downstream.

## The measure

Both statistics run on **one shared luminance stack**, downsampled to 56×56:
mean over RGB, then area-average. A soliton is a large-scale object and bloom
has already destroyed the fine detail, so the reduction costs nothing real and
makes both terms cheap. `structure()` is unchanged and keeps running at full
resolution on the last snapshot.

**56 and not 64**, because `224 = 4 × 56` exactly and `224 = 3.5 × 64` is not.
At a 64 target the box filter would cover only 192 of the 224 pixels and
discard the outer 16px of every edge — which is exactly where a soliton sitting
near a tile boundary lives.

### `locality(crops) -> (n,) float32 in [0, 1]`

Let `g` be the 56×56 luminance of one tile, `N = 3136` its pixel count, and
`M = Σ g`. Sort the pixels descending as `s₁ ≥ s₂ ≥ … ≥ s_N`, take the running
sum `C_k = Σ_{i≤k} sᵢ`, and let `a` be the **fraction of pixels needed to
accumulate `MASS_FRACTION` of the light**:

```
k*  = min { k : C_k >= MASS_FRACTION * M }
a   = k* / N

concentration = 1 - clip(a / MASS_FRACTION, 0, 1)
support       = min(1, a / MIN_SUPPORT)

locality      = concentration * support
```

`MASS_FRACTION = 0.9` and `MIN_SUPPORT = 0.005` are the values development
starts from; both are finalised by the calibration tool below, which is stage
1's whole point.

- A uniform field needs 90% of its pixels, so `a = MASS_FRACTION` and
  `concentration = 0`.
- A compact blob covering a twentieth of the tile gives `a = 0.05`,
  `concentration = 0.94`, `support = 1`.
- A single-pixel dot gives `a ≈ 0.0002`, `concentration ≈ 1`, and `support`
  crushes it to ≈ 0.05. This is what closes the degenerate.
- `M = 0` returns 0 defensively; `is_viable_tile` already rejects that upstream.

One quantity drives both terms, which is why this form was chosen over the
obvious alternatives. A **fixed top-quantile mass fraction** cannot work: the
bright set is a fixed fraction of pixels by construction, so there is no support
size for the ramp to read, and the 95th percentile of a mostly-black tile is
zero, which selects the whole image and scores the single-dot degenerate at 1.0.
The **participation ratio** `(Σx)²/(N·Σx²)` fails on bloom, whose halo inflates
the second moment for precisely the compact objects the measure must reward.

Translation invariance is by construction — this is a histogram statistic, with
no centring step. That is why the torus never enters. Reinke et al. needed
circular recentring only because their criterion was a bounding box.

### `persistence(snapshots) -> (n,) float32 in [0, 1]`

Normalise each snapshot's 56×56 luminance to zero mean and unit L2 norm, then
for each consecutive pair take the **maximum over all cyclic shifts** of their
cross correlation, including `(0, 0)`, and average over pairs:

```
ĝ_s      = (g_s - mean(g_s)) / || g_s - mean(g_s) ||

C_s(u,v) = SUM_{i,j} ĝ_s[i,j] * ĝ_{s+1}[(i+u) mod 56, (j+v) mod 56]
         = IFFT2( conj(FFT2(ĝ_s)) * FFT2(ĝ_{s+1}) )

p_s         = max over (u,v) of C_s(u,v)
persistence = clip( mean over s of p_s, 0, 1 )
```

Unit norm on both sides makes `C_s(u,v)` a cosine, bounded in `[-1, 1]` by
Cauchy-Schwarz, so `p_s` needs no further scaling. The FFT form is the whole
reason the max over all 3136 shifts is affordable: evaluating it directly would
be `O(N²)` per pair.

Cyclic correlation is the exact operation here, not an approximation: a
tournament tile genuinely is a torus under wrap (CLAUDE.md), so a structure that
leaves one edge arrives at the other and must still correlate with itself.

- A frozen pattern scores 1.0. **Intended** — motion is optional by decision.
- A glider scores ≈ 1.0 at its own displacement, which is the case a
  correlation restricted to `(0, 0)` would score as dead. That false negative is
  the direct analogue of the lag-1 failure documented for `structure()`.
- Turbulence and noise score ≈ 0.
- A drifting full-field texture scores ≈ 1.0. It is `locality` that rejects it,
  which is why the two terms multiply rather than either standing alone.
- `S < 2` returns zeros, and callers must not gate on it — the same contract
  `descriptor.liveness` already documents.

A sub-bin drift at 56×56 simply peaks at `(0, 0)` with near-unit correlation.
That is correct rather than a rounding loss, because a stationary structure is a
valid target.

### `structure(crops)` — the existing factor, restated

Unchanged by this spec. It is recorded here because the product below
multiplies it, and because it is the factor `locality` is most easily confused
with. Full 224×224 resolution, last snapshot only:

```
g      <- g - mean(g)                          # per image
energy  = mean(g * g)

r_x(L) = | mean_{i,j} ( g[i,j] * g[i,j+L] ) / energy |
r_y(L) = | mean_{i,j} ( g[i,j] * g[i+L,j] ) / energy |

structure = clip( max over L in LAGS and over both axes of {r_x, r_y}, 0, 1 )
```

`LAGS = (1, 2, 3, 4, 6, 8, 12, 16)`, and a flat tile (`energy ~ 0`) returns 1.0.
The overlap is a valid, non-wrapping one — unlike `persistence`, which shifts
cyclically.

### How the three factors differ

**`locality` is a histogram statistic, `structure` is a correlation statistic,
and `persistence` is the only temporal one.** Scramble a tile's pixels at random
and `locality` does not move at all, while `structure` collapses to zero. Add a
constant pedestal to every pixel and `structure` does not move, while `locality`
falls.

| | reads | invariant to | destroyed by |
|---|---|---|---|
| `locality` | how unevenly the light is distributed, position ignored | any spatial permutation; scaling brightness | a bright pedestal; mass spreading out |
| `structure` | whether the image resembles itself at an offset | affine brightness `a*g + b`; translation, up to edge effects | pixel scrambling |
| `persistence` | whether snapshot `s+1` is snapshot `s` moved | per-frame affine brightness; rigid translation on the torus | the shape changing between snapshots |

`persistence` is **not** `1 - liveness`, and the difference is not only that one
reads pixels and the other CLIP embeddings. `liveness` compares embeddings at
zero spatial offset, and CLIP is strongly position-dependent — enough that
`embed_mean` leaves a positional nuisance above the archive's separation bar
(CLAUDE.md). So `liveness` partly registers a pattern having *moved* as a
pattern having *changed*, while `persistence` quotients translation out by
construction. A glider is therefore the one thing that scores high on both.

Neither of the two spatial terms implies the other, which is why the product
needs both:

| | high `structure` | low `structure` |
|---|---|---|
| **high `locality`** | a compact blob — **the target** | a sparse spray of isolated dots: mass concentrated, no spatial correlation |
| **low `locality`** | full-field stripes, a smooth gradient | white noise |

The top-right cell is what `locality` alone would reward and `structure`
rejects. The bottom-left is what `structure` alone would reward and `locality`
rejects. Only the top-left is a spatially localized pattern, and only the
product selects it.

### The product

```
soliton = locality * persistence * structure
```

A **derived property, never a stored field.** Three stored factors and a
computed product have one home each; a fourth stored number would drift from the
three it claims to summarise.

`structure` is therefore stored per entry alongside the other two, even though
it is not new. It is already computed inside `tell` as `coherence`, so this
costs nothing — and without it the browser's stored column and the expedition's
fitness would be different functions, ranking the same tile differently.

## Where it is computed and stored

`ImgepDriver.precompute` runs off the frame loop and today returns a bare list
of per-snapshot embeddings. It gains the two scores and returns a named tuple
`Precomputed(embeddings, locality, persistence)`. `tell(..., pre=None)` still
recomputes everything when handed nothing, so direct `tell()` calls in the tests
and in `tools/` are unchanged.

This keeps the split the CLAUDE.md caveat defines — everything on the worker
touches only its arguments and the scorer, and everything that mutates the
archive stays on the main thread.

**Storage is `index.jsonl`, not `vectors.npz`.** All three numbers are static
properties of one capture, like `liveness` and unlike `novelty`: they never
change after admission, so an append-only file is the correct home and no
rescore pass is needed. `Candidate` and `ArchiveEntry` each gain `locality`,
`persistence` and `structure`.

**Legacy archives.** `persistence` cannot be backfilled — it needs several
snapshots, and only one thumbnail per entry is kept. `locality` can be, from
that thumbnail, and so can `structure`. Missing values therefore load as
`None`, sort last in the browser under their own label, and are excluded from a
soliton goal's seed sampling rather than being read as zero. Storing the
factors separately rather than only their product is what makes that partial
backfill possible.

## Stage 1 deliverable: `tools/calibrate_soliton.py`

Modelled on `tools/capture_presets.py` — the real `Camera` through a hidden GLFW
window, never a re-derived view path, for the reason that tool's docstring
records — but grabbing S snapshots per preset instead of one.

```
python -m tools.calibrate_soliton --presets
python -m tools.calibrate_soliton --presets --no-bloom
python -m tools.calibrate_soliton --archive <name>
python -m tools.calibrate_soliton --presets --only Zipper,Karst
```

It reports, over the 131-preset library:

1. the distribution of `locality`, `persistence` and their product;
2. the same with bloom disabled, so the bloom sensitivity is measured;
3. a contact sheet of the top and bottom decile, so the measure can be
   falsified by eye rather than trusted.

`--archive` runs `locality` alone over stored thumbnails, which is both the
backfill path and the fastest check of whether the existing archive already
holds these.

Its output sets `MASS_FRACTION` and `MIN_SUPPORT` to their final values, and
that measurement becomes their single home — the CLAUDE.md caveat, with the
constants in code carrying a one-line pointer and no numbers of their own.

It also answers the question the whole design rests on: whether this substrate
produces localized patterns at all, and at what rate.

## Stage 2: the soliton goal kind

A fourth expedition kind alongside latent, text and novelty, mirroring
`novelty_goal` in every structural respect.

- `services/goal_source.py` gains `soliton_goal(archive, rng, alpha)`. No
  embedding; it sets `seed_index`, sampled with `p ∝ soliton^alpha` over the
  stored column, exactly as `novelty_goal` samples on novelty. On an archive
  whose entries carry no column it falls back to novelty sampling rather than
  declining, so an existing archive can start one immediately.
- `ImgepDriver._draw_goal` gains `soliton_share`, **clamped and not
  normalised**, in the same fall-through chain — for the reason already
  recorded: rescaling one share because another moved makes neither mean what it
  says, and an expedition that fails to start wastes a whole cadence interval.
- `ImgepDriver._expedition_fitness` returns `locality * persistence * coherence`
  for `kind == "soliton"`, the same shape as the novelty branch's `nov * c`.
  Both factors are already computed for the generation, so this costs nothing.
- `soliton_share` joins `ArchiveState` and the `PERSISTED_FIELDS` allowlist.
- One slider in the Explore tab beside the other two shares, under
  `layout.push_settings_width()`. Label "Soliton Share".

The existing summit, keeper and record-book machinery needs no change: a
soliton expedition is an expedition, and its best tile reaches the archive by
exactly the paths every other kind uses.

## Browser and map

`sort_by` gains a soliton option; the map's colour ramp gains one alongside
novelty and liveness. Both read the derived product. Entries with no stored
column sort last and draw in the ramp's "unknown" colour rather than at zero.

## Tests

- `tests/test_capture_health.py` — synthetic images with known orderings: a
  Gaussian blob, a full-field sinusoid, white noise, a black frame with one
  bright dot. Assert the ordering, and assert specifically that the single dot
  scores **below** the blob, which is the degenerate the ramp exists to close.
- One test per off-diagonal cell of the 2×2 above, since each is a factor
  covering for the other: a sparse spray of isolated dots must score **high**
  `locality` and **near-zero** `structure`, and a full-field sinusoid the
  reverse. Both must have a low product. A test that only checks the product
  would pass with either factor silently broken.
- A synthetic glider: one blob translated a fixed offset per snapshot,
  **wrapping across the seam**, asserting `persistence ≈ 1`. This is the torus
  case, and the one a non-cyclic correlation fails.
- A frozen stack asserting `persistence ≈ 1`, and an independent-noise stack
  asserting `persistence ≈ 0`.
- `S < 2` returns zeros without raising.
- `tests/test_goal_source.py` — a soliton goal seeds from the column, and falls
  back cleanly on a column-less archive.
- Archive round-trip through `index.jsonl` with the keys present and absent,
  asserting absent loads as `None` and not as `0.0`.
- `tests/test_imgep_driver.py` — `tell()` with `pre=None` still works, and
  `precompute()`'s tuple round-trips into `tell()`.

## Staging

**Stage 1** — the two statistics, the stored column, the calibration tool, and
the tests. Self-contained and falsifiable before any optimizer is pointed at it.
Stage 1 ends with the constants measured and written into CLAUDE.md.

**Stage 2** — the goal kind, the share slider, and the browser and map columns,
with the constants stage 1 produced.

If stage 1 shows the preset library holds nothing localized, stage 2 does not
follow as written: the measure would be describing a region the substrate does
not reach, and the next question would be which physics parameters move it, not
how to climb it.

## Out of scope

- **Any change to archive admission.** Solitonness does not gate, does not
  prune, and does not compete for a niche. Making it a quality axis in a
  MAP-Elites sense contradicts the current design — everything viable is
  admitted and capacity prunes on novelty — and would need its own spec.
- **`vectors.npz` and `Archive._phys`.** Neither widens. No load-time migration
  is needed, because both new fields live in the append-only index.
- **Any CLIP term.** The measure never touches the scorer.
