# Expedition objective redesign

**Date:** 2026-08-08
**Status:** approved design
**Scope:** `ImgepDriver` expedition fitness and latent goal construction, plus a
cancel control. The archive admission gate is deliberately **out of scope** and
gets its own spec.

## The problem, measured

Expeditions do not climb. Latent expeditions produce nothing new; text
expeditions drift toward noise. Both were measured offline against the user's
real `default` archive (4784 descriptors) on 2026-08-08.

### The text encoder is not the bug

`text_model_fp16.onnx` output 0 is `text_embeds`, shape `(batch, 512)`,
L2-normalised, with sane neighbour structure. `embed_text` is correct.

### Finding 1 — the expedition starts on its own optimum

`start_expedition_with` seeds CMA-ES at `Archive.nearest(goal)`, which is
`argmax(embeddings @ goal)`. The seed is therefore the archive's best entry
under the goal **by construction, for any goal**. That is fine in principle —
starting from the best known point is normal — but only if the goal leaves room
beyond it.

It does not. `latent_goal` builds `g = normalise(b + 0.5(b − c))`, and the
archive's mean alignment to its own centroid is `<b,c> = 0.947`, so
analytically `<b, ĝ> = 0.965`. Over 200 trials the goal's nearest archive entry
**was the seed it was built from 199 times**, at mean rank 0.01, with
best-minus-seed fitness of 0.00001. The search spends 50 generations where every
move can only score worse.

β is not the cause and cannot fix it: raising β trades reachability away at a
terrible rate (β=5 → reach 0.640 for a seed rank of only 2.0; β=20 → reach
0.328, as unreachable as a text prompt).

### Finding 2 — raw cosine saturates in a dense cone

The archive covers its reachable region densely: mean pairwise `<b,b'> = 0.897`,
and 50% of its variance lies in **3** of 512 PCA components (80% in 18, 90% in
42). Every reachable goal is therefore already matched to 0.96–0.99 by something
in the archive, and the entire usable range of `<b, g>` is a couple of percent.

For text goals the same saturation appears as the modality gap. Within a prompt
the spread over 4784 tiles is std **0.0111** around 0.21; across prompts for a
fixed tile it is **0.0127**. The prompt you pick moves the number as much as the
creature does.

### Finding 3 — the descriptor is renormalised, so noise gets a bonus

`descriptor()` divides the trajectory centroid by `||m||`, which shrinks as
snapshots decorrelate. Using it as fitness attaches a free multiplier that
rewards *changing* rather than *matching*: a 1.35% swing across the observed
liveness range, or **0.22 SD** of the latent fitness spread, with
`rho(raw, liveness)` reaching 0.23 on text goals. Real, but secondary.

### The Auto (CLIP) tab does none of this

| | Auto (CLIP) tab | IMGEP expedition |
|---|---|---|
| fitness | `softmax(100·cos)` over [prompt + 9 distractors] | raw cosine `b · g` |
| snapshots | score each, then average | average embeddings, renormalise, then score |
| distractors | 9 | none |

`ImgepDriver` never calls `set_prompt()`, so the distractor machinery — which
`clip_scorer.py` documents as dropping a black canvas from 0.37 to 0.03 — is
bypassed entirely.

### What fixes it

Contrastive scoring un-saturates the landscape by turning a 0.03 cosine gap into
a 3.0 logit gap. Whitened extrapolation puts the goal somewhere the seed is not
already the best answer. **Neither alone is sufficient** (60 trials each):

| goal construction | reference set | fit std | top-1 ≠ seed | #beating seed |
|---|---|---|---|---|
| current `g=1.5b−0.5c` | none (raw cosine) | 0.056 | 0/60 | 0.0 |
| current | centroid | 0.112 | 36/60 | 8.5 |
| whitened d=8 +3sd | none (raw cosine) | 0.070 | 0/60 | 0.0 |
| **whitened d=8 +3sd** | **centroid** | **0.139** | **54/60** | **23.8** |
| whitened d=8 +3sd | centroid + 8 random | 0.070 | 10/60 | 17.2 |
| whitened d=8 +3sd | 16 nearest to goal | 0.004 | 4/60 | 76.8 |

"#beating seed" counts archived entries scoring above the CMA-ES starting point
— gradient built from creatures the substrate has already demonstrated.

Two secondary results are load-bearing for the design: **the centroid alone
beats centroid+random** (0.139 vs 0.070 — extra references dilute), and **"16
nearest to goal" collapses the landscape** (std 0.004, too harsh to rank on).
The reference set is exactly one vector.

For text goals the amplification is 1.6×–17.6× and the top-ranked tile moves in
every case.

**Limit of this evidence.** These are offline proxies. They show the landscape
has a gradient made of achievable points; they do not show that CMA-ES finds it
in 50 generations at dim 80. That requires running the app.

## Design

### 1. Descriptor and fitness become separate computations

The conflation is the root of finding 3. `tell()` already computes `per_snap`
embeddings; both consumers read from that same stack:

- **descriptor** — unchanged: `descriptor(snaps)`, the renormalised trajectory
  centroid. Correct for novelty, which needs unit vectors for cosine kNN.
- **fitness** — computed **per snapshot, then averaged**, mirroring
  `PromptDriver.tell`. `||m||` never enters the objective.

No extra CLIP passes. Snapshot embeddings stay at `n_views=1`: three views
would create three competing archive descriptors per tile, which is why
augmentation was turned off here. Augmentation as an anti-adversarial defence
for the *directed* phase was considered and deferred — if a search starts
finding adversarial texture, `embed(..., n_views=3)` for the fitness path alone
is the change.

### 2. `services/expedition_fitness.py` — one contrastive fitness for all goals

New pure module. No GL, no CLIP session, no app state — just numpy.

```python
def contrastive(snaps, goal, references, logit_scale=100.0) -> np.ndarray:
    """(S, n, dim) x (dim,) x (m, dim) -> (n,) in [0, 1].

    mean over snapshots of softmax(logit_scale * [<e, goal>, <e, r_1>, ...])[0]
    """
```

Requires `m >= 1`; an empty reference set makes the softmax degenerate (always
1.0) and is a caller error rather than a silent flat landscape.

References are resolved by the driver from `goal.kind`, so `goal_source.py`
stays free of the fitness concern:

| goal kind | references |
|---|---|
| `text` | the 9 `DEFAULT_DISTRACTORS`, embedded once and cached on the driver |
| `latent` | `archive.centroid()` |
| `chase` | `archive.centroid()` |

Caching the distractors on the driver rather than calling
`scorer.set_prompt()` is deliberate: `set_prompt` writes `scorer._text_emb`,
which the Auto tab owns, and the two modes must not clobber each other.

Known limitation, accepted: a goal that sits *near* the centroid gives a flat
landscape, because `softmax([<e,g>, <e,c>])` approaches 0.5 everywhere as
`g → c`. Latent goals cannot hit this — the +3 sd push is away from the centroid
by construction — but a `chase` on a thoroughly typical tile can. That is
honest rather than broken: chasing an average creature *is* a weak goal, and it
shows up as a visibly flat fitness rather than as silent drift.

### 3. `latent_goal` extrapolates in the archive's principal subspace

Replaces centroid extrapolation. Given a `Projection` fit at 8 components:

1. whiten the seed's coordinates: `y = P(b) / sd`
2. push out along the seed's own whitened direction: `y' = y + 3.0 · ŷ`
3. unwhiten, un-project, renormalise: `g = normalise(mean + (y'·sd) @ P)`

Parent selection is unchanged — the seed is still drawn with `p ∝ NOV^alpha`.

`Projection` gains a `variances` attribute (the eigenvalues of the retained
components) so the whitening scale is available; it currently discards them.

The search owns its **own** `Projection(8)`, separate from `main.py`'s
2-component map projection. It refits at every `start_expedition()` — a fixed
512×512 eigendecomposition regardless of archive size, a few ms against an
expedition interval of ~70 s.

If the fit fails (archive smaller than 9 entries) `latent_goal` returns
**None**, and `_draw_goal` falls through to a text goal exactly as it already
does when the archive is empty. It must NOT fall back to the seed or to the old
centroid extrapolation — both hand back a goal the seed already maximises,
which is the bug this spec exists to remove. Expeditions require `len(archive)
>= seed_n = 256`, so this is a guard, not a path.

`LATENT_DIMS = 8` and `LATENT_PUSH_SD = 3.0` are module constants in
`goal_source.py`. They are tuned to a 4784-entry archive but are not knife-edge:
d=8 worked across +1/+2/+3 sd. d must stay small — **d=32 was nearly a no-op**
(seed rank 0.1), because whitening equalises the components and then most of the
push lands in directions that are geometrically tiny.

`beta` is deleted: from `ArchiveState`, from the `_handle_explore` push list,
from `ImgepDriver`, and from `latent_goal`'s signature. Its UI slider
("Extrapolation (beta)", `ui/archive_window.py`) is removed with it.
`ArchiveState` is not persisted to disk, so no saved config breaks.

### 4. Cancel Expedition

`end_expedition()` already exists and already does the right thing. This is
plumbing in the established one-shot pattern:

- `ArchiveState.cancel_expedition_requested: bool = False`
- a button in the Explore tab, enabled only while `driver.regime ==
  "expedition"`, showing the current goal label so it is clear what is being
  abandoned
- `CommandHandler._handle_explore` calls `drv.end_expedition()`, and
  `_clear_explore_flags` clears the flag

Cancelling leaves `_since_expedition` at 0 (where `start_expedition_with` set
it), so the search returns to expansion and gets a full `expansion_between`
interval before proposing another goal. That is the desired behaviour: cancel
means "not this, and not immediately another".

## Testing

No GL and no CLIP session is needed for any of it; every piece is numpy in,
numpy out.

- **`tests/test_expedition_fitness.py`** — known-answer softmax against
  hand-computed values; a reference set of one vs many; that a uniform shift in
  every similarity leaves the ranking unchanged (this is the modality-gap
  cancellation, stated directly); empty references raises; non-finite input.
- **`tests/test_goal_source.py`** (extend) — the regression test that *is* the
  bug: on a synthetic archive with a controlled cone, the seed must NOT be the
  argmax of the contrastive fitness under a whitened goal, whereas it is under
  the old construction. Plus: goal is unit norm; a failed PCA fit falls back to
  the seed.
- **`tests/test_archive_projection.py`** (extend) — `variances` matches the
  retained eigenvalues, descending, and has length `n_components`.
- **`tests/test_imgep_driver.py`** (extend) — references are selected by goal
  kind; the descriptor written to the archive is unchanged by the fitness
  change; distractors are embedded once, not per generation; `scorer._text_emb`
  is never written.
- **`tests/test_archive_commands.py`** (extend) — the cancel flag ends the
  expedition, is cleared, and returns `regime` to `expansion`.

Manual verification (`docs/testing_checklist.md`, new section): run Explore to
`seed_n`, let an expedition start, confirm the goal label and that fitness moves
off its starting value rather than decaying; cancel mid-expedition and confirm
the regime indicator returns to expansion.

## Out of scope

The archive admission threshold — the thrashing between "nothing admitted" and
"everything admitted" — is a separate spec. It is likely *entangled* with this
one: a search pinned to its own seed feeds the archive near-duplicates, which is
plausibly part of what makes the adaptive threshold oscillate. Assessing it
before expeditions work would measure the wrong system.
