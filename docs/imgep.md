# How Explore (IMGEP) works

Goal-directed exploration over an archive of patterns. One **generation** runs
every tile of the tournament grid at once (16 at 4×4, 64 at 8×8), embeds them
with CLIP, offers them to the archive, and decides what to run next.

This file is the loop and the equations. `ARCHITECTURE.md` covers the app;
`CLAUDE.md` holds the measured caveats — the numbers behind these choices and
the things not to change.

---

## The loop

```
every generation:
    z          = ask()                      # what to run: one vector per tile
    snapshots  = rollout(z)                 # S frames, each (tiles, 224, 224, 3)
    fitness    = tell(z, snapshots)         # embed, admit, score, adapt
```

`ask()` depends on the **regime**, checked in this order:

| regime | when | what it proposes |
|---|---|---|
| expedition | a goal is being chased | `optimizer.ask()` — CMA-ES, popsize = tiles |
| bootstrap | native entries < `seed_n` **and** this layout has spent < `bootstrap_gens` bootstrap generations | `z ~ N(0, sigma0)` — scattered, no parents |
| expansion | otherwise | per tile: one parent `p ~ novelty^alpha`, re-encoded, plus `N(0, sigma_expand)` |

Expansion draws **one parent per tile, independently and with replacement**, so
the grid sets the number of draws, not the number of parents. No crossover, no
covariance, no shared distribution.

An expedition starts when all three hold, and then runs for `expedition_gens`:

```
expansion_between > 0                    # 0 disables expeditions entirely
gens_since_last >= expansion_between     # counted over non-expedition gens only
regime == "expansion"                    # this layout is out of bootstrap
```

**Every brain layout gets its own bootstrap**, and it ends at whichever limit
it reaches first. Random draws are the cheapest exploration there is, and a
layout arrived at by a move holds only what its expedition admitted — all of it
clustered round the one genome that expedition converged on — so expanding from
those alone explores a pinhole.

`seed_n` alone cannot bound that in time. Separation is measured against the
POOLED archive, every layout included, so a layout born into a full archive
admits ever more slowly and takes ever longer to reach the same native count.
`bootstrap_gens` is the ceiling that keeps each layout's bootstrap the same
size however late it arrives.

---

## Quantities

Each tile gets `S` CLIP embeddings, one per snapshot, each unit-norm.

```
e[s][i]        embedding of tile i at snapshot s          (unit vector, 512-d)

descriptor     b[i]  = normalise( mean_s e[s][i] )
                     the trajectory centroid: WHAT this tile is.
                     Everything the archive stores and compares is this.

liveness       L[i]  = 1 - mean_{s>0} max_{s'<s} <e[s][i], e[s'][i]>
                     how much it CHANGED. Frozen -> 0.

novelty        N[i]  = mean distance to the k nearest of (archive ∪ rejects)
                     how far it is from everything already seen.

coherence      C[i]  = max over lags (1,2,3,4,6,8,12,16) and both axes of
                       |spatial autocorrelation of the tile's channel mean|
                     is this a PATTERN or is it static? Purely spatial - it
                     never compares frames, so slow is not penalised. A flat
                     tile scores 1.0; `viable` is what rejects those.

viable         V[i]  = 2 <= mean brightness <= 253      (not blank, not blown)
```

`coherence` and `viable` read the **last snapshot only**; `descriptor`,
`liveness` and the expedition fitness use all `S`.

---

## Fitness

Only an expedition has a fitness — bootstrap and expansion drive no optimizer.

```
contrastive(x, goal, refs, scale):
    logits = scale * [ <x, goal>, <x, refs[0]>, ... ]
    return softmax(logits)[0]           # in [0,1]
```

Contrastive rather than raw `<b, goal>` because the archive occupies a narrow
cone (mean pairwise similarity 0.897), where raw cosine has no usable range —
and because a softmax is invariant to a constant added to every similarity,
which is exactly what CLIP's text-image modality gap is.

**The references and the scale depend on the goal's modality:**

```
goal.kind == "text"                     everything else ("latent", "chase")
    refs  = DEFAULT_DISTRACTORS             refs  = [ archive centroid ]
            (9 text embeddings:                     ("more like the goal than
             "random noise",                         like the average of
             "an abstract texture",                  everything made so far")
             "a blank image",
             "a solid color",
             "a black image", ...)
    scale = TEXT_LOGIT_SCALE = 100      scale = IMAGE_LOGIT_SCALE = 30
```

Two scales because they measure different regimes: text-image similarity sits
in a narrow band near 0.2, which 100 is trained to spread; image-image
similarity sits above 0.9, where 100 saturates.

**The fitnesses:**

```
fitness(i) = coherence[i] * base(i)

    text / latent / chase:
        base(i) = mean_s contrastive( e[s][i], goal, refs, scale )
                  # per snapshot, THEN averaged. Not on the descriptor:
                  # normalising the centroid pays a bonus for CHANGING
                  # rather than for matching.

    novelty:
        base(i) = novelty[i]
                  # no target at all, so nothing to be unreachable.

    (defensive: with no references at all - an empty archive, unreachable
     inside an expedition - base falls back to descriptor(snaps) @ goal.)
```

`coherence` is the noise defence. A latent or chase goal has one reference, so
its fitness is a monotone squash of raw cosine — and raw cosine to an arbitrary
direction is maximised by static, which has energy in every direction. Text
goals were never exposed, because the distractor list contains "random noise"
and "an abstract texture". CMA-ES is rank-based, so multiplying by ~0.95 leaves
real tiles' order intact while ~0.02 buries static at the bottom.

---

## Admission

Every tile of every generation is offered. The base gates are independent of
fitness; the two bypasses below are not.

```
for each tile i:
    if pinned:  admit                  # user selection short-circuits EVERYTHING
    reject unless finite(b[i])
    reject unless V[i]
    reject unless L[i] >= liveness_min                 # unless ignore_liveness
    reject if min distance from b[i] to the archive < min_separation   # unless force
    else admit

    # and after each admission, within the generation:
    sep = min(sep, 1 - b @ b[i])       # the batch must separate from ITSELF,
                                       # or 64 converged tiles all pass
```

Separation is the unstructured-archive rule from quality-diversity: "do we
already have one of these". It is **not** a novelty threshold — there is no
controller and no gain, and it asks a local, parameter-free question.

Four things get past separation, because it asks the wrong question about them.
`keeper`, `summit` and `record` pass `force=True`; `pin` short-circuits the
whole function before any gate runs:

| kind | rule | why |
|---|---|---|
| `pin` | the user selected the tile | human intent outranks everything |
| `keeper` | the generation's most novel viable tile | a run always leaves a trail |
| `summit` | beats this expedition's own best fitness | a converging chase makes tiles that resemble each other, so its best result is exactly what separation discards |
| `record` | beats the **archive's** best match for any enabled text goal, in any regime | a run chasing "pepperoni pizza" can produce the best "a smiley face" ever made |

`summit` and `record` additionally pass `ignore_liveness` — a settled attractor
is a legitimate result, and liveness is measured *higher* during the post-reset
transient than once a pattern settles. `keeper` does not, because it fires every
generation forever. Only `pin` bypasses viability.

Rejections feed the **rejects ring**, which novelty measures against, so the
search remembers regions it was refused. Separation rejections deliberately do
not: that region is in the archive already.

Once per generation, after admission: refresh a slice of stored novelty
(`ceil(len / refresh_sweep_gens)` entries), then evict the least novel down to
`capacity`.

---

## Goals

```
draw_goal():
    nov, lat = max(0, novelty_share), max(0, latent_share)
    if nov + lat > 1:  lat = 1 - nov        # CLAMPED, never normalised: the
                                            # sliders are independent
    u = uniform(0,1)
    u <  nov        -> novelty goal
    u <  nov + lat  -> latent goal
    otherwise       -> a text goal
    ... falling THROUGH to the other kinds on refusal, since an expedition that
        fails to start wastes a whole cadence interval
```

```
text     an embedded phrase from the user's goal list, picked by `goal_order`:
         "round_robin", or "least_matched" = argmin over goals of
         (max_e <e,g> - mean_e <e,g>), the archive's reach past its own
         indifferent baseline for that phrase.

latent   an archive entry b, drawn p ~ novelty^alpha, whitened into the
         archive's 8-component PCA subspace, pushed +3sd along its own
         direction, unwhitened. "Past the frontier, in the direction this
         entry already points."

novelty  no embedding at all. The fitness is novelty itself.
```

An expedition **seeds** on an existing archive entry, sampled (not argmaxed)
with `p ~ fitness^alpha` under its own objective — the same contrastive score it
will be scored on, so a text goal cannot seed on the noise tile that raw cosine
prefers. `banded_alpha` only clips the ends: alpha is left alone when its
effective sample size already falls in `[seed_ess_min, seed_ess_max]`, and the
floor is capped at `N/8`. A novelty goal has no embedding to score against, so
it carries its seed directly.

---

## Settings that matter

| setting | default | effect |
|---|---|---|
| `alpha` | 4.0 | how hard parent choice favours novelty, and seed choice favours goal match. At 8 an archive's parent ESS can fall to ~2 |
| `sigma0` | 0.5 | scatter during bootstrap |
| `sigma_expand` | 0.15 | mutation size in expansion |
| `expedition_sigma` | 0.1 | CMA-ES step size |
| `seed_n` | 256 | native entries at which a layout's bootstrap ends |
| `bootstrap_gens` | 30 | generations a layout may bootstrap for, whichever limit it reaches first |
| `expansion_between` | 25 | generations between expeditions; 0 disables them |
| `expedition_gens` | 50 | how long a chase runs |
| `min_separation` | 0.02 | admission radius; 0 stores everything |
| `liveness_min` | 0.002 | floor on the bulk, never a veto over a chosen entry |
| `k` | 10 | neighbours for novelty |
| `capacity` | 20000 | archive cap; the least novel are evicted past it |
| `n_views` | 3 | CLIP sub-crops averaged per tile; costs `tiles × snapshots × views` passes |
| `refresh_sweep_gens` | 10 | generations for stored novelty to be fully re-scored |
| `novelty_share` / `latent_share` | 0.25 / 0.5 | goal mix; text takes the rest |
| `physics_enabled` | off | search the physics parameters as well as the brain |

### Searching physics

**Search Physics Too** adds `services/physics_genome.PHYSICS_PARAMS` to the
search space. They are searched RELATIVE to the loaded preset — `z = 0` is the
preset exactly — so the sliders still matter while it is on: they set the
CENTRE the search roams around, not each tile's value. The parameters outside
that list (trails, boundary mode, cohorts, hue) are never searched under any
setting.

How far a gene may roam is `SPAN_FRACTION` (0.5) of its nominal range either
way from the preset, **bounded by the parameter's hard limits** — the span is a
distance and not a value, so on its own it does not keep a gene legal.
`hard_min`/`hard_max` in `ui/physics_params.py` are the one home for those
limits, and `physics_genome.reach()` shrinks the roam distance to fit them
instead of clamping the decoded value, which would hand the optimizer a
plateau. The reach is therefore asymmetric, and zero on a side whose limit the
preset already sits on.

Two of the eight searched parameters have hard limits, `DRAG` and
`SENSOR_ANGLE`, both ±1. `DRAG` is the one that matters: `vel = vel*drag +
force`, so above 1 velocity is amplified every step. Over the 131 presets the
unbounded span let **every one of them** reach past ±1 on both parameters, with
22.0% and 21.4% of the reachable interval out of bounds; shrinking removes
exactly that and costs the other six parameters nothing.

`tanh` keeps the decode inside the reach, so nothing clamps on the way to the
GPU and what the archive stores is what ran. The reach is also the limit in the
other direction: seeding an expedition re-encodes
a stored phenotype under the CURRENT preset, and a value further than one span
away has no `z` at all, so the chase starts from the nearest creature the
preset can reach instead. `encode_physics` returns that count and the Explore
tab prints it, the same contract `genome_spec.encode` has for the brain.

Turning it on or off changes the dimension of the search space, so it resets
the search. The archive survives, because it stores decoded phenotypes rather
than `z`. An archive may therefore hold entries from both settings: one
admitted with it off carries no searched physics and replays under the physics
its run was carried out under (`<archive>/runs/<run_id>.json`), and either kind
can be a parent or a seed.
