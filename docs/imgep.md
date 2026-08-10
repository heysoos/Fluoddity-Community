# How Explore (IMGEP) works

Goal-directed exploration over an archive of patterns. One **generation** runs
every tile of the tournament grid at once (16 at 4×4, 64 at 8×8), embeds them
with CLIP, offers them to the archive, and decides what to run next.

`ARCHITECTURE.md` covers the app; `CLAUDE.md` holds the measured caveats. This
file is only the loop and the equations.

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
| bootstrap | archive < `seed_n` | `z ~ N(0, sigma0)` — scattered, no parents |
| expansion | otherwise | per tile: one parent `p ~ novelty^alpha`, re-encoded, plus `N(0, sigma_expand)` |

Expansion draws **one parent per tile, independently and with replacement**, so
the grid sets the number of draws, not the number of parents. No crossover, no
covariance, no shared distribution.

Every `expansion_between` generations an expedition starts and runs for
`expedition_gens`.

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
                       |spatial autocorrelation of the tile's luminance|
                     is this a PATTERN or is it static? Purely spatial —
                     it never compares frames, so slow ≠ penalised.

viable         V[i]  = 2 <= mean brightness <= 253      (not blank, not blown)
```

---

## Fitness

Only an expedition has a fitness — bootstrap and expansion drive no optimizer.

```
contrastive(x, goal, refs, scale):
    logits = scale * [ <x, goal>, <x, refs[0]>, ... ]
    return softmax(logits)[0]           # in [0,1]
```

Why contrastive rather than raw `<b, goal>`: the archive covers a cone whose
mean pairwise similarity is 0.897, so everything already matches everything to
0.96+ and raw cosine has no usable range. A softmax is also invariant to a
constant added to every similarity, which is exactly what CLIP's text-image
modality gap is.

**The references and the scale depend on the goal's modality:**

```
goal.kind == "text"                     goal.kind in ("latent", "chase")
    refs  = DEFAULT_DISTRACTORS             refs  = [ archive centroid ]
            (9 text embeddings:                     ("more like the goal than
             "random noise",                         like the average of
             "an abstract texture",                  everything made so far")
             "a blank image",
             "a solid color",
             "a black image", ...)
    scale = 100   (CLIP's own)          scale = 30
```

Two scales because they measure different regimes. Text-image similarity sits
in a narrow band near 0.2 and 100 is the temperature trained to spread it;
image-image similarity sits above 0.9, where 100 saturates and floors 59.6% of
a generation to zero.

**The three fitnesses:**

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
```

`coherence` is the noise defence. A latent or chase goal has one reference, so
its fitness is a monotone squash of raw cosine — and raw cosine to an arbitrary
direction is maximised by static, which has energy in every direction. Text
goals never had this problem because the distractor list contains "random
noise" and "an abstract texture". CMA-ES is rank-based, so multiplying by
~0.95 leaves real tiles' order intact while ~0.02 buries static at the bottom.

---

## Admission

Independent of fitness. Every tile of every generation is offered.

```
for each tile i:
    reject unless finite(b[i])
    reject unless V[i]                          # not a blank frame
    reject unless L[i] >= liveness_min          # unless forced, see below
    reject if min distance from b[i] to the archive < min_separation
    else admit
```

Separation is the unstructured-archive rule from quality-diversity: "do we
already have one of these". It is **not** a novelty threshold — there is no
controller and no gain, and it asks a local, parameter-free question.

Four things bypass separation, because separation asks the wrong question
about them:

| kind | rule | why |
|---|---|---|
| `pin` | the user selected the tile | human intent outranks everything |
| `keeper` | the generation's most novel viable tile | a run always leaves a trail |
| `summit` | beats this expedition's own best fitness | a converging chase makes tiles that resemble each other, so its best result is exactly what separation discards |
| `record` | beats the **archive's** best match for any enabled text goal, in any regime | a run chasing "pepperoni pizza" can produce the best "a smiley face" ever made |

`summit` and `record` also bypass **liveness** — a settled attractor is a
legitimate result, and liveness is measured higher during the post-reset
transient than once a pattern settles. Neither bypasses viability.

Once per generation, after admission: refresh a slice of stored novelty
(`ceil(len / refresh_sweep_gens)` entries), then evict the least novel down to
`capacity`.

---

## Goals

```
draw_goal():
    u = uniform(0,1)
    u <  novelty_share                  -> novelty goal
    u <  novelty_share + latent_share   -> latent goal
    otherwise                           -> next text goal (round robin)
    ... falling THROUGH on refusal, since an expedition that fails to start
        wastes a whole cadence interval
```

```
text     an embedded phrase from the user's goal list.

latent   an archive entry b, drawn p ~ novelty^alpha, whitened into the
         archive's 8-component PCA subspace, pushed +3sd along its own
         direction, unwhitened. "Past the frontier, in the direction this
         entry already points."

novelty  no embedding at all. The fitness is novelty itself.
```

An expedition **seeds** on an existing archive entry, sampled (not argmaxed)
with `p ~ fitness^alpha` under its own objective, alpha banded to keep the
effective sample size in `[seed_ess_min, seed_ess_max]`. A novelty goal has no
embedding to score against, so it carries its seed directly.

---

## Settings that matter

| setting | default | effect |
|---|---|---|
| `alpha` | 4.0 | how hard parent/seed choice favours novelty. Above ~5 it collapses to one entry |
| `sigma_expand` | 0.15 | mutation size in expansion |
| `min_separation` | 0.02 | admission radius; 0 stores everything |
| `liveness_min` | 0.002 | floor on the bulk, never a veto over a chosen entry |
| `k` | 10 | neighbours for novelty |
| `n_views` | 3 | CLIP sub-crops averaged per tile; costs `tiles × snapshots × views` passes |
| `refresh_sweep_gens` | 10 | generations for stored novelty to be fully re-scored |
| `novelty_share` / `latent_share` | 0.25 / 0.5 | goal mix; text takes the rest |
