"""Contrastive fitness for a directed expedition.

Raw cosine to a goal is not a usable objective on this substrate. Measured
2026-08-08 over a real 4784-entry archive: mean pairwise <b, b'> is 0.897 and
half the variance sits in 3 of 512 components, so the archive covers its
reachable cone densely and every reachable goal is ALREADY matched to 0.96-0.99
by something in it. The whole usable range of <b, g> is a couple of percent.

For a text goal the same saturation wears CLIP's modality gap as a disguise:
within one prompt the spread over the archive is std 0.011 around 0.21, while
the offset BETWEEN prompts is 0.013. The phrase moves the number as much as the
creature does.

    fit = mean_s softmax(scale * [<e_s, g>, <e_s, r_1>, ...])[0]

fixes both at once. The scale turns a 0.03 cosine advantage into a 3.0 logit
advantage, and a softmax is invariant to any constant added to every similarity
- which is exactly what the modality gap is. Measured effect on the thing that
matters: the number of archived entries scoring above an expedition's own
starting point goes from 0.0 to 23.8.

The reference set is ONE vector for latent and chase goals, the archive
centroid: "more like the goal than like the average of everything made so far".
Both richer alternatives measured worse - adding 8 random archive entries halves
the spread (0.139 -> 0.070), and using the 16 entries nearest the goal collapses
it to 0.004, far too flat to rank on.

Scored per snapshot and THEN averaged, never on the trajectory centroid.
descriptor() renormalises that centroid, and 1/||m|| grows as the snapshots
decorrelate, so scoring it hands a pattern a free bonus for CHANGING rather than
for matching - 0.22 standard deviations of the fitness spread, measured. The
descriptor is right for novelty and wrong for fitness; they are separate
computations over the same embeddings.

THE SCALE DEPENDS ON THE GOAL'S MODALITY. See the constants below - using
CLIP's 100 for an image goal silently deletes most of the population.
"""
from __future__ import annotations

import numpy as np

from services.clip_scorer import LOGIT_SCALE

# CLIP's own learned logit_scale.exp(), and the right value for TEXT goals: the
# modality gap compresses text-image similarity into a narrow band near 0.2, and
# 100 is precisely the temperature trained to spread that band out. Measured
# 2026-08-08 over 4784 real descriptors: 0.0% of tiles floored, all 16 of a
# generation distinguishable, and a wider spread than any smaller scale.
TEXT_LOGIT_SCALE = LOGIT_SCALE  # 100.0

# Image-image similarity is a different regime entirely - it sits above 0.9,
# where 100 is far too sharp and the softmax pins to zero. Measured on the same
# archive, a +3sd latent goal scored against the centroid:
#
#     scale 100 -> 59.6% of tiles floored, 10.6 of 16 distinct, spread 0.37
#     scale  30 ->  1.4% of tiles floored, 16.0 of 16 distinct, spread 0.44
#     scale  10 ->  0.0% of tiles floored, 16.0 of 16 distinct, spread 0.33
#
# A floored tile is invisible to a rank-based optimizer, so at 100 a third of
# every generation carries no information. 30 is the peak of that curve.
IMAGE_LOGIT_SCALE = 30.0


def contrastive(snaps, goal, references, logit_scale: float):
    """(S, n, dim) x (dim,) x (m, dim) -> (n,) float32 in [0, 1].

    Every input is assumed L2-normalised, so a dot product is a cosine.

    logit_scale has NO default on purpose: TEXT_LOGIT_SCALE and
    IMAGE_LOGIT_SCALE differ by more than 3x, and picking the wrong one costs
    most of the population to saturation without raising anything.

    A tile whose embedding is non-finite scores 0.0 - the minimum a softmax
    probability can take, so a broken tile ranks last instead of poisoning the
    optimizer with a NaN.
    """
    a = np.asarray(snaps, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected snapshots (S, n, dim), got {a.shape}")
    g = np.asarray(goal, dtype=np.float32).reshape(1, -1)
    r = np.asarray(references, dtype=np.float32).reshape(-1, g.shape[1])
    if len(r) == 0:
        # With nothing to contrast against, softmax([x]) is 1.0 for every tile.
        # A silently flat landscape is far worse than a loud caller error.
        raise ValueError("a contrastive fitness needs at least one reference")

    logits = float(logit_scale) * (a @ np.concatenate([g, r], axis=0).T)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    fit = p[..., 0].mean(axis=0)
    return np.where(np.isfinite(fit), fit, 0.0).astype(np.float32)
