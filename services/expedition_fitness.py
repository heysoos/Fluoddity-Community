"""Contrastive fitness for a directed expedition.

Raw cosine to a goal is not usable: the archive covers its reachable cone
densely, so every reachable goal is already matched almost as well by
something in it, and for a text goal CLIP's modality gap swamps the signal.

    fit = mean_s softmax(scale * [<e_s, g>, <e_s, r_1>, ...])[0]

fixes both - the scale turns a small cosine advantage into a large logit one,
and softmax is invariant to the constant offset the modality gap adds. See
CLAUDE.md.

The reference set is ONE vector for latent and chase goals, the archive
centroid ("more like the goal than like the average of everything made so
far") - richer alternatives measured worse.

Scored per snapshot and THEN averaged, never on the trajectory centroid:
descriptor() renormalises that centroid, which rewards CHANGING over matching.
The descriptor is right for novelty and wrong for fitness; they are separate
computations over the same embeddings.

THE SCALE DEPENDS ON THE GOAL'S MODALITY. See the constants below - using
CLIP's 100 for an image goal silently deletes most of the population.
"""
from __future__ import annotations

import numpy as np

from services.clip_scorer import LOGIT_SCALE

# CLIP's own learned logit_scale.exp(), and the right value for TEXT goals: the
# modality gap compresses text-image similarity into a narrow band, and 100 is
# the temperature trained to spread that band out.
TEXT_LOGIT_SCALE = LOGIT_SCALE  # 100.0

# Image-image similarity sits above 0.9, a different regime - 100 is far too
# sharp there and floors most of the population to zero. See CLAUDE.md.
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
