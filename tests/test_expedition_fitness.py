"""The expedition objective.

Two properties raw cosine does not have, and both are why expeditions did not
climb: adding a constant to every similarity (the modality gap) must not change
the ranking, and a couple of percent of cosine must survive as a rankable gap.

The second one is why the logit scale is a per-modality constant rather than
CLIP's 100 everywhere. See services/expedition_fitness.py for the measurements.
"""
import numpy as np
import pytest

from services.expedition_fitness import (
    IMAGE_LOGIT_SCALE,
    TEXT_LOGIT_SCALE,
    contrastive,
)


def unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def snaps_of(*per_snapshot):
    """Each argument is one snapshot's (n, dim) embeddings."""
    return np.stack([unit(np.asarray(s, dtype=np.float32))
                     for s in per_snapshot], axis=0)


def cone(rng, n=64, dim=64, spread=0.04):
    """A tight cluster of unit vectors, which is the regime the real archive is
    in: mean pairwise <b, b'> = 0.897, measured over 4784 descriptors.

    cos ~ 1 / (1 + spread^2 * dim) for independent noise, so 0.04 at dim 64
    lands near 0.9. At 0.15 it would be 0.41 - not a cone, and any test built on
    it would be testing a regime the app never sees.
    """
    axis = unit(rng.normal(size=dim).astype(np.float32))
    return unit(axis + spread * rng.normal(size=(n, dim)).astype(np.float32))


# ---- the arithmetic -----------------------------------------------------

def test_it_is_the_softmax_probability_of_the_goal():
    goal, ref = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    e = unit(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    fit = contrastive(snaps_of(e), goal, ref[None, :], logit_scale=1.0)

    def expect(v):
        lg = np.array([v @ goal, v @ ref])
        p = np.exp(lg - lg.max())
        return p[0] / p.sum()

    assert fit == pytest.approx([expect(v) for v in e], abs=1e-6)


def test_a_tile_on_the_goal_beats_one_on_the_reference():
    goal, ref = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    e = np.stack([goal, ref]).astype(np.float32)
    fit = contrastive(snaps_of(e), goal, ref[None, :],
                      logit_scale=IMAGE_LOGIT_SCALE)
    assert fit[0] > 0.99 and fit[1] < 0.01


def test_snapshots_are_averaged_after_scoring_not_before():
    """Scoring the trajectory CENTROID divides by ||m||, which shrinks as the
    snapshots decorrelate - a free bonus for changing rather than matching."""
    goal, ref = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    a = unit(np.array([[1.0, 0.2]]))
    b = unit(np.array([[1.0, -0.2]]))
    fit = contrastive(snaps_of(a, b), goal, ref[None, :], logit_scale=1.0)
    each = [contrastive(snaps_of(s), goal, ref[None, :], logit_scale=1.0)[0]
            for s in (a, b)]
    assert fit[0] == pytest.approx(float(np.mean(each)), abs=1e-6)


# ---- the property that fixes text goals ---------------------------------

def test_a_constant_offset_on_every_similarity_leaves_the_ranking_alone():
    """THE point. CLIP's modality gap is a per-prompt constant added to every
    image's similarity - measured at 0.013 between prompts against a within-
    prompt spread of 0.011, so under raw cosine the phrase moves the number as
    much as the creature does. A softmax cancels it exactly.
    """
    rng = np.random.default_rng(0)
    dim = 16
    e = unit(rng.normal(size=(32, dim)).astype(np.float32))
    goal = unit(rng.normal(size=dim).astype(np.float32))
    refs = unit(rng.normal(size=(3, dim)).astype(np.float32))
    base = contrastive(snaps_of(e), goal, refs, logit_scale=TEXT_LOGIT_SCALE)

    # Shift every similarity by the same amount, by appending a component the
    # goal, the references and every tile all share.
    pad = np.full((len(e), 1), 0.5, dtype=np.float32)
    e2 = np.concatenate([e, pad], axis=1)
    goal2 = np.concatenate([goal, [0.4]]).astype(np.float32)
    refs2 = np.concatenate([refs, np.full((len(refs), 1), 0.4, np.float32)], axis=1)
    shifted = contrastive(snaps_of(e2), goal2, refs2, logit_scale=TEXT_LOGIT_SCALE)

    assert np.argsort(base).tolist() == np.argsort(shifted).tolist()


def test_it_separates_tiles_that_raw_cosine_squashes_together():
    """In a tight cone every descriptor is within a couple of percent of any
    reachable goal. The logit scale is what makes that couple of percent
    rankable - measured on the real archive, contrastive-vs-centroid roughly
    doubles the spread of a latent objective."""
    rng = np.random.default_rng(7)
    e = cone(rng)
    goal = unit(e[0] + 0.05 * rng.normal(size=e.shape[1]).astype(np.float32))
    centroid = unit(e.mean(axis=0))

    assert float((e @ e.T)[np.triu_indices(len(e), 1)].mean()) > 0.85, \
        "the fixture has to actually be a tight cone for this to mean anything"

    raw = e @ goal
    fit = contrastive(snaps_of(e), goal, centroid[None, :],
                      logit_scale=IMAGE_LOGIT_SCALE)
    assert raw.std() < 0.05, "raw cosine really is this squashed"
    assert fit.std() > 1.5 * raw.std()


# ---- the scale is per-modality, and getting it wrong is silent ----------

def test_the_image_scale_keeps_a_generation_rankable_where_clips_does_not():
    """MEASURED 2026-08-08 on 4784 real descriptors: a +3sd latent goal scored
    against the archive centroid floors 59.6% of tiles at CLIP's scale of 100,
    leaving only 10.6 of a 16-tile generation distinguishable. A floored tile is
    invisible to a rank-based optimizer, so a third of every generation carried
    no information. At 30 the floored fraction is 1.4%.
    """
    rng = np.random.default_rng(11)
    e = cone(rng)
    c = unit(e.mean(axis=0))
    # a goal pushed away from the centroid, as the whitened construction does
    goal = unit(c + 3.0 * (e[0] - c))

    def distinct(scale):
        f = contrastive(snaps_of(e), goal, c[None, :], logit_scale=scale)
        return len(np.unique(np.round(f[:16], 6)))

    assert distinct(IMAGE_LOGIT_SCALE) > distinct(TEXT_LOGIT_SCALE)
    assert distinct(IMAGE_LOGIT_SCALE) == 16, "no tile lost to saturation"


def test_the_two_scales_are_far_enough_apart_to_matter():
    """If these ever converge, the per-modality split is dead weight and the
    comment justifying it is a lie."""
    assert TEXT_LOGIT_SCALE / IMAGE_LOGIT_SCALE > 3.0


def test_the_scale_is_required():
    """A wrong scale saturates the landscape without raising, so there is no
    safe default to fall back on."""
    with pytest.raises(TypeError):
        contrastive(snaps_of(np.eye(2, dtype=np.float32)),
                    np.array([1.0, 0.0]), np.array([0.0, 1.0])[None, :])


# ---- guards -------------------------------------------------------------

def test_an_empty_reference_set_is_refused():
    """softmax over one entry is 1.0 for every tile. A flat landscape that
    looks like a working one is the worst possible failure here."""
    with pytest.raises(ValueError, match="at least one reference"):
        contrastive(snaps_of(np.eye(2, dtype=np.float32)),
                    np.array([1.0, 0.0]), np.zeros((0, 2), np.float32),
                    logit_scale=IMAGE_LOGIT_SCALE)


def test_a_single_reference_vector_is_the_normal_case():
    """Latent and chase goals contrast against the archive centroid alone."""
    fit = contrastive(snaps_of(np.eye(3, dtype=np.float32)),
                      np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0])[None, :],
                      logit_scale=IMAGE_LOGIT_SCALE)
    assert fit.shape == (3,)
    assert np.all(np.isfinite(fit))


def test_wrongly_shaped_snapshots_are_refused():
    with pytest.raises(ValueError, match=r"\(S, n, dim\)"):
        contrastive(np.eye(3, dtype=np.float32), np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0])[None, :],
                    logit_scale=IMAGE_LOGIT_SCALE)


def test_a_broken_tile_scores_zero_rather_than_nan():
    """A NaN reaching the optimizer corrupts the whole generation's ranking."""
    e = np.eye(3, dtype=np.float32)
    e[1] = np.nan
    fit = contrastive(np.stack([e]), np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0])[None, :],
                      logit_scale=IMAGE_LOGIT_SCALE)
    assert np.all(np.isfinite(fit))
    assert fit[1] == 0.0
    assert fit[0] == max(fit), "the healthy tile still ranks top"


def test_the_output_is_a_probability():
    rng = np.random.default_rng(1)
    e = unit(rng.normal(size=(20, 12)).astype(np.float32))
    fit = contrastive(snaps_of(e, e), unit(rng.normal(size=12).astype(np.float32)),
                      unit(rng.normal(size=(9, 12)).astype(np.float32)),
                      logit_scale=TEXT_LOGIT_SCALE)
    assert fit.dtype == np.float32
    assert np.all(fit >= 0.0) and np.all(fit <= 1.0)
