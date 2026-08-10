"""Where an expedition's target comes from.

Two sources, both fully offline:

  LATENT - extrapolate past the archive's frontier, away from its centroid.
  Entirely within image space, so CLIP's modality gap is irrelevant and no
  phrase can propose something the substrate has no vocabulary for.

  TEXT - the user's own list. E&E has o4-mini read 25 archive thumbnails and
  write a 5-15 word description of a hypothetical pattern; here the human writes
  that list and the app cycles it.

Ordering over text goals is ROUND ROBIN by default. CLIP's modality gap means
different phrases have different baseline affinities to any image, so 'which
goal is the archive least able to match' is not comparable across phrases
without normalisation. Within a single expedition the goal is fixed, so the gap
is a constant offset on every tile's score and CMA-ES's ranking is unaffected -
which is why raw <b, g> is a fine expedition fitness and a poor cross-goal
comparison. least_matched() therefore subtracts the archive mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.novelty import sample_by_novelty


@dataclass
class Goal:
    kind: str                 # "latent" | "text" | "chase" | "novelty"
    text: str
    embedding: np.ndarray | None   # (dim,) float32 unit norm; None for novelty
    # Where the expedition should START, for a goal that cannot be asked.
    #
    # Only a novelty goal sets this: it has no embedding, so there is nothing
    # for _seed_index to score the archive against. Text, chase and latent
    # goals all leave it None and go through _seed_index, which samples with a
    # banded alpha - measured better than any shortcut tried here, including
    # seeding a latent goal at its own anchor. See latent_goal.
    seed_index: int | None = None

    @property
    def label(self) -> str:
        return self.text if self.text else f"({self.kind})"


class GoalList:
    """The user's editable text goals, persisted in archive/goals.json."""

    def __init__(self, store=None):
        self.store = store
        self.items: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._embedded_texts: tuple[str, ...] = ()
        self._cursor = 0

    # ---- editing -------------------------------------------------------

    def add(self, text: str) -> bool:
        t = str(text).strip()
        if not t or any(i["text"] == t for i in self.items):
            return False
        self.items.append({"text": t, "enabled": True})
        self._embeddings = None
        return True

    def remove(self, i: int) -> None:
        if 0 <= i < len(self.items):
            self.items.pop(i)
            self._embeddings = None

    def set_text(self, i: int, text: str) -> None:
        t = str(text).strip()
        if 0 <= i < len(self.items) and t:
            self.items[i]["text"] = t
            self._embeddings = None

    def set_enabled(self, i: int, enabled: bool) -> None:
        if 0 <= i < len(self.items):
            self.items[i]["enabled"] = bool(enabled)

    def move(self, i: int, delta: int) -> None:
        j = i + int(delta)
        if 0 <= i < len(self.items) and 0 <= j < len(self.items):
            self.items[i], self.items[j] = self.items[j], self.items[i]
            self._embeddings = None

    def enabled_items(self) -> list[tuple[int, dict]]:
        return [(i, it) for i, it in enumerate(self.items) if it.get("enabled", True)]

    # ---- embedding -----------------------------------------------------

    def ensure_embedded(self, scorer) -> None:
        """Embed once and cache. Re-embedding an unchanged list would spend a
        text-encoder pass per expedition for nothing."""
        texts = tuple(i["text"] for i in self.items)
        if self._embeddings is not None and texts == self._embedded_texts:
            return
        if not texts or scorer is None:
            self._embeddings = None
            self._embedded_texts = texts
            return
        self._embeddings = np.asarray(scorer.embed_text(list(texts)), dtype=np.float32)
        self._embedded_texts = texts

    # ---- selection -----------------------------------------------------

    def next_goal(self) -> Goal | None:
        """Round robin over the enabled entries."""
        live = self.enabled_items()
        if not live or self._embeddings is None:
            return None
        pos = self._cursor % len(live)
        self._cursor = (pos + 1) % len(live)
        idx, item = live[pos]
        return Goal("text", item["text"], self._embeddings[idx])

    def enabled_goals(self) -> list[Goal]:
        """Every enabled goal, embedded, in list order.

        For callers that need to score against ALL of them at once rather than
        pick one - the archive's per-goal record book. [] when the list is
        empty or ensure_embedded has not run, so a caller can treat "no goals"
        and "not embedded yet" the same way.
        """
        live = self.enabled_items()
        if not live or self._embeddings is None:
            return []
        return [Goal("text", it["text"], self._embeddings[i]) for i, it in live]

    def least_matched(self, embeddings: np.ndarray) -> Goal | None:
        """The enabled goal the archive extends toward least.

            reach(g) = max_e <e, g> - mean_e <e, g>

        The mean is the archive's indifferent baseline for that phrase, which is
        exactly what the modality gap contributes, so subtracting it makes two
        phrases comparable. What is left is in cosine units and still positional
        - it says how far past that baseline the archive actually reaches.

        NOT z-scored. Dividing by the per-goal std makes the statistic
        scale-free, and a scale-free max is a property of the distribution's
        tail rather than of where it sits: on an archive that fans out around
        goal A and barely brushes goal B, the z-score ranks A as the worse-
        covered of the two. Verified 2026-08-07 - reach picks B, z picks A.
        """
        live = self.enabled_items()
        e = np.asarray(embeddings, dtype=np.float32)
        if not live or self._embeddings is None or len(e) == 0:
            return None
        best, best_score = None, np.inf
        for idx, item in live:
            sims = e @ self._embeddings[idx]
            score = float(sims.max()) - float(sims.mean())
            if score < best_score:
                best, best_score = Goal("text", item["text"], self._embeddings[idx]), score
        return best

    # ---- persistence ---------------------------------------------------

    def load(self) -> None:
        if self.store is None:
            return
        items = self.store.load_goals()
        self.items = [{"text": str(i.get("text", "")),
                       "enabled": bool(i.get("enabled", True))}
                      for i in items if str(i.get("text", "")).strip()]
        self._embeddings = None

    def save(self) -> None:
        if self.store is not None:
            self.store.save_goals(self.items)


# The archive is a low-dimensional cloud inside a 512-d space: measured
# 2026-08-08 over 4784 descriptors, HALF its variance lies in 3 components and
# 80% in 18. Extrapolating in the top 8 therefore moves along the manifold the
# creatures actually occupy.
#
# d must stay SMALL. At d=32 this construction is nearly a no-op - seed rank
# 0.1, against 41.3 at d=8 - because whitening equalises the components and the
# unit direction then puts most of its energy in the minor ones, which unwhiten
# back to almost nothing. Raising d to "capture more variance" silently turns
# the push off.
LATENT_DIMS = 8

# In whitened units, so it is 3 standard deviations along the seed's own
# direction whichever axes that direction uses. Measured seed rank: 21.0 at
# +1sd, 29.1 at +2sd, 41.3 at +3sd, with reach still 0.962 - far enough to
# leave real room past the seed, near enough to stay on the manifold.
LATENT_PUSH_SD = 3.0


def latent_goal(archive, rng, projection, alpha: float = 4.0,
                push_sd: float = LATENT_PUSH_SD) -> Goal | None:
    """Extrapolate past the frontier IN THE ARCHIVE'S PRINCIPAL SUBSPACE.

    The source b is drawn with p proportional to NOV^alpha, so the goal is
    anchored on a genuinely novel entry rather than a random one. It is then
    whitened, pushed out along its own direction, and unwhitened.

    This replaces g = normalise(b + beta*(b - c)), which did not work and could
    not be made to work by tuning beta. Archive.nearest() returns
    argmax(embeddings @ g), so the expedition always seeds on the archive's best
    entry under its goal; the old construction then put the goal 0.965 cosine
    from that very seed, because <b, c> = 0.947 makes b - c a tiny vector.
    Measured over 200 trials: the goal's nearest entry WAS the seed it was built
    from 199 times, at mean rank 0.01. CMA-ES started on the optimum of its own
    objective and every subsequent move could only score worse. Raising beta
    bought climbability at a ruinous rate - beta=20 reaches only 0.328, as
    unreachable as a text prompt.

    Returns None when the projection cannot be fit. It must NOT fall back to the
    seed or to centroid extrapolation: both hand back a goal the seed already
    maximises, which is the bug this function exists to remove. The caller falls
    through to a text goal, exactly as it does for an empty archive.
    """
    if len(archive) == 0 or projection is None:
        return None
    if not projection.fit(archive.embeddings):
        return None

    nov = np.array([e.novelty for e in archive.entries], dtype=np.float32)
    i = int(sample_by_novelty(nov, 1, rng, alpha)[0])
    b = archive.embeddings[i]

    sd = np.sqrt(np.maximum(projection.variances, 1e-12))
    y = ((b - projection.mean) @ projection.components.T) / sd
    n = float(np.linalg.norm(y))
    if not np.isfinite(n) or n < 1e-6:
        # The seed sits on the centroid, so it has no direction of its own.
        return None
    y = y + float(push_sd) * (y / n)

    g = projection.mean + ((y * sd) @ projection.components)
    nrm = float(np.linalg.norm(g))
    if not np.isfinite(nrm) or nrm < 1e-6:
        return None
    # NO seed_index, deliberately. Seeding at the anchor looks obviously right
    # and measured as a REGRESSION - 15.8% -> 27.0% of seeds in the archive's
    # roughest decile - because _seed_index samples with banded_alpha, which
    # already spreads the draw, while p ~ NOV^alpha concentrates hard. The
    # noisy-TOP-PICK half is answered by the coherence factor in
    # ImgepDriver._expedition_fitness, not here.
    return Goal("latent", "", (g / nrm).astype(np.float32))


def novelty_goal(archive, rng, alpha: float = 4.0) -> Goal | None:
    """An expedition with NO target: climb novelty itself.

    Novelty search (Lehman & Stanley) as the inner loop, where E&E uses a
    directed chase. Well-posed by construction: there is no point in embedding
    space that might not be realisable, only a landscape defined by what the
    archive already holds. It is also the only thing in the search that
    optimises novelty WITHIN a generation - expansion samples parents by
    novelty and then mutates blindly.

    The objective is deliberately non-stationary: entries admitted during the
    expedition lower the novelty of everything near them, so CMA-ES adapts a
    covariance on shifting ground. For an explorer that is desired.

    Carries no embedding - there is nothing to point at - so the seed is drawn
    here, the same way expansion draws a parent.
    """
    if len(archive) == 0:
        return None
    nov = np.array([e.novelty for e in archive.entries], dtype=np.float32)
    i = int(sample_by_novelty(nov, 1, rng, alpha)[0])
    return Goal("novelty", "", None, seed_index=i)
