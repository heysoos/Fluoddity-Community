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
    kind: str                 # "latent" | "text" | "chase"
    text: str
    embedding: np.ndarray     # (dim,) float32, unit norm

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


def latent_goal(archive, rng, alpha: float = 4.0, beta: float = 0.5) -> Goal | None:
    """Extrapolate past the frontier: g = normalise(b + beta*(b - c)).

    'Keep going in the direction that already looks unlike everything else.'
    The source b is drawn with p proportional to NOV^alpha, so the goal is
    anchored on a genuinely novel entry rather than a random one.
    """
    if len(archive) == 0:
        return None
    c = archive.centroid()
    if c is None:
        return None
    nov = np.array([e.novelty for e in archive.entries], dtype=np.float32)
    i = int(sample_by_novelty(nov, 1, rng, alpha)[0])
    b = archive.embeddings[i]
    g = b + float(beta) * (b - c)
    nrm = float(np.linalg.norm(g))
    g = (g / nrm) if nrm > 1e-6 else b.copy()
    return Goal("latent", "", g.astype(np.float32))
