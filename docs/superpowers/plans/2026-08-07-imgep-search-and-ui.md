# IMGEP Search and Archive UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Expedition & Expansion exploration loop on top of the archive foundation, expose it as a third tournament mode, and make the archive browsable as a thumbnail gallery and a 2-D semantic map.

**Architecture:** Stages 3–4 of `docs/superpowers/specs/2026-08-07-imgep-novelty-archive-design.md`. `ImgepDriver` is a `SearchDriver`, so it reuses `AutoTournamentService`'s frame-spread rollout machine untouched. Expansion samples archive parents with `p ∝ NOV^α` and mutates them independently per tile; expeditions build a fresh CMA-ES per goal.

**Tech Stack:** Python 3.12, NumPy, Pillow, ModernGL (thumbnail textures only), imgui_bundle, pytest.

## Global Constraints

- **Prerequisite:** `docs/superpowers/plans/2026-08-07-imgep-archive-foundation.md` is complete and its Task 1 spread check printed `VERDICT: PASS`.
- **No new runtime dependencies.** PCA is a 512×512 eigendecomposition in NumPy. No `scikit-learn`, no `umap-learn`.
- **`sim.py` and every file in `shaders/` are untouched.**
- **Manual and Auto (CLIP) modes must behave identically.** The eight gate files from foundation Task 7 stay green and unmodified.
- **CLIP/evolution imports stay lazy.** Never imported at app startup.
- **UI is passive.** Widgets render and set one-shot flags; `CommandHandler` clears them. No logic in `ui/`.
- **Embeddings are L2-normalised float32, shape `(n, 512)`.** Higher novelty/fitness is better.
- **Windows platform.** Forward slashes or `os.path`; `rm` not `del` in bash.
- Run tests with `python -m pytest` from the repo root.

---

## File Structure

| File | Responsibility |
|---|---|
| `services/archive_projection.py` | PCA to 2-D with sign alignment across refits |
| `services/goal_source.py` | `Goal`, `GoalList` (user text goals), `latent_goal()` |
| `services/imgep_driver.py` | Bootstrap / expansion / expedition; the admission call site |
| `state/archive_state.py` | `ArchiveState` — Explore settings and one-shot flags |
| `ui/auto_tournament_window.py` | *(modify)* factor out the shared rollout controls |
| `ui/tournament_window.py` | *(modify)* third tab |
| `ui/archive_window.py` | Explore tab body, gallery, map |
| `main.py` | *(modify)* build `ImgepDriver` lazily, swap drivers on mode change |
| `command_handler.py` | *(modify)* Explore one-shot flags |

---

### Task 1: `archive_projection.py` — PCA with sign alignment

Spec §8.3. The projection is a *view*; nothing in the search depends on it.

**Files:**
- Create: `services/archive_projection.py`
- Test: `tests/test_archive_projection.py`

**Interfaces:**
- Produces: `Projection(n_components=2)` with `.components` (`(2, dim)` or `None`), `.mean` (`(dim,)` or `None`), `.fitted` (bool), `.fit(embeddings) -> bool`, `.transform(embeddings) -> (n, 2)`, `.fit_transform(embeddings)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_archive_projection.py
import numpy as np
import pytest

from services.archive_projection import Projection


def _blob(n, dim, rng, scale):
    return (rng.normal(size=(n, dim)) * scale).astype(np.float32)


def test_transform_before_fit_returns_zeros():
    p = Projection()
    out = p.transform(np.zeros((5, 8), np.float32))
    assert out.shape == (5, 2)
    assert np.all(out == 0.0)
    assert p.fitted is False


def test_fit_needs_more_rows_than_components():
    p = Projection(n_components=2)
    assert p.fit(np.zeros((1, 8), np.float32)) is False
    assert p.fitted is False


def test_components_are_orthonormal():
    rng = np.random.default_rng(0)
    p = Projection()
    p.fit(_blob(200, 16, rng, 1.0))
    c = p.components
    assert c.shape == (2, 16)
    assert np.allclose(c @ c.T, np.eye(2), atol=1e-4)


def test_the_first_component_follows_the_direction_of_greatest_variance():
    rng = np.random.default_rng(1)
    x = _blob(300, 4, rng, 0.01)
    x[:, 2] += rng.normal(size=300) * 5.0          # variance dominates axis 2
    p = Projection()
    p.fit(x.astype(np.float32))
    assert abs(p.components[0, 2]) > 0.9


def test_transform_is_centred_and_deterministic():
    rng = np.random.default_rng(2)
    x = _blob(100, 8, rng, 1.0)
    p = Projection()
    p.fit(x)
    a = p.transform(x)
    b = p.transform(x)
    assert a.shape == (100, 2)
    assert np.array_equal(a, b)
    assert np.allclose(a.mean(axis=0), 0.0, atol=1e-4)


def test_a_refit_sign_aligns_to_the_previous_components():
    """Eigenvectors have arbitrary sign. Without alignment the map mirrors
    itself at every refit and reads as a bug."""
    rng = np.random.default_rng(3)
    x = _blob(200, 8, rng, 1.0)
    p = Projection()
    p.fit(x)
    first = p.components.copy()
    before = p.transform(x[:20])

    p.fit(np.concatenate([x, _blob(20, 8, rng, 1.0)]))
    assert np.all(np.sum(first * p.components, axis=1) > 0), "signs must agree"
    after = p.transform(x[:20])
    # a small data change must not flip the layout
    assert np.mean(np.sign(before) == np.sign(after)) > 0.8


def test_a_deliberately_flipped_refit_is_corrected():
    rng = np.random.default_rng(4)
    x = _blob(200, 6, rng, 1.0)
    p = Projection()
    p.fit(x)
    p._prev = -p.components.copy()       # pretend the last fit was the mirror
    p.fit(x)
    assert np.all(np.sum(p._prev * p.components, axis=1) > 0)


def test_fit_transform_matches_fit_then_transform():
    rng = np.random.default_rng(5)
    x = _blob(60, 8, rng, 1.0)
    a = Projection().fit_transform(x)
    p = Projection()
    p.fit(x)
    assert np.allclose(a, p.transform(x), atol=1e-6)


def test_degenerate_input_does_not_raise():
    """Every embedding identical - zero variance in every direction."""
    x = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (50, 1))
    p = Projection()
    p.fit(x)
    out = p.transform(x)
    assert out.shape == (50, 2)
    assert np.all(np.isfinite(out))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_archive_projection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.archive_projection'`

- [ ] **Step 3: Write the implementation**

```python
# services/archive_projection.py
"""2-D PCA of the archive, for the map view.

Via the eigendecomposition of the dim x dim covariance rather than an SVD of the
data: at dim=512 that is a fixed 512x512 problem regardless of archive size, so
a refit costs the same at 200 entries as at 20,000.

This is a VIEW. Nothing in the search depends on it - novelty is always computed
in the full 512-d space.
"""
from __future__ import annotations

import numpy as np


class Projection:
    def __init__(self, n_components: int = 2):
        self.n_components = int(n_components)
        self.components: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self._prev: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.components is not None

    def fit(self, embeddings: np.ndarray) -> bool:
        """-> did it fit? False when there is not enough data yet."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] <= self.n_components:
            return False

        mean = x.mean(axis=0)
        centred = x - mean
        cov = (centred.T @ centred) / max(1, x.shape[0] - 1)
        # eigh, not eig: the covariance is symmetric, so this is both faster and
        # guaranteed to return real orthonormal vectors.
        vals, vecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(vals)[::-1][: self.n_components]
        comp = np.ascontiguousarray(vecs[:, order].T, dtype=np.float32)

        # Eigenvectors have arbitrary sign. Aligning each new component to the
        # previous one is what stops the map mirroring itself on every refit.
        if self._prev is not None and self._prev.shape == comp.shape:
            flip = np.sign(np.sum(self._prev * comp, axis=1))
            flip[flip == 0] = 1.0
            comp = comp * flip[:, None]

        self.components = comp
        self.mean = mean.astype(np.float32)
        self._prev = comp.copy()
        return True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if not self.fitted:
            return np.zeros((x.shape[0], self.n_components), dtype=np.float32)
        return ((x - self.mean) @ self.components.T).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        self.fit(embeddings)
        return self.transform(embeddings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_archive_projection.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add services/archive_projection.py tests/test_archive_projection.py
git commit -m "feat: 2-D archive projection with sign-aligned refits"
```

---

### Task 2: `goal_source.py` — latent goals and the user's text goal list

Spec §7. Two sources, both fully offline. The human is the goal generator that E&E delegates to a VLM.

**Files:**
- Create: `services/goal_source.py`
- Test: `tests/test_goal_source.py`

**Interfaces:**
- Consumes: `services.novelty.sample_by_novelty`, `services.archive.Archive`
- Produces:
  - `Goal(kind: str, text: str, embedding: np.ndarray)` — `kind` ∈ `{"latent", "text", "chase"}`
  - `GoalList(store=None)` with `.items: list[dict]`, `.add(text)`, `.remove(i)`, `.move(i, delta)`, `.set_enabled(i, bool)`, `.set_text(i, text)`, `.enabled_items() -> list[tuple[int, dict]]`, `.ensure_embedded(scorer)`, `.next_goal() -> Goal | None`, `.least_matched(embeddings) -> Goal | None`, `.save()`, `.load()`
  - `latent_goal(archive, rng, alpha=4.0, beta=0.5) -> Goal | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_goal_source.py
import numpy as np
import pytest

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore
from services.goal_source import Goal, GoalList, latent_goal


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


class FakeScorer:
    """One axis per distinct prompt, so goals are trivially distinguishable."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = 0

    def embed_text(self, prompts):
        self.calls += 1
        out = np.zeros((len(prompts), self.dim), dtype=np.float32)
        for i, p in enumerate(prompts):
            out[i, hash(p) % self.dim] = 1.0
        return _unit(out)


def archive_with(vectors, dim=4, novelties=None):
    a = Archive(store=None, dim=dim, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
    for j, v in enumerate(vectors):
        e = np.zeros(dim, dtype=np.float32)
        e[: len(v)] = v
        a.consider(
            Candidate(brain=np.zeros((10, 8), np.float32),
                      physics=np.zeros(8, np.float32),
                      embedding=_unit(e[None])[0], liveness=1.0, spec="brain:80"),
            novelty=(novelties[j] if novelties else 1.0),
        )
    return a


# ---- the goal list ------------------------------------------------------

def test_add_remove_and_reorder():
    g = GoalList()
    g.add("coral reef")
    g.add("lightning")
    assert [i["text"] for i in g.items] == ["coral reef", "lightning"]
    g.move(0, +1)
    assert [i["text"] for i in g.items] == ["lightning", "coral reef"]
    g.remove(0)
    assert [i["text"] for i in g.items] == ["coral reef"]


def test_move_at_the_boundary_is_a_no_op():
    g = GoalList()
    g.add("a")
    g.add("b")
    g.move(0, -1)
    g.move(1, +1)
    assert [i["text"] for i in g.items] == ["a", "b"]


def test_new_goals_are_enabled_and_can_be_disabled():
    g = GoalList()
    g.add("a")
    assert g.items[0]["enabled"] is True
    g.set_enabled(0, False)
    assert g.enabled_items() == []


def test_blank_and_duplicate_goals_are_ignored():
    g = GoalList()
    assert g.add("   ") is False
    assert g.add("coral") is True
    assert g.add("coral") is False
    assert len(g.items) == 1


def test_next_goal_cycles_round_robin_over_enabled_entries():
    g = GoalList()
    for t in ("a", "b", "c"):
        g.add(t)
    g.set_enabled(1, False)
    g.ensure_embedded(FakeScorer())
    assert [g.next_goal().text for _ in range(4)] == ["a", "c", "a", "c"]


def test_next_goal_is_none_when_the_list_is_empty_or_all_disabled():
    g = GoalList()
    assert g.next_goal() is None
    g.add("a")
    g.set_enabled(0, False)
    g.ensure_embedded(FakeScorer())
    assert g.next_goal() is None


def test_goal_embeddings_are_cached_until_the_text_changes():
    s = FakeScorer()
    g = GoalList()
    g.add("a")
    g.ensure_embedded(s)
    g.ensure_embedded(s)
    assert s.calls == 1, "re-embedding an unchanged list wastes a text pass"
    g.add("b")
    g.ensure_embedded(s)
    assert s.calls == 2


def test_a_goal_embedding_is_unit_norm():
    g = GoalList()
    g.add("coral")
    g.ensure_embedded(FakeScorer())
    goal = g.next_goal()
    assert goal.kind == "text"
    assert np.linalg.norm(goal.embedding) == pytest.approx(1.0, abs=1e-5)


def test_least_matched_picks_the_goal_the_archive_cannot_reach():
    g = GoalList()
    g.add("reachable")
    g.add("unreachable")
    g.ensure_embedded(FakeScorer())
    # build an archive sitting exactly on the first goal's axis
    reach = g._embeddings[0]
    a = archive_with([reach.tolist(), reach.tolist(), (reach * 0.9).tolist()])
    picked = g.least_matched(a.embeddings)
    assert picked.text == "unreachable"


def test_least_matched_is_none_with_an_empty_archive():
    g = GoalList()
    g.add("a")
    g.ensure_embedded(FakeScorer())
    assert g.least_matched(np.zeros((0, 4), np.float32)) is None


def test_goals_persist_through_the_store(tmp_path):
    store = ArchiveStore(tmp_path)
    g = GoalList(store=store)
    g.add("coral reef")
    g.set_enabled(0, False)
    g.save()
    store.close()

    g2 = GoalList(store=ArchiveStore(tmp_path))
    g2.load()
    assert g2.items == [{"text": "coral reef", "enabled": False}]


# ---- latent goals -------------------------------------------------------

def test_latent_goal_is_unit_norm_and_labelled():
    a = archive_with([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    goal = latent_goal(a, np.random.default_rng(0))
    assert goal.kind == "latent"
    assert goal.text == ""
    assert np.linalg.norm(goal.embedding) == pytest.approx(1.0, abs=1e-5)


def test_latent_goal_points_further_out_than_its_source():
    """The whole point: extrapolate past the frontier, do not sit on it."""
    a = archive_with([[1, 0, 0, 0], [1, 0.1, 0, 0], [1, 0.2, 0, 0], [0, 0, 1, 0]],
                     novelties=[0.1, 0.1, 0.1, 0.9])
    c = a.centroid()
    goal = latent_goal(a, np.random.default_rng(0), alpha=8.0, beta=0.5)
    src = a.embeddings[3]                      # the novel outlier is the source
    assert float(goal.embedding @ c) < float(src @ c)


def test_latent_goal_on_an_empty_archive_is_none():
    a = Archive(store=None, dim=4)
    assert latent_goal(a, np.random.default_rng(0)) is None


def test_latent_goal_on_a_single_entry_returns_that_entry():
    a = archive_with([[1, 0, 0, 0]])
    goal = latent_goal(a, np.random.default_rng(0))
    assert np.allclose(goal.embedding, a.embeddings[0], atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_goal_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.goal_source'`

- [ ] **Step 3: Write the implementation**

```python
# services/goal_source.py
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
comparison. least_matched() therefore z-scores.
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
        """The enabled goal the archive is least able to reach.

        Each goal's archive-max cosine is z-scored against that GOAL's OWN
        distribution over the archive. Comparing raw cosines across phrases
        would rank the modality gap, not the archive.
        """
        live = self.enabled_items()
        e = np.asarray(embeddings, dtype=np.float32)
        if not live or self._embeddings is None or len(e) == 0:
            return None
        best, best_score = None, np.inf
        for idx, item in live:
            g = self._embeddings[idx]
            sims = e @ g
            sd = float(sims.std())
            score = ((float(sims.max()) - float(sims.mean())) / sd) if sd > 1e-6 else 0.0
            if score < best_score:
                best, best_score = Goal("text", item["text"], g), score
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_goal_source.py -v`
Expected: PASS, 15 tests

- [ ] **Step 5: Commit**

```bash
git add services/goal_source.py tests/test_goal_source.py
git commit -m "feat: latent goals and a user-editable text goal list"
```

---

### Task 3: `imgep_driver.py` — bootstrap, expansion, and the admission call site

Spec §6.1, §6.2. Expeditions come in Task 4; this task ends with a driver that bootstraps and expands.

Note the `_sync_driver` extension: `AutoTournamentService._sync_driver()` currently pushes `algorithm`, `sigma0`, `base_seed`. `ImgepDriver` also needs `physics_origin`, `physics_enabled` and `run_id`. The loop is `hasattr`-guarded, so adding keys is invisible to `PromptDriver`.

**Files:**
- Create: `services/imgep_driver.py`
- Modify: `services/auto_tournament_service.py` (two small additions)
- Test: `tests/test_imgep_driver.py`

**Interfaces:**
- Consumes: `Archive`, `Candidate` (foundation Task 9), `descriptor`/`liveness`/`stack_snapshots` (Task 4), `sample_by_novelty` (Task 5), `is_viable_tile` (Task 3), `GoalList`/`latent_goal` (Task 2), `genome_spec.encode`, `physics_genome.encode_physics`
- Produces: `ImgepDriver(tournament, scorer, archive, goals=None, spec=BRAIN_SPEC, rng=None)` — a `SearchDriver` with extra attributes `sigma_expand`, `alpha`, `k`, `seed_n`, `liveness_min`, `refresh_per_gen`, `physics_origin`, `physics_enabled`, `run_id`, and read-only `regime`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_imgep_driver.py
import numpy as np
import pytest

from services.archive import Archive
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.imgep_driver import ImgepDriver
from services.tournament_service import TournamentService

DIM = 8


class FakeScorer:
    """Embeds each tile onto an axis chosen by its mean brightness, so tests
    control exactly where a tile lands in behaviour space."""

    def __init__(self, dim=DIM):
        self.dim = dim

    def embed(self, images, n_views=1):
        n = len(images)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, int(img.reshape(-1).mean()) % self.dim] = 1.0
        return out

    def embed_text(self, prompts):
        out = np.zeros((len(prompts), self.dim), dtype=np.float32)
        for i, p in enumerate(prompts):
            out[i, hash(p) % self.dim] = 1.0
        return out


def make(grid=2, **kw):
    ts = TournamentService(grid=grid)
    ts.init_population()
    arc = Archive(store=None, dim=DIM, seed_n=kw.pop("seed_n", 4),
                  liveness_min=kw.pop("liveness_min", 0.0), capacity=100)
    arc.threshold.value = 0.0
    d = ImgepDriver(ts, FakeScorer(), arc, rng=np.random.default_rng(0))
    for k, v in kw.items():
        setattr(d, k, v)
    return d, arc, ts


def snaps(n, values):
    """One (n,224,224,3) crop array per snapshot; values[s][i] sets tile i."""
    out = []
    for row in values:
        a = np.zeros((n, 224, 224, 3), dtype=np.uint8)
        for i, v in enumerate(row):
            a[i] = v
        out.append(a)
    return out


def moving(n, base=10):
    """Two snapshots that differ, so liveness is high."""
    return snaps(n, [[base + i for i in range(n)], [base + 40 + i for i in range(n)]])


# ---- regimes -----------------------------------------------------------

def test_starts_in_bootstrap_with_an_empty_archive():
    d, _, _ = make()
    assert d.regime == "bootstrap"


def test_bootstrap_asks_for_scattered_vectors_not_archive_children():
    d, _, _ = make()
    z = d.ask(4)
    assert z.shape == (4, BRAIN_SPEC.dim)
    assert z.dtype == np.float32
    assert z.std() > 0.1, "a bootstrap population must be scattered"


def test_regime_becomes_expansion_once_the_archive_reaches_seed_n():
    d, arc, _ = make(seed_n=4)
    z = d.ask(4)
    d.tell(z, moving(4))
    assert len(arc) == 4
    assert d.regime == "expansion"


def test_expansion_children_stay_near_their_parents():
    d, arc, _ = make(seed_n=4, sigma_expand=0.01)
    d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"
    z = d.ask(4)
    # every child decodes to a brain close to some archived parent
    for row in z:
        brain = BRAIN_SPEC.decode(row)["brain"]
        gaps = np.abs(arc.brains - brain).max(axis=(1, 2))
        assert gaps.min() < 0.3


def test_sigma_expand_controls_how_far_children_travel():
    near, arc_n, _ = make(seed_n=4, sigma_expand=0.01)
    near.tell(near.ask(4), moving(4))
    far, arc_f, _ = make(seed_n=4, sigma_expand=1.0)
    far.tell(far.ask(4), moving(4))
    assert far.ask(16).std() > near.ask(16).std()


# ---- tell / the admission call site ------------------------------------

def test_tell_admits_one_entry_per_viable_live_novel_tile():
    d, arc, _ = make(seed_n=0)
    arc.threshold.value = 0.0
    d.tell(d.ask(4), moving(4))
    assert len(arc) == 4


def test_a_dead_tile_is_rejected_by_the_viability_gate():
    d, arc, _ = make(seed_n=0)
    s = moving(4)
    for a in s:
        a[2] = 0                       # tile 2 is pure black
    d.tell(d.ask(4), s)
    assert len(arc) == 3
    assert all(e.tile != 2 for e in arc.entries)


def test_a_frozen_tile_is_rejected_by_the_liveness_gate():
    d, arc, _ = make(seed_n=0, liveness_min=0.5)
    arc.liveness_min = 0.5
    same = [v for v in range(10, 14)]
    d.tell(d.ask(4), snaps(4, [same, same]))    # identical snapshots
    assert len(arc) == 0


def test_the_liveness_gate_is_disabled_with_a_single_snapshot():
    """One snapshot is no evidence of change; rejecting everything would be
    wrong, not conservative."""
    d, arc, _ = make(seed_n=0, liveness_min=0.5)
    d.liveness_min = 0.5
    d.tell(d.ask(4), snaps(4, [[10, 11, 12, 13]]))
    assert len(arc) == 4


def test_tell_with_no_snapshots_returns_zeros_and_admits_nothing():
    d, arc, _ = make()
    assert np.array_equal(d.tell(d.ask(4), []), np.zeros(4, np.float32))
    assert len(arc) == 0


def test_expansion_returns_novelty_as_the_display_score():
    d, arc, _ = make(seed_n=0)
    out = d.tell(d.ask(4), moving(4))
    assert out.shape == (4,)
    assert np.all(out >= 0.0)
    assert d.status()["score_label"] == "novelty"


def test_a_selected_tile_is_pinned_and_the_selection_is_cleared():
    d, arc, ts = make(seed_n=0, liveness_min=0.9)
    arc.liveness_min = 0.9
    ts.toggle_select(1)
    same = [10, 11, 12, 13]
    d.tell(d.ask(4), snaps(4, [same, same]))     # everything frozen
    assert len(arc) == 1, "only the pinned tile gets in"
    assert arc.entries[0].pinned is True
    assert arc.entries[0].tile == 1
    assert ts.selected == set()


def test_entries_record_their_generation_tile_and_spec():
    d, arc, _ = make(seed_n=0)
    d.gen = 7
    d.tell(d.ask(4), moving(4))
    assert {e.tile for e in arc.entries} == {0, 1, 2, 3}
    assert all(e.gen == 7 for e in arc.entries)
    assert all(e.spec == "brain:80" for e in arc.entries)


def test_refresh_runs_every_generation():
    d, arc, _ = make(seed_n=0, refresh_per_gen=4)
    d.tell(d.ask(4), moving(4))
    before = [e.novelty for e in arc.entries]
    d.tell(d.ask(4), moving(4, base=100))
    assert [e.novelty for e in arc.entries[:4]] != before


# ---- physics -----------------------------------------------------------

def test_physics_genes_are_stored_as_absolute_values():
    from services.physics_genome import PHYSICS_PARAMS, decode_physics

    d, arc, _ = make(seed_n=0)
    d.set_spec(BRAIN_PHYSICS_SPEC)
    d.physics_enabled = True
    d.physics_origin = {n: (lo + hi) / 2 for n, _g, lo, hi in PHYSICS_PARAMS}
    z = d.ask(4)
    d.tell(z, moving(4))
    expected = decode_physics(z[0][BRAIN_SPEC.dim:], d.physics_origin)
    stored = arc.physics[[e.tile for e in arc.entries].index(0)]
    for j, (n, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS):
        assert stored[j] == pytest.approx(expected[n], abs=1e-4)
    assert all(e.spec == "brain:80,physics:8" for e in arc.entries)


def test_a_brain_only_parent_gets_zero_physics_genes_in_a_physics_run():
    """z=0 for the physics block decodes to the current origin exactly, so a
    brain-only archive entry is well-defined rather than an error."""
    d, arc, _ = make(seed_n=4)
    d.tell(d.ask(4), moving(4))                  # 4 brain-only entries
    d.set_spec(BRAIN_PHYSICS_SPEC)
    d.physics_enabled = True
    d.sigma_expand = 0.0                         # no mutation, so z is exact
    z = d.ask(4)
    assert np.allclose(z[:, BRAIN_SPEC.dim:], 0.0, atol=1e-6)


def test_set_spec_changes_the_ask_dimension():
    d, _, _ = make()
    assert d.ask(4).shape[1] == BRAIN_SPEC.dim
    d.set_spec(BRAIN_PHYSICS_SPEC)
    assert d.ask(4).shape[1] == BRAIN_PHYSICS_SPEC.dim


# ---- lifecycle ---------------------------------------------------------

def test_reset_clears_the_search_but_keeps_the_archive():
    d, arc, _ = make(seed_n=0)
    d.tell(d.ask(4), moving(4))
    n = len(arc)
    d.reset()
    assert len(arc) == n, "the archive is the product; reset is about the search"
    assert d.regime in ("bootstrap", "expansion")


def test_status_reports_what_the_ui_needs():
    d, _, _ = make()
    st = d.status()
    assert set(st) >= {"regime", "goal", "archive_size", "threshold",
                       "admission_rate", "score_label", "sigma"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_imgep_driver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.imgep_driver'`

- [ ] **Step 3: Write the driver (bootstrap + expansion only; expeditions are Task 4)**

```python
# services/imgep_driver.py
"""Intrinsically-motivated goal exploration over the archive.

Expedition & Expansion (arXiv:2509.03863) on a substrate that runs N^2 rollouts
per generation. E&E's loop is serial - one theta, one rollout, one embedding -
so one generation here is a batch of 4 to 64 IMGEP samples at no extra cost.

Three regimes, checked in order:

  BOOTSTRAP    archive < seed_n. z ~ N(x0, sigma0): a scattered population, and
               the archive admits on viability and liveness only (5.2.2).
  EXPEDITION   an expedition is active: optimizer.ask, fitness <b, g>.
  EXPANSION    otherwise: per tile, sample a parent with p ~ NOV^alpha,
               re-encode it under the CURRENT physics origin, add Gaussian
               noise. No optimizer, no covariance, no shared state between
               tiles - the population is a sample, not a distribution.

Deliberate deviation from E&E: every expedition generation's tiles also go
through the admission gate, not just the endpoint. E&E archives only the final
optimised solution because its expeditions are a serial inner loop; here the
embeddings are already computed and the path toward a goal is itself territory.
"""
from __future__ import annotations

import numpy as np

from services.archive import Candidate
from services.capture_health import is_viable_tile
from services.descriptor import descriptor, liveness, stack_snapshots
from services.genome_spec import BRAIN_SPEC, encode
from services.novelty import sample_by_novelty
from services.physics_genome import PHYSICS_DIM, PHYSICS_PARAMS, encode_physics


def _phys_dict(vec) -> dict[str, float]:
    return {n: float(v) for (n, _g, _lo, _hi), v in zip(PHYSICS_PARAMS, vec)}


class ImgepDriver:
    name = "imgep"

    def __init__(self, tournament, scorer, archive, goals=None,
                 spec=BRAIN_SPEC, rng=None):
        self.tournament = tournament
        self.scorer = scorer
        self.archive = archive
        self.goals = goals
        self.spec = spec
        self.rng = rng if rng is not None else np.random.default_rng()

        # pushed by AutoTournamentService._sync_driver
        self.algorithm = "CMA-ES"
        self.sigma0 = 0.5
        self.base_seed = 1000
        self.physics_origin: dict[str, float] = {}
        self.physics_enabled = False
        self.run_id = ""

        # exploration settings (spec 7.4)
        self.sigma_expand = 0.15
        self.alpha = 4.0
        self.k = 10
        self.seed_n = 256
        self.liveness_min = 0.02
        self.refresh_per_gen = 64
        self.flush_every = 200

        self.gen = 0
        self._last_score_label = "novelty"

    # ---- state ---------------------------------------------------------

    @property
    def regime(self) -> str:
        if len(self.archive) < self.seed_n:
            return "bootstrap"
        return "expansion"

    @property
    def optimizer(self):
        return None

    @property
    def sigma(self) -> float:
        return float(self.sigma_expand)

    @property
    def goal_label(self) -> str:
        return ""

    def status(self) -> dict:
        st = self.archive.stats()
        return {
            "regime": self.regime,
            "goal": self.goal_label,
            "archive_size": st["size"],
            "threshold": st["threshold"],
            "admission_rate": st["admission_rate"],
            "n_pinned": st["n_pinned"],
            "blocked_by_pins": st["blocked_by_pins"],
            "score_label": self._last_score_label,
            "sigma": self.sigma,
            "algorithm": self.algorithm,
            "prompt": self.goal_label,
        }

    # ---- driver interface ----------------------------------------------

    def set_spec(self, spec) -> None:
        self.spec = spec

    def reset(self) -> None:
        """Clears the SEARCH, never the archive. The archive is the product;
        Reset is about abandoning the current trajectory through it."""
        self.gen = 0

    def ask(self, n: int) -> np.ndarray:
        n = int(n)
        if self.regime == "bootstrap":
            return self._ask_bootstrap(n)
        return self._ask_expansion(n)

    def _ask_bootstrap(self, n: int) -> np.ndarray:
        return (self.sigma0 * self.rng.normal(size=(n, self.spec.dim))
                ).astype(np.float32)

    def _ask_expansion(self, n: int) -> np.ndarray:
        nov = np.array([e.novelty for e in self.archive.entries], dtype=np.float32)
        idx = sample_by_novelty(nov, n, self.rng, self.alpha)
        out = np.empty((n, self.spec.dim), dtype=np.float32)
        for j, i in enumerate(idx):
            out[j] = self._parent_z(int(i))
        out += (self.sigma_expand * self.rng.normal(size=out.shape)).astype(np.float32)
        return out

    def _parent_z(self, i: int) -> np.ndarray:
        """Re-encode an archived PHENOTYPE into z under the CURRENT origin.

        The archive stores decoded values precisely so this works: a z archived
        under one preset would mean a different creature under another."""
        zb, _clamped = encode(self.archive.brains[i])
        if self.spec.dim <= len(zb):
            return zb[: self.spec.dim].astype(np.float32)
        entry = self.archive.entries[i]
        if "physics" in entry.spec:
            zp = encode_physics(_phys_dict(self.archive.physics[i]),
                                self.physics_origin)
        else:
            # z = 0 decodes to the current origin exactly, so a brain-only
            # entry is well-defined rather than an error.
            zp = np.zeros(PHYSICS_DIM, dtype=np.float32)
        return np.concatenate([zb, zp]).astype(np.float32)

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray]) -> np.ndarray:
        n = len(z)
        if not snapshots:
            return np.zeros(n, dtype=np.float32)

        per_snap = [np.asarray(self.scorer.embed(c, n_views=1), dtype=np.float32)
                    for c in snapshots]
        snaps = stack_snapshots(per_snap)
        b = descriptor(snaps)
        live = liveness(snaps)
        nov = self.archive.novelty_of(b)

        # One snapshot is no evidence of change; rejecting everything would be
        # wrong rather than conservative.
        self.archive.liveness_min = (
            float(self.liveness_min) if len(snapshots) >= 2 else 0.0)
        self.archive.k = int(self.k)
        self.archive.seed_n = int(self.seed_n)

        last = snapshots[-1]
        pinned = set(self.tournament.selected)
        source = self.regime
        goal_text = self.goal_label
        parts = [self.spec.decode(zi) for zi in z]

        for i in range(n):
            phys = parts[i].get("physics")
            if phys is not None and self.physics_enabled:
                from services.physics_genome import decode_physics

                values = decode_physics(phys, self.physics_origin)
                phys_vec = np.array([values[nm] for nm, _g, _lo, _hi in PHYSICS_PARAMS],
                                    dtype=np.float32)
            else:
                phys_vec = np.zeros(PHYSICS_DIM, dtype=np.float32)

            self.archive.consider(
                Candidate(
                    brain=np.asarray(parts[i]["brain"], dtype=np.float32),
                    physics=phys_vec,
                    embedding=b[i],
                    liveness=float(live[i]),
                    viable=bool(is_viable_tile(last[i])),
                    spec=self.spec.signature(),
                    gen=int(self.gen),
                    tile=int(i),
                    run_id=self.run_id,
                    goal=goal_text,
                ),
                float(nov[i]),
                pinned=(i in pinned),
                source=source,
                thumb_crop=last[i],
            )

        self.tournament.selected.clear()
        self.gen += 1
        self.archive.refresh(self.refresh_per_gen)
        self.archive.maybe_flush(every=self.flush_every)

        self._last_score_label = "novelty"
        return np.asarray(nov, dtype=np.float32)

    # ---- checkpointing -------------------------------------------------

    def checkpoint_state(self) -> dict:
        return {
            "optimizer_name": self.algorithm,
            "optimizer_state": {},
            "prompt": self.goal_label,
            "distractors": [],
            "best_z": np.zeros(self.spec.dim, dtype=np.float32),
            "best_fitness": 0.0,
            "imgep_gen": int(self.gen),
            "imgep_threshold": float(self.archive.threshold.value),
        }

    def restore(self, d: dict) -> None:
        self.gen = int(d.get("imgep_gen", 0))
        if "imgep_threshold" in d:
            self.archive.threshold.value = float(d["imgep_threshold"])
```

- [ ] **Step 4: Extend `_sync_driver` and add `run_id`**

In `services/auto_tournament_service.py`, replace the `_sync_driver` loop tuple and add a `run_id` property:

```python
    @property
    def run_id(self) -> str:
        return getattr(self.logger, "run_id", "") if self.logger else ""

    def _sync_driver(self) -> None:
        """Push the UI-owned settings the driver understands, and the current
        genome layout. Only attributes the driver already has are set, so
        drivers may ignore settings that mean nothing to them."""
        for k in ("algorithm", "sigma0", "base_seed",
                  "physics_origin", "physics_enabled", "run_id"):
            if hasattr(self.driver, k):
                setattr(self.driver, k, getattr(self, k))
        self.driver.set_spec(self.spec)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_imgep_driver.py -v`
Expected: PASS, 19 tests

- [ ] **Step 6: Confirm Auto mode is still untouched**

Run:

```bash
python -m pytest tests/test_auto_tournament_service.py tests/test_search_driver.py -v
```

Expected: PASS, both files unmodified. `_sync_driver`'s new keys are `hasattr`-guarded, so `PromptDriver` never sees them.

- [ ] **Step 7: Commit**

```bash
git add services/imgep_driver.py services/auto_tournament_service.py tests/test_imgep_driver.py
git commit -m "feat: ImgepDriver bootstrap and novelty-driven expansion"
```

---

### Task 4: Expeditions and "Chase this"

Spec §6.3, §7.3. A fresh CMA-ES per goal — nothing is carried across goals, because a covariance learned climbing toward "coral reef" says nothing about "lightning".

**Files:**
- Modify: `services/imgep_driver.py`
- Test: `tests/test_imgep_driver.py` (append)

**Interfaces:**
- Consumes: `services.optimizers.make_optimizer`, `GoalList`/`latent_goal` (Task 2)
- Produces: on `ImgepDriver` — settings `expansion_between`, `expedition_gens`, `expedition_sigma`, `latent_share`, `beta`, `goal_order`; methods `chase(tile) -> bool`, `end_expedition()`; `regime` now also returns `"expedition"`

- [ ] **Step 1: Write the failing tests (append to the existing file)**

```python
# tests/test_imgep_driver.py  --  APPEND
from services.goal_source import GoalList


def seeded(n_entries=8, **kw):
    """A driver past bootstrap, with a populated archive."""
    d, arc, ts = make(seed_n=4, **kw)
    d.tell(d.ask(4), moving(4))
    d.tell(d.ask(4), moving(4, base=100))
    return d, arc, ts


def test_no_expedition_fires_while_expansion_between_is_zero():
    d, _, _ = seeded(expansion_between=0)
    for _ in range(10):
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"


def test_an_expedition_fires_after_expansion_between_generations():
    d, _, _ = seeded(expansion_between=2, expedition_gens=5, latent_share=1.0)
    d.expansion_between = 2
    for _ in range(3):
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expedition"


def test_an_expedition_lasts_exactly_expedition_gens_generations():
    d, _, _ = seeded(expansion_between=1, expedition_gens=3, latent_share=1.0)
    d.expansion_between = 1
    d.expedition_gens = 3
    while d.regime != "expedition":
        d.tell(d.ask(4), moving(4))
    for _ in range(3):
        assert d.regime == "expedition"
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"


def test_an_expedition_builds_a_fresh_optimizer_per_goal():
    d, _, _ = seeded(expansion_between=1, expedition_gens=2, latent_share=1.0)
    d.expansion_between = 1
    d.expedition_gens = 2
    while d.regime != "expedition":
        d.tell(d.ask(4), moving(4))
    first = d.optimizer
    assert first is not None and first.popsize == 4
    for _ in range(2):
        d.tell(d.ask(4), moving(4))
    while d.regime != "expedition":
        d.tell(d.ask(4), moving(4))
    assert d.optimizer is not first, "a new goal must not inherit a covariance"


def test_an_expedition_seeds_at_the_archive_entry_nearest_the_goal():
    d, arc, _ = seeded(latent_share=1.0)
    goal = arc.embeddings[2].copy()
    d.start_expedition_with(goal, kind="chase", text="")
    assert d._x0_index == 2


def test_expedition_fitness_is_alignment_with_the_goal():
    d, arc, _ = seeded(latent_share=1.0, expedition_gens=5)
    goal = np.zeros(DIM, dtype=np.float32)
    goal[3] = 1.0
    d.start_expedition_with(goal, kind="latent", text="")
    z = d.ask(4)
    # FakeScorer puts tile i on axis (mean brightness % DIM); brightness 3 -> axis 3
    out = d.tell(z, snaps(4, [[1, 2, 3, 4], [41, 42, 43, 44]]))
    assert d.status()["score_label"] == "goal match"
    assert int(np.argmax(out)) == 2, "the tile landing on the goal axis wins"


def test_expedition_tiles_still_reach_the_archive():
    """Deliberate deviation from E&E: the path to a goal is territory too."""
    d, arc, _ = seeded(expedition_gens=5)
    before = len(arc)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.tell(d.ask(4), moving(4, base=150))
    assert len(arc) > before
    assert any(e.source == "expedition" for e in arc.entries)


def test_latent_share_of_one_always_picks_a_latent_goal():
    g = GoalList()
    g.add("coral")
    g.ensure_embedded(FakeScorer())
    d, _, _ = seeded(latent_share=1.0)
    d.goals = g
    d.start_expedition()
    assert d._goal.kind == "latent"


def test_latent_share_of_zero_uses_the_text_goal_list():
    g = GoalList()
    g.add("coral")
    g.ensure_embedded(FakeScorer())
    d, _, _ = seeded(latent_share=0.0)
    d.goals = g
    d.start_expedition()
    assert d._goal.kind == "text"
    assert d._goal.text == "coral"


def test_an_empty_goal_list_falls_back_to_latent():
    d, _, _ = seeded(latent_share=0.0)
    d.goals = GoalList()
    d.start_expedition()
    assert d._goal.kind == "latent"


def test_goal_order_least_matched_is_selectable():
    g = GoalList()
    g.add("a")
    g.add("b")
    g.ensure_embedded(FakeScorer())
    d, _, _ = seeded(latent_share=0.0)
    d.goals = g
    d.goal_order = "least_matched"
    d.start_expedition()
    assert d._goal.kind == "text"


def test_an_expedition_on_an_empty_archive_falls_back_to_expansion():
    d, arc, _ = make(seed_n=0)
    d.goals = GoalList()
    assert d.start_expedition() is False
    assert d.regime == "bootstrap"


def test_chase_starts_an_expedition_on_that_tiles_descriptor():
    d, arc, _ = seeded()
    d.tell(d.ask(4), snaps(4, [[1, 2, 3, 4], [41, 42, 43, 44]]))
    assert d.chase(2) is True
    assert d.regime == "expedition"
    assert d._goal.kind == "chase"
    assert float(d._goal.embedding[3]) == pytest.approx(1.0, abs=1e-5)


def test_chase_before_any_generation_is_refused():
    d, _, _ = make()
    assert d.chase(0) is False


def test_chase_pre_empts_the_expansion_cadence():
    d, _, _ = seeded(expansion_between=1000)
    d.expansion_between = 1000
    d.tell(d.ask(4), moving(4))
    d.chase(0)
    assert d.regime == "expedition"


def test_the_goal_label_appears_in_status_and_on_entries():
    g = GoalList()
    g.add("coral reef")
    g.ensure_embedded(FakeScorer())
    d, arc, _ = seeded(latent_share=0.0, expedition_gens=5)
    d.goals = g
    d.start_expedition()
    assert d.status()["goal"] == "coral reef"
    d.tell(d.ask(4), moving(4, base=150))
    assert any(e.goal == "coral reef" for e in arc.entries)


def test_a_grid_change_ends_the_expedition_rather_than_crashing():
    """cmaes fixes popsize at construction and asserts on it in tell()."""
    d, arc, ts = seeded(expedition_gens=50)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.ask(4)
    ts.set_grid(4)
    z = d.ask(16)
    assert z.shape == (16, BRAIN_SPEC.dim)
    d.tell(z, moving(16))          # must not raise


def test_set_spec_ends_an_active_expedition():
    d, arc, _ = seeded(expedition_gens=50)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.set_spec(BRAIN_PHYSICS_SPEC)
    assert d.regime != "expedition"
    assert d.optimizer is None


def test_reset_ends_an_expedition_but_keeps_the_archive():
    d, arc, _ = seeded(expedition_gens=50)
    n = len(arc)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.reset()
    assert d.regime != "expedition"
    assert len(arc) == n
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_imgep_driver.py -v`
Expected: the nineteen Task-3 tests PASS; the nineteen new ones FAIL with `AttributeError: 'ImgepDriver' object has no attribute 'start_expedition'`

- [ ] **Step 3: Add the expedition settings and state to `__init__`**

In `services/imgep_driver.py`, add to the imports:

```python
from services.goal_source import Goal, latent_goal
from services.optimizers import make_optimizer
```

and append to `__init__`, after `self.flush_every = 200`:

```python
        # expedition settings (spec 7.4)
        self.expansion_between = 25      # 0 disables expeditions entirely
        self.expedition_gens = 50        # E&E uses 350; that is ~16 min at 2000 steps
        self.expedition_sigma = 0.1      # E&E's value; deliberately << sigma0
        self.latent_share = 0.5
        self.beta = 0.5                  # latent extrapolation distance
        self.goal_order = "round_robin"  # or "least_matched"

        self._optimizer = None
        self._goal: Goal | None = None
        self._remaining = 0
        self._since_expedition = 0
        self._x0_index: int | None = None
        self._last_descriptors: np.ndarray | None = None
```

- [ ] **Step 4: Replace `regime`, `optimizer`, `sigma`, `goal_label`, `set_spec`, `reset` and `ask`**

```python
    @property
    def regime(self) -> str:
        if self._remaining > 0 and self._goal is not None:
            return "expedition"
        if len(self.archive) < self.seed_n:
            return "bootstrap"
        return "expansion"

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def sigma(self) -> float:
        if self._optimizer is not None:
            return float(self._optimizer.sigma)
        return float(self.sigma_expand)

    @property
    def goal_label(self) -> str:
        return self._goal.label if self._goal is not None else ""

    def set_spec(self, spec) -> None:
        if spec is not self.spec:
            self.spec = spec
            # The search dimension changed; an optimizer for the old one is
            # meaningless, and the archive is unaffected because it stores
            # phenotypes rather than z.
            self.end_expedition()

    def reset(self) -> None:
        """Clears the SEARCH, never the archive. The archive is the product;
        Reset is about abandoning the current trajectory through it."""
        self.gen = 0
        self._since_expedition = 0
        self.end_expedition()

    def end_expedition(self) -> None:
        self._optimizer = None
        self._goal = None
        self._remaining = 0
        self._x0_index = None

    def ask(self, n: int) -> np.ndarray:
        n = int(n)
        if self.regime == "expedition":
            return self._ask_expedition(n)
        if self.regime == "bootstrap":
            return self._ask_bootstrap(n)
        return self._ask_expansion(n)

    def _ask_expedition(self, n: int) -> np.ndarray:
        # The population size is fixed at construction (cmaes asserts on it in
        # tell()). A grid change mid-expedition ends the expedition rather than
        # crashing on the next tell.
        if self._optimizer is not None and self._optimizer.popsize != n:
            self.end_expedition()
            return self._ask_expansion(n) if len(self.archive) else self._ask_bootstrap(n)
        return self._optimizer.ask(n)
```

- [ ] **Step 5: Add expedition start, and the goal draw**

```python
    def start_expedition(self) -> bool:
        """Draw a goal from the configured sources and begin. -> did it start?"""
        goal = self._draw_goal()
        if goal is None:
            return False
        return self.start_expedition_with(goal.embedding, goal.kind, goal.text)

    def start_expedition_with(self, embedding, kind: str, text: str) -> bool:
        """Begin an expedition toward a specific embedding. An expedition needs
        a seed, so an empty archive falls back to expansion rather than
        starting a search from nowhere."""
        i = self.archive.nearest(np.asarray(embedding, dtype=np.float32))
        if i is None or self.expedition_gens <= 0:
            return False
        self._goal = Goal(kind, text, np.asarray(embedding, dtype=np.float32))
        self._x0_index = int(i)
        self._remaining = int(self.expedition_gens)
        self._since_expedition = 0
        # A FRESH optimizer per goal: a covariance learned climbing toward
        # "coral reef" is not informative about "lightning". sigma is
        # deliberately much smaller than sigma0 - an expedition is a local
        # refinement from an already-relevant seed, not a fresh search.
        self._optimizer = make_optimizer(
            self.algorithm, self.spec.dim, self.tournament.tiles,
            self.expedition_sigma, self.base_seed + self.gen,
            self._parent_z(self._x0_index).astype(np.float64),
        )
        return True

    def _draw_goal(self) -> Goal | None:
        want_latent = float(self.rng.random()) < float(self.latent_share)
        text_goal = None
        if self.goals is not None:
            self.goals.ensure_embedded(self.scorer)
            text_goal = (self.goals.least_matched(self.archive.embeddings)
                         if self.goal_order == "least_matched"
                         else self.goals.next_goal())
        if want_latent or text_goal is None:
            return latent_goal(self.archive, self.rng, self.alpha, self.beta) or text_goal
        return text_goal

    def chase(self, tile: int) -> bool:
        """Start an expedition toward one tile's own descriptor.

        The direct analogue of manual mode's 'more like that one', expressed as
        a goal rather than a selection. The descriptor is already computed for
        that generation, so this costs no extra CLIP work."""
        d = self._last_descriptors
        if d is None or not (0 <= int(tile) < len(d)):
            return False
        return self.start_expedition_with(d[int(tile)].copy(), "chase", "")
```

- [ ] **Step 6: Update `tell` to score against the goal and drive the cadence**

Replace the tail of `tell` — everything from `self.tournament.selected.clear()` onward — with:

```python
        self.tournament.selected.clear()
        self._last_descriptors = b
        self.gen += 1
        self.archive.refresh(self.refresh_per_gen)
        self.archive.maybe_flush(every=self.flush_every)

        if self.regime == "expedition":
            fit = (b @ self._goal.embedding).astype(np.float32)
            self._optimizer.tell(z, fit)
            self._remaining -= 1
            if self._remaining <= 0:
                self.end_expedition()
            self._last_score_label = "goal match"
            return fit

        self._since_expedition += 1
        if (self.expansion_between > 0
                and self._since_expedition >= self.expansion_between
                and len(self.archive) >= self.seed_n):
            self.start_expedition()

        self._last_score_label = "novelty"
        return np.asarray(nov, dtype=np.float32)
```

Also set `source` before the loop so expedition tiles are labelled correctly — it is already `self.regime`, evaluated while `_remaining > 0`, which is what makes expedition entries record `source="expedition"`.

- [ ] **Step 7: Update `checkpoint_state` / `restore` to carry the cadence**

```python
    def checkpoint_state(self) -> dict:
        return {
            "optimizer_name": self.algorithm,
            "optimizer_state": {},
            "prompt": self.goal_label,
            "distractors": [],
            "best_z": np.zeros(self.spec.dim, dtype=np.float32),
            "best_fitness": 0.0,
            "imgep_gen": int(self.gen),
            "imgep_threshold": float(self.archive.threshold.value),
            "imgep_since_expedition": int(self._since_expedition),
        }

    def restore(self, d: dict) -> None:
        # An expedition is deliberately NOT resumed: its optimizer is one
        # goal's local refinement, and the archive - which is the thing worth
        # preserving - is on disk independently of any checkpoint.
        self.end_expedition()
        self.gen = int(d.get("imgep_gen", 0))
        self._since_expedition = int(d.get("imgep_since_expedition", 0))
        if "imgep_threshold" in d:
            self.archive.threshold.value = float(d["imgep_threshold"])
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_imgep_driver.py -v`
Expected: PASS, 38 tests

- [ ] **Step 9: Run the whole suite**

Run: `python -m pytest -q`
Expected: everything green.

- [ ] **Step 10: Commit**

```bash
git add services/imgep_driver.py tests/test_imgep_driver.py
git commit -m "feat: goal-directed expeditions and Chase this"
```

---

### Task 5: `ArchiveState` and the shared rollout controls

Spec §8.1, §6.8-equivalent. Extract the widgets both auto modes share before adding a third tab that would otherwise duplicate them.

**Files:**
- Create: `state/archive_state.py`
- Modify: `state/__init__.py`, `state/ui_state.py`
- Modify: `ui/auto_tournament_window.py`
- Test: `tests/test_archive_state.py`

**Interfaces:**
- Produces:
  - `state.archive_state.ArchiveState` — persistent settings mirroring spec §7.4, plus one-shot flags
  - `AutoTournamentWindowMixin._render_rollout_controls(ats)` — the shared grid / steps / snapshots / sim-steps block
  - `UIState.archive: ArchiveState`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_archive_state.py
from dataclasses import fields

from state.archive_state import ArchiveState
from state.ui_state import UIState


ONE_SHOTS = (
    "start_requested", "pause_requested", "reset_requested",
    "add_goal_requested", "remove_goal_index", "move_goal_index",
    "move_goal_delta", "chase_tile", "pin_tile", "export_entry_id",
    "seed_entry_id", "delete_entry_id", "refit_projection_requested",
)


def test_defaults_match_the_spec_table():
    s = ArchiveState()
    assert s.enabled is False
    assert s.steps_per_gen == 2000
    assert s.snapshots_per_gen == 6
    assert s.seed_n == 256
    assert s.sigma0 == 0.5
    assert s.sigma_expand == 0.15
    assert s.alpha == 4.0
    assert s.k == 10
    assert s.expansion_between == 25
    assert s.expedition_gens == 50
    assert s.expedition_sigma == 0.1
    assert s.latent_share == 0.5
    assert s.liveness_min == 0.02
    assert s.target_rate == 0.15
    assert s.capacity == 20000
    assert s.refresh_per_gen == 64
    assert s.goal_order == "round_robin"


def test_one_shot_flags_default_to_inert():
    s = ArchiveState()
    for name in ONE_SHOTS:
        v = getattr(s, name)
        assert v in (False, -1), f"{name} defaults to {v!r}"


def test_the_new_goal_text_buffer_starts_empty():
    assert ArchiveState().new_goal_text == ""


def test_ui_state_carries_an_archive_block():
    assert isinstance(UIState().archive, ArchiveState)


def test_every_declared_one_shot_actually_exists():
    names = {f.name for f in fields(ArchiveState)}
    assert set(ONE_SHOTS) <= names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_archive_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'state.archive_state'`

- [ ] **Step 3: Write the state container**

```python
# state/archive_state.py
"""UI state for Explore (IMGEP) mode and the archive browser.

One-shot request flags are set by the UI and cleared by the consuming side in
CommandHandler - never inside UI.get_state(), which returns the live object.

Defaults are spec 7.4. Note steps_per_gen=2000 and snapshots_per_gen=6: at 2000
steps a generation is ~2.8 s, so 6 snapshots land ~333 steps apart, which is
enough for a slow pattern to visibly change. Six snapshots is 96 CLIP images at
N=4, about 7 ms against 2800 ms of simulation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArchiveState:
    # persistent settings
    enabled: bool = False
    running: bool = False
    show_browser: bool = False

    # rollout (shared with Auto mode's widgets, own defaults)
    grid: int = 4
    steps_per_gen: int = 2000
    sim_steps_per_frame: int = 10
    snapshots_per_gen: int = 6
    physics_enabled: bool = False
    tile_mutation_enabled: bool = False
    variants_per_tile: int = 4
    tile_mutation_strength: float = 0.1

    # exploration
    sigma0: float = 0.5              # bootstrap scatter only
    sigma_expand: float = 0.15
    alpha: float = 4.0
    k: int = 10
    seed_n: int = 256
    liveness_min: float = 0.02
    target_rate: float = 0.15
    capacity: int = 20000
    refresh_per_gen: int = 64

    # expeditions
    expansion_between: int = 25      # 0 disables expeditions
    expedition_gens: int = 50
    expedition_sigma: float = 0.1
    latent_share: float = 0.5
    beta: float = 0.5
    goal_order: str = "round_robin"  # or "least_matched"

    # browser view
    sort_by: str = "novelty"         # novelty | recency | liveness
    pinned_only: bool = False
    selected_entry_id: int = -1

    # persistent, not a one-shot: dismissed explicitly by the user
    warning: str = ""

    # editing buffer for the goal list
    new_goal_text: str = ""

    # one-shot request flags, cleared by CommandHandler
    start_requested: bool = False
    pause_requested: bool = False
    reset_requested: bool = False
    add_goal_requested: bool = False
    remove_goal_index: int = -1
    move_goal_index: int = -1
    move_goal_delta: int = 0
    chase_tile: int = -1
    pin_tile: int = -1
    export_entry_id: int = -1
    seed_entry_id: int = -1
    delete_entry_id: int = -1
    refit_projection_requested: bool = False
```

- [ ] **Step 4: Wire it into the aggregate state**

In `state/ui_state.py`, add the import and the field:

```python
from .archive_state import ArchiveState
```

```python
    archive: ArchiveState = field(default_factory=ArchiveState)
```

In `state/__init__.py`, add `ArchiveState` to the imports and `__all__` alongside `AutoTournamentState`.

- [ ] **Step 5: Extract the shared rollout controls**

In `ui/auto_tournament_window.py`, replace the inline block in `render_auto_tournament_tab` (currently lines 97–113, from the Grid slider through the Initial Sigma slider) with a call, and add the method. The Auto tab keeps its own algorithm combo and sigma slider above it; only the four rollout sliders and the grid move.

Replace those lines with:

```python
        self._render_rollout_controls(ats)
        _, ats.sigma0 = imgui.slider_float("Initial Sigma", ats.sigma0, 0.05, 1.5)
```

and add this method to `AutoTournamentWindowMixin`:

```python
    def _render_rollout_controls(self, ats, grid_note=None):
        """Grid and rollout timing. Shared verbatim by Auto and Explore - both
        drive the same AutoTournamentService rollout machine, so duplicating
        these widgets would let the two tabs disagree about what a generation
        is.

        grid_note overrides the last hint line because the consequence of a
        grid change differs: Auto mode loses its accumulated covariance, while
        Explore mode only ends any expedition in flight."""
        ch, g = imgui.slider_int("Grid", ats.grid, 2, 8)
        if ch and g != ats.grid:
            ats.grid = g
            ats.grid_changed = True
        self._render_grid_hints(ats, grid_note)

        _, ats.steps_per_gen = imgui.slider_int(
            "Steps per Gen", ats.steps_per_gen, 50, 2000)
        _, ats.sim_steps_per_frame = imgui.slider_int(
            "Sim Steps per Frame", ats.sim_steps_per_frame, 1, 50)
        _, ats.snapshots_per_gen = imgui.slider_int(
            "Snapshots per Gen", ats.snapshots_per_gen, 1, 8)
```

`_render_grid_hints` currently reads `ats.grid` only — verified — so it works unchanged for `ArchiveState`. Its last line, however, hardcodes "changing the grid resets the optimizer", which is false in Explore mode (there is no persistent optimizer; only an expedition in flight ends). Give it the override:

```python
    def _render_grid_hints(self, ats, note=None):
        tiles = ats.grid * ats.grid
        src_px = 1024 // ats.grid
        imgui.text_disabled(f"population {tiles}   source {src_px}px/tile")
        if src_px < 224:
            imgui.text_disabled(
                "  upscaled to 224 for CLIP - consider a larger canvas")
        if ats.grid == 2:
            imgui.text_disabled("  popsize 4 is small for 80-D CMA-ES")
        imgui.text_disabled(note or "changing the grid resets the optimizer")
```

The Explore tab passes `grid_note="changing the grid ends any expedition in flight"`.

`ArchiveState` needs a `grid_changed` field for this shared widget, so add it:

```python
    grid_changed: bool = False
```

and add `"grid_changed"` to the `ONE_SHOTS` tuple in `tests/test_archive_state.py`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_archive_state.py -v`
Expected: PASS, 5 tests

- [ ] **Step 7: Confirm the Auto tab still renders**

Run: `python -m pytest tests/test_tournament_window_render.py -v`
Expected: PASS. Then launch the app, open Tournament → Auto (CLIP), and confirm the Grid, Steps per Gen, Sim Steps per Frame, Snapshots per Gen and Initial Sigma sliders are all present and still move.

- [ ] **Step 8: Commit**

```bash
git add state/archive_state.py state/ui_state.py state/__init__.py ui/auto_tournament_window.py tests/test_archive_state.py
git commit -m "feat: ArchiveState and shared rollout controls for both auto tabs"
```

---

### Task 6: The Explore tab

Spec §2, §8.1. Passive: widgets render and set one-shot flags, nothing else.

**Files:**
- Create: `ui/archive_window.py`
- Modify: `ui/core.py`, `ui/tournament_window.py`
- Test: `tests/test_archive_window_render.py`

**Interfaces:**
- Consumes: `ArchiveState` (Task 5), `AutoTournamentWindowMixin._render_rollout_controls`
- Produces: `ArchiveWindowMixin.render_explore_tab()`, `.render_archive_window()`, `.archive_service` / `.archive_goals` / `.archive_unavailable` attributes set by the orchestrator

- [ ] **Step 1: Write the failing test**

This mirrors `tests/test_tournament_window_render.py` exactly: a module-scoped `gui` fixture that creates a headless ImGui context, and a `frame()` helper that runs **two** full frames because ImGui emits no geometry for a window on the frame it is first created. Do not invent a different fixture — that file is the working reference for how this project renders widgets under test.

```python
# tests/test_archive_window_render.py
"""Actually render the Explore tab and the archive browser.

An ImGui begin/end imbalance or a None dereference does not fail loudly - it
corrupts the whole frame, so every window in the app disappears at once and the
cause is invisible. ImGui asserts on an unbalanced stack inside EndFrame, so a
test that completes at all has proved the stack balances.

No window and no renderer: ImGui only needs a display size and a frame.
"""
import numpy as np
import pytest
from imgui_bundle import imgui

from services.goal_source import GoalList
from state.archive_state import ArchiveState
from state.auto_tournament_state import AutoTournamentState
from state.tournament_state import TournamentState
from ui.archive_window import ArchiveWindowMixin
from ui.auto_tournament_window import AutoTournamentWindowMixin


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    for _ in range(2):
        imgui.new_frame()
        imgui.render()
    yield
    imgui.destroy_context()


def frame(fn, n=2):
    """Run full ImGui frames around fn and return the last vertex count."""
    for _ in range(n):
        imgui.new_frame()
        imgui.begin("host", True)
        fn()
        imgui.end()
        imgui.render()
    return imgui.get_draw_data().total_vtx_count


class _State:
    def __init__(self):
        self.archive = ArchiveState()
        self.auto_tournament = AutoTournamentState()
        self.tournament = TournamentState()


class _FakeArchive:
    def __init__(self, n=0, dim=8):
        self.entries = []
        self.embeddings = np.zeros((n, dim), dtype=np.float32)

    def __len__(self):
        return len(self.entries)

    def stats(self):
        return {"size": len(self.entries), "threshold": 0.05,
                "admission_rate": 0.15, "n_nonfinite": 0, "n_rejected": 0,
                "n_pinned": 0, "blocked_by_pins": False, "rejects_ring": 0}


class _FakeDriver:
    def __init__(self, **over):
        self._st = {"regime": "expansion", "goal": "coral reef",
                    "archive_size": 3, "threshold": 0.05, "admission_rate": 0.15,
                    "n_pinned": 0, "blocked_by_pins": False,
                    "score_label": "novelty", "sigma": 0.15,
                    "algorithm": "CMA-ES", "prompt": "coral reef"}
        self._st.update(over)

    def status(self):
        return dict(self._st)


class Harness(ArchiveWindowMixin, AutoTournamentWindowMixin):
    """The two mixins under test, with only the attributes they reach for."""

    def __init__(self, driver=None, archive=None, goals=None, unavailable=""):
        self.state = _State()
        self.archive_driver = driver
        self.archive_obj = archive
        self.archive_goals = goals
        self.archive_service = None
        self.archive_unavailable = unavailable
        self.archive_projection = None
        self.archive_goal_point = None
        self.thumb_cache = None


def test_explore_tab_renders_before_the_service_exists(gui):
    h = Harness()
    assert frame(h.render_explore_tab) > 0


def test_explore_tab_renders_when_clip_is_unavailable(gui):
    h = Harness(unavailable="missing package: onnxruntime")
    assert frame(h.render_explore_tab) > 0


def test_explore_tab_renders_with_a_live_driver(gui):
    goals = GoalList()
    goals.add("coral reef")
    goals.add("lightning")
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive(), goals=goals)
    assert frame(h.render_explore_tab) > 0


def test_explore_tab_warns_when_the_archive_is_full_of_pins(gui):
    h = Harness(driver=_FakeDriver(blocked_by_pins=True, archive_size=10),
                archive=_FakeArchive())
    assert frame(h.render_explore_tab) > 0


def test_explore_tab_warns_when_nothing_is_being_admitted(gui):
    h = Harness(driver=_FakeDriver(archive_size=500, admission_rate=0.0),
                archive=_FakeArchive())
    assert frame(h.render_explore_tab) > 0


def test_rendering_the_tab_marks_the_mode_enabled(gui):
    h = Harness()
    frame(h.render_explore_tab)
    assert h.state.archive.enabled is True


def test_a_dismissable_warning_renders(gui):
    h = Harness()
    h.state.archive.warning = "capture is nearly black"
    assert frame(h.render_explore_tab) > 0


def test_the_archive_window_is_hidden_until_asked_for(gui):
    h = Harness(archive=_FakeArchive())
    assert frame(h.render_archive_window) == 0


def test_the_archive_window_renders_with_no_archive(gui):
    h = Harness()
    h.state.archive.show_browser = True
    assert frame(h.render_archive_window) > 0
```

The last two tests cover `render_archive_window`, whose body arrives in Task 8; add the stub described in Step 4 so they pass now and keep passing.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_archive_window_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.archive_window'`

- [ ] **Step 3: Write the Explore tab**

```python
# ui/archive_window.py
"""Explore (IMGEP) tab and the archive browser.

Passive: renders widgets and sets one-shot flags, runs no logic. Every value
displayed comes from driver.status() or archive.stats(), so the UI cannot
disagree with the search about what is happening.
"""
from __future__ import annotations

from imgui_bundle import imgui

_BAD = (1.0, 0.4, 0.3, 1.0)
_WARN = (1.0, 0.6, 0.2, 1.0)
_OK = (0.4, 0.9, 0.5, 1.0)
_DIM = (0.6, 0.6, 0.6, 1.0)

GOAL_TOOLTIP = (
    "Expeditions alternate between two goal sources.\n\n"
    "LATENT goals extrapolate past the archive's frontier, away from its "
    "centroid: 'keep going in the direction that already looks unlike "
    "everything else'. No text involved, so nothing is lost in translation.\n\n"
    "TEXT goals are this list, cycled in order. E&E has a language model write "
    "these from archive thumbnails; here you write them, which is strictly more "
    "controllable and needs no network.\n\n"
    "Leave the list empty and every expedition is latent."
)

ALPHA_TOOLTIP = (
    "How strongly parent selection favours novel archive entries: p ~ NOV^alpha.\n\n"
    "alpha = 4 is the tuned value from Expedition & Expansion. alpha = 0 samples "
    "parents uniformly, which with Expansion Between = 0 reduces the whole search "
    "to random archive mutation - E&E's own baseline, reachable here without a "
    "code change."
)


class ArchiveWindowMixin:
    archive_service = None      # AutoTournamentService driving Explore mode
    archive_driver = None       # ImgepDriver
    archive_obj = None          # Archive
    archive_goals = None        # GoalList
    archive_unavailable = ""

    # ---- the tab -------------------------------------------------------

    def render_explore_tab(self):
        ast = self.state.archive
        ast.enabled = True

        if ast.warning:
            imgui.text_colored(imgui.ImVec4(*_BAD), ast.warning)
            imgui.same_line()
            if imgui.button("Dismiss##explore"):
                ast.warning = ""

        if self.archive_unavailable:
            imgui.text_colored(imgui.ImVec4(*_WARN), self.archive_unavailable)
            imgui.text_wrapped(
                "Explore mode needs the same CLIP model and packages as Auto mode. "
                "Manual mode is unaffected.")
            return

        self._render_explore_status(ast)
        imgui.separator()
        self._render_explore_transport(ast)
        imgui.separator()
        self._render_goal_list(ast)
        imgui.separator()
        self._render_rollout_controls(
            ast, grid_note="changing the grid ends any expedition in flight")
        imgui.separator()
        self._render_exploration_settings(ast)
        imgui.separator()
        self._render_expedition_settings(ast)
        imgui.separator()
        if imgui.button("Open Archive Browser"):
            ast.show_browser = True

    def _render_explore_status(self, ast):
        d = self.archive_driver
        if d is None:
            imgui.text_colored(imgui.ImVec4(*_DIM), "Not started")
            return
        st = d.status()
        imgui.text(f"Regime: {st['regime']}")
        goal = st.get("goal") or "-"
        imgui.text(f"Goal: {goal}")
        imgui.text(f"Archive: {st['archive_size']}   "
                   f"threshold {st['threshold']:.3f}   "
                   f"admitting {100.0 * st['admission_rate']:.0f}%")
        if st.get("blocked_by_pins"):
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "Archive is full and entirely pinned - nothing new can be added.")
        # spec 10: a persistently zero admission rate is a broken capture or a
        # dead preset, not a hard search.
        if st["archive_size"] > 0 and st["admission_rate"] <= 0.0:
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "Nothing has been admitted recently - check the preset is alive, "
                "the capture is not black, and Liveness Floor is not too high.")

    def _render_explore_transport(self, ast):
        if imgui.button("Start##explore"):
            ast.start_requested = True
        imgui.same_line()
        if imgui.button("Pause##explore"):
            ast.pause_requested = True
        imgui.same_line()
        if imgui.button("Reset Search##explore"):
            ast.reset_requested = True
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(*_DIM), "(Reset keeps the archive)")
        self._render_cycle_estimate(ast)

    def _render_cycle_estimate(self, ast):
        """One expansion+expedition cycle in wall-clock, at the measured
        716 sim steps/s. expedition_gens=350 (the paper's value) is ~16 minutes
        on a single goal, which is worth seeing before choosing it."""
        per_gen = ast.steps_per_gen / 716.0
        cycle = (ast.expansion_between + ast.expedition_gens) * per_gen
        imgui.text_colored(
            imgui.ImVec4(*_DIM),
            f"~{per_gen:.1f}s per generation, ~{cycle / 60.0:.1f} min per cycle")

    def _render_goal_list(self, ast):
        imgui.text("Goals")
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.set_tooltip(GOAL_TOOLTIP)

        goals = self.archive_goals
        if goals is not None:
            for i, item in enumerate(list(goals.items)):
                changed, enabled = imgui.checkbox(f"##goal_on{i}", item["enabled"])
                if changed:
                    item["enabled"] = enabled
                imgui.same_line()
                if imgui.button(f"^##goal_up{i}"):
                    ast.move_goal_index, ast.move_goal_delta = i, -1
                imgui.same_line()
                if imgui.button(f"v##goal_dn{i}"):
                    ast.move_goal_index, ast.move_goal_delta = i, +1
                imgui.same_line()
                if imgui.button(f"x##goal_rm{i}"):
                    ast.remove_goal_index = i
                imgui.same_line()
                imgui.text(item["text"])

        _, ast.new_goal_text = imgui.input_text("##new_goal", ast.new_goal_text)
        imgui.same_line()
        if imgui.button("Add Goal") and ast.new_goal_text.strip():
            ast.add_goal_requested = True

        order = ["round_robin", "least_matched"]
        idx = order.index(ast.goal_order) if ast.goal_order in order else 0
        ch, idx = imgui.combo("Goal Order", idx, ["Round robin", "Least matched"])
        if ch:
            ast.goal_order = order[idx]

    def _render_exploration_settings(self, ast):
        _, ast.sigma_expand = imgui.slider_float(
            "Expansion Sigma", ast.sigma_expand, 0.01, 1.0)
        _, ast.alpha = imgui.slider_float("Novelty Exponent", ast.alpha, 0.0, 8.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(ALPHA_TOOLTIP)
        _, ast.k = imgui.slider_int("Neighbours (k)", ast.k, 1, 50)
        _, ast.seed_n = imgui.slider_int("Seed Entries", ast.seed_n, 64, 2048)
        _, ast.sigma0 = imgui.slider_float("Bootstrap Sigma", ast.sigma0, 0.05, 1.5)
        _, ast.liveness_min = imgui.slider_float(
            "Liveness Floor", ast.liveness_min, 0.0, 0.5)
        _, ast.target_rate = imgui.slider_float(
            "Target Admission Rate", ast.target_rate, 0.01, 1.0)
        _, ast.refresh_per_gen = imgui.slider_int(
            "Novelty Refresh / Gen", ast.refresh_per_gen, 0, 512)
        _, ast.capacity = imgui.slider_int("Capacity", ast.capacity, 1000, 100000)

    def _render_expedition_settings(self, ast):
        _, ast.expansion_between = imgui.slider_int(
            "Expansion Between", ast.expansion_between, 0, 500)
        if imgui.is_item_hovered():
            imgui.set_tooltip("0 disables expeditions entirely - pure novelty "
                              "search, which is E&E's own ablation.")
        _, ast.expedition_gens = imgui.slider_int(
            "Expedition Gens", ast.expedition_gens, 5, 400)
        _, ast.expedition_sigma = imgui.slider_float(
            "Expedition Sigma", ast.expedition_sigma, 0.01, 1.0)
        _, ast.latent_share = imgui.slider_float(
            "Latent Goal Share", ast.latent_share, 0.0, 1.0)
        _, ast.beta = imgui.slider_float("Extrapolation (beta)", ast.beta, 0.0, 2.0)
```

- [ ] **Step 4: Add the third tab**

In `ui/tournament_window.py`, inside the tab bar, after the Auto tab:

```python
            if imgui.begin_tab_item("Explore (IMGEP)")[0]:
                self.state.auto_tournament.enabled = False
                self.render_explore_tab()
                imgui.end_tab_item()
            else:
                self.state.archive.enabled = False
```

and set `self.state.archive.enabled = False` in the Manual and Auto branches too, so exactly one mode is ever enabled.

In `ui/core.py`, add `ArchiveWindowMixin` to the `UI` class bases, import it from `ui.archive_window`, and add `self.render_archive_window()` to the render dispatch.

The browser body lands in Task 8, so add this stub to `ArchiveWindowMixin` now — the last two tests in Step 1 cover it, and it keeps `ui/core.py`'s dispatch valid from this task onward:

```python
    def render_archive_window(self):
        """Body arrives in Task 8; the early return is the permanent guard."""
        ast = self.state.archive
        if not ast.show_browser:
            return
        expanded, opened = imgui.begin("Archive", True)
        if not opened:
            ast.show_browser = False
            imgui.end()
            return
        imgui.text_colored(imgui.ImVec4(*_DIM), "No archive yet.")
        imgui.end()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_archive_window_render.py tests/test_tournament_window_render.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ui/archive_window.py ui/core.py ui/tournament_window.py tests/test_archive_window_render.py
git commit -m "feat: Explore (IMGEP) tab with goal list and exploration controls"
```

---

### Task 7: Orchestrator wiring

Spec §2, §10. Build `ImgepDriver` lazily, swap drivers on mode change, handle the one-shot flags.

**Files:**
- Modify: `main.py`, `command_handler.py`

**Interfaces:**
- Consumes: everything above
- Produces: `App._ensure_archive_service()`, `CommandHandler._handle_explore(ui_state)`

- [ ] **Step 1: Add the lazy builder to `main.py`**

After `_ensure_auto_service`, add:

```python
    def _ensure_archive_service(self):
        """Build the archive, goal list and IMGEP driver on first use.

        Explore mode reuses the SAME AutoTournamentService instance - the
        rollout machine is identical - and only swaps its driver. Imports stay
        lazy: onnxruntime and cmaes must not be imported at startup.
        """
        if not self._ensure_auto_service():
            self.ui.archive_unavailable = self.ui.auto_unavailable
            return False
        if self.imgep_driver is not None:
            return True

        from services.archive import Archive
        from services.archive_io import ArchiveStore
        from services.goal_source import GoalList
        from services.imgep_driver import ImgepDriver
        from utilities.paths import get_archive_dir

        store = ArchiveStore(get_archive_dir())
        archive = Archive(store=store)
        loaded, dropped = archive.load_from_store()
        print(f"[archive] loaded {loaded} entries ({dropped} dropped)")

        goals = GoalList(store=store)
        goals.load()

        self.archive_store = store
        self.archive = archive
        self.goal_list = goals
        self.imgep_driver = ImgepDriver(
            self.tournament_service, self.clip_scorer, archive, goals)
        self.prompt_driver = self.auto_service.driver

        self.ui.archive_driver = self.imgep_driver
        self.ui.archive_obj = archive
        self.ui.archive_goals = goals
        self.ui.archive_service = self.auto_service
        self.ui.archive_unavailable = ""
        self.command_handler.imgep_driver = self.imgep_driver
        self.command_handler.archive = archive
        self.command_handler.goal_list = goals
        return True
```

Add to `App.__init__`, next to the other lazy handles:

```python
        self.imgep_driver = None
        self.prompt_driver = None
        self.archive = None
        self.archive_store = None
        self.goal_list = None
        self._explore_was_enabled = False
```

- [ ] **Step 2: Swap the driver on the mode edge**

In `orchestrate_frame`, immediately after the existing Auto enable-edge block (step 1.5), add:

```python
        # Explore mode reuses Auto mode's rollout machine, canvas forcing and
        # capture path; only the driver differs. Swapping on the edge - rather
        # than constructing a second service - is what keeps abort-on-resize,
        # snapshot scheduling and the capture wiring in exactly one place.
        expl = ui_state.archive
        if expl.enabled and not self._explore_was_enabled:
            self._auto_prev_aspect = ui_state.preferences.canvas_aspect_ratio
            self._auto_prev_speedmult = ui_state.preferences.speedmult
            if ui_state.preferences.canvas_aspect_ratio != "1:1":
                ui_state.preferences.canvas_aspect_ratio = "1:1"
                ui_state.request_world_size_change = True
            if self._ensure_archive_service():
                self.auto_service.driver = self.imgep_driver
                self.auto_service.abort_generation()
        elif not expl.enabled and self._explore_was_enabled:
            if self.auto_service is not None and self.prompt_driver is not None:
                self.auto_service.pause()
                self.auto_service.driver = self.prompt_driver
            if self.archive is not None:
                self.archive.maybe_flush(force=True)
            if self._auto_prev_aspect and self._auto_prev_aspect != "1:1":
                ui_state.preferences.canvas_aspect_ratio = self._auto_prev_aspect
                ui_state.request_world_size_change = True
            if self._auto_prev_speedmult is not None:
                ui_state.preferences.speedmult = self._auto_prev_speedmult
        self._explore_was_enabled = expl.enabled
```

- [ ] **Step 3: Drive the loop in Explore mode**

Replace the `auto_running` block with one that covers both modes:

```python
        auto_running = ((auto.enabled or ui_state.archive.enabled)
                        and self.auto_service is not None
                        and ui_state.tournament.enabled)
        if auto_running:
            steps = self._drive_auto_tournament(ui_state)
            ui_state.preferences.speedmult = max(0, steps)
```

And in the `apply_state` block, extend the two Explore-relevant conditions — `plain_colour` and `physics_origin` — to fire for either mode:

```python
        _auto_on = ui_state.auto_tournament.enabled or ui_state.archive.enabled
```

then use `_auto_on` in place of `ui_state.auto_tournament.enabled` in the `plain_colour=`, `physics=` and `physics_origin` expressions.

- [ ] **Step 4: Flush the archive on exit**

In `cleanup()`, before `save_preferences`:

```python
        if self.archive is not None:
            self.archive.maybe_flush(force=True)
        if self.goal_list is not None:
            self.goal_list.save()
        if self.archive_store is not None:
            self.archive_store.close()
```

- [ ] **Step 5: Handle the one-shot flags**

In `command_handler.py`, add `imgep_driver = None`, `archive = None`, `goal_list = None` to `__init__`, call `self._handle_explore(ui_state)` from `process_commands` after `_handle_auto_tournament`, and add:

```python
    @staticmethod
    def _clear_explore_flags(ast):
        ast.start_requested = False
        ast.pause_requested = False
        ast.reset_requested = False
        ast.add_goal_requested = False
        ast.remove_goal_index = -1
        ast.move_goal_index = -1
        ast.move_goal_delta = 0
        ast.grid_changed = False
        ast.chase_tile = -1
        ast.pin_tile = -1
        ast.export_entry_id = -1
        ast.seed_entry_id = -1
        ast.delete_entry_id = -1
        ast.refit_projection_requested = False

    def _handle_explore(self, ui_state):
        ast = ui_state.archive
        svc, drv = self.auto_service, self.imgep_driver
        if drv is None or svc is None or svc.driver is not drv:
            self._clear_explore_flags(ast)
            return

        if ast.grid_changed and self.tournament_service is not None:
            self.tournament_service.set_grid(int(ast.grid))
            svc.abort_generation()

        svc.configure(
            steps_per_gen=ast.steps_per_gen,
            snapshots_per_gen=ast.snapshots_per_gen,
            sim_steps_per_frame=ast.sim_steps_per_frame,
            sigma0=ast.sigma0,
            physics_enabled=ast.physics_enabled,
            tile_mutation_enabled=ast.tile_mutation_enabled,
            variants_per_tile=ast.variants_per_tile,
            tile_mutation_strength=ast.tile_mutation_strength,
        )
        for name in ("sigma_expand", "alpha", "k", "seed_n", "liveness_min",
                     "refresh_per_gen", "expansion_between", "expedition_gens",
                     "expedition_sigma", "latent_share", "beta", "goal_order"):
            setattr(drv, name, getattr(ast, name))
        if self.archive is not None:
            self.archive.capacity = int(ast.capacity)
            self.archive.threshold.target_rate = float(ast.target_rate)

        if self.goal_list is not None:
            if ast.add_goal_requested and ast.new_goal_text.strip():
                if self.goal_list.add(ast.new_goal_text):
                    ast.new_goal_text = ""
                self.goal_list.save()
            if ast.remove_goal_index >= 0:
                self.goal_list.remove(ast.remove_goal_index)
                self.goal_list.save()
            if ast.move_goal_index >= 0 and ast.move_goal_delta:
                self.goal_list.move(ast.move_goal_index, ast.move_goal_delta)
                self.goal_list.save()

        if ast.pause_requested:
            svc.pause()
        if ast.reset_requested:
            # Resets the SEARCH. The archive is the product and survives.
            svc.reset()
        if ast.start_requested:
            svc.start()
        if ast.chase_tile >= 0 and not drv.chase(int(ast.chase_tile)):
            ast.warning = "nothing captured yet - chase needs one generation first"

        ast.running = svc.phase.value == "rollout"
        self._clear_explore_flags(ast)
```

`svc.start()` is called with no prompt, which is why `AutoTournamentService.start` takes `prompt: str | None = None` — Explore mode has no prompt, and the Start button must not be gated on one.

- [ ] **Step 6: Wire tile clicks to pin and chase**

In `ui/tournament_window.py`'s manual grid renderer, the left click already sets `state.clicked_tile` and right click sets `save_tile_requested`. In Explore mode the tournament `selected` set *is* the pin set, so left-click needs no change. Add right-click chase in `_handle_tournament`'s click path in `command_handler.py`:

```python
        # In Explore mode a right-click on a tile means 'chase this' rather
        # than 'save this tile as a config'.
        if ui_state.archive.enabled and self.imgep_driver is not None:
            if ui_state.auto_tournament.save_tile_requested >= 0:
                ui_state.archive.chase_tile = ui_state.auto_tournament.save_tile_requested
                ui_state.auto_tournament.save_tile_requested = -1
```

Place this immediately before `self._handle_auto_tournament(ui_state)` so the flag is redirected before that handler clears it.

- [ ] **Step 7: Verify by hand**

Launch the app. Enable Tournament, open the Explore tab, press Start.

Expected, in order:
1. The canvas switches to 1:1 and the grid populates with scattered genomes.
2. Regime reads `bootstrap`; archive size climbs by up to 16 per generation.
3. At 256 entries the regime becomes `expansion` and the admission rate settles near 15%.
4. `Documents/Fluoddity/archive/index.jsonl` grows; `thumbs/` fills with 160×160 JPEGs.
5. Add a goal, set Expansion Between to 3, and confirm the regime becomes `expedition` with your goal named in the status line.
6. Right-click a tile: the regime becomes `expedition` with goal `(chase)`.
7. Quit and relaunch, re-enter Explore: the archive size is preserved and the goal list is still there.

- [ ] **Step 8: Run the whole suite and commit**

```bash
python -m pytest -q
git add main.py command_handler.py ui/tournament_window.py
git commit -m "feat: wire Explore mode into the orchestrator and command handler"
```

---

### Task 8: The archive gallery

Spec §8.2. Thumbnails behind an LRU of GL textures; per-entry export, seed, pin, delete.

**Files:**
- Modify: `ui/archive_window.py`
- Create: `services/thumb_cache.py`
- Test: `tests/test_thumb_cache.py`

**Interfaces:**
- Produces: `ThumbCache(ctx, store, capacity=256)` with `.get(name) -> texture | None`, `.invalidate(name)`, `.release()`; `ArchiveWindowMixin.render_archive_window()`

- [ ] **Step 1: Write the failing test**

The cache's eviction policy is testable without a GL context by injecting a fake loader.

```python
# tests/test_thumb_cache.py
import numpy as np

from services.thumb_cache import ThumbCache


class _FakeTex:
    def __init__(self, name):
        self.name = name
        self.released = False

    def release(self):
        self.released = True


def cache(capacity=3):
    made = []

    def loader(name):
        t = _FakeTex(name)
        made.append(t)
        return t

    return ThumbCache(loader=loader, capacity=capacity), made


def test_get_loads_once_and_then_hits():
    c, made = cache()
    a = c.get("000001.jpg")
    b = c.get("000001.jpg")
    assert a is b
    assert len(made) == 1


def test_capacity_evicts_the_least_recently_used():
    c, _ = cache(capacity=2)
    first = c.get("a")
    c.get("b")
    c.get("a")            # 'a' is now the most recent
    c.get("c")            # evicts 'b'
    assert first.released is False
    assert len(c) == 2


def test_evicted_textures_are_released():
    c, made = cache(capacity=1)
    c.get("a")
    c.get("b")
    assert made[0].released is True


def test_a_failed_load_is_not_cached_and_is_retried():
    """A missing thumbnail must not become a permanent hole in the gallery -
    it may simply not have been written yet when the entry was admitted."""
    calls = []

    def loader(name):
        calls.append(name)
        return None

    c = ThumbCache(loader=loader, capacity=3)
    assert c.get("missing.jpg") is None
    assert c.get("missing.jpg") is None
    assert len(calls) == 2
    assert len(c) == 0


def test_invalidate_releases_and_forgets():
    c, made = cache()
    c.get("a")
    c.invalidate("a")
    assert made[0].released is True
    assert len(c) == 0


def test_release_frees_everything():
    c, made = cache()
    c.get("a")
    c.get("b")
    c.release()
    assert all(t.released for t in made)
    assert len(c) == 0


def test_an_empty_name_is_a_miss():
    c, made = cache()
    assert c.get("") is None
    assert made == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_thumb_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.thumb_cache'`

- [ ] **Step 3: Write the cache**

```python
# services/thumb_cache.py
"""LRU of GL textures for archive thumbnails.

An archive of 20,000 entries cannot hold 20,000 live textures. The gallery only
ever shows a screenful, so a small LRU with explicit release() is enough - and
explicit release matters because ModernGL's gc_mode='auto' would otherwise free
them at unpredictable moments during a frame.

The loader is injected so the eviction policy is testable without a GL context.
"""
from __future__ import annotations

from collections import OrderedDict


def gl_loader(ctx, store):
    """The real loader: read a thumbnail JPEG into an RGB texture."""

    def load(name: str):
        try:
            from PIL import Image

            path = store.thumb_path(name)
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                return ctx.texture(rgb.size, 3, rgb.tobytes())
        except Exception:
            # A missing or unreadable thumbnail costs a placeholder, never a
            # crash mid-frame.
            return None

    return load


class ThumbCache:
    def __init__(self, loader, capacity: int = 256):
        self._load = loader
        self.capacity = int(capacity)
        self._items: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def _order(self):
        return self._items

    def get(self, name: str):
        if not name:
            return None
        if name in self._items:
            self._items.move_to_end(name)
            return self._items[name]
        tex = self._load(name)
        if tex is None:
            return None            # not cached, so a transient failure retries
        self._items[name] = tex
        while len(self._items) > self.capacity:
            _, old = self._items.popitem(last=False)
            _release(old)
        return tex

    def invalidate(self, name: str) -> None:
        tex = self._items.pop(name, None)
        _release(tex)

    def release(self) -> None:
        for tex in self._items.values():
            _release(tex)
        self._items.clear()


def _release(tex) -> None:
    if tex is not None:
        try:
            tex.release()
        except Exception:
            pass
```

- [ ] **Step 4: Add the gallery to `ui/archive_window.py`**

This **replaces** the Task 6 stub of `render_archive_window` with the full body. The two archive-window tests written in Task 6 must still pass afterward.

```python
    def render_archive_window(self):
        ast = self.state.archive
        if not ast.show_browser:
            return
        expanded, opened = imgui.begin("Archive", True)
        if not opened:
            ast.show_browser = False
            imgui.end()
            return
        arc = self.archive_obj
        if arc is None:
            imgui.text_colored(imgui.ImVec4(*_DIM), "No archive yet.")
            imgui.end()
            return

        st = arc.stats()
        imgui.text(f"{st['size']} entries   {st['n_pinned']} pinned   "
                   f"threshold {st['threshold']:.3f}   "
                   f"admitting {100.0 * st['admission_rate']:.0f}%")
        imgui.separator()

        if imgui.begin_tab_bar("archive_views"):
            if imgui.begin_tab_item("Gallery")[0]:
                self._render_gallery(ast, arc)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Map")[0]:
                self._render_map(ast, arc)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.end()

    def _sorted_entries(self, ast, arc):
        entries = list(enumerate(arc.entries))
        if ast.pinned_only:
            entries = [(i, e) for i, e in entries if e.pinned]
        key = {"novelty": lambda p: -p[1].novelty,
               "liveness": lambda p: -p[1].liveness,
               "recency": lambda p: -p[1].ts}.get(ast.sort_by,
                                                  lambda p: -p[1].novelty)
        return sorted(entries, key=key)

    def _render_gallery(self, ast, arc):
        modes = ["novelty", "recency", "liveness"]
        idx = modes.index(ast.sort_by) if ast.sort_by in modes else 0
        ch, idx = imgui.combo("Sort", idx, ["Novelty", "Recency", "Liveness"])
        if ch:
            ast.sort_by = modes[idx]
        imgui.same_line()
        _, ast.pinned_only = imgui.checkbox("Pinned only", ast.pinned_only)

        cache = getattr(self, "thumb_cache", None)
        per_row = 6
        imgui.begin_child("gallery", imgui.ImVec2(0, 360))
        for n, (i, e) in enumerate(self._sorted_entries(ast, arc)[:240]):
            tex = cache.get(e.thumb) if cache is not None else None
            if tex is not None:
                imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(96, 96))
            else:
                imgui.button(f"#{e.id}", imgui.ImVec2(96, 96))
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"#{e.id}  {e.source}\nnovelty {e.novelty:.3f}\n"
                    f"liveness {e.liveness:.3f}\ngoal: {e.goal or '-'}")
            if imgui.is_item_clicked():
                ast.selected_entry_id = e.id
            if n % per_row != per_row - 1:
                imgui.same_line()
        imgui.end_child()

        if ast.selected_entry_id >= 0:
            imgui.separator()
            imgui.text(f"Selected #{ast.selected_entry_id}")
            if imgui.button("Export as config"):
                ast.export_entry_id = ast.selected_entry_id
            imgui.same_line()
            if imgui.button("Seed a run from here"):
                ast.seed_entry_id = ast.selected_entry_id
            imgui.same_line()
            if imgui.button("Delete"):
                ast.delete_entry_id = ast.selected_entry_id
```

- [ ] **Step 5: Construct the cache and handle the entry actions**

In `main.py._ensure_archive_service`, after building the store:

```python
        from services.thumb_cache import ThumbCache, gl_loader

        self.thumb_cache = ThumbCache(gl_loader(self.ctx, store), capacity=256)
        self.ui.thumb_cache = self.thumb_cache
```

and release it in `cleanup()` with `if self.thumb_cache is not None: self.thumb_cache.release()`.

In `command_handler._handle_explore`, before clearing the flags:

```python
        if ast.export_entry_id >= 0:
            self._export_archive_entry(ui_state, ast.export_entry_id)
        if ast.seed_entry_id >= 0:
            self._seed_from_archive(ast.seed_entry_id)
        if ast.delete_entry_id >= 0:
            self._delete_archive_entry(ast)
```

with:

```python
    def _archive_index(self, entry_id):
        for i, e in enumerate(self.archive.entries):
            if e.id == int(entry_id):
                return i
        return None

    def _export_archive_entry(self, ui_state, entry_id):
        """Write an archive entry as an ordinary Fluoddity config, so it opens
        in the normal single-simulation view at any resolution."""
        from services.config_saver import ConfigSaver
        from services.genome_io import export_genome
        from services.genome_spec import encode
        from services.physics_genome import PHYSICS_PARAMS

        i = self._archive_index(entry_id)
        if i is None:
            return
        e = self.archive.entries[i]
        z, _clamped = encode(self.archive.brains[i])
        sim_state = ui_state.sim
        if "physics" in e.spec:
            # The archive stores ABSOLUTE physics, so applying it needs no origin.
            for j, (name, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS):
                setattr(sim_state, name, float(self.archive.physics[i][j]))
        meta = {"archive_id": int(e.id), "novelty": float(e.novelty),
                "liveness": float(e.liveness), "source": e.source,
                "goal": e.goal, "run_id": e.run_id, "spec": e.spec}
        path = self.user_configs_dir / f"archive_{e.id:06d}.json"
        export_genome(path, z, sim_state, meta)
        print(f"[archive] saved {path}")

    def _seed_from_archive(self, entry_id):
        """Load an archive entry as a search starting point.

        Only Auto mode has an x0 - an IMGEP expansion draws its parents from the
        archive by novelty, so 'seed from here' has no meaning there and must
        say so rather than silently do nothing.
        """
        from services.genome_spec import encode

        i = self._archive_index(entry_id)
        if i is None or self.auto_service is None:
            return
        if not hasattr(self.auto_service.driver, "set_x0"):
            print("[archive] seeding applies to Auto (CLIP) mode; "
                  "Explore draws its parents from the archive already")
            return
        z, _ = encode(self.archive.brains[i])
        self.auto_service.set_x0(z)

    def _delete_archive_entry(self, ast):
        i = self._archive_index(ast.delete_entry_id)
        if i is None:
            return
        self.archive._remove(i)
        self.archive.maybe_flush(force=True)
        if ast.selected_entry_id == ast.delete_entry_id:
            ast.selected_entry_id = -1
```

`_seed_from_archive` only sets `x0`, which is a no-op for `ImgepDriver` (it has no `set_x0`); guard with `hasattr(self.auto_service.driver, "set_x0")` and otherwise report that seeding applies to Auto mode.

- [ ] **Step 6: Run tests and verify by hand**

Run: `python -m pytest tests/test_thumb_cache.py tests/test_archive_window_render.py -v`
Expected: PASS

Then launch, run Explore until the archive has ~50 entries, open the Archive browser, and confirm: thumbnails render, sorting changes the order, hover shows novelty/liveness, Export writes a config that loads in the normal view, Delete removes the tile from the grid.

- [ ] **Step 7: Commit**

```bash
git add services/thumb_cache.py ui/archive_window.py main.py command_handler.py tests/test_thumb_cache.py
git commit -m "feat: archive gallery with an LRU thumbnail cache and entry export"
```

---

### Task 9: The 2-D semantic map

Spec §8.3. This is what makes "expanding the boundaries" visible.

**Files:**
- Modify: `ui/archive_window.py`, `main.py`

**Interfaces:**
- Consumes: `services.archive_projection.Projection` (Task 1)
- Produces: `ArchiveWindowMixin._render_map(ast, arc)`

- [ ] **Step 1: Add the map renderer**

```python
    _MAP_COLORS = {
        "bootstrap": imgui.IM_COL32(120, 120, 130, 200),
        "expansion": imgui.IM_COL32(90, 200, 120, 220),
        "expedition": imgui.IM_COL32(255, 170, 60, 230),
        "pin": imgui.IM_COL32(90, 170, 255, 255),
    }

    def _render_map(self, ast, arc):
        proj = getattr(self, "archive_projection", None)
        if proj is None or len(arc) < 3:
            imgui.text_colored(imgui.ImVec4(*_DIM),
                               "Not enough entries to project yet.")
            return
        if imgui.button("Refit projection"):
            ast.refit_projection_requested = True
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(*_DIM),
                           "PCA of the CLIP embeddings. The map is a view - "
                           "novelty is always measured in the full 512-d space.")

        pts = proj.transform(arc.embeddings)
        if not len(pts):
            return
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)

        size = imgui.ImVec2(imgui.get_content_region_avail().x, 320)
        origin = imgui.get_cursor_screen_pos()
        imgui.invisible_button("map_canvas", size)
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(origin,
                             imgui.ImVec2(origin.x + size.x, origin.y + size.y),
                             imgui.IM_COL32(20, 20, 24, 255))

        mouse = imgui.get_mouse_pos()
        hovered_id, best_d = -1, 1e9
        for i, e in enumerate(arc.entries):
            u = (pts[i] - lo) / span
            x = origin.x + 8.0 + float(u[0]) * (size.x - 16.0)
            y = origin.y + 8.0 + (1.0 - float(u[1])) * (size.y - 16.0)
            key = "pin" if e.pinned else e.source
            draw.add_circle_filled(
                imgui.ImVec2(x, y), 3.0,
                self._MAP_COLORS.get(key, self._MAP_COLORS["expansion"]))
            d = abs(mouse.x - x) + abs(mouse.y - y)
            if d < best_d:
                hovered_id, best_d = e.id, d

        goal_pt = getattr(self, "archive_goal_point", None)
        if goal_pt is not None:
            u = (proj.transform(goal_pt[None])[0] - lo) / span
            gx = origin.x + 8.0 + float(u[0]) * (size.x - 16.0)
            gy = origin.y + 8.0 + (1.0 - float(u[1])) * (size.y - 16.0)
            draw.add_circle(imgui.ImVec2(gx, gy), 7.0,
                            imgui.IM_COL32(255, 90, 90, 255), 0, 2.0)

        if best_d < 12.0 and imgui.is_item_hovered():
            i = self._archive_pos(arc, hovered_id)
            if i is not None:
                e = arc.entries[i]
                imgui.set_tooltip(f"#{e.id}  {e.source}\n"
                                  f"novelty {e.novelty:.3f}\ngoal: {e.goal or '-'}")
                if imgui.is_item_clicked():
                    ast.selected_entry_id = e.id

    @staticmethod
    def _archive_pos(arc, entry_id):
        for i, e in enumerate(arc.entries):
            if e.id == entry_id:
                return i
        return None
```

Add `import numpy as np` to the top of `ui/archive_window.py`.

- [ ] **Step 2: Own the projection in the orchestrator and refit on a schedule**

In `main.py._ensure_archive_service`:

```python
        from services.archive_projection import Projection

        self.archive_projection = Projection()
        self.archive_projection.fit(archive.embeddings)
        self._last_projection_size = len(archive)
        self.ui.archive_projection = self.archive_projection
```

In `_after_generation`, add:

```python
        # A refit every 500 admissions, not per frame. Projection.fit
        # sign-aligns to the previous components, so the map does not mirror
        # itself when this fires.
        if self.archive is not None and self.archive_projection is not None:
            if len(self.archive) - self._last_projection_size >= 500:
                self.archive_projection.fit(self.archive.embeddings)
                self._last_projection_size = len(self.archive)
        if self.imgep_driver is not None:
            g = getattr(self.imgep_driver, "_goal", None)
            self.ui.archive_goal_point = g.embedding if g is not None else None
```

Handle `ast.refit_projection_requested` in `_handle_explore` by calling `self.archive_projection.fit(self.archive.embeddings)` — expose the projection to `CommandHandler` the same way the archive is.

Add `self.archive_projection = None` and `self._last_projection_size = 0` to `App.__init__`, and `archive_projection = None` / `archive_goal_point = None` as class attributes on `ArchiveWindowMixin`.

- [ ] **Step 3: Verify by hand**

Run Explore until the archive has ~300 entries, then open Archive → Map.

Expected: a scatter that visibly grows outward over time; expansion points in green, expedition points in orange, pins in blue; the red ring marks the current goal and sits *outside* the main cloud during a latent expedition; hovering a point shows its stats; "Refit projection" does not mirror the layout.

- [ ] **Step 4: Commit**

```bash
git add ui/archive_window.py main.py command_handler.py
git commit -m "feat: 2-D semantic map of the archive with the live goal marker"
```

---

### Task 10: Calibration and documentation

Spec §12.2. Three constants are stated as defaults but must be **measured**, exactly as `sigma0` was for Auto mode. This is a one-off measurement recorded in the repo, not a runtime feature.

**Files:**
- Create: `tools/calibrate_imgep.py`
- Modify: `docs/testing_checklist.md`, `ARCHITECTURE.md`, `CLAUDE.md`

- [ ] **Step 1: Write the calibration tool**

```python
# tools/calibrate_imgep.py
"""One-off measurements for the three constants spec 12.2 says must not be
guessed. Prints a block to paste into the implementation log.

    python -m tools.calibrate_imgep

sigma_expand: the decoded spread of a mutated population should be visibly
related to, but narrower than, a fresh random population. This compares the
per-coefficient std of genomes decoded from N(0, sigma0) against genomes
decoded from a single parent plus N(0, sigma_expand).
"""
from __future__ import annotations

import numpy as np

from services.genome import random_genome
from services.genome_spec import DIM, decode, encode


def measure_expansion_spread(sigma0=0.5, sigma_expand=0.15, n=512, seed=0):
    rng = np.random.default_rng(seed)
    fresh = np.stack([random_genome(rng) for _ in range(n)])
    boot = np.stack([decode(z) for z in sigma0 * rng.normal(size=(n, DIM))])

    parent = random_genome(rng)
    pz, _ = encode(parent)
    kids = np.stack([decode(pz + sigma_expand * rng.normal(size=DIM))
                     for _ in range(n)])

    return {
        "sigma0": sigma0,
        "sigma_expand": sigma_expand,
        "std_random_genome": float(fresh.std()),
        "std_bootstrap": float(boot.std()),
        "std_expansion_children": float(kids.std()),
        "ratio_children_to_bootstrap": float(kids.std() / max(boot.std(), 1e-8)),
    }


def main() -> int:
    r = measure_expansion_spread()
    print("--- sigma_expand calibration ---")
    for k, v in r.items():
        print(f"{k:32s} {v:.4f}")
    print()
    print("Bar: bootstrap spread should be close to std_random_genome, and")
    print("children should sit at roughly 0.2-0.4 of it - related but narrower.")
    print("If ratio_children_to_bootstrap is above ~0.6, lower sigma_expand;")
    print("below ~0.1, expansion cannot escape its parent and will stall.")
    print()
    print("liveness_min and the NOV scale must be measured IN THE APP:")
    print("  1. Load a preset you know freezes; run one Explore generation;")
    print("     read the liveness of its tiles from the archive index.")
    print("  2. Do the same with a preset you know stays lively.")
    print("  3. Set Liveness Floor between them, nearer the frozen end.")
    print("  4. Read the settled adaptive threshold from the status line;")
    print("     that is the sensible centre of the novelty scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it and record the numbers**

```bash
python -m tools.calibrate_imgep
```

Expected: `ratio_children_to_bootstrap` between roughly 0.2 and 0.4. If it is outside that band, change the `sigma_expand` default in `state/archive_state.py` and `services/imgep_driver.py` to the value that lands inside it, and record both the measurement and the chosen value in the commit message.

- [ ] **Step 3: Measure `liveness_min` in the app**

Run one Explore generation on a preset that freezes and one on a lively preset. Read the `liveness` field from `Documents/Fluoddity/archive/index.jsonl` for each. Set the `liveness_min` default between the two, nearer the frozen end, in `state/archive_state.py` and `services/imgep_driver.py`.

Record both measured distributions in the commit message. **If they overlap, say so** — that means the descriptor is not separating frozen from lively at this snapshot count, and `snapshots_per_gen` should go up before the threshold is tuned.

- [ ] **Step 4: Extend the testing checklist**

Append to `docs/testing_checklist.md`:

```markdown
## Explore (IMGEP) mode

- [ ] Enabling Explore forces a 1:1 canvas; disabling restores the previous ratio
- [ ] Start with an empty archive: regime reads `bootstrap`, archive grows by up to N^2 per generation
- [ ] At `seed_n` the regime becomes `expansion` and the admission rate settles near the target
- [ ] `Documents/Fluoddity/archive/index.jsonl` gains one line per admission; `thumbs/` fills
- [ ] With `Expansion Between` = 3 and a goal in the list, an expedition fires and names the goal
- [ ] With the goal list empty, every expedition reads `(latent)`
- [ ] `Expansion Between` = 0 never expeditions
- [ ] Right-click a tile: an expedition starts immediately with goal `(chase)`
- [ ] Left-click a tile: it enters the archive as pinned, bypassing the gates
- [ ] Reset Search keeps the archive; the size readout does not drop
- [ ] Quit and relaunch: archive size and goal list are preserved
- [ ] Kill the app mid-run (no clean quit): the archive reloads, reporting the dropped trailing entries
- [ ] Archive browser: sorting, pinned-only filter, hover stats, Export writes a loadable config
- [ ] Map: the cloud grows outward; the goal marker sits outside it during a latent expedition
- [ ] "Refit projection" does not mirror the layout
- [ ] Switching Explore -> Auto -> Manual and back leaves each mode working
- [ ] Toggling physics search mid-run ends any expedition without crashing
- [ ] Changing the grid mid-expedition does not crash
```

- [ ] **Step 5: Update the architecture docs**

Add the new services to `ARCHITECTURE.md`'s Services table (`archive.py`, `archive_io.py`, `novelty.py`, `descriptor.py`, `goal_source.py`, `imgep_driver.py`, `prompt_driver.py`, `search_driver.py`, `archive_projection.py`, `thumb_cache.py`), `ArchiveState` to the State table, and `ui/archive_window.py` to the UI table.

Add to `CLAUDE.md`'s Important Caveats:

```markdown
- **The archive stores decoded phenotypes, never `z`.** With physics search on,
  `z` is relative to `physics_origin` — the preset loaded at the time. A `z`
  archived under one preset decodes to a different creature under another. See
  `tests/test_physics_origin_roundtrip.py`.

- **`Archive.refresh()` is what makes eviction cheap.** Eviction drops the
  lowest *stored* novelty, which is only meaningful because 64 entries per
  generation are re-scored against the full archive. Turning `refresh_per_gen`
  down to 0 silently degrades eviction into "drop whatever was least novel when
  it was admitted".

- **Novelty is measured against archive ∪ rejects ring.** The archive is gated,
  so without the ring the search has no memory of the regions it just rejected
  and re-explores them forever.
```

- [ ] **Step 6: Run the whole suite and commit**

```bash
python -m pytest -q
git add tools/calibrate_imgep.py docs/testing_checklist.md ARCHITECTURE.md CLAUDE.md state/archive_state.py services/imgep_driver.py
git commit -m "feat: IMGEP calibration tool, testing checklist and docs"
```

---

## Done when

- `python -m pytest -q` is green, and the eight gate files from the foundation plan's Task 7 remain unmodified.
- An unattended Explore run bootstraps, expands, expeditions, and grows `Documents/Fluoddity/archive/` across an app restart.
- The gallery and map both render, and an exported archive entry opens in the normal single-simulation view.
- `sigma_expand` and `liveness_min` were **measured**, not left at their guessed defaults, with the numbers in the commit messages.
- Nothing in `sim.py` or `shaders/` changed.
