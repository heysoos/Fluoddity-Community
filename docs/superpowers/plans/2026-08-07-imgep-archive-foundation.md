# IMGEP Archive Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the headless foundation for IMGEP novelty search — validate that CLIP spreads Fluoddity tiles, extract a `SearchDriver` abstraction from `AutoTournamentService` without changing Auto mode's behaviour, and implement the behaviour descriptor, novelty index and persistent archive.

**Architecture:** Stages 0–2 of `docs/superpowers/specs/2026-08-07-imgep-novelty-archive-design.md`. Nothing here has a UI and nothing here changes what the app does — Task 6 is a pure refactor whose gate is that eight existing test files pass unmodified. The exploration loop that consumes all of this is the second plan, `2026-08-07-imgep-search-and-ui.md`.

**Tech Stack:** Python 3.12, NumPy, Pillow, ONNX Runtime (CLIP ViT-B/32, 512-d embeddings), pytest.

## Global Constraints

- **No new runtime dependencies.** NumPy and Pillow only. No `scikit-learn`, no `umap-learn`, no ANN index.
- **`sim.py` and every file in `shaders/` are untouched.** Not one line.
- **CLIP/evolution imports stay lazy.** `onnxruntime`, `tokenizers` and `cmaes` must never be imported at app startup.
- **All embeddings are L2-normalised float32**, so cosine similarity is a plain dot product everywhere. Shape convention: `(n, 512)`.
- **Fitness/novelty sign convention: higher is better.** Never "loss".
- **Windows platform.** Forward slashes or `os.path`; `rm` not `del` in bash.
- **Atomic writes** for any file that replaces a good copy: write `<name>.tmp`, then `os.replace`.
- **Never `pickle`.** `np.load` always with `allow_pickle=False`.
- Run tests with `python -m pytest` from the repo root. `tests/conftest.py` already fixes `sys.path` and the `ui`-before-`services` import order.

---

## File Structure

| File | Responsibility |
|---|---|
| `tools/clip_spread_check.py` | Stage-0 measurement: does CLIP spread these tiles? |
| `services/descriptor.py` | Trajectory centroid + ASAL liveness from snapshot embeddings |
| `services/novelty.py` | k-NN distances, novelty merge, rejects ring, `NOV^α` sampling |
| `services/archive_io.py` | `ArchiveStore` — index.jsonl, vectors.npz, thumbnails, goals.json |
| `services/archive.py` | `Archive` — admission gates, adaptive threshold, capacity, refresh |
| `services/search_driver.py` | The `SearchDriver` Protocol |
| `services/prompt_driver.py` | Today's optimizer + softmax behaviour, extracted |
| `services/clip_scorer.py` | *(modify)* `embed()` / `embed_text()`; `score()` rebuilt on them |
| `services/capture_health.py` | *(modify)* `is_viable_tile()` |
| `services/auto_tournament_service.py` | *(modify)* keeps the rollout machine, delegates the rest |

---

### Task 1: CLIP spread check (Stage 0 — the kill switch)

Spec §12.1. This decides whether the rest of the plan is worth building. Auto mode proved CLIP has *directed* signal; what is unproven is that these embeddings **spread** enough for k-NN novelty to have a gradient.

**Files:**
- Create: `tools/clip_spread_check.py`
- Test: `tests/test_clip_spread.py`

**Interfaces:**
- Consumes: `services.clip_scorer.CLIPScorer` (existing), `tools.fetch_clip_onnx.MODEL_DIR`
- Produces: `spread_report(embeddings: np.ndarray) -> dict` with keys `n`, `mean_pairwise`, `std_pairwise`, `p05`, `p50`, `p95`, `passes`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clip_spread.py
import numpy as np

from tools.clip_spread_check import spread_report


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def test_identical_embeddings_have_zero_spread_and_fail():
    e = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (32, 1))
    r = spread_report(e)
    assert r["n"] == 32
    assert r["mean_pairwise"] == 0.0
    assert r["passes"] is False


def test_orthogonal_embeddings_have_full_spread_and_pass():
    e = np.eye(8, dtype=np.float32)
    r = spread_report(e)
    # every pair is orthogonal -> cosine distance 1.0
    assert r["mean_pairwise"] == 1.0
    assert r["passes"] is True


def test_report_uses_only_the_upper_triangle():
    """Including the zero self-distances would halve every mean."""
    e = _unit(np.random.default_rng(0).normal(size=(64, 16)).astype(np.float32))
    r = spread_report(e)
    sim = e @ e.T
    iu = np.triu_indices(64, k=1)
    assert r["mean_pairwise"] == float(np.mean(1.0 - sim[iu]))


def test_pass_bar_is_mean_over_015_and_std_over_005():
    rng = np.random.default_rng(1)
    e = _unit(rng.normal(size=(128, 32)).astype(np.float32))
    r = spread_report(e)
    assert r["passes"] == (r["mean_pairwise"] > 0.15 and r["std_pairwise"] > 0.05)


def test_fewer_than_two_embeddings_reports_zero_and_fails():
    r = spread_report(np.zeros((1, 4), dtype=np.float32))
    assert r["n"] == 1
    assert r["mean_pairwise"] == 0.0
    assert r["passes"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clip_spread.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.clip_spread_check'`

- [ ] **Step 3: Write the implementation**

```python
# tools/clip_spread_check.py
"""Stage-0 kill switch: does CLIP spread Fluoddity tiles enough to navigate?

Auto mode already proved CLIP has usable DIRECTED signal on these tiles (a prompt
can be climbed). That is a weaker property than what novelty search needs: a
representation could give a prompt enough gradient to climb while still packing
every tile into a tiny cone, in which case k-NN novelty is sampling noise.

Run this against real captures BEFORE building anything that depends on the
answer:

    python -m tools.clip_spread_check runs/*/frames/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Spec 12.1. Both must hold.
MEAN_BAR = 0.15
STD_BAR = 0.05


def spread_report(embeddings: np.ndarray) -> dict:
    """(N, dim) L2-normalised embeddings -> distribution of pairwise cosine distance.

    Only the strict upper triangle is used. Including the diagonal would fold N
    zero self-distances into the mean and halve it at N=2, which would make a
    perfectly good representation look flat.
    """
    e = np.asarray(embeddings, dtype=np.float32)
    n = int(e.shape[0])
    if n < 2:
        return {"n": n, "mean_pairwise": 0.0, "std_pairwise": 0.0,
                "p05": 0.0, "p50": 0.0, "p95": 0.0, "passes": False}
    d = 1.0 - (e @ e.T)
    iu = np.triu_indices(n, k=1)
    v = d[iu]
    mean = float(np.mean(v))
    std = float(np.std(v))
    return {
        "n": n,
        "mean_pairwise": mean,
        "std_pairwise": std,
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "passes": bool(mean > MEAN_BAR and std > STD_BAR),
    }


def _load_crops(paths: list[Path]) -> np.ndarray:
    from PIL import Image

    out = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.uint8))
    return np.stack(out) if out else np.zeros((0, 224, 224, 3), np.uint8)


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv if Path(a).is_file()]
    if len(paths) < 2:
        print("usage: python -m tools.clip_spread_check <image> <image> ...")
        return 2

    from services.clip_scorer import CLIPScorer
    from tools.fetch_clip_onnx import MODEL_DIR

    scorer = CLIPScorer(MODEL_DIR)
    emb = scorer.embed(_load_crops(paths), n_views=1)
    r = spread_report(emb)

    print(f"images            {r['n']}")
    print(f"mean pairwise     {r['mean_pairwise']:.4f}   (bar > {MEAN_BAR})")
    print(f"std  pairwise     {r['std_pairwise']:.4f}   (bar > {STD_BAR})")
    print(f"p05 / p50 / p95   {r['p05']:.4f} / {r['p50']:.4f} / {r['p95']:.4f}")
    print("VERDICT:          " + ("PASS" if r["passes"] else "FAIL"))
    if not r["passes"]:
        print("\nDo not build the archive on this. Adjust the eval image first -")
        print("colormap, exposure, zoom - and re-measure. See spec 12.1.")
    return 0 if r["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clip_spread.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add tools/clip_spread_check.py tests/test_clip_spread.py
git commit -m "feat: CLIP embedding-spread check, the stage-0 kill switch"
```

- [ ] **Step 6: Run the real measurement and record it**

This step requires the CLIP model to be present and needs at least 200 varied tile images. Collect them from existing run frames plus fresh Auto-mode captures across several different presets:

```bash
python -m tools.clip_spread_check runs/*/frames/*.png
```

Expected: `VERDICT: PASS`, with `mean pairwise > 0.15` and `std pairwise > 0.05`.

**This step is a gate, not a formality.** If it prints FAIL, stop and report the numbers — the remaining tasks in both plans are built on the assumption that it passes, and spec §12.1 requires revisiting the eval image before continuing. Record the printed block verbatim in the commit message of the next task.

---

### Task 2: `CLIPScorer.embed()` and `embed_text()`

Spec §3.4. This is the whole "extend the CLIP comparison" step. `score()` is rebuilt on top and must produce identical numbers.

**Files:**
- Modify: `services/clip_scorer.py:157-196`
- Test: `tests/test_clip_scorer.py` (**additive only** — do not edit existing tests)

**Interfaces:**
- Consumes: existing `_embed_images`, `augment`, `_l2`, `LOGIT_SCALE`, `DEFAULT_DISTRACTORS`
- Produces:
  - `CLIPScorer.embed(images: np.ndarray, n_views: int | None = None) -> np.ndarray` — uint8 `(B,224,224,3)` → float32 `(B*n_views, 512)`, L2-normalised, image-major. `None` means "use the instance default".
  - `CLIPScorer.embed_text(prompts: list[str]) -> np.ndarray` — float32 `(P, 512)`, L2-normalised. Does **not** touch the cached `self._text_emb`.

- [ ] **Step 1: Write the failing tests (append to the existing file)**

```python
# tests/test_clip_scorer.py  --  APPEND, do not modify anything above
def test_embed_returns_l2_normalised_rows():
    s = _scorer_with_stubs()
    e = s.embed(np.zeros((3, 224, 224, 3), dtype=np.uint8), n_views=1)
    assert e.shape == (3, 4)
    assert e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


def test_embed_with_one_view_skips_augmentation():
    s = _scorer_with_stubs()
    s.embed(np.zeros((5, 224, 224, 3), dtype=np.uint8), n_views=1)
    assert sum(s._vision.batch_sizes) == 5, "n_views=1 must not expand the batch"


def test_embed_defaults_to_the_instance_view_count():
    s = _scorer_with_stubs()          # _n_views == 3
    e = s.embed(np.zeros((2, 224, 224, 3), dtype=np.uint8))
    assert len(e) == 6


def test_embed_chunks_large_batches():
    s = _scorer_with_stubs()
    s.embed(np.zeros((200, 224, 224, 3), dtype=np.uint8), n_views=1)
    assert max(s._vision.batch_sizes) <= CLIPScorer.MAX_CHUNK
    assert sum(s._vision.batch_sizes) == 200


def test_score_is_recoverable_from_embed():
    """score() must be exactly the softmax over embed(); if these ever disagree,
    the archive and the prompt objective are measuring different things."""
    s = _scorer_with_stubs()
    emb = np.zeros((3, 4), dtype=np.float32)
    emb[0] = [1.0, 0, 0, 0]
    emb[1] = [0, 1.0, 0, 0]
    emb[2] = [0, 0, 1.0, 0]
    s._text_emb = emb

    images = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    s._rng = np.random.default_rng(7)
    got = s.score(images)

    s._rng = np.random.default_rng(7)
    e = s.embed(images, s._n_views)
    logits = LOGIT_SCALE * (e @ emb.T)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    expected = p[:, 0].reshape(4, -1).mean(axis=1)

    assert np.allclose(got, expected, atol=1e-6)


def test_embed_text_does_not_disturb_the_cached_prompt_embedding():
    s = _scorer_with_stubs()

    class _Tok:
        def encode_batch(self, prompts):
            return [type("E", (), {"ids": [0] * 77, "attention_mask": [1] * 77})()
                    for _ in prompts]

    s._tokenizer = _Tok()
    cached = np.full((7, 4), 0.5, dtype=np.float32)
    s._text_emb = cached
    out = s.embed_text(["a", "b"])
    assert out.shape == (2, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert s._text_emb is cached, "embed_text must not overwrite the score() cache"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clip_scorer.py -v`
Expected: the six new tests FAIL with `AttributeError: 'CLIPScorer' object has no attribute 'embed'`; the fourteen existing tests still PASS.

- [ ] **Step 3: Replace `set_prompt` and `score`, and add the two new methods**

In `services/clip_scorer.py`, replace the body of `set_prompt` (currently lines 157–172) and `score` (currently lines 182–196) with:

```python
    def set_prompt(self, text: str, distractors: list[str] | None = None) -> None:
        """Embed the target prompt (index 0) plus distractors. Cached until the
        next call, so retyping the goal costs one text-encoder pass."""
        prompts = [text] + list(
            DEFAULT_DISTRACTORS if distractors is None else distractors
        )
        self._text_emb = self.embed_text(prompts)
        self._prompt = text

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """(P,) strings -> float32 (P, 512), L2-normalised.

        Deliberately does NOT write self._text_emb: the archive's goal
        embeddings and score()'s prompt+distractor cache are different things
        and must not be able to clobber each other.
        """
        encoded = self._tokenizer.encode_batch(list(prompts))
        feed = {self._text_in: np.array([e.ids for e in encoded], dtype=np.int64)}
        for inp in self._text.get_inputs()[1:]:
            if inp.name == "attention_mask":
                feed["attention_mask"] = np.array(
                    [e.attention_mask for e in encoded], dtype=np.int64
                )
        emb = self._text.run([self._text_out], feed)[0].astype(np.float32)
        return _l2(emb)

    def embed(self, images: np.ndarray, n_views: int | None = None) -> np.ndarray:
        """uint8 (B,224,224,3) -> float32 (B*n_views, 512), L2-normalised.

        Output is image-major: [img0 v0, img0 v1, ..., img1 v0, ...].

        n_views=1 skips augmentation entirely. Augmentation is an
        anti-adversarial defence for a DIRECTED objective; for novelty it would
        turn three random sub-crops of one tile into three competing archive
        descriptors for one behaviour.
        """
        v = self._n_views if n_views is None else int(n_views)
        return self._embed_images(augment(images, v, self._rng))

    def score(self, images: np.ndarray) -> np.ndarray:
        """uint8 (B,224,224,3) -> float32 (B,). Softmax probability of the
        target prompt against the distractor set, averaged over augmented views.
        """
        if self._text_emb is None:
            raise RuntimeError("set_prompt() must be called before score()")
        b = len(images)
        emb = self.embed(images, self._n_views)
        logits = LOGIT_SCALE * (emb @ self._text_emb.T)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        target = probs[:, 0].reshape(b, -1)  # (B, n_views)
        return target.mean(axis=1).astype(np.float32)
```

- [ ] **Step 4: Run the whole scorer suite**

Run: `python -m pytest tests/test_clip_scorer.py -v`
Expected: PASS, 20 tests. The fourteen pre-existing tests must be **unmodified** and still green.

- [ ] **Step 5: Commit**

```bash
git add services/clip_scorer.py tests/test_clip_scorer.py
git commit -m "feat: CLIPScorer.embed and embed_text; score rebuilt on top"
```

---

### Task 3: `is_viable_tile()`

Spec §5.2. A per-tile version of the existing batch-level `check_capture`, so the viability gate can reject one dead tile without discarding the generation.

**Files:**
- Modify: `services/capture_health.py`
- Test: `tests/test_capture_health.py` (**additive only**)

**Interfaces:**
- Produces: `services.capture_health.is_viable_tile(crop: np.ndarray) -> bool`

- [ ] **Step 1: Write the failing tests (append to the existing file)**

```python
# tests/test_capture_health.py  --  APPEND
import numpy as np

from services.capture_health import check_capture, is_viable_tile


def test_viable_tile_accepts_an_ordinary_crop():
    crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    assert is_viable_tile(crop) is True


def test_viable_tile_rejects_pure_black():
    assert is_viable_tile(np.zeros((224, 224, 3), dtype=np.uint8)) is False


def test_viable_tile_rejects_nearly_black():
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    crop[0, 0] = 255                      # max > 0 but the mean is ~0.02
    assert is_viable_tile(crop) is False


def test_viable_tile_rejects_blown_out():
    assert is_viable_tile(np.full((224, 224, 3), 255, dtype=np.uint8)) is False


def test_viable_tile_rejects_empty():
    assert is_viable_tile(np.zeros((0, 224, 3), dtype=np.uint8)) is False


def test_viable_tile_agrees_with_check_capture_on_single_tile_batches():
    for value in (0, 1, 128, 254, 255):
        crop = np.full((224, 224, 3), value, dtype=np.uint8)
        batch = crop[None, ...]
        assert is_viable_tile(crop) == (check_capture(batch) is None), value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_capture_health.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_viable_tile'`

- [ ] **Step 3: Add the function**

Insert into `services/capture_health.py` immediately after `check_capture`:

```python
def is_viable_tile(crop: np.ndarray) -> bool:
    """One tile's crop -> is it worth measuring at all?

    Same thresholds as check_capture, applied per tile rather than per batch.
    In Explore mode one dead tile must be rejected on its own; discarding the
    whole generation because a single genome died would throw away 15 useful
    samples with it.
    """
    if crop.size == 0:
        return False
    if int(crop.max()) == 0:
        return False
    return 2.0 <= float(crop.mean()) <= 253.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_capture_health.py -v`
Expected: PASS (the 6 new tests plus the pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add services/capture_health.py tests/test_capture_health.py
git commit -m "feat: per-tile viability check for the archive admission gate"
```

---

### Task 4: `descriptor.py` — trajectory centroid and ASAL liveness

Spec §4. Turns a generation's S snapshot embeddings into one behaviour descriptor per tile plus a liveness scalar.

**Files:**
- Create: `services/descriptor.py`
- Test: `tests/test_descriptor.py`

**Interfaces:**
- Produces:
  - `descriptor(snaps: np.ndarray) -> np.ndarray` — `(S, n, dim)` → `(n, dim)` float32, L2-normalised
  - `liveness(snaps: np.ndarray) -> np.ndarray` — `(S, n, dim)` → `(n,)` float32 in `[0, 2]`, 0 = frozen. Returns zeros when `S < 2`.
  - `stack_snapshots(per_snapshot: list[np.ndarray]) -> np.ndarray` — list of `(n, dim)` → `(S, n, dim)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_descriptor.py
import numpy as np
import pytest

from services.descriptor import descriptor, liveness, stack_snapshots


def _basis(i, dim=4, n=1):
    e = np.zeros((n, dim), dtype=np.float32)
    e[:, i] = 1.0
    return e


def test_stack_snapshots_shape():
    s = stack_snapshots([np.zeros((3, 4), np.float32) for _ in range(5)])
    assert s.shape == (5, 3, 4)
    assert s.dtype == np.float32


def test_descriptor_is_unit_norm():
    rng = np.random.default_rng(0)
    snaps = rng.normal(size=(4, 6, 8)).astype(np.float32)
    snaps /= np.linalg.norm(snaps, axis=-1, keepdims=True)
    b = descriptor(snaps)
    assert b.shape == (6, 8)
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-5)


def test_descriptor_of_identical_snapshots_is_that_snapshot():
    e = _basis(1, n=3)
    snaps = np.stack([e, e, e, e])
    assert np.allclose(descriptor(snaps), e, atol=1e-6)


def test_descriptor_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(S, n, dim\)"):
        descriptor(np.zeros((4, 8), dtype=np.float32))


def test_liveness_is_zero_for_a_frozen_pattern():
    e = _basis(0, n=2)
    snaps = np.stack([e, e, e, e])
    assert np.allclose(liveness(snaps), 0.0, atol=1e-6)


def test_liveness_is_one_for_mutually_orthogonal_snapshots():
    snaps = np.stack([_basis(0), _basis(1), _basis(2), _basis(3)])
    assert np.allclose(liveness(snaps), 1.0, atol=1e-6)


def test_liveness_matches_the_asal_formula_by_hand():
    """ASAL Eq.3: L = 1 - mean_{s>0} max_{s'<s} <e_s, e_s'>.

    e0 == e1, e2 orthogonal to both:
      s=1 -> max(<e1,e0>)        = 1.0
      s=2 -> max(<e2,e0>,<e2,e1>) = 0.0
      mean = 0.5  ->  L = 0.5
    """
    snaps = np.stack([_basis(0), _basis(0), _basis(1)])
    assert liveness(snaps) == pytest.approx([0.5], abs=1e-6)


def test_liveness_uses_historical_max_not_consecutive_pairs():
    """A pattern that returns to an earlier state is not novel just because the
    step before it differed. e2 == e0 must score 0 on its own term."""
    snaps = np.stack([_basis(0), _basis(1), _basis(0)])
    # s=1 -> <e1,e0> = 0 ; s=2 -> max(<e2,e0>=1, <e2,e1>=0) = 1 ; mean 0.5
    assert liveness(snaps) == pytest.approx([0.5], abs=1e-6)


def test_liveness_is_zero_when_there_is_only_one_snapshot():
    """A single snapshot carries no evidence of change. The caller must disable
    the liveness gate at snapshots_per_gen == 1."""
    assert np.array_equal(liveness(_basis(0, n=5)[None]), np.zeros(5, np.float32))


def test_liveness_is_per_tile():
    frozen = np.stack([_basis(0), _basis(0), _basis(0)])[:, 0]        # (3, 4)
    moving = np.stack([_basis(0), _basis(1), _basis(2)])[:, 0]        # (3, 4)
    snaps = np.stack([frozen, moving], axis=1)                        # (3, 2, 4)
    out = liveness(snaps)
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_descriptor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.descriptor'`

- [ ] **Step 3: Write the implementation**

```python
# services/descriptor.py
"""Behaviour descriptor and liveness, from one generation's snapshot embeddings.

The descriptor is the TRAJECTORY CENTROID, not the final frame. E&E uses the
final state; Fluoddity patterns are watched in motion, and a single frame
discards what the pattern does - which is most of what distinguishes one
creature from another here. The centroid is also insensitive to the exact
stopping step, which matters because steps_per_gen is user-adjustable.

Known failure mode, accepted: a pattern oscillating between two dissimilar
states gets a centroid resembling neither, and may land next to an unrelated
pattern. Liveness is stored alongside precisely so this case is visible.

Liveness is ASAL's open-endedness objective (arXiv:2412.17799 Eq. 3),
sign-flipped so that higher is better like everything else in this codebase.
"""
from __future__ import annotations

import numpy as np


def stack_snapshots(per_snapshot: list[np.ndarray]) -> np.ndarray:
    """[(n, dim)] * S -> (S, n, dim) float32."""
    return np.stack([np.asarray(e, dtype=np.float32) for e in per_snapshot], axis=0)


def _check(snaps: np.ndarray) -> np.ndarray:
    a = np.asarray(snaps, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected (S, n, dim), got {a.shape}")
    return a


def descriptor(snaps: np.ndarray) -> np.ndarray:
    """(S, n, dim) unit-norm embeddings -> (n, dim) unit-norm centroid."""
    a = _check(snaps)
    m = a.mean(axis=0)
    return (m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8)
            ).astype(np.float32)


def liveness(snaps: np.ndarray) -> np.ndarray:
    """(S, n, dim) -> (n,) float32.

        L = 1 - mean_{s>0} max_{s'<s} <e_s, e_s'>

    A frozen pattern scores 0; a pattern whose every state is orthogonal to all
    its predecessors scores 1. The max is over ALL earlier snapshots, not just
    the previous one, so returning to an earlier state is correctly not novel.

    S < 2 returns zeros: one snapshot is no evidence of change. Callers must
    disable the liveness gate at snapshots_per_gen == 1 rather than reject
    everything.
    """
    a = _check(snaps)
    s, n, _ = a.shape
    if s < 2:
        return np.zeros(n, dtype=np.float32)
    e = np.transpose(a, (1, 0, 2))              # (n, S, dim)
    sim = np.einsum("nsd,ntd->nst", e, e)       # (n, S, S)
    worst = np.empty((n, s - 1), dtype=np.float32)
    for i in range(1, s):
        worst[:, i - 1] = sim[:, i, :i].max(axis=1)
    return (1.0 - worst.mean(axis=1)).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_descriptor.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add services/descriptor.py tests/test_descriptor.py
git commit -m "feat: trajectory-centroid descriptor and ASAL liveness"
```

---

### Task 5: `novelty.py` — k-NN novelty, rejects ring, NOV^alpha sampling

Spec §5.1. Note `knn_novelty` takes references **separately** rather than concatenated: the archive is 20k×512 and copying it every generation to glue on a 2k rejects ring would cost more than the novelty computation itself.

**Files:**
- Create: `services/novelty.py`
- Test: `tests/test_novelty.py`

**Interfaces:**
- Produces:
  - `knn_distances(queries, reference, k=10, exclude_self=False) -> np.ndarray` — `(n, m')` sorted ascending, `m' = min(k, len(reference))`, or `min(k, len-1)` when excluding self. Empty reference → `(n, 0)`.
  - `novelty_from_distances(dists: list[np.ndarray], k=10) -> np.ndarray` — merge per-reference distance blocks, take the `k` smallest overall, return their mean. All-empty → ones.
  - `knn_novelty(queries, reference, k=10, exclude_self=False) -> np.ndarray` — the single-reference convenience wrapper.
  - `sample_by_novelty(novelty, n, rng, alpha=4.0) -> np.ndarray` — `(n,)` int indices, `p ∝ NOV^α`.
  - `class RejectsRing(capacity=2048, dim=512)` with `.add(embeddings)`, `.view() -> (m, dim)`, `.clear()`, `len()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_novelty.py
import numpy as np
import pytest

from services.novelty import (
    RejectsRing,
    knn_distances,
    knn_novelty,
    novelty_from_distances,
    sample_by_novelty,
)


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def test_knn_distances_are_sorted_ascending():
    ref = _unit(np.eye(6))
    q = _unit(np.array([[1.0, 0.2, 0.0, 0.0, 0.0, 0.0]]))
    d = knn_distances(q, ref, k=4)
    assert d.shape == (1, 4)
    assert np.all(np.diff(d[0]) >= -1e-6)


def test_knn_distances_match_a_brute_force_reference():
    rng = np.random.default_rng(3)
    ref = _unit(rng.normal(size=(50, 8)))
    q = _unit(rng.normal(size=(7, 8)))
    got = knn_distances(q, ref, k=5)
    full = np.sort(1.0 - (q @ ref.T), axis=1)[:, :5]
    assert np.allclose(got, full, atol=1e-6)


def test_knn_distances_clamps_k_to_the_reference_size():
    ref = _unit(np.eye(3))
    assert knn_distances(_unit(np.eye(3)[:1]), ref, k=10).shape == (1, 3)


def test_knn_distances_on_an_empty_reference_returns_no_columns():
    assert knn_distances(_unit(np.eye(2)), np.zeros((0, 2), np.float32), k=5).shape == (2, 0)


def test_exclude_self_drops_the_zero_distance():
    ref = _unit(np.eye(5))
    d = knn_distances(ref, ref, k=2, exclude_self=True)
    assert d.shape == (5, 2)
    assert np.all(d > 0.5), "the self-match (distance 0) must be dropped"


def test_exclude_self_on_a_single_entry_returns_no_columns():
    ref = _unit(np.eye(4)[:1])
    assert knn_distances(ref, ref, k=3, exclude_self=True).shape == (1, 0)


def test_novelty_from_distances_takes_the_k_smallest_across_blocks():
    a = np.array([[0.1, 0.9]], dtype=np.float32)
    b = np.array([[0.2, 0.8]], dtype=np.float32)
    # merged and sorted: 0.1 0.2 0.8 0.9 -> mean of the 3 smallest
    assert novelty_from_distances([a, b], k=3) == pytest.approx([(0.1 + 0.2 + 0.8) / 3])


def test_novelty_from_distances_with_no_reference_is_maximally_novel():
    out = novelty_from_distances([np.zeros((4, 0), np.float32)], k=10)
    assert np.array_equal(out, np.ones(4, np.float32))


def test_knn_novelty_is_the_mean_of_the_k_nearest_distances():
    ref = _unit(np.eye(4))
    q = _unit(np.array([[1.0, 0.0, 0.0, 0.0]]))
    # distances: 0, 1, 1, 1 -> k=3 mean = (0 + 1 + 1) / 3
    assert knn_novelty(q, ref, k=3) == pytest.approx([2.0 / 3.0], abs=1e-6)


def test_knn_novelty_on_an_empty_archive_is_one():
    out = knn_novelty(_unit(np.eye(3)), np.zeros((0, 3), np.float32), k=10)
    assert np.array_equal(out, np.ones(3, np.float32))


def test_rejects_ring_reports_its_contents():
    r = RejectsRing(capacity=4, dim=3)
    assert len(r) == 0
    r.add(_unit(np.eye(3)[:2]))
    assert len(r) == 2
    assert r.view().shape == (2, 3)
    assert r.view().dtype == np.float32


def test_rejects_ring_overwrites_oldest_first():
    r = RejectsRing(capacity=3, dim=2)
    for i in range(5):
        r.add(np.array([[float(i), 0.0]], dtype=np.float32))
    assert len(r) == 3
    assert sorted(r.view()[:, 0].tolist()) == [2.0, 3.0, 4.0]


def test_rejects_ring_clear():
    r = RejectsRing(capacity=4, dim=2)
    r.add(np.zeros((3, 2), np.float32))
    r.clear()
    assert len(r) == 0
    assert r.view().shape == (0, 2)


def test_sample_by_novelty_favours_high_novelty():
    rng = np.random.default_rng(0)
    nov = np.array([0.1, 0.9], dtype=np.float32)
    idx = sample_by_novelty(nov, 4000, rng, alpha=4.0)
    share = float((idx == 1).mean())
    # 0.9^4 / (0.9^4 + 0.1^4) = 0.99985
    assert share > 0.99


def test_sample_by_novelty_with_alpha_zero_is_uniform():
    rng = np.random.default_rng(1)
    nov = np.array([0.01, 0.5, 0.99], dtype=np.float32)
    idx = sample_by_novelty(nov, 6000, rng, alpha=0.0)
    counts = np.bincount(idx, minlength=3) / 6000.0
    assert np.allclose(counts, 1 / 3, atol=0.03)


def test_sample_by_novelty_falls_back_to_uniform_when_all_weights_vanish():
    rng = np.random.default_rng(2)
    idx = sample_by_novelty(np.zeros(3, np.float32), 3000, rng, alpha=4.0)
    counts = np.bincount(idx, minlength=3) / 3000.0
    assert np.allclose(counts, 1 / 3, atol=0.04)


def test_sample_by_novelty_ignores_negative_novelty():
    rng = np.random.default_rng(4)
    idx = sample_by_novelty(np.array([-1.0, 1.0], np.float32), 500, rng, alpha=4.0)
    assert np.all(idx == 1)


def test_sample_by_novelty_on_an_empty_archive_raises():
    with pytest.raises(ValueError, match="empty"):
        sample_by_novelty(np.zeros(0, np.float32), 3, np.random.default_rng(0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_novelty.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.novelty'`

- [ ] **Step 3: Write the implementation**

```python
# services/novelty.py
"""k-nearest-neighbour novelty in CLIP embedding space.

    NOV(b) = (1/k) * sum_{j in kNN(b)} (1 - <b, b_j>)

k=10 and the parent-sampling exponent alpha=4 are E&E's tuned values. All
vectors are L2-normalised, so cosine similarity is a plain dot product and the
whole thing is one matmul.

References are taken SEPARATELY rather than concatenated. Novelty is measured
against archive UNION rejects-ring (the classic Lehman-Stanley formulation:
archive plus current population), and gluing a 2k ring onto a 20k archive every
generation would copy 40 MB - more than the novelty computation costs. Instead
each reference block is reduced to its k nearest distances and the blocks are
merged.
"""
from __future__ import annotations

import numpy as np


def knn_distances(queries: np.ndarray, reference: np.ndarray, k: int = 10,
                  exclude_self: bool = False) -> np.ndarray:
    """(n, dim) x (m, dim) -> (n, k') cosine distances, sorted ascending.

    k' = min(k, m), or min(k, m-1) when exclude_self. An empty reference (or a
    single-entry one with exclude_self) yields (n, 0), which
    novelty_from_distances treats as 'no evidence'.

    exclude_self drops the single nearest match, which for a query drawn from
    the reference itself is the zero-distance self-match. Used when refreshing
    an archive entry's own novelty.
    """
    q = np.asarray(queries, dtype=np.float32).reshape(len(queries), -1)
    r = np.asarray(reference, dtype=np.float32)
    n, m = len(q), len(r)
    if m == 0 or (exclude_self and m == 1):
        return np.zeros((n, 0), dtype=np.float32)

    want = int(min(k + (1 if exclude_self else 0), m))
    dist = 1.0 - (q @ r.T)
    if want < m:
        idx = np.argpartition(dist, want - 1, axis=1)[:, :want]
        d = np.take_along_axis(dist, idx, axis=1)
    else:
        d = dist
    d = np.sort(d, axis=1)
    if exclude_self:
        d = d[:, 1:]
    return np.ascontiguousarray(d, dtype=np.float32)


def novelty_from_distances(dists: list[np.ndarray], k: int = 10) -> np.ndarray:
    """Merge per-reference distance blocks into one novelty value per query.

    With no reference at all every query is maximally novel by convention (1.0),
    which is what makes a cold archive bootstrap instead of dividing by zero.
    """
    blocks = [np.asarray(d, dtype=np.float32) for d in dists if d.size or d.shape[0]]
    blocks = [b for b in blocks if b.shape[1] > 0]
    if not blocks:
        n = int(dists[0].shape[0]) if dists else 0
        return np.ones(n, dtype=np.float32)
    d = np.concatenate(blocks, axis=1)
    d = np.sort(d, axis=1)[:, : int(min(k, d.shape[1]))]
    return d.mean(axis=1).astype(np.float32)


def knn_novelty(queries: np.ndarray, reference: np.ndarray, k: int = 10,
                exclude_self: bool = False) -> np.ndarray:
    """Single-reference convenience wrapper."""
    return novelty_from_distances(
        [knn_distances(queries, reference, k, exclude_self)], k
    )


def sample_by_novelty(novelty: np.ndarray, n: int, rng, alpha: float = 4.0):
    """Indices sampled with p proportional to NOV^alpha, with replacement.

    alpha=0 is uniform - E&E's 'Random GA' baseline, reachable from the UI
    without a separate code path. Negative novelty cannot occur for unit
    vectors but is clamped anyway so a NaN-repaired value cannot flip a weight
    negative and poison the distribution.
    """
    v = np.asarray(novelty, dtype=np.float64).reshape(-1)
    m = v.size
    if m == 0:
        raise ValueError("cannot sample a parent from an empty archive")
    w = np.clip(v, 0.0, None) ** float(alpha)
    total = float(w.sum())
    p = (w / total) if (total > 0.0 and np.isfinite(total)) else np.full(m, 1.0 / m)
    return rng.choice(m, size=int(n), replace=True, p=p)


class RejectsRing:
    """The most recent rejected descriptors, for novelty only.

    Without this, novelty is measured against a filtered history: a gated
    archive has no memory of the regions it just rejected, so the search
    re-explores them forever. Never persisted - it is about the last few
    minutes, not the run.
    """

    def __init__(self, capacity: int = 2048, dim: int = 512):
        self.capacity = int(capacity)
        self._buf = np.zeros((self.capacity, int(dim)), dtype=np.float32)
        self._n = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._n

    def add(self, embeddings: np.ndarray) -> None:
        e = np.asarray(embeddings, dtype=np.float32).reshape(-1, self._buf.shape[1])
        for row in e:
            self._buf[self._pos] = row
            self._pos = (self._pos + 1) % self.capacity
            self._n = min(self._n + 1, self.capacity)

    def view(self) -> np.ndarray:
        return self._buf[: self._n]

    def clear(self) -> None:
        self._n = 0
        self._pos = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_novelty.py -v`
Expected: PASS, 18 tests

- [ ] **Step 5: Commit**

```bash
git add services/novelty.py tests/test_novelty.py
git commit -m "feat: kNN novelty, rejects ring, and NOV^alpha parent sampling"
```

---

### Task 6: `SearchDriver` protocol and `PromptDriver`

Spec §3.1. Create the abstraction and the extraction *alongside* the existing service. Nothing is rewired yet — that is Task 7 — so this task cannot break Auto mode.

**Files:**
- Create: `services/search_driver.py`
- Create: `services/prompt_driver.py`
- Test: `tests/test_search_driver.py`

**Interfaces:**
- Consumes: `services.optimizers.make_optimizer`, `services.genome_spec.BRAIN_SPEC`
- Produces:
  - `services.search_driver.SearchDriver` — Protocol with `name`, `set_spec(spec)`, `ask(n) -> (n, dim) float32`, `tell(z, snapshots) -> (n,) float32`, `status() -> dict`, `checkpoint_state() -> dict`, `restore(d)`, `reset()`
  - `services.prompt_driver.PromptDriver(tournament, scorer=None, spec=BRAIN_SPEC)` with mutable attributes `algorithm`, `sigma0`, `base_seed`, `scorer`, and read-only `optimizer`, `prompt`, `sigma`; plus `set_prompt(text)`, `set_x0(z)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_search_driver.py
import numpy as np
import pytest

from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.prompt_driver import PromptDriver
from services.tournament_service import TournamentService


class FakeScorer:
    """Scores each tile by mean brightness, matching test_auto_tournament_service."""

    def __init__(self):
        self.prompt = ""

    def set_prompt(self, text, distractors=None):
        self.prompt = text

    def score(self, images):
        return images.reshape(len(images), -1).mean(axis=1).astype(np.float32) / 255.0


def make(grid=2):
    ts = TournamentService(grid=grid)
    ts.init_population()
    return PromptDriver(ts, FakeScorer()), ts


def crops(n, value=128):
    return np.full((n, 224, 224, 3), value, dtype=np.uint8)


def test_ask_returns_the_requested_shape_and_dtype():
    d, _ = make()
    z = d.ask(4)
    assert z.shape == (4, BRAIN_SPEC.dim)
    assert z.dtype == np.float32


def test_optimizer_is_built_lazily_on_first_ask():
    d, _ = make()
    assert d.optimizer is None
    d.ask(4)
    assert d.optimizer is not None
    assert d.optimizer.popsize == 4


def test_ask_rebuilds_when_the_population_size_changes():
    """cmaes fixes popsize at construction and asserts on it in tell()."""
    d, _ = make()
    d.ask(4)
    d.ask(16)
    assert d.optimizer.popsize == 16


def test_tell_averages_the_score_over_snapshots():
    d, _ = make()
    z = d.ask(4)
    fit = d.tell(z, [crops(4, 0), crops(4, 200)])
    assert fit[0] == pytest.approx(100 / 255.0, rel=1e-3)


def test_tell_replaces_non_finite_scores_with_the_generation_minimum():
    d, _ = make()
    d.scorer.score = lambda imgs: np.array([np.nan, 0.2, 0.5, 0.1], dtype=np.float32)
    z = d.ask(4)
    fit = d.tell(z, [crops(4)])
    assert np.all(np.isfinite(fit))
    assert fit[0] == pytest.approx(0.1)
    assert d.status()["nan_replaced"] == 1


def test_tell_with_no_snapshots_returns_zeros():
    d, _ = make()
    z = d.ask(4)
    assert np.array_equal(d.tell(z, []), np.zeros(4, np.float32))


def test_elite_injection_ranks_selected_tiles_first_and_clears_the_selection():
    d, ts = make()
    z = d.ask(4)
    imgs = crops(4, 10)
    imgs[2] = 250                 # tile 2 is genuinely brightest
    ts.toggle_select(0)           # but the human picks tile 0
    fit = d.tell(z, [imgs])
    assert fit[0] == fit.max()
    assert ts.selected == set()
    assert d.status()["elites_injected"] == 1


def test_set_prompt_forwards_to_the_scorer():
    d, _ = make()
    d.set_prompt("coral")
    assert d.prompt == "coral"
    assert d.scorer.prompt == "coral"


def test_set_x0_builds_an_optimizer_centred_on_that_genome():
    d, _ = make()
    z0 = np.full(BRAIN_SPEC.dim, 0.25, dtype=np.float32)
    d.set_x0(z0)
    assert d.optimizer is not None
    assert np.allclose(d.ask(4).mean(axis=0), 0.25, atol=0.6)


def test_set_spec_discards_the_optimizer():
    """The search dimension changed; a covariance for the old one is meaningless."""
    d, _ = make()
    d.ask(4)
    d.set_spec(BRAIN_PHYSICS_SPEC)
    assert d.optimizer is None
    assert d.ask(4).shape == (4, BRAIN_PHYSICS_SPEC.dim)


def test_set_spec_to_the_same_spec_keeps_the_optimizer():
    d, _ = make()
    d.ask(4)
    before = d.optimizer
    d.set_spec(BRAIN_SPEC)
    assert d.optimizer is before


def test_reset_discards_the_optimizer_and_the_loaded_start_point():
    d, _ = make()
    d.set_x0(np.full(BRAIN_SPEC.dim, 2.0, dtype=np.float32))
    d.reset()
    assert d.optimizer is None
    assert np.allclose(d.ask(64).mean(axis=0), 0.0, atol=0.4)


def test_sigma_falls_back_to_sigma0_before_the_first_ask():
    d, _ = make()
    d.sigma0 = 0.3
    assert d.sigma == pytest.approx(0.3)


def test_checkpoint_roundtrip_restores_the_optimizer_and_prompt():
    d, _ = make()
    d.set_prompt("coral")
    z = d.ask(4)
    d.tell(z, [crops(4)])
    st = d.checkpoint_state()
    assert st["optimizer_name"] == "CMA-ES"
    assert st["prompt"] == "coral"

    d2, _ = make()
    d2.restore(st)
    assert d2.prompt == "coral"
    assert d2.optimizer is not None
    assert np.allclose(d2.optimizer.ask(4), d.optimizer.ask(4), atol=1e-5)


def test_status_reports_what_the_log_and_ui_need():
    d, _ = make()
    d.set_prompt("coral")
    d.tell(d.ask(4), [crops(4)])
    st = d.status()
    assert set(st) >= {"prompt", "algorithm", "sigma", "nan_replaced", "elites_injected"}
    assert st["sigma"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_search_driver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.prompt_driver'`

- [ ] **Step 3: Write the protocol**

```python
# services/search_driver.py
"""The interface AutoTournamentService searches through.

The service owns the rollout state machine - WRITE_RULES, STEP, CAPTURE x S,
SCORE, spread across real application frames, with abort-on-resize. That machine
is identical for every search. What differs is only two decisions: which genomes
to run next, and what to do with the resulting images.

A driver receives RAW CROPS rather than scores, which is what lets PromptDriver
call scorer.score() while ImgepDriver calls scorer.embed(). The consequence is
that the service no longer imports or knows about CLIP at all.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class SearchDriver(Protocol):
    name: str

    def set_spec(self, spec) -> None:
        """The genome layout changed (physics search toggled). Any optimizer
        state is for a different dimension and must be discarded. Must not
        raise."""

    def ask(self, n: int) -> np.ndarray:
        """(n, spec.dim) float32 - one search vector per tile."""

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray]) -> np.ndarray:
        """snapshots is the service's crop buffer verbatim: a list of S arrays
        of shape (n, 224, 224, 3) uint8, in capture order. May be empty when a
        generation produced no captures.

        Returns (n,) float32 - a per-tile score for the UI and the log ONLY.
        Whatever learning the driver does, it does internally."""

    def status(self) -> dict:
        """Free-form, for the UI status line and the JSONL record."""

    def checkpoint_state(self) -> dict: ...

    def restore(self, d: dict) -> None: ...

    def reset(self) -> None:
        """Discard all learned state. The archive, if any, survives."""
```

- [ ] **Step 4: Write `PromptDriver`**

```python
# services/prompt_driver.py
"""Today's Auto (CLIP) behaviour, extracted behind SearchDriver.

This is a mechanical extraction of AutoTournamentService's optimizer half, not a
rewrite. Its correctness gate is that the eight pre-existing auto-mode test files
pass UNMODIFIED after Task 7 rewires the service.
"""
from __future__ import annotations

import numpy as np

from services.genome_spec import BRAIN_SPEC
from services.optimizers import make_optimizer


class PromptDriver:
    name = "prompt"

    def __init__(self, tournament, scorer=None, spec=BRAIN_SPEC):
        self.tournament = tournament
        self.scorer = scorer
        self.spec = spec
        self.algorithm = "CMA-ES"
        self.sigma0 = 0.5
        self.base_seed = 1000
        self.prompt = ""
        self._optimizer = None
        self._x0 = None
        self._n_nan = 0
        self._n_elites = 0

    # ---- state exposed to the service and the UI -----------------------

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def sigma(self) -> float:
        return float(self._optimizer.sigma) if self._optimizer else float(self.sigma0)

    def status(self) -> dict:
        return {
            "prompt": self.prompt,
            "algorithm": self.algorithm,
            "sigma": self.sigma,
            "nan_replaced": int(self._n_nan),
            "elites_injected": int(self._n_elites),
        }

    # ---- configuration -------------------------------------------------

    def set_spec(self, spec) -> None:
        if spec is not self.spec:
            self.spec = spec
            self._optimizer = None

    def set_prompt(self, text: str) -> None:
        """Changing the prompt keeps the learned covariance and simply starts
        climbing a new landscape."""
        self.prompt = text
        if self.scorer is not None:
            self.scorer.set_prompt(text)

    def set_x0(self, z: np.ndarray) -> None:
        """Load a genome as the search starting point. Discards optimizer state;
        sigma, algorithm and grid come from the UI, not the file."""
        self._optimizer = None
        self._x0 = np.asarray(z, dtype=np.float64).reshape(-1)
        self._ensure(self.tournament.tiles)

    def reset(self) -> None:
        self._optimizer = None
        self._x0 = None

    # ---- the driver interface ------------------------------------------

    def _ensure(self, popsize: int) -> None:
        # The population size is fixed at construction (cmaes asserts on it in
        # tell()). If the grid changed by any route that did not reset us,
        # rebuild rather than crash on the next tell.
        if self._optimizer is not None and self._optimizer.popsize != popsize:
            self._optimizer = None
        if self._optimizer is None:
            self._optimizer = make_optimizer(
                self.algorithm, self.spec.dim, popsize,
                self.sigma0, self.base_seed, self._x0,
            )

    def ask(self, n: int) -> np.ndarray:
        self._ensure(int(n))
        return self._optimizer.ask(int(n))

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray]) -> np.ndarray:
        n = len(z)
        if self.scorer is None or not snapshots:
            fit = np.zeros(n, dtype=np.float32)
        else:
            per_snap = [np.asarray(self.scorer.score(c), dtype=np.float32)
                        for c in snapshots]
            fit = np.mean(np.stack(per_snap, axis=0), axis=0).astype(np.float32)

        bad = ~np.isfinite(fit)
        self._n_nan = int(bad.sum())
        if bad.any():
            finite = fit[~bad]
            fit[bad] = float(finite.min()) if finite.size else 0.0

        # Elite injection: selected tiles are already in this population, so
        # this overwrites their fitness to force them to the top ranks.
        selected = sorted(self.tournament.selected)
        self._n_elites = len(selected)
        if selected:
            top = float(fit.max())
            for rank, tile in enumerate(selected):
                if 0 <= tile < n:
                    fit[tile] = top + 1e-3 * (len(selected) - rank)

        self._ensure(n)
        self._optimizer.tell(z, fit)
        self.tournament.selected.clear()
        return fit

    # ---- checkpointing -------------------------------------------------

    def checkpoint_state(self) -> dict:
        best_z, best_f = (self._optimizer.best() if self._optimizer
                          else (np.zeros(self.spec.dim, np.float32), -np.inf))
        return {
            "optimizer_name": self.algorithm,
            "optimizer_state": self._optimizer.state_dict() if self._optimizer else {},
            "prompt": self.prompt,
            "distractors": [],
            "best_z": best_z,
            "best_fitness": float(best_f) if np.isfinite(best_f) else 0.0,
        }

    def restore(self, d: dict) -> None:
        self.algorithm = str(d.get("optimizer_name", self.algorithm))
        self._optimizer = None
        self._x0 = None
        self._ensure(self.tournament.tiles)
        if d.get("optimizer_state"):
            self._optimizer.load_state_dict(d["optimizer_state"])
        self.set_prompt(str(d.get("prompt", "")))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_search_driver.py -v`
Expected: PASS, 15 tests

- [ ] **Step 6: Verify nothing else moved**

Run: `python -m pytest -q`
Expected: the whole suite passes. Nothing is wired to `PromptDriver` yet, so this is purely a "no accidental import breakage" check.

- [ ] **Step 7: Commit**

```bash
git add services/search_driver.py services/prompt_driver.py tests/test_search_driver.py
git commit -m "feat: SearchDriver protocol and PromptDriver, not yet wired in"
```

---

### Task 7: Rewire `AutoTournamentService` onto the driver

Spec §3.2, §3.3. **This is the risk in the whole plan.** It touches working, shipped code. The gate is that eight existing test files pass *unmodified*.

The public surface must be preserved exactly, because the existing tests and `command_handler.py` reach into it: `svc.optimizer`, `svc.scorer` (mutated in place by a test), `svc.prompt`, `svc.set_prompt()`, `svc.set_x0()`, `svc.sigma`, `svc.current_z`, `svc.fitness`, `svc.configure()`.

**Files:**
- Modify: `services/auto_tournament_service.py`
- Test: no new test file — the gate is the existing suite

**Interfaces:**
- Consumes: `PromptDriver`, `SearchDriver` from Task 6
- Produces: `AutoTournamentService(tournament_service, scorer=None, logger=None, spec=BRAIN_SPEC, base_seed=1000, driver=None)` — same behaviour, plus a `driver` injection point that plan 2 uses for `ImgepDriver`.

- [ ] **Step 1: Record the baseline**

```bash
python -m pytest -q > ../baseline.txt 2>&1; tail -3 ../baseline.txt
```

Expected: a passing summary line. Write down the exact test count — Step 6 must match it.

- [ ] **Step 2: Replace the imports and `__init__`**

In `services/auto_tournament_service.py`, replace the import block (currently lines 16–18) with:

```python
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.physics_genome import decode_physics
from services.prompt_driver import PromptDriver
```

`make_optimizer` and `decode` are no longer used here — the driver owns the optimizer, and the service uses `spec.decode`.

Replace `__init__`'s first block (currently lines 43–49) with:

```python
    def __init__(self, tournament_service, scorer=None, logger=None,
                 spec=BRAIN_SPEC, base_seed: int = 1000, driver=None):
        self.tournament = tournament_service
        self.logger = logger
        self.spec = spec
        self.base_seed = int(base_seed)
        # The service owns the rollout machine; the driver owns what to run and
        # what the pictures mean. Injecting it is how Explore mode reuses this
        # whole state machine without a second copy of it.
        self.driver = (driver if driver is not None
                       else PromptDriver(tournament_service, scorer, spec))
        self.algorithm = "CMA-ES"
        self.sigma0 = 0.5
```

Then delete the two now-dead lines from the state block (currently lines 73 and 74):

```python
        self.prompt = ""        # DELETE - now a property
        self.optimizer = None   # DELETE - now a property
```

- [ ] **Step 3: Add the delegating properties**

Insert immediately after `__init__`, before `configure`:

```python
    # ---- delegated to the driver ---------------------------------------
    # These exist so the service's public surface is unchanged by the
    # extraction. command_handler.py and every existing test reach through
    # them; a rename here is a silent break there.

    @property
    def optimizer(self):
        """Read-only on purpose: any leftover `self.optimizer = None` in this
        file must fail loudly rather than shadow the driver's."""
        return getattr(self.driver, "optimizer", None)

    @property
    def scorer(self):
        return getattr(self.driver, "scorer", None)

    @scorer.setter
    def scorer(self, value):
        self.driver.scorer = value

    @property
    def prompt(self) -> str:
        return getattr(self.driver, "prompt", "")

    @property
    def sigma(self) -> float:
        return float(getattr(self.driver, "sigma", self.sigma0))

    def set_prompt(self, text: str) -> None:
        self.driver.set_prompt(text)

    def set_x0(self, z) -> None:
        self.driver.set_x0(z)

    def _sync_driver(self) -> None:
        """Push the UI-owned settings the driver understands, and the current
        genome layout. Only attributes the driver already has are set, so
        drivers may ignore settings that mean nothing to them."""
        for k in ("algorithm", "sigma0", "base_seed"):
            if hasattr(self.driver, k):
                setattr(self.driver, k, getattr(self, k))
        self.driver.set_spec(self.spec)
```

- [ ] **Step 4: Replace `_resolve_spec`, `_ensure_optimizer`, `start`, `reset`, `_begin_generation`, `score_and_tell`, `checkpoint_state`, `restore`**

Delete `_ensure_optimizer` entirely — the driver builds its optimizer lazily on `ask`. Replace the rest with:

```python
    def _resolve_spec(self) -> None:
        """Point self.spec at the space the current settings imply."""
        want = BRAIN_PHYSICS_SPEC if self.physics_enabled else BRAIN_SPEC
        if self.spec is not want:
            self.spec = want

    def start(self, prompt: str | None = None) -> None:
        if prompt is not None:
            self.set_prompt(prompt)
        self._resolve_spec()
        self._sync_driver()
        if self.phase is Phase.IDLE:
            self._begin_generation()
        if self.phase in (Phase.IDLE, Phase.PAUSED):
            self.phase = Phase.ROLLOUT

    def reset(self) -> None:
        # The plot reads logger.history(); leaving it would draw the abandoned
        # search's curve in front of the new one.
        if self.logger is not None:
            self.logger.start_new_run()
        self.driver.reset()
        self.generation = 0
        self.step_in_gen = 0
        self.fitness = None
        self._z = None
        self.tile_physics = []
        self._buffer.clear()
        self._needs_write = False
        self.phase = Phase.IDLE

    def _begin_generation(self) -> None:
        # Turning physics search on or off changes the dimension of the search
        # space; _sync_driver hands the new spec to the driver, which discards
        # any optimizer built for the old one.
        self._resolve_spec()
        self._sync_driver()
        self._z = self.driver.ask(self.popsize)
        parts = [self.spec.decode(z) for z in self._z]
        self.tournament.population = [p["brain"] for p in parts]
        self.tile_physics = (
            [decode_physics(p["physics"], self.physics_origin) for p in parts]
            if self.physics_enabled else [])
        self.tournament.mark_dirty()
        self._snaps = snapshot_steps(self.steps_per_gen, self.snapshots_per_gen)
        self._next_snap = 0
        self.step_in_gen = 0
        self._buffer.clear()
        self._needs_write = True

    def score_and_tell(self) -> np.ndarray:
        fit = np.asarray(self.driver.tell(self._z, self._buffer), dtype=np.float32)
        self.fitness = fit
        self.generation += 1
        st = self.driver.status()

        if self.logger is not None:
            self.logger.log_generation({
                "gen": self.generation,
                "prompt": st.get("prompt", ""),
                "algorithm": st.get("algorithm", self.algorithm),
                "sigma": st.get("sigma", self.sigma),
                "popsize": len(fit),
                "grid": self.tournament.grid,
                "seed": self.gen_seed,
                "fit_best": float(fit.max()),
                "fit_mean": float(fit.mean()),
                "fit_median": float(np.median(fit)),
                "fit_min": float(fit.min()),
                "fit_std": float(fit.std()),
                "best_tile": int(np.argmax(fit)),
                "steps_per_gen": self.steps_per_gen,
                "snapshots": self.snapshots_per_gen,
                "elites_injected": int(st.get("elites_injected", 0)),
                "tile_mutation": bool(self.tile_mutation_enabled),
                "physics_search": bool(self.physics_enabled),
                "nan_replaced": int(st.get("nan_replaced", 0)),
                "driver": self.driver.name,
            })

        self._begin_generation()
        self.phase = Phase.ROLLOUT
        return fit

    def checkpoint_state(self) -> dict:
        d = self.driver.checkpoint_state()
        return {
            "genome_spec_signature": self.spec.signature(),
            "generation": self.generation,
            "optimizer_name": d.get("optimizer_name", self.algorithm),
            "optimizer_state": d.get("optimizer_state", {}),
            "base_seed": self.base_seed,
            "prompt": d.get("prompt", ""),
            "distractors": d.get("distractors", []),
            "settings": {
                "grid": self.tournament.grid,
                "steps_per_gen": self.steps_per_gen,
                "snapshots_per_gen": self.snapshots_per_gen,
                "sim_steps_per_frame": self.sim_steps_per_frame,
                "sigma0": self.sigma0,
                "autosave_every": self.autosave_every,
                "tile_mutation_enabled": self.tile_mutation_enabled,
                "physics_enabled": self.physics_enabled,
                "variants_per_tile": self.variants_per_tile,
                "tile_mutation_strength": self.tile_mutation_strength,
            },
            "history": self.logger.history() if self.logger else {},
            "best_z": d.get("best_z", np.zeros(self.spec.dim, np.float32)),
            "best_fitness": float(d.get("best_fitness", 0.0)),
        }

    def restore(self, state: dict) -> None:
        s = state["settings"]
        # Grid and popsize are inseparable from the checkpoint: cmaes.CMA fixes
        # its population at construction.
        self.tournament.set_grid(int(s["grid"]))
        self.configure(
            steps_per_gen=int(s["steps_per_gen"]),
            snapshots_per_gen=int(s["snapshots_per_gen"]),
            sim_steps_per_frame=int(s["sim_steps_per_frame"]),
            sigma0=float(s["sigma0"]),
            autosave_every=int(s.get("autosave_every", 10)),
            tile_mutation_enabled=bool(s.get("tile_mutation_enabled", False)),
            physics_enabled=bool(s.get("physics_enabled", False)),
            variants_per_tile=int(s.get("variants_per_tile", 4)),
            tile_mutation_strength=float(s.get("tile_mutation_strength", 0.1)),
            algorithm=str(state["optimizer_name"]),
        )
        self.base_seed = int(state["base_seed"])
        self.generation = int(state["generation"])
        self._resolve_spec()
        self._sync_driver()
        self.driver.reset()
        self.driver.restore(state)
        if self.logger is not None and state.get("history"):
            self.logger.load_history(state["history"])
        self.phase = Phase.PAUSED
```

- [ ] **Step 5: Run the eight gate files**

Run:

```bash
python -m pytest tests/test_auto_tournament_service.py tests/test_auto_tournament_state.py tests/test_optimizers.py tests/test_run_checkpoint.py tests/test_run_logger.py tests/test_genome_spec.py tests/test_clip_scorer.py tests/test_auto_tournament_plot.py -v
```

Expected: all PASS, with **zero edits** to any of those files.

**If any of them needs editing to pass, the refactor changed behaviour and is wrong.** Revert with `git checkout services/auto_tournament_service.py` and either fix the extraction or fall back to spec §12.3's alternative — a mode flag inside `AutoTournamentService` — reporting why.

- [ ] **Step 6: Run the whole suite and compare to the baseline**

```bash
python -m pytest -q
```

Expected: the same passing count as Step 1's baseline, plus the tests added in Tasks 1–6.

- [ ] **Step 7: Verify Auto mode by hand**

Launch the app, open Tournament → Auto (CLIP), type a goal, press Start, and let three generations run.

Expected: the grid repopulates each generation, the fitness sparkline rises, the sigma readout is non-zero, and `runs/<id>/log.jsonl` gains one line per generation. Save a checkpoint, press Reset, load the checkpoint back, and confirm the generation counter and prompt return.

- [ ] **Step 8: Commit**

```bash
git add services/auto_tournament_service.py
git commit -m "refactor: AutoTournamentService delegates search to a SearchDriver

The rollout state machine is identical for every search; the optimizer,
scoring and elite injection are not. Extracting the second half lets
Explore mode reuse the frame-spread machine rather than copy it.

Behaviour is unchanged: the eight pre-existing auto-mode test files pass
unmodified, and Auto mode was verified by hand across three generations
plus a checkpoint save/load cycle."
```

---

### Task 8: `archive_io.py` — persistence

Spec §5.6, §10. `index.jsonl` is the authority for which entries exist; `vectors.npz` supplies their arrays.

**Files:**
- Create: `services/archive_io.py`
- Modify: `utilities/paths.py`
- Test: `tests/test_archive_io.py`

**Interfaces:**
- Produces:
  - `services.archive_io.FORMAT_VERSION = 1`
  - `ArchiveStore(root)` with `.enabled`, `.append_index(row: dict)`, `.flush_vectors(ids, embeddings, brains, physics)`, `.load() -> (rows: list[dict], arrays: dict)`, `.write_thumb(entry_id: int, crop) -> str`, `.thumb_path(name) -> Path`, `.load_goals() -> list[dict]`, `.save_goals(items)`, `.close()`
  - `utilities.paths.get_archive_dir() -> Path`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_archive_io.py
import json

import numpy as np
import pytest

from services.archive_io import FORMAT_VERSION, ArchiveStore


def _arrays(n=3, dim=4):
    return (
        np.arange(n, dtype=np.int64),
        np.eye(n, dim, dtype=np.float32),
        np.zeros((n, 10, 8), dtype=np.float32),
        np.ones((n, 8), dtype=np.float32),
    )


def test_creates_the_directory_layout(tmp_path):
    s = ArchiveStore(tmp_path / "archive")
    assert s.enabled
    assert (tmp_path / "archive" / "thumbs").is_dir()
    s.close()


def test_append_index_writes_one_json_object_per_line(tmp_path):
    s = ArchiveStore(tmp_path)
    s.append_index({"id": 0, "novelty": 0.5})
    s.append_index({"id": 1, "novelty": 0.6})
    s.close()
    lines = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["id"] for x in lines] == [0, 1]


def test_vectors_roundtrip(tmp_path):
    s = ArchiveStore(tmp_path)
    ids, emb, brains, phys = _arrays()
    s.flush_vectors(ids, emb, brains, phys)
    for i in range(3):
        s.append_index({"id": int(i)})
    rows, arrays = ArchiveStore(tmp_path).load()
    assert [r["id"] for r in rows] == [0, 1, 2]
    assert np.array_equal(arrays["ids"], ids)
    assert np.allclose(arrays["embeddings"].astype(np.float32), emb, atol=1e-3)
    assert arrays["embeddings"].dtype == np.float16, "embeddings are stored fp16"
    assert arrays["brains"].shape == (3, 10, 8)
    assert arrays["physics"].shape == (3, 8)
    s.close()


def test_flush_leaves_no_temp_file(tmp_path):
    s = ArchiveStore(tmp_path)
    s.flush_vectors(*_arrays())
    assert list(tmp_path.glob("*.tmp")) == []
    s.close()


def test_a_failed_flush_leaves_the_previous_vectors_intact(tmp_path, monkeypatch):
    """A crash mid-write must never destroy a good archive."""
    s = ArchiveStore(tmp_path)
    ids, emb, brains, phys = _arrays()
    s.flush_vectors(ids, emb, brains, phys)
    good = (tmp_path / "vectors.npz").read_bytes()

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("numpy.savez", boom)
    s.flush_vectors(ids, emb * 0, brains, phys)
    assert (tmp_path / "vectors.npz").read_bytes() == good
    s.close()


def test_a_corrupt_vectors_file_is_quarantined_not_overwritten(tmp_path):
    (tmp_path / "index.jsonl").write_text('{"id": 0}\n', encoding="utf-8")
    (tmp_path / "vectors.npz").write_bytes(b"not an npz at all")
    rows, arrays = ArchiveStore(tmp_path).load()
    assert rows == [{"id": 0}]
    assert arrays == {}
    assert not (tmp_path / "vectors.npz").exists()
    assert list(tmp_path.glob("vectors.npz.bad-*")), "the bad file must be kept"


def test_a_format_version_mismatch_is_quarantined(tmp_path):
    s = ArchiveStore(tmp_path)
    ids, emb, brains, phys = _arrays()
    with open(tmp_path / "vectors.npz", "wb") as fh:
        np.savez(fh, format_version=np.array(FORMAT_VERSION + 1), ids=ids,
                 embeddings=emb.astype(np.float16), brains=brains, physics=phys)
    _rows, arrays = ArchiveStore(tmp_path).load()
    assert arrays == {}
    assert list(tmp_path.glob("vectors.npz.bad-*"))
    s.close()


def test_a_torn_trailing_index_line_costs_one_entry_not_the_file(tmp_path):
    (tmp_path / "index.jsonl").write_text(
        '{"id": 0}\n{"id": 1}\n{"id": 2, "nov', encoding="utf-8")
    rows, _ = ArchiveStore(tmp_path).load()
    assert [r["id"] for r in rows] == [0, 1]


def test_load_on_an_empty_directory_is_empty_not_an_error(tmp_path):
    rows, arrays = ArchiveStore(tmp_path).load()
    assert rows == []
    assert arrays == {}


def test_write_thumb_returns_the_filename_and_writes_160px(tmp_path):
    from PIL import Image

    s = ArchiveStore(tmp_path)
    crop = np.full((224, 224, 3), 200, dtype=np.uint8)
    name = s.write_thumb(7, crop)
    assert name == "000007.jpg"
    with Image.open(s.thumb_path(name)) as img:
        assert img.size == (160, 160)
    s.close()


def test_goals_roundtrip(tmp_path):
    s = ArchiveStore(tmp_path)
    items = [{"text": "coral reef", "enabled": True},
             {"text": "lightning", "enabled": False}]
    s.save_goals(items)
    assert ArchiveStore(tmp_path).load_goals() == items
    s.close()


def test_missing_goals_file_loads_as_empty(tmp_path):
    assert ArchiveStore(tmp_path).load_goals() == []


def test_an_unwritable_root_disables_persistence_without_raising(tmp_path):
    blocker = tmp_path / "archive"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    s = ArchiveStore(blocker)
    assert s.enabled is False
    # every operation must be a no-op rather than an exception
    s.append_index({"id": 0})
    s.flush_vectors(*_arrays())
    assert s.write_thumb(0, np.zeros((8, 8, 3), np.uint8)) == ""
    assert s.load_goals() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_archive_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.archive_io'`

- [ ] **Step 3: Write the implementation**

```python
# services/archive_io.py
"""Persistence for the exploration archive.

index.jsonl is the AUTHORITY for which entries exist; vectors.npz supplies their
arrays. They are reconciled on load, so a crash between the last flush and quit
costs the trailing entries rather than the archive.

Write strategies mirror the two already in this codebase: append-and-flush per
generation for the index (RunLogger), tmp + os.replace for the vectors
(run_checkpoint). Never a pickle; np.load always with allow_pickle=False.

A disk problem must never block evolution. Every method degrades to a no-op with
one printed warning instead of raising.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
THUMB_PX = 160
THUMB_QUALITY = 85

_ARRAY_KEYS = ("ids", "embeddings", "brains", "physics")


class ArchiveStore:
    def __init__(self, root):
        self.root = Path(root)
        self.enabled = True
        self._fh = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "thumbs").mkdir(exist_ok=True)
            self._fh = open(self.index_path, "a", encoding="utf-8")
        except OSError as exc:
            self.enabled = False
            print(f"[Archive] persistence disabled ({exc}); the run continues")

    # ---- paths ---------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.root / "vectors.npz"

    @property
    def goals_path(self) -> Path:
        return self.root / "goals.json"

    def thumb_path(self, name: str) -> Path:
        return self.root / "thumbs" / name

    # ---- writing -------------------------------------------------------

    def append_index(self, row: dict) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()  # a crash must lose at most one entry
        except (OSError, TypeError) as exc:
            self.enabled = False
            print(f"[Archive] index write failed ({exc}); persistence disabled")

    def flush_vectors(self, ids, embeddings, brains, physics) -> None:
        """Rewrite vectors.npz atomically. Embeddings go to disk as fp16 -
        half the bytes, and the precision loss is far below the scale any
        novelty decision turns on."""
        if not self.enabled:
            return
        tmp = self.vectors_path.with_suffix(self.vectors_path.suffix + ".tmp")
        try:
            # A file handle, not a path: np.savez appends ".npz" to any path
            # that does not already end in it.
            with open(tmp, "wb") as fh:
                np.savez(
                    fh,
                    format_version=np.array(FORMAT_VERSION),
                    ids=np.asarray(ids, dtype=np.int64),
                    embeddings=np.asarray(embeddings, dtype=np.float16),
                    brains=np.asarray(brains, dtype=np.float32),
                    physics=np.asarray(physics, dtype=np.float32),
                )
            os.replace(tmp, self.vectors_path)
        except (OSError, ValueError) as exc:
            print(f"[Archive] vector flush failed ({exc}); previous file kept")
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def write_thumb(self, entry_id: int, crop: np.ndarray) -> str:
        """-> the filename, or "" if it could not be written. A missing
        thumbnail costs a gallery placeholder, never an admission."""
        if not self.enabled:
            return ""
        name = f"{int(entry_id):06d}.jpg"
        try:
            from PIL import Image

            img = Image.fromarray(np.asarray(crop, dtype=np.uint8))
            img = img.resize((THUMB_PX, THUMB_PX), Image.BILINEAR)
            img.save(self.thumb_path(name), quality=THUMB_QUALITY)
        except (OSError, ValueError) as exc:
            print(f"[Archive] thumbnail {name} failed ({exc})")
            return ""
        return name

    def save_goals(self, items: list[dict]) -> None:
        if not self.enabled:
            return
        tmp = self.goals_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(list(items), indent=2), encoding="utf-8")
            os.replace(tmp, self.goals_path)
        except OSError as exc:
            print(f"[Archive] goal list not saved ({exc})")

    # ---- reading -------------------------------------------------------

    def load_goals(self) -> list[dict]:
        try:
            return json.loads(self.goals_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def load(self) -> tuple[list[dict], dict]:
        """-> (index rows, arrays). Either may be empty; the caller reconciles."""
        rows: list[dict] = []
        try:
            text = self.index_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A torn trailing line is one lost entry, not a lost archive.
                continue

        arrays: dict = {}
        if self.vectors_path.is_file():
            try:
                with np.load(self.vectors_path, allow_pickle=False) as z:
                    got = int(z["format_version"])
                    if got != FORMAT_VERSION:
                        raise ValueError(
                            f"vectors.npz format_version is {got}, "
                            f"this build expects {FORMAT_VERSION}")
                    arrays = {k: z[k] for k in _ARRAY_KEYS}
            except (OSError, ValueError, KeyError) as exc:
                self._quarantine(exc)
                arrays = {}
        return rows, arrays

    def _quarantine(self, exc) -> None:
        """Move a bad vectors file aside. Never overwrite it - if the archive
        represents hours of exploration, a recoverable file is worth more than
        a tidy directory."""
        bad = self.vectors_path.with_name(f"vectors.npz.bad-{int(time.time())}")
        try:
            os.replace(self.vectors_path, bad)
            print(f"[Archive] {exc}; moved to {bad.name}, starting empty")
        except OSError:
            print(f"[Archive] {exc}; could not quarantine {self.vectors_path}")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
```

- [ ] **Step 4: Add the path helper**

In `utilities/paths.py`, add after `get_videos_dir`:

```python
def get_archive_dir() -> Path:
    """Get path to the exploration archive directory."""
    return get_user_data_dir() / "archive"
```

and add this line inside `initialize_user_data()`, next to the other `mkdir` calls:

```python
    get_archive_dir().mkdir(exist_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_archive_io.py -v`
Expected: PASS, 13 tests

- [ ] **Step 6: Commit**

```bash
git add services/archive_io.py utilities/paths.py tests/test_archive_io.py
git commit -m "feat: ArchiveStore - index.jsonl, atomic vectors.npz, thumbnails"
```

---

### Task 9: `archive.py` — gates, adaptive threshold, capacity, refresh

Spec §5.2–§5.5. The last piece of the foundation.

**Files:**
- Create: `services/archive.py`
- Test: `tests/test_archive.py`
- Test: `tests/test_physics_origin_roundtrip.py`

**Interfaces:**
- Consumes: `services.novelty` (Task 5), `services.archive_io.ArchiveStore` (Task 8)
- Produces:
  - `AdaptiveThreshold(initial=0.05, target_rate=0.15, window=100, lo=1e-4, hi=1.0)` with `.value`, `.observe(admitted: bool)`, `.rate`
  - `Candidate(brain, physics, embedding, liveness, spec, viable=True, gen=0, tile=0, run_id="", goal="")`
  - `ArchiveEntry` dataclass — `id, novelty, liveness, pinned, source, spec, goal, run_id, gen, tile, ts, thumb`
  - `Archive(store=None, capacity=20000, k=10, liveness_min=0.02, target_rate=0.15, seed_n=256, dim=512)` with `.embeddings`, `.entries`, `.rejects`, `.threshold`, `.brains`, `.physics`, `__len__`, `.novelty_of(queries)`, `.consider(cand, novelty, pinned=False, source=..., thumb_crop=None) -> ArchiveEntry | None`, `.refresh(n)`, `.centroid()`, `.maybe_flush(every=200, force=False)`, `.load_from_store()`, `.nearest(goal) -> int | None`, `.stats() -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_archive.py
import numpy as np
import pytest

from services.archive import AdaptiveThreshold, Archive, Candidate
from services.archive_io import ArchiveStore


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def cand(vec, liveness=1.0, viable=True, dim=4, spec="brain:80"):
    e = np.zeros(dim, dtype=np.float32)
    e[: len(vec)] = vec
    return Candidate(
        brain=np.zeros((10, 8), np.float32),
        physics=np.zeros(8, np.float32),
        embedding=_unit(e[None])[0],
        liveness=liveness,
        viable=viable,
        spec=spec,
    )


def fresh(**kw):
    kw.setdefault("dim", 4)
    kw.setdefault("seed_n", 0)      # novelty gate on from the first candidate
    kw.setdefault("capacity", 100)
    return Archive(store=None, **kw)


# ---- adaptive threshold ------------------------------------------------

def test_threshold_rises_when_admission_is_too_generous():
    t = AdaptiveThreshold(initial=0.1, target_rate=0.15, window=10)
    for _ in range(50):
        t.observe(True)
    assert t.value > 0.1


def test_threshold_falls_when_nothing_gets_in():
    t = AdaptiveThreshold(initial=0.1, target_rate=0.15, window=10)
    for _ in range(50):
        t.observe(False)
    assert t.value < 0.1


def test_threshold_converges_toward_the_target_rate():
    """A stream where exactly 15% clear any threshold below 0.85 must settle."""
    rng = np.random.default_rng(0)
    t = AdaptiveThreshold(initial=0.5, target_rate=0.15, window=100)
    for _ in range(3000):
        t.observe(bool(rng.random() < 0.15))
    assert t.rate == pytest.approx(0.15, abs=0.08)


def test_threshold_is_clamped_to_its_bounds():
    t = AdaptiveThreshold(initial=0.5, target_rate=0.15, window=5, lo=0.01, hi=0.6)
    for _ in range(500):
        t.observe(True)
    assert t.value <= 0.6
    for _ in range(2000):
        t.observe(False)
    assert t.value >= 0.01


# ---- the three gates ---------------------------------------------------

def test_a_novel_live_viable_candidate_is_admitted():
    a = fresh()
    e = a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert e is not None
    assert len(a) == 1
    assert e.source == "expansion"


def test_the_viability_gate_rejects_a_dead_tile():
    a = fresh()
    assert a.consider(cand([1, 0, 0, 0], viable=False), novelty=1.0) is None
    assert len(a) == 0


def test_the_liveness_gate_rejects_a_frozen_pattern():
    a = fresh(liveness_min=0.02)
    assert a.consider(cand([1, 0, 0, 0], liveness=0.0), novelty=1.0) is None
    assert len(a) == 0


def test_the_novelty_gate_rejects_a_near_duplicate():
    a = fresh(liveness_min=0.0)
    a.threshold.value = 0.5
    assert a.consider(cand([1, 0, 0, 0]), novelty=0.4) is None
    assert len(a) == 0


def test_a_non_finite_descriptor_is_rejected_and_counted():
    a = fresh()
    c = cand([1, 0, 0, 0])
    c.embedding = np.array([np.nan, 0, 0, 0], dtype=np.float32)
    assert a.consider(c, novelty=1.0) is None
    assert a.stats()["n_nonfinite"] == 1


def test_each_gate_fires_independently():
    """Failing one gate must not depend on the others passing."""
    a = fresh(liveness_min=0.5)
    a.threshold.value = 0.5
    assert a.consider(cand([1, 0, 0, 0], liveness=1.0, viable=False), 1.0) is None
    assert a.consider(cand([0, 1, 0, 0], liveness=0.1), 1.0) is None
    assert a.consider(cand([0, 0, 1, 0], liveness=1.0), 0.1) is None
    assert a.consider(cand([0, 0, 0, 1], liveness=1.0), 1.0) is not None


def test_rejected_descriptors_go_into_the_rejects_ring():
    a = fresh(liveness_min=0.5)
    a.consider(cand([1, 0, 0, 0], liveness=0.0), novelty=1.0)
    assert len(a.rejects) == 1


# ---- bootstrap ---------------------------------------------------------

def test_bootstrap_skips_only_the_novelty_gate():
    a = fresh(seed_n=5, liveness_min=0.02)
    a.threshold.value = 0.9
    assert a.consider(cand([1, 0, 0, 0]), novelty=0.0) is not None, "novelty off"
    assert a.consider(cand([0, 1, 0, 0], liveness=0.0), novelty=1.0) is None, "liveness on"
    assert a.consider(cand([0, 0, 1, 0], viable=False), novelty=1.0) is None, "viability on"


def test_the_novelty_gate_switches_on_at_seed_n():
    a = fresh(seed_n=2, liveness_min=0.0)
    a.threshold.value = 0.9
    a.consider(cand([1, 0, 0, 0]), novelty=0.0)
    a.consider(cand([0, 1, 0, 0]), novelty=0.0)
    assert len(a) == 2
    assert a.consider(cand([0, 0, 1, 0]), novelty=0.0) is None


def test_the_threshold_is_not_updated_during_bootstrap():
    """A cold start would drive the threshold on evidence that means nothing."""
    a = fresh(seed_n=10, liveness_min=0.0)
    before = a.threshold.value
    for i in range(5):
        a.consider(cand([float(i), 1, 0, 0]), novelty=0.0)
    assert a.threshold.value == before


# ---- pins --------------------------------------------------------------

def test_a_pin_bypasses_every_gate():
    a = fresh(liveness_min=0.9)
    a.threshold.value = 0.99
    e = a.consider(cand([1, 0, 0, 0], liveness=0.0, viable=False), 0.0, pinned=True)
    assert e is not None
    assert e.pinned is True
    assert e.source == "pin"


def test_pins_survive_eviction():
    a = fresh(capacity=3, liveness_min=0.0, seed_n=0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=0.0, pinned=True)
    for i in range(10):
        a.consider(cand([0, float(i + 1), 0, 0]), novelty=0.9)
    assert len(a) == 3
    assert any(e.pinned for e in a.entries)


# ---- capacity ----------------------------------------------------------

def test_eviction_drops_the_least_novel_entry():
    a = fresh(capacity=2, liveness_min=0.0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=0.9)
    a.consider(cand([0, 1, 0, 0]), novelty=0.1)      # the redundant one
    a.consider(cand([0, 0, 1, 0]), novelty=0.8)
    assert len(a) == 2
    assert sorted(round(e.novelty, 1) for e in a.entries) == [0.8, 0.9]


def test_eviction_keeps_arrays_and_entries_in_lockstep():
    a = fresh(capacity=3, liveness_min=0.0)
    a.threshold.value = 0.0
    for i in range(8):
        a.consider(cand([float(i), 0, 0, 0]), novelty=float(i) / 10.0)
    assert len(a) == 3
    assert a.embeddings.shape == (3, 4)
    assert a.brains.shape == (3, 10, 8)
    assert a.physics.shape == (3, 8)
    assert len(a.entries) == 3


def test_an_all_pinned_archive_at_capacity_stops_admitting():
    a = fresh(capacity=2, liveness_min=0.0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), 0.0, pinned=True)
    a.consider(cand([0, 1, 0, 0]), 0.0, pinned=True)
    a.consider(cand([0, 0, 1, 0]), novelty=0.9)
    assert len(a) == 2
    assert a.stats()["blocked_by_pins"] is True


# ---- novelty bookkeeping ----------------------------------------------

def test_novelty_of_uses_both_the_archive_and_the_rejects_ring():
    a = fresh(liveness_min=0.0, k=1)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    far = _unit(np.array([[0, 0, 0, 1]], np.float32))
    before = float(a.novelty_of(far)[0])
    a.rejects.add(far)
    after = float(a.novelty_of(far)[0])
    assert after < before, "a recent rejection must suppress novelty there"


def test_novelty_of_an_empty_archive_is_one():
    a = fresh()
    assert a.novelty_of(_unit(np.eye(4)[:2]))[0] == pytest.approx(1.0)


def test_refresh_rescores_entries_against_the_current_archive():
    a = fresh(liveness_min=0.0, k=1)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    for i in range(1, 4):
        v = [0.0, 0.0, 0.0, 0.0]
        v[i] = 1.0
        a.consider(cand(v), novelty=1.0)
    assert a.refresh(4) == 4
    assert all(e.novelty < 1.0 for e in a.entries), "novelty must have dropped"


def test_refresh_excludes_the_entry_from_its_own_neighbours():
    """Without exclude_self every entry's nearest neighbour is itself at
    distance 0, and every novelty collapses toward zero."""
    a = fresh(liveness_min=0.0, k=1)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    a.refresh(2)
    assert all(e.novelty == pytest.approx(1.0, abs=1e-5) for e in a.entries)


def test_refresh_walks_round_robin():
    a = fresh(liveness_min=0.0)
    a.threshold.value = 0.0
    for i in range(6):
        v = np.zeros(4)
        v[i % 4] = float(i + 1)
        a.consider(cand(v.tolist()), novelty=1.0)
    assert a.refresh(2) == 2
    assert a.refresh(2) == 2
    assert a._refresh_cursor == 4


def test_refresh_on_an_empty_archive_is_a_no_op():
    assert fresh().refresh(10) == 0


def test_centroid_is_unit_norm_and_none_when_empty():
    a = fresh(liveness_min=0.0)
    a.threshold.value = 0.0
    assert a.centroid() is None
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    c = a.centroid()
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-5)


def test_nearest_returns_the_best_match_for_a_goal():
    a = fresh(liveness_min=0.0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    goal = _unit(np.array([[0.1, 1.0, 0, 0]], np.float32))[0]
    assert a.nearest(goal) == 1
    assert fresh().nearest(goal) is None


# ---- persistence -------------------------------------------------------

def test_load_from_store_reconciles_index_against_vectors(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
    for i in range(3):
        v = np.zeros(4)
        v[i] = 1.0
        a.consider(cand(v.tolist()), novelty=1.0)
    a.maybe_flush(force=True)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4)
    loaded, dropped = b.load_from_store()
    assert loaded == 3
    assert dropped == 0
    assert len(b) == 3
    assert [e.id for e in b.entries] == [0, 1, 2]


def test_entries_with_no_matching_vector_row_are_dropped(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.maybe_flush(force=True)
    # a second entry reaches the index but the process dies before the flush
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4)
    loaded, dropped = b.load_from_store()
    assert loaded == 1
    assert dropped == 1


def test_next_id_continues_after_a_reload(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
    for i in range(3):
        v = np.zeros(4)
        v[i] = 1.0
        a.consider(cand(v.tolist()), novelty=1.0)
    a.maybe_flush(force=True)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4, seed_n=0, liveness_min=0.0)
    b.load_from_store()
    b.threshold.value = 0.0
    e = b.consider(cand([0, 0, 0, 1]), novelty=1.0)
    assert e.id == 3, "ids must not collide with the reloaded entries"


def test_maybe_flush_only_writes_on_the_interval(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert a.maybe_flush(every=5) is False
    assert a.maybe_flush(every=5, force=True) is True
    store.close()


def test_an_archive_with_no_store_works_entirely_in_memory():
    a = fresh()
    assert a.consider(cand([1, 0, 0, 0]), novelty=1.0) is not None
    assert a.maybe_flush(force=True) is False
    assert a.load_from_store() == (0, 0)
```

```python
# tests/test_physics_origin_roundtrip.py
"""The archive stores DECODED phenotypes, not z.

With physics search on, z means 'this far from the preset that happened to be
loaded' (physics_genome: value = origin + span*tanh(z)). A z archived under one
preset and re-run under another is a different creature. These tests pin the
property that makes the archive origin-independent.
"""
import numpy as np
import pytest

from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.physics_genome import PHYSICS_PARAMS, decode_physics, encode_physics


def origin_a():
    return {n: (lo + hi) / 2.0 for n, _g, lo, hi in PHYSICS_PARAMS}


def origin_b():
    return {n: lo + 0.25 * (hi - lo) for n, _g, lo, hi in PHYSICS_PARAMS}


def test_the_same_z_means_different_physics_under_different_origins():
    """This is the trap the design avoids. If this test ever fails, storing z
    would have been safe - and it is not."""
    z = np.full(len(PHYSICS_PARAMS), 0.5, dtype=np.float32)
    a = decode_physics(z, origin_a())
    b = decode_physics(z, origin_b())
    assert any(not np.isclose(a[k], b[k]) for k in a)


def test_a_stored_phenotype_reproduces_under_any_origin():
    z = np.full(len(PHYSICS_PARAMS), 0.4, dtype=np.float32)
    phenotype = decode_physics(z, origin_a())          # what the archive stores

    z_under_b = encode_physics(phenotype, origin_b())  # re-encoded on reuse
    got = decode_physics(z_under_b, origin_b())

    for k, v in phenotype.items():
        assert got[k] == pytest.approx(v, abs=1e-3), k


def test_zero_physics_genes_reproduce_the_current_origin_exactly():
    """A brain-only archive entry used in a physics-enabled run gets z=0 for the
    physics block, which must be the preset as loaded - not a slider midpoint."""
    o = origin_b()
    got = decode_physics(np.zeros(len(PHYSICS_PARAMS), np.float32), o)
    for k, v in o.items():
        assert got[k] == pytest.approx(v, abs=1e-6), k


def test_a_brain_spec_entry_is_shorter_than_a_brain_physics_entry():
    """The signature difference is what tells the driver to pad with zeros."""
    assert BRAIN_SPEC.signature() == "brain:80"
    assert BRAIN_PHYSICS_SPEC.signature() == "brain:80,physics:8"
    assert BRAIN_PHYSICS_SPEC.dim == BRAIN_SPEC.dim + len(PHYSICS_PARAMS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_archive.py tests/test_physics_origin_roundtrip.py -v`
Expected: `test_archive.py` FAILS with `ModuleNotFoundError: No module named 'services.archive'`. `test_physics_origin_roundtrip.py` should already PASS — it pins existing behaviour that the archive design depends on. **If it fails, stop:** the origin-relative encoding does not behave as the spec assumes and §5.5 must be revisited.

- [ ] **Step 3: Write the implementation**

```python
# services/archive.py
"""The exploration archive: admission gates, capacity, and novelty bookkeeping.

Three gates, all of which must pass (5.2):
  viable   - the tile is not black or blown out
  alive    - liveness >= liveness_min (ASAL Eq.3; a frozen canvas scores ~0)
  novel    - kNN novelty >= an ADAPTIVE threshold

The novelty threshold adapts because a fixed one either floods a rich region or
starves a barren one, and which happens depends on the preset the user loaded.
Tracking the admission rate and nudging toward a target makes archive growth
predictable, and self-lowers in a barren region so the search never goes blind.

While the archive is below seed_n the novelty gate is OFF (5.2.2). With it on
from a cold start, reaching 256 entries at a 15% target takes ~107 generations -
five minutes before expansion begins - and the threshold would be adapting
against no distribution at all.

Entries store the DECODED PHENOTYPE, never z. See
tests/test_physics_origin_roundtrip.py for why.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from services.novelty import RejectsRing, knn_distances, knn_novelty, novelty_from_distances


class AdaptiveThreshold:
    """Lehman-Stanley dynamic novelty threshold, driven to a target rate."""

    def __init__(self, initial: float = 0.05, target_rate: float = 0.15,
                 window: int = 100, lo: float = 1e-4, hi: float = 1.0,
                 step: float = 0.05):
        self.value = float(initial)
        self.target_rate = float(target_rate)
        self.window = int(window)
        self.lo = float(lo)
        self.hi = float(hi)
        self.step = float(step)
        self._hits = 0
        self._seen = 0
        self._recent: list[int] = []

    @property
    def rate(self) -> float:
        return (sum(self._recent) / len(self._recent)) if self._recent else 0.0

    def observe(self, admitted: bool) -> None:
        self._recent.append(1 if admitted else 0)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        if len(self._recent) < max(4, self.window // 10):
            return          # too little evidence to steer on
        if self.rate > self.target_rate:
            self.value *= (1.0 + self.step)
        elif self.rate < self.target_rate:
            self.value *= (1.0 - self.step)
        self.value = float(min(max(self.value, self.lo), self.hi))

    def state_dict(self) -> dict:
        return {"value": self.value, "recent": list(self._recent)}

    def load_state_dict(self, d: dict) -> None:
        self.value = float(d.get("value", self.value))
        self._recent = [int(x) for x in d.get("recent", [])]


@dataclass
class Candidate:
    """One tile's result for one generation, before the gates run."""
    brain: np.ndarray               # (10, 8) float32
    physics: np.ndarray             # (8,) float32, ABSOLUTE values
    embedding: np.ndarray           # (dim,) float32, unit norm
    liveness: float
    spec: str
    viable: bool = True
    gen: int = 0
    tile: int = 0
    run_id: str = ""
    goal: str = ""


@dataclass
class ArchiveEntry:
    id: int
    novelty: float
    liveness: float
    pinned: bool
    source: str
    spec: str
    goal: str
    run_id: str
    gen: int
    tile: int
    ts: float
    thumb: str = ""


class Archive:
    def __init__(self, store=None, capacity: int = 20000, k: int = 10,
                 liveness_min: float = 0.02, target_rate: float = 0.15,
                 seed_n: int = 256, dim: int = 512):
        self.store = store
        self.capacity = int(capacity)
        self.k = int(k)
        self.liveness_min = float(liveness_min)
        self.seed_n = int(seed_n)
        self._dim = int(dim)

        self.entries: list[ArchiveEntry] = []
        self.threshold = AdaptiveThreshold(target_rate=target_rate)
        self.rejects = RejectsRing(dim=self._dim)

        self._emb = np.zeros((0, self._dim), dtype=np.float32)
        self._brain = np.zeros((0, 10, 8), dtype=np.float32)
        self._phys = np.zeros((0, 8), dtype=np.float32)
        self._n = 0

        self._next_id = 0
        self._refresh_cursor = 0
        self._since_flush = 0
        self.n_nonfinite = 0
        self.n_rejected = 0
        self.blocked_by_pins = False

    # ---- views ---------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    @property
    def embeddings(self) -> np.ndarray:
        return self._emb[: self._n]

    @property
    def brains(self) -> np.ndarray:
        return self._brain[: self._n]

    @property
    def physics(self) -> np.ndarray:
        return self._phys[: self._n]

    def stats(self) -> dict:
        return {
            "size": self._n,
            "threshold": self.threshold.value,
            "admission_rate": self.threshold.rate,
            "n_nonfinite": self.n_nonfinite,
            "n_rejected": self.n_rejected,
            "n_pinned": sum(1 for e in self.entries if e.pinned),
            "blocked_by_pins": self.blocked_by_pins,
            "rejects_ring": len(self.rejects),
        }

    # ---- novelty -------------------------------------------------------

    def novelty_of(self, queries: np.ndarray) -> np.ndarray:
        """kNN novelty against archive UNION rejects ring.

        The two references are reduced separately and merged; concatenating a
        2k ring onto a 20k archive every generation would copy 40 MB - more
        than the novelty computation costs.
        """
        return novelty_from_distances(
            [knn_distances(queries, self.embeddings, self.k),
             knn_distances(queries, self.rejects.view(), self.k)],
            self.k,
        )

    def refresh(self, n: int) -> int:
        """Re-score n entries against the FULL archive, round-robin by position.

        Full archive, not a subsample: a subsampled neighbour set inflates kNN
        distances, so subsampled and full novelty values sit on different
        scales and could not be ranked against each other for parent sampling
        or eviction.
        """
        if n <= 0 or self._n == 0:
            return 0
        n = int(min(n, self._n))
        start = self._refresh_cursor % self._n
        idx = (np.arange(start, start + n) % self._n).astype(np.int64)
        nov = knn_novelty(self._emb[idx], self.embeddings, k=self.k, exclude_self=True)
        for j, i in enumerate(idx):
            self.entries[int(i)].novelty = float(nov[j])
        self._refresh_cursor = int((start + n) % self._n)
        return n

    def centroid(self) -> np.ndarray | None:
        if self._n == 0:
            return None
        c = self.embeddings.mean(axis=0)
        nrm = float(np.linalg.norm(c))
        return (c / nrm).astype(np.float32) if nrm > 1e-8 else None

    def nearest(self, goal: np.ndarray) -> int | None:
        """Index of the entry most aligned with a goal embedding."""
        if self._n == 0:
            return None
        return int(np.argmax(self.embeddings @ np.asarray(goal, np.float32)))

    # ---- admission -----------------------------------------------------

    def consider(self, cand: Candidate, novelty: float, *, pinned: bool = False,
                 source: str = "expansion", thumb_crop=None) -> ArchiveEntry | None:
        """Run the gates and add on success. Returns the entry, or None."""
        if pinned:
            return self._add(cand, novelty, "pin", True, thumb_crop)

        if not np.isfinite(cand.embedding).all():
            self.n_nonfinite += 1
            self.n_rejected += 1
            return None

        if not cand.viable or cand.liveness < self.liveness_min:
            self._reject(cand)
            return None

        if self._n >= self.seed_n:
            ok = float(novelty) >= self.threshold.value
            self.threshold.observe(ok)
            if not ok:
                self._reject(cand)
                return None

        return self._add(cand, novelty, source, False, thumb_crop)

    def _reject(self, cand: Candidate) -> None:
        self.n_rejected += 1
        self.rejects.add(cand.embedding[None, :])

    def _add(self, cand, novelty, source, pinned, thumb_crop) -> ArchiveEntry | None:
        if self._n >= self.capacity and not self._evict_one():
            self.blocked_by_pins = True
            return None

        self._grow(1)
        i = self._n
        self._emb[i] = cand.embedding
        self._brain[i] = cand.brain
        self._phys[i] = cand.physics
        self._n += 1

        eid = self._next_id
        self._next_id += 1
        thumb = ""
        if thumb_crop is not None and self.store is not None:
            thumb = self.store.write_thumb(eid, thumb_crop)

        entry = ArchiveEntry(
            id=eid, novelty=float(novelty), liveness=float(cand.liveness),
            pinned=bool(pinned), source=source, spec=cand.spec, goal=cand.goal,
            run_id=cand.run_id, gen=int(cand.gen), tile=int(cand.tile),
            ts=time.time(), thumb=thumb,
        )
        self.entries.append(entry)
        if self.store is not None:
            self.store.append_index(asdict(entry))
        self._since_flush += 1
        return entry

    # ---- capacity ------------------------------------------------------

    def _evict_one(self) -> bool:
        """Drop the least novel non-pinned entry. -> did anything go?

        ASAL's illumination criterion (remove the least novel) applied as an
        eviction rule. It is O(N) and free BECAUSE refresh() keeps stored
        novelty current; a sampled nearest-neighbour pass here would be ~5
        GFLOP and would fire on every admission once the cap is reached.
        """
        best, best_v = None, np.inf
        for i, e in enumerate(self.entries):
            if not e.pinned and e.novelty < best_v:
                best, best_v = i, e.novelty
        if best is None:
            return False
        self._remove(best)
        return True

    def _remove(self, i: int) -> None:
        last = self._n - 1
        if i != last:
            self._emb[i] = self._emb[last]
            self._brain[i] = self._brain[last]
            self._phys[i] = self._phys[last]
            self.entries[i] = self.entries[last]
        self.entries.pop()
        self._n -= 1
        if self._refresh_cursor > self._n:
            self._refresh_cursor = 0

    def _grow(self, extra: int) -> None:
        need = self._n + int(extra)
        cap = len(self._emb)
        if need <= cap:
            return
        new_cap = max(256, need, cap * 2)

        def _re(a, tail):
            b = np.zeros((new_cap,) + tail, dtype=a.dtype)
            b[: self._n] = a[: self._n]
            return b

        self._emb = _re(self._emb, (self._dim,))
        self._brain = _re(self._brain, (10, 8))
        self._phys = _re(self._phys, (8,))

    # ---- persistence ---------------------------------------------------

    def maybe_flush(self, every: int = 200, force: bool = False) -> bool:
        if self.store is None:
            return False
        if not force and self._since_flush < int(every):
            return False
        ids = np.array([e.id for e in self.entries], dtype=np.int64)
        self.store.flush_vectors(ids, self.embeddings, self.brains, self.physics)
        self._since_flush = 0
        return True

    def load_from_store(self) -> tuple[int, int]:
        """-> (loaded, dropped). index.jsonl is the authority for WHICH entries
        exist; vectors.npz supplies their arrays. Keeping only the intersection
        means a crash between the last flush and quit costs the trailing
        entries, never the archive."""
        if self.store is None:
            return 0, 0
        rows, arrays = self.store.load()
        by_id = {int(r["id"]): r for r in rows if isinstance(r, dict) and "id" in r}
        if not arrays or not by_id:
            return 0, len(by_id)

        ids = np.asarray(arrays["ids"], dtype=np.int64)
        emb = np.asarray(arrays["embeddings"], dtype=np.float32)
        brains = np.asarray(arrays["brains"], dtype=np.float32)
        phys = np.asarray(arrays["physics"], dtype=np.float32)
        keep = [j for j, i in enumerate(ids) if int(i) in by_id]
        dropped = (len(ids) - len(keep)) + (len(by_id) - len(keep))

        self._n = 0
        self.entries = []
        self._grow(len(keep))
        for j in keep:
            i = self._n
            self._emb[i] = emb[j]
            self._brain[i] = brains[j]
            self._phys[i] = phys[j]
            self._n += 1
            r = by_id[int(ids[j])]
            self.entries.append(ArchiveEntry(
                id=int(r["id"]),
                novelty=float(r.get("novelty", 0.0)),
                liveness=float(r.get("liveness", 0.0)),
                pinned=bool(r.get("pinned", False)),
                source=str(r.get("source", "expansion")),
                spec=str(r.get("spec", "")),
                goal=str(r.get("goal", "")),
                run_id=str(r.get("run_id", "")),
                gen=int(r.get("gen", 0)),
                tile=int(r.get("tile", 0)),
                ts=float(r.get("ts", 0.0)),
                thumb=str(r.get("thumb", "")),
            ))
        self._next_id = max((e.id for e in self.entries), default=-1) + 1
        if dropped:
            print(f"[Archive] dropped {dropped} entries with no matching "
                  "index/vector row")
        return len(keep), dropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_archive.py tests/test_physics_origin_roundtrip.py -v`
Expected: PASS, 32 tests

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: everything green.

- [ ] **Step 6: Commit**

```bash
git add services/archive.py tests/test_archive.py tests/test_physics_origin_roundtrip.py
git commit -m "feat: Archive - three gates, adaptive threshold, capacity, refresh"
```

---

## Done when

- `python -m pytest -q` is green, and the eight gate files from Task 7 are byte-identical to their state before this plan.
- `python -m tools.clip_spread_check runs/*/frames/*.png` printed `VERDICT: PASS`.
- Auto (CLIP) mode was verified by hand: three generations, rising sparkline, one JSONL line per generation, checkpoint save/load roundtrip.
- Nothing in `sim.py` or `shaders/` changed.

Continue with `docs/superpowers/plans/2026-08-07-imgep-search-and-ui.md`.

