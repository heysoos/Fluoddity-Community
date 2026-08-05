# CLIP-Guided Automatic Tournament — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an automatic tournament mode where a user-typed text prompt is turned into a fitness score by CLIP, and CMA-ES evolves the tile population toward it.

**Architecture:** A generation state machine (`AutoTournamentService`) drives the existing tiled tournament. Each generation asks an `Optimizer` for N² genomes, writes them through the existing rule-upload path, runs the sim for K steps spread across real frames, captures the assembled grid into an offscreen uint8 FBO, splits it into exact 224×224 tiles, scores them with a `CLIPScorer`, and tells the optimizer. Every unit is injected, so all logic is testable without a GPU.

**Tech Stack:** Python 3.12, ModernGL, imgui_bundle, NumPy, onnxruntime-directml, tokenizers, cmaes, Pillow.

**Spec:** `docs/superpowers/specs/2026-08-04-clip-guided-tournament-design.md`

## Global Constraints

- **`sim.py` is user-owned.** Only one additive change is permitted (Task 4, Step 6): extra `tryset` calls inside the existing `if self._tournament_enabled:` block at `sim.py:226`. No restructuring, no new methods, no reordering.
- **No shader changes.** All of `shaders/*` stays as-is. `TOURNAMENT_GRID` is already a uniform; per-tile mutation reuses `mutate_rule` as it stands.
- **Manual tournament mode must behave identically** after every task, including with `onnxruntime`/`cmaes`/`tokenizers` absent.
- **New imports must be lazy.** `services/__init__.py` must not import `clip_scorer` or `optimizers` at startup. Import them inside functions, guarded by `try/except ImportError`.
- **Dependency versions:** `onnxruntime-directml>=1.20`, `tokenizers>=0.20`, `cmaes>=0.11`.
- **Fitness sign: higher is better, everywhere.** `Optimizer.tell` takes fitness where higher wins; implementations negate internally. Logs say `fit_*`, never `loss`.
- **Grid is always N×N**, N ∈ [2, 8]. `popsize == N²`. Auto mode forces a 1:1 canvas.
- **Model files** live in `models/clip-vit-b32/`; runs live in `runs/`. Both gitignored.
- **Naming:** SimState fields and GLSL uniforms are `ALL_CAPS_UNDERSCORE`; UI labels are Title Case; private UI state is `_snake_case`.
- **Run tests with** `./.venv/Scripts/python.exe -m pytest`. The venv already exists.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility |
|---|---|
| `tools/fetch_clip_onnx.py` | One-time model download. Stdlib only. |
| `services/clip_scorer.py` | Text/image → fitness. Knows nothing about tournaments or GL. |
| `services/genome_spec.py` | `z ∈ ℝ^80` ↔ genome `(10,8)`. Pure numpy. |
| `services/optimizers.py` | `Optimizer` protocol + CMA-ES / Sep-CMA-ES / GA / Random. Pure numpy. |
| `services/genome_io.py` | genome ↔ `PhysicsConfig` file. |
| `services/tile_capture.py` | Offscreen FBO → `uint8 (N², 224, 224, 3)`. The only GL unit. |
| `services/run_logger.py` | Per-generation JSONL metrics. |
| `services/run_checkpoint.py` | Atomic `.npz` save/load of full run state. |
| `services/auto_tournament_service.py` | The generation state machine. No GL, no ImGui. |
| `state/auto_tournament_state.py` | UI state + one-shot flags. |
| `ui/auto_tournament_window.py` | Auto tab widgets. Passive. |

---

## Task 1: Dependencies and model fetch

**Files:**
- Create: `tools/fetch_clip_onnx.py`
- Modify: `requirements.txt`, `.gitignore`
- Test: `tests/test_fetch_clip_onnx.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MODEL_DIR = "models/clip-vit-b32"`, `FILES: dict[str, str]` (local filename → HF path), `fetch(dest_dir: str, progress=None) -> None`, `is_present(dest_dir: str) -> bool`.

- [ ] **Step 1: Add dependencies and gitignore entries**

Append to `requirements.txt`:

```
onnxruntime-directml>=1.20
tokenizers>=0.20
cmaes>=0.11
```

Append to `.gitignore`:

```
models/
runs/
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_fetch_clip_onnx.py`:

```python
import json
from pathlib import Path

import pytest

from tools.fetch_clip_onnx import FILES, MODEL_DIR, is_present, _target_url


def test_files_map_covers_three_assets():
    assert set(FILES) == {
        "vision_model_fp16.onnx",
        "text_model_fp16.onnx",
        "tokenizer.json",
    }


def test_target_url_points_at_the_pinned_repo():
    url = _target_url("tokenizer.json")
    assert url == (
        "https://huggingface.co/Xenova/clip-vit-base-patch32"
        "/resolve/main/tokenizer.json"
    )
    assert _target_url("vision_model_fp16.onnx").endswith("/onnx/vision_model_fp16.onnx")


def test_is_present_false_when_a_file_is_missing(tmp_path):
    assert is_present(tmp_path) is False
    for name in FILES:
        (tmp_path / name).write_bytes(b"x")
    assert is_present(tmp_path) is True


def test_is_present_ignores_partial_downloads(tmp_path):
    """A .part file must never be mistaken for a finished download."""
    for name in FILES:
        (tmp_path / (name + ".part")).write_bytes(b"x")
    assert is_present(tmp_path) is False
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_fetch_clip_onnx.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools'`.

- [ ] **Step 4: Write the implementation**

Create `tools/__init__.py` (empty file).

Create `tools/fetch_clip_onnx.py`:

```python
"""One-time download of the CLIP ViT-B/32 ONNX assets.

Stdlib only - this runs before onnxruntime is necessarily importable, and we do
not want a huggingface_hub dependency for three static URLs.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

MODEL_DIR = "models/clip-vit-b32"

_REPO = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"

# local filename -> path within the HF repo
FILES: dict[str, str] = {
    "vision_model_fp16.onnx": "onnx/vision_model_fp16.onnx",
    "text_model_fp16.onnx": "onnx/text_model_fp16.onnx",
    "tokenizer.json": "tokenizer.json",
}


def _target_url(name: str) -> str:
    return f"{_REPO}/{FILES[name]}"


def is_present(dest_dir: str | Path) -> bool:
    """True only when every asset is fully downloaded."""
    d = Path(dest_dir)
    return all((d / name).is_file() for name in FILES)


def fetch(dest_dir: str | Path = MODEL_DIR, progress=None) -> None:
    """Download any missing assets. Writes to .part then renames, so an
    interrupted download can never be mistaken for a finished one."""
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        final = d / name
        if final.is_file():
            continue
        part = d / (name + ".part")
        url = _target_url(name)
        if progress:
            progress(name, 0.0)
        with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(name, done / total)
        os.replace(part, final)
        if progress:
            progress(name, 1.0)


if __name__ == "__main__":
    def _cli(name: str, frac: float) -> None:
        print(f"\r{name}: {frac * 100:5.1f}%", end="", flush=True)
        if frac >= 1.0:
            print()

    fetch(progress=_cli)
    print(f"CLIP assets ready in {MODEL_DIR}")
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_fetch_clip_onnx.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Install dependencies and download the model for real**

```bash
./.venv/Scripts/python.exe -m pip install onnxruntime-directml tokenizers cmaes
```

```bash
./.venv/Scripts/python.exe -m tools.fetch_clip_onnx
```

Expected: three files in `models/clip-vit-b32/`, roughly 176 MB, 127 MB and 2 MB.

- [ ] **Step 7: Confirm the ONNX output tensor names**

The spec requires verifying that the split exports emit *projected* embeddings.

```bash
./.venv/Scripts/python.exe -c "import onnxruntime as ort; s=ort.InferenceSession('models/clip-vit-b32/vision_model_fp16.onnx', providers=['CPUExecutionProvider']); print('IN ', [(i.name,i.shape) for i in s.get_inputs()]); print('OUT', [(o.name,o.shape) for o in s.get_outputs()])"
```

**If the outputs include `image_embeds`:** proceed as planned.

**If the outputs are only `last_hidden_state` / `pooler_output`:** the projection layer is missing. Change `FILES` in `tools/fetch_clip_onnx.py` to fetch the combined `onnx/model_fp16.onnx` as `model_fp16.onnx` instead of the two split files, re-run the fetch, and note the deviation. The combined model exposes `image_embeds` and `text_embeds` on one session. Task 2 must then use one session for both.

Record the actual input and output names in a comment at the top of `services/clip_scorer.py` in Task 2.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .gitignore tools/ tests/test_fetch_clip_onnx.py
git commit -m "feat: CLIP ONNX model fetch script and optional dependencies

Stdlib-only downloader for the three ViT-B/32 assets, writing to .part and
renaming so an interrupted download is never mistaken for a complete one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: CLIPScorer

**Files:**
- Create: `services/clip_scorer.py`
- Test: `tests/test_clip_scorer.py`

**Interfaces:**
- Consumes: `tools.fetch_clip_onnx.MODEL_DIR`, `is_present`.
- Produces:
  - `CLIP_MEAN: np.ndarray (3,)`, `CLIP_STD: np.ndarray (3,)`, `LOGIT_SCALE: float = 100.0`
  - `DEFAULT_DISTRACTORS: list[str]`
  - `preprocess(crops: np.ndarray) -> np.ndarray` — uint8 `(B,224,224,3)` → float16 `(B,3,224,224)`, C-contiguous
  - `augment(crops: np.ndarray, n_views: int, rng) -> np.ndarray` — `(B,224,224,3)` → `(B*n_views,224,224,3)`
  - `class CLIPScorer` with `available: bool`, `set_prompt(text, distractors=None)`, `score(images) -> np.ndarray (B,) float32`, `MAX_CHUNK = 64`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clip_scorer.py`:

```python
import numpy as np
import pytest

from services.clip_scorer import (
    CLIP_MEAN,
    CLIP_STD,
    DEFAULT_DISTRACTORS,
    LOGIT_SCALE,
    CLIPScorer,
    augment,
    preprocess,
)


def test_preprocess_shape_dtype_and_contiguity():
    crops = np.full((2, 224, 224, 3), 255, dtype=np.uint8)
    x = preprocess(crops)
    assert x.shape == (2, 3, 224, 224)
    assert x.dtype == np.float16
    assert x.flags["C_CONTIGUOUS"], "ONNX Runtime needs a contiguous NCHW buffer"


def test_preprocess_applies_clip_normalisation():
    crops = np.full((1, 224, 224, 3), 255, dtype=np.uint8)
    x = preprocess(crops).astype(np.float32)
    expected = (1.0 - CLIP_MEAN) / CLIP_STD
    for c in range(3):
        assert np.allclose(x[0, c], expected[c], atol=1e-2)


def test_augment_produces_n_views_per_image():
    rng = np.random.default_rng(0)
    crops = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    out = augment(crops, n_views=3, rng=rng)
    assert out.shape == (9, 224, 224, 3)
    assert out.dtype == np.uint8


def test_augment_first_view_is_the_untouched_frame():
    rng = np.random.default_rng(0)
    crops = np.random.default_rng(1).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)
    out = augment(crops, n_views=3, rng=rng)
    # views are laid out image-major: [img0 v0, img0 v1, img0 v2, img1 v0, ...]
    assert np.array_equal(out[0], crops[0])
    assert np.array_equal(out[3], crops[1])


class _StubSession:
    """Stands in for an onnxruntime InferenceSession.

    Returns an embedding whose direction is controlled by the mean pixel value,
    so we can make specific images 'match' specific prompts deterministically.
    """

    def __init__(self, kind):
        self.kind = kind
        self.calls = 0
        self.batch_sizes = []

    def get_inputs(self):
        name = "pixel_values" if self.kind == "vision" else "input_ids"
        return [type("I", (), {"name": name})()]

    def get_outputs(self):
        name = "image_embeds" if self.kind == "vision" else "text_embeds"
        return [type("O", (), {"name": name})()]

    def run(self, _out_names, feed):
        self.calls += 1
        arr = next(iter(feed.values()))
        n = arr.shape[0]
        self.batch_sizes.append(n)
        emb = np.zeros((n, 4), dtype=np.float32)
        emb[:, 0] = 1.0
        return [emb]


def _scorer_with_stubs(monkeypatch, vision=None, text=None):
    s = CLIPScorer.__new__(CLIPScorer)
    s._vision = vision or _StubSession("vision")
    s._text = text or _StubSession("text")
    s._vision_in = "pixel_values"
    s._vision_out = "image_embeds"
    s._text_in = "input_ids"
    s._text_out = "text_embeds"
    s._tokenizer = None
    s._text_emb = None
    s._prompt = ""
    s._n_views = 3
    s._rng = np.random.default_rng(0)
    s._available = True
    return s


def test_score_returns_probability_per_image(monkeypatch):
    s = _scorer_with_stubs(monkeypatch)
    # 1 target + 6 distractors, all identical -> uniform softmax = 1/7
    s._text_emb = np.tile(np.array([[1.0, 0, 0, 0]], dtype=np.float32), (7, 1))
    out = s.score(np.zeros((4, 224, 224, 3), dtype=np.uint8))
    assert out.shape == (4,)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0 / 7.0, atol=1e-5)


def test_score_target_is_index_zero(monkeypatch):
    s = _scorer_with_stubs(monkeypatch)
    # target aligned with the image embedding, distractors orthogonal
    emb = np.zeros((3, 4), dtype=np.float32)
    emb[0] = [1.0, 0, 0, 0]
    emb[1] = [0, 1.0, 0, 0]
    emb[2] = [0, 0, 1.0, 0]
    s._text_emb = emb
    out = s.score(np.zeros((1, 224, 224, 3), dtype=np.uint8))
    assert out[0] > 0.99, "an image aligned with prompt 0 must score near 1"


def test_score_chunks_large_batches(monkeypatch):
    s = _scorer_with_stubs(monkeypatch)
    s._text_emb = np.tile(np.array([[1.0, 0, 0, 0]], dtype=np.float32), (7, 1))
    # 64 tiles x 3 views = 192 images -> must not be one 192-image call
    s.score(np.zeros((64, 224, 224, 3), dtype=np.uint8))
    assert max(s._vision.batch_sizes) <= CLIPScorer.MAX_CHUNK
    assert sum(s._vision.batch_sizes) == 192


def test_default_distractors_include_abstract_texture():
    """Without this one, every trail pattern scores highly on every prompt."""
    assert "an abstract texture" in DEFAULT_DISTRACTORS


def test_logit_scale_matches_clip():
    assert LOGIT_SCALE == pytest.approx(100.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_clip_scorer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'services.clip_scorer'`.

- [ ] **Step 3: Write the implementation**

Create `services/clip_scorer.py`:

```python
"""CLIP text-image scoring for the automatic tournament.

Depends only on onnxruntime, tokenizers, numpy and PIL. Knows nothing about
tournaments, tiles or OpenGL.

ONNX tensor names confirmed in Task 1 Step 7:
  vision: in 'pixel_values' (N,3,224,224) fp16 -> out 'image_embeds' (N,512)
  text:   in 'input_ids' (N,77) int64 [+ 'attention_mask'] -> out 'text_embeds'
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# CLIP preprocessing constants, from preprocessor_config.json
CLIP_MEAN = np.array([0.48145466, 0.45782750, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# CLIP's learned logit_scale.exp()
LOGIT_SCALE = 100.0

CONTEXT_LENGTH = 77

DEFAULT_DISTRACTORS = [
    "a blank image",
    "random noise",
    "a blurry photograph",
    "a screenshot of text",
    "a solid color",
    # Without this one, nearly every trail pattern scores highly on nearly every
    # prompt, because "abstract texture" is the honest description of most of
    # the search space.
    "an abstract texture",
]


def preprocess(crops: np.ndarray) -> np.ndarray:
    """uint8 (B,224,224,3) -> float16 (B,3,224,224), C-contiguous."""
    x = crops.astype(np.float32) / 255.0
    x = (x - CLIP_MEAN) / CLIP_STD
    # ascontiguousarray is required, not cosmetic: a non-contiguous array either
    # forces a silent copy inside ONNX Runtime or errors, depending on provider.
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    return x.astype(np.float16)


def augment(crops: np.ndarray, n_views: int, rng) -> np.ndarray:
    """Expand each image into n_views. View 0 is the untouched frame; the rest
    are random 160-224px sub-crops resampled back to 224.

    Averaging over random views is the standard defense against a search process
    finding adversarial texture that satisfies one fixed view.
    Output is image-major: [img0 v0, img0 v1, ..., img1 v0, ...].
    """
    if n_views <= 1:
        return crops
    b, h, w, _ = crops.shape
    out = np.empty((b * n_views, h, w, 3), dtype=np.uint8)
    for i in range(b):
        out[i * n_views] = crops[i]
        for v in range(1, n_views):
            size = int(rng.integers(160, h + 1))
            y = int(rng.integers(0, h - size + 1))
            x = int(rng.integers(0, w - size + 1))
            patch = crops[i, y:y + size, x:x + size]
            if size == h:
                out[i * n_views + v] = patch
            else:
                img = Image.fromarray(patch).resize((w, h), Image.BILINEAR)
                out[i * n_views + v] = np.asarray(img)
    return out


def _l2(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


class CLIPScorer:
    """Turns a text prompt into a per-image fitness in [0, 1]."""

    MAX_CHUNK = 64  # images per session.run; 64 fp16 NCHW inputs ~= 19 MB

    def __init__(self, model_dir, providers=None, n_views: int = 3, seed: int = 0):
        self._available = False
        self._text_emb = None
        self._prompt = ""
        self._n_views = n_views
        self._rng = np.random.default_rng(seed)

        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self._vision = ort.InferenceSession(str(d / "vision_model_fp16.onnx"), providers=providers)
        self._text = ort.InferenceSession(str(d / "text_model_fp16.onnx"), providers=providers)

        self._vision_in = self._vision.get_inputs()[0].name
        self._vision_out = self._vision.get_outputs()[0].name
        self._text_in = self._text.get_inputs()[0].name
        self._text_out = self._text.get_outputs()[0].name

        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tokenizer.enable_truncation(CONTEXT_LENGTH)
        self._tokenizer.enable_padding(length=CONTEXT_LENGTH)

        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def prompt(self) -> str:
        return self._prompt

    def set_prompt(self, text: str, distractors: list[str] | None = None) -> None:
        """Embed the target prompt (index 0) plus distractors. Cached until the
        next call, so retyping the goal costs one text-encoder pass."""
        prompts = [text] + list(DEFAULT_DISTRACTORS if distractors is None else distractors)
        encoded = self._tokenizer.encode_batch(prompts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        feed = {self._text_in: ids}
        for inp in self._text.get_inputs()[1:]:
            if inp.name == "attention_mask":
                feed["attention_mask"] = np.array(
                    [e.attention_mask for e in encoded], dtype=np.int64
                )
        emb = self._text.run([self._text_out], feed)[0].astype(np.float32)
        self._text_emb = _l2(emb)
        self._prompt = text

    def _embed_images(self, crops: np.ndarray) -> np.ndarray:
        chunks = []
        for i in range(0, len(crops), self.MAX_CHUNK):
            batch = preprocess(crops[i:i + self.MAX_CHUNK])
            out = self._vision.run([self._vision_out], {self._vision_in: batch})[0]
            chunks.append(out.astype(np.float32))
        return _l2(np.concatenate(chunks, axis=0))

    def score(self, images: np.ndarray) -> np.ndarray:
        """uint8 (B,224,224,3) -> float32 (B,). Softmax probability of the
        target prompt against the distractor set, averaged over augmented views.
        """
        if self._text_emb is None:
            raise RuntimeError("set_prompt() must be called before score()")
        b = len(images)
        views = augment(images, self._n_views, self._rng)
        emb = self._embed_images(views)
        logits = LOGIT_SCALE * (emb @ self._text_emb.T)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        target = probs[:, 0].reshape(b, -1)  # (B, n_views)
        return target.mean(axis=1).astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_clip_scorer.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Add the real-model GPU test**

Create `tests/test_clip_real_model.py`:

```python
"""Requires the downloaded CLIP model. Run with: pytest -m gpu"""
import numpy as np
import pytest

from tools.fetch_clip_onnx import MODEL_DIR, is_present

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def scorer():
    if not is_present(MODEL_DIR):
        pytest.skip("CLIP model not downloaded")
    from services.clip_scorer import CLIPScorer

    return CLIPScorer(MODEL_DIR)


def _square(color):
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[56:168, 56:168] = color
    return img


def test_red_square_prefers_its_own_prompt(scorer):
    red = _square((220, 30, 30))[None]
    scorer.set_prompt("a red square")
    red_on_red = scorer.score(red)[0]
    scorer.set_prompt("a blue circle")
    red_on_blue = scorer.score(red)[0]
    assert red_on_red > red_on_blue


def test_blank_image_scores_lower_than_content(scorer):
    blank = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    red = _square((220, 30, 30))[None]
    scorer.set_prompt("a red square")
    assert scorer.score(red)[0] > scorer.score(blank)[0]
```

Register the marker in `pytest.ini` (create it if absent, otherwise add the line to the existing `[pytest]` section):

```ini
[pytest]
markers =
    gpu: requires a GPU and/or downloaded model weights (deselect with '-m "not gpu"')
```

- [ ] **Step 6: Run the GPU test**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_clip_real_model.py -v -m gpu
```

Expected: 2 passed. If `DmlExecutionProvider` is unavailable it will silently use CPU and still pass, just slower.

- [ ] **Step 7: Commit**

```bash
git add services/clip_scorer.py tests/test_clip_scorer.py tests/test_clip_real_model.py pytest.ini
git commit -m "feat: CLIPScorer with softmax-vs-distractor fitness

Scores images as the softmax probability of the target prompt against a
distractor set rather than raw cosine similarity, which has a narrow ~0.15-0.35
band and is easily exploited. Averages over 3 augmented views and chunks vision
inference at 64 images to bound input tensor memory.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Kill-switch — does CLIP have signal on trail patterns?

This task is a **gate**. It builds no production code. If it fails, stop and report to the user before continuing.

**Files:**
- Create: `tools/clip_signal_check.py`

**Interfaces:**
- Consumes: `services.clip_scorer.CLIPScorer`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the check script**

Create `tools/clip_signal_check.py`:

```python
"""Kill-switch: does CLIP produce a usable fitness landscape on trail patterns?

Usage:
    python -m tools.clip_signal_check <dir-of-pngs>

Bar to pass: the std-dev of scores across the sample must exceed 0.05 for at
least half the prompts. A flat landscape means no optimizer can climb it, and
we should fix the eval image before building the loop on top of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from services.clip_scorer import CLIPScorer
from tools.fetch_clip_onnx import MODEL_DIR

PROMPTS = [
    "glowing coral",
    "a dense city map",
    "a swirling galaxy",
    "tree branches",
    "flowing water",
    "a spider web",
]

STD_BAR = 0.05


def load_tiles(d: Path) -> tuple[np.ndarray, list[str]]:
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
    if not paths:
        raise SystemExit(f"no PNGs in {d}")
    imgs = [
        np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR))
        for p in paths
    ]
    return np.stack(imgs).astype(np.uint8), [p.name for p in paths]


def main() -> int:
    tiles, names = load_tiles(Path(sys.argv[1]))
    scorer = CLIPScorer(MODEL_DIR)
    print(f"{len(tiles)} tiles, {len(PROMPTS)} prompts\n")

    passes = 0
    for prompt in PROMPTS:
        scorer.set_prompt(prompt)
        s = scorer.score(tiles)
        order = np.argsort(-s)
        ok = s.std() > STD_BAR
        passes += ok
        print(f"{prompt!r:24} std={s.std():.4f} range=[{s.min():.3f},{s.max():.3f}] "
              f"{'PASS' if ok else 'flat'}")
        print(f"    best : {names[order[0]]} ({s[order[0]]:.3f})")
        print(f"    worst: {names[order[-1]]} ({s[order[-1]]:.3f})")

    verdict = passes >= len(PROMPTS) / 2
    print(f"\n{passes}/{len(PROMPTS)} prompts above std {STD_BAR} -> "
          f"{'GATE PASSED' if verdict else 'GATE FAILED'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Collect ~20 varied sample tiles**

Run the app, enable manual tournament mode, and use Shift+P (or the existing screenshot path) to capture the grid across 2 generations. Crop into individual tiles, or simply save whole-grid screenshots and crop with:

```bash
./.venv/Scripts/python.exe -c "
from PIL import Image; import sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path('scratch_tiles'); out.mkdir(exist_ok=True)
im = Image.open(src); w,h = im.size; s = min(w,h)//4
for ty in range(4):
    for tx in range(4):
        im.crop((tx*s, ty*s, (tx+1)*s, (ty+1)*s)).save(out/f'{src.stem}_{ty}{tx}.png')
" <path-to-screenshot.png>
```

Aim for genuine variety: dense, sparse, filamentary, blobby, near-dead.

- [ ] **Step 3: Run the gate**

```bash
./.venv/Scripts/python.exe -m tools.clip_signal_check scratch_tiles
```

- [ ] **Step 4: Interpret and report**

**GATE PASSED** → continue to Task 4.

**GATE FAILED** → **stop**. Report the numbers to the user before writing any more code. The likely remedies, in order of cost:
1. Raise contrast/exposure in the captured view — CLIP may be seeing near-black frames.
2. Capture at a lower grid N so each tile has more real pixels.
3. Try a different view mode (particles+trails composite, view 6) as the eval image.
4. Reconsider the prompt set — very abstract prompts have less signal than concrete ones.

Re-run the gate after each change. Do not proceed on a failed gate.

- [ ] **Step 5: Commit the tool (regardless of outcome)**

```bash
git add tools/clip_signal_check.py
git commit -m "tools: CLIP signal check for trail patterns

Gate before building the evolution loop: confirms CLIP scores vary enough
across sample tiles to be optimizable. A flat landscape would make every
layer above it undebuggable.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Variable grid, cohort tiling, and the mutation suppression fix

**Files:**
- Modify: `services/tournament_service.py` (`TILES` → constructor arg)
- Modify: `ui/tournament_window.py:40` (hardcoded `grid = 4`)
- Modify: `command_handler.py:230` (hardcoded grid in click mapping)
- Modify: `sim.py:226-228` (**additive only**)
- Create: `services/cohort_tiling.py`
- Test: `tests/test_cohort_tiling.py`, `tests/test_tournament_service.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `TournamentService(grid: int = 4, rng=None)` with `self.grid`, `self.tiles` (== grid²). `TILES` class attribute is removed.
  - `services/cohort_tiling.py`: `tile_of(index, active, grid) -> int`, `cohort_of(index, active, cohorts) -> int`, `max_variants(grid) -> int`, `cohorts_for(grid, variants) -> int`, `MAX_COHORTS = 144`

- [ ] **Step 1: Write the failing cohort tiling test**

Create `tests/test_cohort_tiling.py`:

```python
import pytest

from services.cohort_tiling import (
    MAX_COHORTS,
    cohort_of,
    cohorts_for,
    max_variants,
    tile_of,
)

ACTIVE = 100_000


@pytest.mark.parametrize("grid", [2, 3, 4, 5, 6, 7, 8])
def test_each_tile_contains_exactly_k_cohorts(grid):
    """Tile and cohort are both slices of the same particle numbering, so
    COHORTS must be a multiple of the tile count. If they were equal, every
    tile would hold one cohort - one mutation - and stay a monoculture."""
    tiles = grid * grid
    for k in range(1, max_variants(grid) + 1):
        cohorts = cohorts_for(grid, k)
        assert cohorts == k * tiles
        seen = {t: set() for t in range(tiles)}
        for i in range(0, ACTIVE, 7):
            seen[tile_of(i, ACTIVE, grid)].add(cohort_of(i, ACTIVE, cohorts))
        for t, s in seen.items():
            assert len(s) == k, f"grid={grid} k={k} tile={t} saw {len(s)} cohorts"


def test_the_monoculture_trap():
    """cohorts == tiles is the specific failure the feature must avoid."""
    grid, active = 4, ACTIVE
    seen = {t: set() for t in range(16)}
    for i in range(0, active, 7):
        seen[tile_of(i, active, grid)].add(cohort_of(i, active, 16))
    assert all(len(s) == 1 for s in seen.values())


@pytest.mark.parametrize(
    "grid,expected", [(2, 36), (3, 16), (4, 9), (5, 5), (6, 4), (7, 2), (8, 2)]
)
def test_max_variants_matches_the_144_cohort_cap(grid, expected):
    assert max_variants(grid) == expected
    assert cohorts_for(grid, expected) <= MAX_COHORTS


def test_tile_of_covers_every_tile_and_stays_in_range():
    for grid in (2, 4, 8):
        tiles = grid * grid
        got = {tile_of(i, ACTIVE, grid) for i in range(ACTIVE)}
        assert got == set(range(tiles))
    assert tile_of(ACTIVE - 1, ACTIVE, 4) == 15
    assert tile_of(ACTIVE, ACTIVE, 4) == 15, "must clamp, matching the shader"
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_cohort_tiling.py -v
```

Expected: FAIL, `No module named 'services.cohort_tiling'`.

- [ ] **Step 3: Implement cohort_tiling**

Create `services/cohort_tiling.py`:

```python
"""Tile and cohort indexing, mirroring entity_update.glsl exactly.

Both `tournament_home_tile(index)` and `get_cohort(index)` are slices of the
same particle numbering, so they nest: with COHORTS = k * tiles, each tile
contains exactly k cohorts, and each cohort gets its own deterministic tweak of
that tile's genome.

If COHORTS == tiles, each tile holds exactly one cohort - one tweak applied to
every particle in it - so the tile stays uniform, just displaced. That is the
trap this module exists to prevent.
"""
from __future__ import annotations

import math

# The Number of Cohorts slider is capped at 144 in ui/physics_window.py.
MAX_COHORTS = 144


def tile_of(index: int, active: int, grid: int) -> int:
    """Mirrors tournament_home_tile() in entity_update.glsl."""
    n = grid * grid
    tile = int(math.floor(float(index) / float(active) * float(n)))
    return max(0, min(tile, n - 1))


def cohort_of(index: int, active: int, cohorts: int) -> int:
    """Mirrors floor(get_cohort(index)) in entity_update.glsl."""
    return int(math.floor(float(cohorts) * float(index) / float(active)))


def max_variants(grid: int) -> int:
    """Largest variants-per-tile that keeps COHORTS within the 144 cap."""
    return max(1, MAX_COHORTS // (grid * grid))


def cohorts_for(grid: int, variants: int) -> int:
    """Cohort count that gives exactly `variants` distinct cohorts per tile."""
    return variants * grid * grid
```

- [ ] **Step 4: Run to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_cohort_tiling.py -v
```

Expected: 17 passed.

- [ ] **Step 5: Make TournamentService grid-configurable**

In `services/tournament_service.py`, remove the `TILES = 16` class attribute and change the constructor:

```python
    def __init__(self, grid: int = 4, rng: np.random.Generator | None = None):
        self.grid = grid
        self.tiles = grid * grid
        ...
```

Replace every `self.TILES` with `self.tiles` throughout the file (constructor, `init_population`, `reset`, `toggle_select`, `next_generation`).

Add a method to change grid at a generation boundary:

```python
    def set_grid(self, grid: int) -> None:
        """Resize the population. Callers must only do this between
        generations - cmaes.CMA fixes popsize at construction, so the optimizer
        is reset separately by AutoTournamentService."""
        if grid == self.grid:
            return
        self.grid = grid
        self.tiles = grid * grid
        self.selected.clear()
        self._undo_stack.clear()
        self.population = [random_genome(self._rng) for _ in range(self.tiles)]
        self.mark_dirty()
```

Add to `tests/test_tournament_service.py`:

```python
def test_grid_is_configurable():
    svc = TournamentService(grid=6)
    svc.init_population()
    assert svc.tiles == 36
    assert len(svc.population) == 36


def test_set_grid_resizes_and_clears_selection():
    svc = TournamentService(grid=4)
    svc.init_population()
    svc.toggle_select(3)
    svc.set_grid(2)
    assert svc.tiles == 4
    assert len(svc.population) == 4
    assert svc.selected == set()
    assert svc.is_dirty()
```

- [ ] **Step 6: Make the UI grid and click mapping N-driven**

In `ui/tournament_window.py`, replace the hardcoded `grid = 4` at line 40 with:

```python
        grid = self.tournament_service.grid
```

In `command_handler.py`, in `_handle_mouse_clicks`, replace the hardcoded grid with:

```python
            grid = self.tournament_service.grid if self.tournament_service else 4
```

before the line that computes `ui_state.tournament.clicked_tile = ty * grid + tx`.

In `main.py:233`, replace:

```python
        self.sim.apply_tournament(ui_state.tournament.enabled, grid=4)
```

with:

```python
        self.sim.apply_tournament(
            ui_state.tournament.enabled, grid=self.tournament_service.grid
        )
```

- [ ] **Step 7: Fix mutation suppression in sim.py (additive only)**

`sim.py:228` zeroes `MUTATION_SCALE_SETTING.slider_value` to disable per-particle mutation. That is insufficient: `calculate_setting` (`entity_update.glsl:171`) returns `slider_value` **only** when all sweeps and jitter are zero, and otherwise computes from the sweeps and ignores `slider_value` entirely.

Replace the body of the existing `if self._tournament_enabled:` block at `sim.py:226-228` with:

```python
        if self._tournament_enabled:
            # Per-particle mutation is owned by tournament mode. Zeroing
            # slider_value alone is NOT enough: calculate_setting() ignores
            # slider_value whenever any sweep or jitter is non-zero, so a user
            # sweep on Mutation Scale would survive the suppression.
            tryset(self.entity_update_program,
                   'MUTATION_SCALE_SETTING.slider_value',
                   float(self._tournament_mutation))
            for _field in ('x_sweep', 'y_sweep', 'cohort_sweep', 'jitter'):
                tryset(self.entity_update_program,
                       f'MUTATION_SCALE_SETTING.{_field}', 0.0)
```

Add the backing attribute next to `self._tournament_enabled = False` at `sim.py:37`:

```python
        self._tournament_mutation = 0.0
```

And extend `apply_tournament` (at `sim.py:739`) to accept it:

```python
    def apply_tournament(self, enabled: bool, grid: int = 4, mutation: float = 0.0):
        self._tournament_enabled = enabled
        self._tournament_grid = grid
        self._tournament_mutation = mutation
```

(Keep whatever the existing body does; only add the `mutation` parameter and the assignment.)

- [ ] **Step 8: Run the full suite**

```bash
./.venv/Scripts/python.exe -m pytest -m "not gpu" -q
```

Expected: all prior tests still pass, plus the new ones.

- [ ] **Step 9: Verify manual tournament still works in the app**

```bash
./.venv/Scripts/python.exe main.py
```

Enable tournament mode, confirm the 4×4 grid renders, clicking a tile selects the correct one, and Next Gen advances. Manual mode behaviour must be unchanged.

- [ ] **Step 10: Commit**

```bash
git add services/cohort_tiling.py services/tournament_service.py ui/tournament_window.py command_handler.py main.py sim.py tests/test_cohort_tiling.py tests/test_tournament_service.py
git commit -m "feat: variable tournament grid, and fix mutation suppression

TILES=16 becomes a constructor arg; the UI grid, click mapping and shader
uniform all follow TournamentService.grid.

Adds services/cohort_tiling.py, which mirrors the shader's tile and cohort
indexing. Both derive from the same particle numbering, so COHORTS must be a
multiple of the tile count - if they are equal, every tile holds exactly one
cohort and stays a monoculture with a displaced genome.

Fixes a real bug: sim.py zeroed MUTATION_SCALE_SETTING.slider_value to suppress
per-particle mutation, but calculate_setting ignores slider_value entirely when
any sweep or jitter is non-zero, so mutation survived suppression in that case.
Suppression now zeroes the sweeps too.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: GenomeSpec

**Files:**
- Create: `services/genome_spec.py`
- Test: `tests/test_genome_spec.py`

**Interfaces:**
- Consumes: `services.genome.GENOME_SHAPE`, `N_CENTERS`.
- Produces:
  - `FREQ_SCALE = 3.0`, `AMP_SCALE = 1.0`, `EPS = 1e-4`
  - `decode(z: np.ndarray) -> np.ndarray` — `(80,)` → `(10,8)` float32
  - `encode(genome: np.ndarray) -> tuple[np.ndarray, int]` — `(10,8)` → `((80,), n_clamped)`
  - `class Block(name: str, size: int)`, `class GenomeSpec` with `.dim`, `.signature()`, `.decode(z)`, `.encode(parts)`
  - `BRAIN_SPEC = GenomeSpec([Block("brain", 80)])`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_genome_spec.py`:

```python
import numpy as np
import pytest

from services.genome_spec import (
    AMP_SCALE,
    BRAIN_SPEC,
    FREQ_SCALE,
    decode,
    encode,
)


def test_decode_shape_and_dtype():
    g = decode(np.zeros(80, dtype=np.float32))
    assert g.shape == (10, 8)
    assert g.dtype == np.float32


def test_decode_is_bounded_for_extreme_z():
    """tanh squash means the optimizer can never wander to freq=50."""
    g = decode(np.full(80, 1e6, dtype=np.float32))
    assert np.all(np.abs(g[:, :4]) <= FREQ_SCALE + 1e-5)
    assert np.all(np.abs(g[:, 4:]) <= AMP_SCALE + 1e-5)
    g = decode(np.full(80, -1e6, dtype=np.float32))
    assert np.all(np.abs(g[:, :4]) <= FREQ_SCALE + 1e-5)


def test_encode_decode_roundtrip_for_in_range_values():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.7, 80).astype(np.float32)
    g = decode(z)
    z2, clamped = encode(g)
    assert clamped == 0
    assert np.allclose(z, z2, atol=1e-3)


def test_encode_clamps_out_of_range_and_reports_count():
    g = np.zeros((10, 8), dtype=np.float32)
    g[0, 0] = 99.0     # frequency far beyond +3
    g[1, 4] = -5.0     # amplitude far beyond -1
    z, clamped = encode(g)
    assert clamped == 2
    assert np.all(np.isfinite(z)), "must never produce inf in x0"
    assert z[0] == pytest.approx(np.arctanh(1 - 1e-4), rel=1e-3)


def test_spec_dim_and_signature():
    assert BRAIN_SPEC.dim == 80
    assert BRAIN_SPEC.signature() == "brain:80"


def test_spec_decode_returns_named_parts():
    parts = BRAIN_SPEC.decode(np.zeros(80, dtype=np.float32))
    assert set(parts) == {"brain"}
    assert parts["brain"].shape == (10, 8)
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_genome_spec.py -v
```

Expected: FAIL, `No module named 'services.genome_spec'`.

- [ ] **Step 3: Implement**

Create `services/genome_spec.py`:

```python
"""Mapping between the optimizer's search space and real genomes.

The optimizer searches an unbounded z in R^80. Decoding applies a bounded tanh
squash, so it can never wander to freq=50, and - unlike clipping - no repair
bias is introduced at the boundary. Clipping would map many distinct z to the
same genome, which distorts CMA-ES's covariance estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.genome import N_CENTERS

FREQ_SCALE = 3.0
AMP_SCALE = 1.0
EPS = 1e-4

DIM = N_CENTERS * 8  # 80


def decode(z: np.ndarray) -> np.ndarray:
    """(80,) -> (10, 8) float32, matching services.genome.random_genome ranges."""
    z = np.asarray(z, dtype=np.float32).reshape(N_CENTERS * 2, 4)
    freq = FREQ_SCALE * np.tanh(z[:N_CENTERS])
    amp = AMP_SCALE * np.tanh(z[N_CENTERS:])
    return np.concatenate([freq, amp], axis=1).astype(np.float32)


def encode(genome: np.ndarray) -> tuple[np.ndarray, int]:
    """(10, 8) -> ((80,) float32, n_clamped).

    A config file may hold values outside the squash range (legacy generator,
    hand editing). Those are clamped, which is lossy at the extremes but keeps
    x0 finite. The clamp count is returned so a badly out-of-range file is
    visible rather than silent.
    """
    g = np.asarray(genome, dtype=np.float32).reshape(N_CENTERS, 8)
    freq = g[:, :4] / FREQ_SCALE
    amp = g[:, 4:] / AMP_SCALE
    raw = np.concatenate([freq, amp], axis=0)
    n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
    z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
    return z.reshape(-1).astype(np.float32), n_clamped


@dataclass(frozen=True)
class Block:
    name: str
    size: int


class GenomeSpec:
    """Ordered named blocks of the search vector.

    Adding physics later means appending Block("physics", 12); the optimizer,
    scorer, logger and loop are all dimension-agnostic.
    """

    def __init__(self, blocks: list[Block]):
        self.blocks = list(blocks)

    @property
    def dim(self) -> int:
        return sum(b.size for b in self.blocks)

    def signature(self) -> str:
        """Stable string used to reject incompatible checkpoints."""
        return ",".join(f"{b.name}:{b.size}" for b in self.blocks)

    def decode(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z = np.asarray(z, dtype=np.float32)
        out: dict[str, np.ndarray] = {}
        off = 0
        for b in self.blocks:
            chunk = z[off:off + b.size]
            out[b.name] = decode(chunk) if b.name == "brain" else chunk.copy()
            off += b.size
        return out

    def encode(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        pieces = []
        for b in self.blocks:
            v = parts[b.name]
            pieces.append(encode(v)[0] if b.name == "brain" else np.asarray(v).reshape(-1))
        return np.concatenate(pieces).astype(np.float32)


BRAIN_SPEC = GenomeSpec([Block("brain", DIM)])
```

- [ ] **Step 4: Run to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_genome_spec.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add services/genome_spec.py tests/test_genome_spec.py
git commit -m "feat: GenomeSpec - bounded tanh mapping between z-space and genomes

Squash rather than clip, so no repair bias distorts the CMA-ES covariance.
encode() reports how many coefficients it clamped, so an out-of-range config
file is visible instead of silently producing inf in x0.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: Optimizers

**Files:**
- Create: `services/optimizers.py`
- Test: `tests/test_optimizers.py`

**Interfaces:**
- Consumes: `services.genome.mutate`, `crossover`.
- Produces:
  - `class Optimizer(Protocol)` with `name`, `ask(n)`, `tell(z, fitness)`, `best()`, `state_dict()`, `load_state_dict(d)`, `sigma` property
  - `CMAESOptimizer`, `SepCMAESOptimizer`, `GAOptimizer`, `RandomSearchOptimizer` — all `(dim, popsize, sigma0=0.5, seed=0, x0=None)`
  - `ALGORITHMS: dict[str, type]` keyed by `"CMA-ES"`, `"Sep-CMA-ES"`, `"GA"`, `"Random Search"`
  - `make_optimizer(name, dim, popsize, sigma0, seed, x0=None) -> Optimizer`

- [ ] **Step 1: Inspect the cmaes internals before writing state_dict**

```bash
./.venv/Scripts/python.exe -c "
from cmaes import CMA
import numpy as np
c = CMA(mean=np.zeros(4), sigma=0.5, population_size=8)
print(sorted(k for k in vars(c)))
"
```

Record the printed attribute names. The implementation below uses an allowlist and skips absent names, and Step 4's roundtrip test fails loudly if the set is wrong — but knowing the real names first avoids a confusing first run.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_optimizers.py`:

```python
import numpy as np
import pytest

from services.optimizers import ALGORITHMS, make_optimizer

DIM = 80
POP = 16
NAMES = list(ALGORITHMS)


def sphere(z):
    """Fitness, higher is better. Optimum 0.0 at the origin."""
    return -float(np.sum(z * z))


def rastrigin(z):
    return -float(10 * len(z) + np.sum(z * z - 10 * np.cos(2 * np.pi * z)))


def run(name, fn, gens, seed=0, sigma0=0.5):
    opt = make_optimizer(name, DIM, POP, sigma0, seed)
    best = -np.inf
    for _ in range(gens):
        z = opt.ask(POP)
        f = np.array([fn(zi) for zi in z], dtype=np.float32)
        opt.tell(z, f)
        best = max(best, float(f.max()))
    return best, opt


@pytest.mark.parametrize("name", NAMES)
def test_ask_returns_correct_shape_and_dtype(name):
    opt = make_optimizer(name, DIM, POP, 0.5, 0)
    z = opt.ask(POP)
    assert z.shape == (POP, DIM)
    assert z.dtype == np.float32


@pytest.mark.parametrize("name", NAMES)
def test_higher_fitness_is_better(name):
    """The sign convention lives in one place. Every optimizer must improve on
    sphere, whose optimum is at the origin with fitness 0."""
    start, _ = run(name, sphere, 1)
    end, _ = run(name, sphere, 40)
    assert end > start


@pytest.mark.parametrize("name", ["CMA-ES", "Sep-CMA-ES"])
def test_cma_converges_on_sphere(name):
    best, _ = run(name, sphere, 150)
    assert best > -5.0, f"{name} failed to converge on 80-D sphere: {best}"


def test_cma_beats_random_search_on_rastrigin():
    cma, _ = run("CMA-ES", rastrigin, 120)
    rnd, _ = run("Random Search", rastrigin, 120)
    assert cma > rnd


@pytest.mark.parametrize("name", NAMES)
def test_best_returns_the_best_seen(name):
    _, opt = run(name, sphere, 20)
    z, f = opt.best()
    assert z.shape == (DIM,)
    assert f == pytest.approx(sphere(z), rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_state_dict_roundtrip_reproduces_the_next_ask(name):
    """Explicit named arrays, not a pickle - so a checkpoint is not tied to the
    installed cmaes version. This test is what pins the allowlist."""
    _, opt = run(name, sphere, 12, seed=3)
    d = opt.state_dict()
    expected = opt.ask(POP)

    fresh = make_optimizer(name, DIM, POP, 0.5, 999)
    fresh.load_state_dict(d)
    assert np.allclose(fresh.ask(POP), expected, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_state_dict_has_no_pickled_objects(name):
    _, opt = run(name, sphere, 3)
    for k, v in opt.state_dict().items():
        assert isinstance(v, (int, float, str, bool, np.ndarray, list, tuple)), (
            f"{name}.state_dict()[{k!r}] is {type(v)}; must be npz-serializable"
        )


def test_x0_seeds_the_initial_mean():
    x0 = np.full(DIM, 0.4, dtype=np.float32)
    opt = make_optimizer("CMA-ES", DIM, POP, 0.05, 0, x0=x0)
    z = opt.ask(POP)
    assert np.allclose(z.mean(axis=0), x0, atol=0.1)


@pytest.mark.parametrize("name", NAMES)
def test_sigma_is_exposed(name):
    opt = make_optimizer(name, DIM, POP, 0.5, 0)
    assert opt.sigma > 0
```

- [ ] **Step 3: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_optimizers.py -v
```

Expected: FAIL, `No module named 'services.optimizers'`.

- [ ] **Step 4: Implement**

Create `services/optimizers.py`:

```python
"""Evolution strategies for the automatic tournament.

Pure numpy plus the `cmaes` library. No OpenGL, no CLIP, no app state.

Sign convention: `tell` takes fitness where HIGHER IS BETTER. Implementations
negate internally (cmaes minimizes). Keeping the convention in one place stops
it leaking into the loop.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from services.genome import crossover, mutate


class Optimizer(Protocol):
    name: str

    def ask(self, n: int) -> np.ndarray: ...
    def tell(self, z: np.ndarray, fitness: np.ndarray) -> None: ...
    def best(self) -> tuple[np.ndarray, float]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...
    @property
    def sigma(self) -> float: ...


class _BaseOptimizer:
    name = "base"

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None):
        self._dim = int(dim)
        self._popsize = int(popsize)
        self._sigma0 = float(sigma0)
        self._seed = int(seed)
        self._x0 = (
            np.zeros(self._dim, dtype=np.float64)
            if x0 is None
            else np.asarray(x0, dtype=np.float64).reshape(-1)
        )
        self._best_z = None
        self._best_f = -np.inf

    def _track_best(self, z, fitness):
        i = int(np.argmax(fitness))
        if float(fitness[i]) > self._best_f:
            self._best_f = float(fitness[i])
            self._best_z = np.asarray(z[i], dtype=np.float32).copy()

    def best(self):
        if self._best_z is None:
            return np.zeros(self._dim, dtype=np.float32), -np.inf
        return self._best_z.copy(), self._best_f

    def _base_state(self):
        return {
            "dim": self._dim,
            "popsize": self._popsize,
            "sigma0": self._sigma0,
            "seed": self._seed,
            "best_z": (
                self._best_z
                if self._best_z is not None
                else np.zeros(self._dim, dtype=np.float32)
            ),
            "best_f": self._best_f,
            "has_best": self._best_z is not None,
        }

    def _load_base(self, d):
        self._dim = int(d["dim"])
        self._popsize = int(d["popsize"])
        self._sigma0 = float(d["sigma0"])
        self._seed = int(d["seed"])
        self._best_f = float(d["best_f"])
        self._best_z = (
            np.asarray(d["best_z"], dtype=np.float32) if bool(d["has_best"]) else None
        )


# Attributes of cmaes.CMA / cmaes.SepCMA that constitute its search state.
# The roundtrip test in tests/test_optimizers.py fails loudly if this list stops
# matching the installed library, which is the point - we never pickle.
_CMA_ARRAYS = ("_mean", "_C", "_D", "_B", "_p_sigma", "_pc", "_p_c", "_sigma")
_CMA_SCALARS = ("_g", "_funhist_term", "_tolx", "_tolxup", "_tolfun", "_tolconditioncov")


class _CMAFamily(_BaseOptimizer):
    _cls = None

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None):
        super().__init__(dim, popsize, sigma0, seed, x0)
        self._cma = self._new()

    def _new(self):
        return self._cls(
            mean=self._x0.copy(),
            sigma=self._sigma0,
            population_size=self._popsize,
            seed=self._seed,
        )

    @property
    def sigma(self) -> float:
        return float(getattr(self._cma, "_sigma", self._sigma0))

    def ask(self, n: int) -> np.ndarray:
        return np.array(
            [self._cma.ask() for _ in range(n)], dtype=np.float32
        ).reshape(n, self._dim)

    def tell(self, z: np.ndarray, fitness: np.ndarray) -> None:
        self._track_best(z, fitness)
        # cmaes minimizes, so negate.
        solutions = [
            (np.asarray(zi, dtype=np.float64), -float(f)) for zi, f in zip(z, fitness)
        ]
        self._cma.tell(solutions)

    def state_dict(self) -> dict:
        d = self._base_state()
        present = []
        for k in _CMA_ARRAYS + _CMA_SCALARS:
            if hasattr(self._cma, k):
                v = getattr(self._cma, k)
                d["cma" + k] = np.asarray(v) if not np.isscalar(v) else float(v)
                present.append(k)
        d["cma_present"] = present
        rng = getattr(self._cma, "_rng", None)
        if rng is not None and hasattr(rng, "get_state"):
            st = rng.get_state()
            d["rng_kind"] = st[0]
            d["rng_keys"] = np.asarray(st[1])
            d["rng_pos"] = int(st[2])
            d["rng_has_gauss"] = int(st[3])
            d["rng_cached_gauss"] = float(st[4])
        return d

    def load_state_dict(self, d: dict) -> None:
        self._load_base(d)
        self._cma = self._new()
        for k in list(d["cma_present"]):
            k = str(k)
            setattr(self._cma, k, d["cma" + k])
        if "rng_keys" in d:
            rng = getattr(self._cma, "_rng", None)
            if rng is not None and hasattr(rng, "set_state"):
                rng.set_state((
                    str(d["rng_kind"]),
                    np.asarray(d["rng_keys"], dtype=np.uint32),
                    int(d["rng_pos"]),
                    int(d["rng_has_gauss"]),
                    float(d["rng_cached_gauss"]),
                ))


class CMAESOptimizer(_CMAFamily):
    name = "CMA-ES"

    def __init__(self, *a, **kw):
        from cmaes import CMA

        self._cls = CMA
        super().__init__(*a, **kw)


class SepCMAESOptimizer(_CMAFamily):
    name = "Sep-CMA-ES"

    def __init__(self, *a, **kw):
        from cmaes import SepCMA

        self._cls = SepCMA
        super().__init__(*a, **kw)


class RandomSearchOptimizer(_BaseOptimizer):
    """The control. If CMA-ES cannot beat this, the fitness signal is not real."""

    name = "Random Search"

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None):
        super().__init__(dim, popsize, sigma0, seed, x0)
        self._rng = np.random.default_rng(seed)

    @property
    def sigma(self) -> float:
        return self._sigma0

    def ask(self, n: int) -> np.ndarray:
        return (
            self._x0 + self._sigma0 * self._rng.normal(0, 1, (n, self._dim))
        ).astype(np.float32)

    def tell(self, z, fitness) -> None:
        self._track_best(z, fitness)

    def state_dict(self) -> dict:
        d = self._base_state()
        st = self._rng.bit_generator.state
        d["bg_state"] = int(st["state"]["state"])
        d["bg_inc"] = int(st["state"]["inc"])
        d["bg_has_uint32"] = int(st["has_uint32"])
        d["bg_uinteger"] = int(st["uinteger"])
        return d

    def load_state_dict(self, d: dict) -> None:
        self._load_base(d)
        self._rng = np.random.default_rng(self._seed)
        st = self._rng.bit_generator.state
        st["state"]["state"] = int(d["bg_state"])
        st["state"]["inc"] = int(d["bg_inc"])
        st["has_uint32"] = int(d["bg_has_uint32"])
        st["uinteger"] = int(d["bg_uinteger"])
        self._rng.bit_generator.state = st


class GAOptimizer(_BaseOptimizer):
    """Manual mode's operator, driven by CLIP instead of by a human.

    Reuses services.genome.mutate/crossover, applied in z-space by reshaping to
    the (10, 8) layout those operators expect.
    """

    name = "GA"
    ELITES = 4
    MUT = 0.25

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None):
        super().__init__(dim, popsize, sigma0, seed, x0)
        self._rng = np.random.default_rng(seed)
        self._pop = (
            self._x0 + self._sigma0 * self._rng.normal(0, 1, (popsize, dim))
        ).astype(np.float32)
        self._fit = None

    @property
    def sigma(self) -> float:
        return float(self._pop.std()) if self._fit is not None else self._sigma0

    def ask(self, n: int) -> np.ndarray:
        if len(self._pop) != n:
            self._pop = (
                self._x0 + self._sigma0 * self._rng.normal(0, 1, (n, self._dim))
            ).astype(np.float32)
        return self._pop.copy()

    def tell(self, z, fitness) -> None:
        z = np.asarray(z, dtype=np.float32)
        fitness = np.asarray(fitness, dtype=np.float32)
        self._track_best(z, fitness)
        self._fit = fitness
        order = np.argsort(-fitness)
        elites = z[order[: self.ELITES]]
        n = len(z)
        nxt = [e.copy() for e in elites]
        while len(nxt) < n:
            a = elites[self._rng.integers(len(elites))].reshape(-1, 8)
            b = elites[self._rng.integers(len(elites))].reshape(-1, 8)
            child = crossover(a, b, self._rng)
            child = mutate(child, self.MUT, self._rng)
            nxt.append(child.reshape(-1).astype(np.float32))
        self._pop = np.array(nxt[:n], dtype=np.float32)

    def state_dict(self) -> dict:
        d = self._base_state()
        d["pop"] = self._pop
        st = self._rng.bit_generator.state
        d["bg_state"] = int(st["state"]["state"])
        d["bg_inc"] = int(st["state"]["inc"])
        d["bg_has_uint32"] = int(st["has_uint32"])
        d["bg_uinteger"] = int(st["uinteger"])
        return d

    def load_state_dict(self, d: dict) -> None:
        self._load_base(d)
        self._pop = np.asarray(d["pop"], dtype=np.float32)
        self._rng = np.random.default_rng(self._seed)
        st = self._rng.bit_generator.state
        st["state"]["state"] = int(d["bg_state"])
        st["state"]["inc"] = int(d["bg_inc"])
        st["has_uint32"] = int(d["bg_has_uint32"])
        st["uinteger"] = int(d["bg_uinteger"])
        self._rng.bit_generator.state = st


ALGORITHMS: dict[str, type] = {
    "CMA-ES": CMAESOptimizer,
    "Sep-CMA-ES": SepCMAESOptimizer,
    "GA": GAOptimizer,
    "Random Search": RandomSearchOptimizer,
}


def make_optimizer(name, dim, popsize, sigma0=0.5, seed=0, x0=None) -> Optimizer:
    return ALGORITHMS[name](dim, popsize, sigma0, seed, x0)
```

- [ ] **Step 5: Run and iterate until green**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_optimizers.py -v
```

Expected: all pass. If `test_state_dict_roundtrip_reproduces_the_next_ask` fails for CMA-ES or Sep-CMA-ES, the `_CMA_ARRAYS` / `_CMA_SCALARS` allowlist is missing an attribute — compare against the names printed in Step 1 and add the missing ones. This is exactly the failure the test exists to catch.

- [ ] **Step 6: Commit**

```bash
git add services/optimizers.py tests/test_optimizers.py
git commit -m "feat: CMA-ES, Sep-CMA-ES, GA and Random Search behind one protocol

tell() takes fitness where higher is better; implementations negate internally
so the sign convention never leaks into the loop.

state_dict() returns explicit named arrays rather than a pickle, so checkpoints
are not tied to the installed cmaes version. A roundtrip test asserts a restored
optimizer produces the identical next ask(), which pins the attribute allowlist.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: genome_io — export/import as ordinary config files

**Files:**
- Create: `services/genome_io.py`
- Test: `tests/test_genome_io.py`

**Interfaces:**
- Consumes: `services.config_saver.ConfigSaver`, `PhysicsConfig`; `services.genome_spec.encode`, `decode`.
- Produces:
  - `META_KEY = "fluoddity_evolution"`
  - `export_genome(path, z, sim_state, meta: dict) -> None`
  - `import_genome(path) -> tuple[np.ndarray, int, dict]` — `(z (80,), n_clamped, meta)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_genome_io.py`:

```python
import json

import numpy as np
import pytest

from services.config_saver import ConfigSaver
from services.genome_io import META_KEY, export_genome, import_genome
from services.genome_spec import decode
from state import SimState


@pytest.fixture
def sim_state():
    return SimState()


def test_exported_file_loads_with_the_normal_config_loader(tmp_path, sim_state):
    """The whole point: an evolved genome must open in the normal single-sim
    view at any resolution, with no new load path."""
    z = np.random.default_rng(0).normal(0, 0.5, 80).astype(np.float32)
    p = tmp_path / "creature.json"
    export_genome(p, z, sim_state, {"generation": 42})

    cfg = ConfigSaver().load_from_file(p)
    assert cfg is not None
    assert np.allclose(cfg.rule, decode(z), atol=1e-5)


def test_export_records_provenance_metadata(tmp_path, sim_state):
    z = np.zeros(80, dtype=np.float32)
    p = tmp_path / "c.json"
    export_genome(p, z, sim_state, {
        "generation": 7,
        "evolved_at_canvas_px": 1024,
        "evolved_at_tile_px": 256,
        "evolved_with_tile_mutation": True,
        "mutation_strength": 0.1,
        "variants_per_tile": 4,
    })
    data = json.loads(p.read_text())
    assert data[META_KEY]["generation"] == 7
    assert data[META_KEY]["evolved_with_tile_mutation"] is True
    assert data[META_KEY]["variants_per_tile"] == 4


def test_metadata_does_not_break_the_normal_loader(tmp_path, sim_state):
    """from_dict uses .get(), so unknown top-level keys must be ignored."""
    export_genome(tmp_path / "c.json", np.zeros(80, dtype=np.float32),
                  sim_state, {"generation": 1})
    assert ConfigSaver().load_from_file(tmp_path / "c.json") is not None


def test_import_roundtrips_z(tmp_path, sim_state):
    z = np.random.default_rng(1).normal(0, 0.6, 80).astype(np.float32)
    p = tmp_path / "c.json"
    export_genome(p, z, sim_state, {"generation": 3})
    z2, clamped, meta = import_genome(p)
    assert clamped == 0
    assert np.allclose(z, z2, atol=1e-3)
    assert meta["generation"] == 3


def test_import_of_out_of_range_config_clamps_and_reports(tmp_path, sim_state):
    saver = ConfigSaver()
    rule = np.zeros((10, 8), dtype=np.float32)
    rule[0, 0] = 500.0
    cfg = saver.create_config(sim_state, rule)
    p = tmp_path / "wild.json"
    saver.save_to_file(cfg, p)

    z, clamped, meta = import_genome(p)
    assert clamped >= 1
    assert np.all(np.isfinite(z))
    assert meta == {}


def test_import_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_genome(tmp_path / "nope.json")
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_genome_io.py -v
```

Expected: FAIL, `No module named 'services.genome_io'`.

- [ ] **Step 3: Implement**

Create `services/genome_io.py`:

```python
"""Genome export/import using the existing Fluoddity config format.

Deliberately not a new file format. PhysicsConfig already carries the (10, 8)
rule alongside every physics slider, so an exported genome opens in the normal
single-simulation view at any resolution through the existing config browser,
and is copy-pasteable through the existing clipboard string.

Provenance is stored under a single extra top-level key. PhysicsConfig.from_dict
reads with .get(), so the extra key is ignored by every existing reader.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from services.config_saver import ConfigSaver, PhysicsConfig
from services.genome_spec import decode, encode

META_KEY = "fluoddity_evolution"


def export_genome(path, z: np.ndarray, sim_state, meta: dict | None = None) -> None:
    """Write z as an ordinary config file, plus a provenance block."""
    rule = decode(np.asarray(z, dtype=np.float32))
    config = ConfigSaver().create_config(sim_state, rule)
    data = json.loads(config.to_json())
    data[META_KEY] = dict(meta or {})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def import_genome(path) -> tuple[np.ndarray, int, dict]:
    """Read any config file as a search starting point.

    Returns (z, n_clamped, meta). Only x0 comes from the file - sigma, algorithm
    and grid are always taken from the current UI, which is what makes this
    'load the model, not the optimizer'.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text())
    meta = data.pop(META_KEY, {})
    config = PhysicsConfig.from_dict(data)
    z, clamped = encode(config.rule)
    return z, clamped, meta
```

- [ ] **Step 4: Run to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_genome_io.py -v
```

Expected: 6 passed. If `create_config` requires more positional arguments than `(sim_state, rule)`, check its signature at `services/config_saver.py:243` and pass what it needs.

- [ ] **Step 5: Commit**

```bash
git add services/genome_io.py tests/test_genome_io.py
git commit -m "feat: export evolved genomes as ordinary Fluoddity config files

PhysicsConfig already carries the (10,8) rule, so an evolved creature opens in
the normal single-simulation view at full resolution with no new load path.
Provenance lives under one extra top-level key that existing readers ignore.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: TileCapture and tile geometry

**Files:**
- Create: `services/tile_capture.py`
- Test: `tests/test_tile_geometry.py`, `tests/test_tile_capture.py` (gpu)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TILE_PX = 224`
  - `crop_bounds(tile: int, grid: int, tile_px: int = 224) -> tuple[int, int, int, int]` — `(row0, row1, col0, col1)` into a **top-down** image
  - `split_grid(img: np.ndarray, grid: int, tile_px: int = 224) -> np.ndarray` — top-down `(G*px, G*px, 3)` → `(G², px, px, 3)`
  - `class TileCapture` with `grid`, `size`, `capture(render_fn) -> np.ndarray`, `resize(grid)`, `release()`

- [ ] **Step 1: Write the failing geometry tests**

Create `tests/test_tile_geometry.py`:

```python
import numpy as np
import pytest

from services.cohort_tiling import tile_of
from services.tile_capture import TILE_PX, crop_bounds, split_grid


@pytest.mark.parametrize("grid", [2, 3, 4, 5, 6, 7, 8])
def test_crops_are_unique_and_non_overlapping(grid):
    seen = set()
    for tile in range(grid * grid):
        b = crop_bounds(tile, grid)
        assert b not in seen
        seen.add(b)
        r0, r1, c0, c1 = b
        assert r1 - r0 == TILE_PX
        assert c1 - c0 == TILE_PX
        assert 0 <= r0 < r1 <= grid * TILE_PX
        assert 0 <= c0 < c1 <= grid * TILE_PX


@pytest.mark.parametrize("grid", [2, 4, 8])
def test_tile_zero_is_bottom_left(grid):
    """tournament_home_tile() numbers tile 0 as bottom-left, but a top-down
    image has row 0 at the top. Getting this backwards has already caused one
    user-visible bug."""
    r0, _, c0, _ = crop_bounds(0, grid)
    assert c0 == 0, "tile 0 must be in the leftmost column"
    assert r0 == (grid - 1) * TILE_PX, "tile 0 must be in the bottom row"

    top_left_tile = (grid - 1) * grid
    r0, _, c0, _ = crop_bounds(top_left_tile, grid)
    assert (r0, c0) == (0, 0)


@pytest.mark.parametrize("grid", [2, 4])
def test_split_grid_returns_tiles_in_shader_order(grid):
    """Paint tile t with intensity t, laid out with ty counted from the bottom,
    then assert split_grid recovers them in index order."""
    size = grid * TILE_PX
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for tile in range(grid * grid):
        tx, ty = tile % grid, tile // grid
        row0 = (grid - 1 - ty) * TILE_PX
        img[row0:row0 + TILE_PX, tx * TILE_PX:(tx + 1) * TILE_PX] = tile + 1

    tiles = split_grid(img, grid)
    assert tiles.shape == (grid * grid, TILE_PX, TILE_PX, 3)
    for tile in range(grid * grid):
        assert tiles[tile].min() == tile + 1
        assert tiles[tile].max() == tile + 1


def test_tile_index_formula_agrees_with_the_shader_partition():
    """crop_bounds indexes by tile = ty*grid + tx, the same numbering that
    tile_of() derives from the particle index."""
    grid, active = 4, 100_000
    for i in range(0, active, 997):
        t = tile_of(i, active, grid)
        assert 0 <= t < grid * grid
        assert crop_bounds(t, grid) == crop_bounds(
            (t // grid) * grid + (t % grid), grid
        )
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_tile_geometry.py -v
```

Expected: FAIL, `No module named 'services.tile_capture'`.

- [ ] **Step 3: Implement**

Create `services/tile_capture.py`:

```python
"""Offscreen capture of the tournament grid as exact CLIP-sized crops.

The framebuffer is grid*224 square regardless of grid size, so CLIP always
receives exact 224x224 crops with no resampling, padding or cropping. This is
only correct because Auto mode forces a 1:1 canvas - tiles inherit the canvas
aspect ratio, and a 16:9 tile could not be squared without loss.
"""
from __future__ import annotations

import numpy as np

TILE_PX = 224


def crop_bounds(tile: int, grid: int, tile_px: int = TILE_PX):
    """(row0, row1, col0, col1) into a TOP-DOWN image.

    tournament_home_tile() numbers tile 0 as bottom-left, so ty is measured from
    the bottom and the row is inverted here. Keeping the inversion in exactly
    one place is what stops it being applied twice or cancelled out.
    """
    tx = tile % grid
    ty = tile // grid
    row0 = (grid - 1 - ty) * tile_px
    col0 = tx * tile_px
    return row0, row0 + tile_px, col0, col0 + tile_px


def split_grid(img: np.ndarray, grid: int, tile_px: int = TILE_PX) -> np.ndarray:
    """Top-down (G*px, G*px, 3) -> (G*G, px, px, 3) in shader tile order."""
    n = grid * grid
    out = np.empty((n, tile_px, tile_px, 3), dtype=np.uint8)
    for tile in range(n):
        r0, r1, c0, c1 = crop_bounds(tile, grid, tile_px)
        out[tile] = img[r0:r1, c0:c1]
    return out


class TileCapture:
    """Owns the offscreen framebuffer. The only GL-dependent unit."""

    def __init__(self, ctx, grid: int, tile_px: int = TILE_PX):
        self._ctx = ctx
        self._tile_px = tile_px
        self._tex = None
        self._fbo = None
        self.grid = 0
        self.resize(grid)

    @property
    def size(self) -> int:
        return self.grid * self._tile_px

    def resize(self, grid: int) -> None:
        if grid == self.grid and self._fbo is not None:
            return
        self.release()
        self.grid = grid
        size = grid * self._tile_px
        # dtype='f1' (GL_RGBA8), NOT 'f4'. frame_assembly.frag can emit values
        # above 1.0 after exposure; a float attachment would store those
        # unclamped, so the readback would contain values the display never
        # shows. An 8-bit unsigned-normalized attachment makes the GPU clamp
        # exactly as it does for the visible framebuffer.
        self._tex = self._ctx.texture((size, size), 4, dtype="f1")
        self._fbo = self._ctx.framebuffer(color_attachments=[self._tex])

    def capture(self, render_fn) -> np.ndarray:
        """render_fn(fbo) draws the assembled grid. Returns uint8
        (grid*grid, tile_px, tile_px, 3)."""
        size = self.size
        self._fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        render_fn(self._fbo)
        # alignment=1 is mandatory. The default GL_PACK_ALIGNMENT of 4 pads each
        # row to a 4-byte boundary; it happens to be a no-op at these widths and
        # would silently corrupt the image if tile_px or the channel count
        # changed.
        buf = self._fbo.read(components=3, alignment=1)
        expected = size * size * 3
        if len(buf) != expected:
            raise RuntimeError(f"readback was {len(buf)} bytes, expected {expected}")
        # fbo.read() returns rows bottom-up; flip once here to top-down.
        img = np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)[::-1]
        return split_grid(np.ascontiguousarray(img), self.grid, self._tile_px)

    def release(self) -> None:
        for obj in (self._fbo, self._tex):
            if obj is not None:
                obj.release()
        self._fbo = None
        self._tex = None
```

- [ ] **Step 4: Run the geometry tests**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_tile_geometry.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Write the GPU capture test**

Create `tests/test_tile_capture.py`:

```python
"""Pins the readback contract: uint8 clamping, pack alignment, and the flip.
Run with: pytest -m gpu"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

moderngl = pytest.importorskip("moderngl")

from services.tile_capture import TILE_PX, TileCapture, crop_bounds


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL context: {exc}")
    yield c
    c.release()


def _paint(ctx, fbo, grid, colors):
    """Fill each tile with its colour using scissored clears, with ty counted
    from the BOTTOM - i.e. in GL's own bottom-up viewport coordinates."""
    for tile, col in enumerate(colors):
        tx, ty = tile % grid, tile // grid
        ctx.scissor = (tx * TILE_PX, ty * TILE_PX, TILE_PX, TILE_PX)
        ctx.clear(col[0] / 255, col[1] / 255, col[2] / 255, 1.0)
    ctx.scissor = None


@pytest.mark.parametrize("grid", [2, 4, 8])
def test_capture_returns_tiles_in_shader_order(ctx, grid):
    n = grid * grid
    colors = [((i * 7) % 256, (i * 13) % 256, (i * 29) % 256) for i in range(n)]
    cap = TileCapture(ctx, grid)
    tiles = cap.capture(lambda fbo: _paint(ctx, fbo, grid, colors))
    assert tiles.shape == (n, TILE_PX, TILE_PX, 3)
    for i, col in enumerate(colors):
        got = tuple(int(v) for v in tiles[i, TILE_PX // 2, TILE_PX // 2])
        assert got == col, f"tile {i}: got {got}, expected {col}"
    cap.release()


def test_capture_clamps_values_above_one(ctx):
    """An f4 attachment would store 4.0; f1 must clamp to 255."""
    cap = TileCapture(ctx, 2)
    tiles = cap.capture(lambda fbo: ctx.clear(4.0, 0.0, 0.0, 1.0))
    assert tiles[..., 0].max() == 255
    cap.release()


def test_resize_reallocates(ctx):
    cap = TileCapture(ctx, 2)
    assert cap.size == 2 * TILE_PX
    cap.resize(6)
    assert cap.size == 6 * TILE_PX
    assert cap.capture(lambda fbo: None).shape[0] == 36
    cap.release()
```

- [ ] **Step 6: Run the GPU test**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_tile_capture.py -v -m gpu
```

Expected: 5 passed. A failure in `test_capture_returns_tiles_in_shader_order` means the flip in `capture()` and the inversion in `crop_bounds()` are cancelling or doubling — fix `capture()`, not `crop_bounds()`, since `crop_bounds` is pinned by the pure-python tests.

- [ ] **Step 7: Commit**

```bash
git add services/tile_capture.py tests/test_tile_geometry.py tests/test_tile_capture.py
git commit -m "feat: TileCapture - offscreen grid readback as exact CLIP crops

Three things that fail silently, pinned by tests:
- uint8 (GL_RGBA8) attachment, so the GPU clamps exposure overflow exactly as
  the display does rather than handing CLIP values the screen never showed
- explicit pack alignment, which is a no-op at these widths and would corrupt
  rows the moment tile_px changed
- the bottom-up readback flip, applied in exactly one place, since GL rows and
  tournament_home_tile count from opposite ends

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: RunLogger

**Files:**
- Create: `services/run_logger.py`
- Test: `tests/test_run_logger.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `new_run_id(now=None) -> str` — `"YYYYmmdd-HHMMSS"`
  - `class RunLogger(root="runs", run_id=None, config=None)` with `.dir`, `.run_id`, `.enabled`, `log_generation(rec: dict)`, `history() -> dict[str, list]`, `save_frame(img, gen)`, `close()`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_logger.py`:

```python
import json

import numpy as np
import pytest

from services.run_logger import RunLogger, new_run_id


def test_run_id_format():
    import datetime as dt

    rid = new_run_id(dt.datetime(2026, 8, 4, 14, 30, 22))
    assert rid == "20260804-143022"


def test_each_generation_is_one_json_line(tmp_path):
    log = RunLogger(root=tmp_path, config={"grid": 4})
    for g in range(3):
        log.log_generation({"gen": g, "fit_best": 0.1 * g})
    log.close()

    lines = (log.dir / "log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    recs = [json.loads(x) for x in lines]
    assert [r["gen"] for r in recs] == [0, 1, 2]


def test_config_is_written_at_run_start(tmp_path):
    log = RunLogger(root=tmp_path, config={"grid": 6, "prompt": "coral"})
    data = json.loads((log.dir / "config.json").read_text())
    assert data["grid"] == 6
    assert data["prompt"] == "coral"
    log.close()


def test_lines_are_flushed_so_a_crash_loses_at_most_one(tmp_path):
    log = RunLogger(root=tmp_path)
    log.log_generation({"gen": 0, "fit_best": 0.5})
    # deliberately not closed
    assert (log.dir / "log.jsonl").read_text().strip() != ""


def test_a_new_field_does_not_break_reading_earlier_lines(tmp_path):
    log = RunLogger(root=tmp_path)
    log.log_generation({"gen": 0, "fit_best": 0.1})
    log.log_generation({"gen": 1, "fit_best": 0.2, "novelty": 0.9})
    log.close()
    recs = [json.loads(x) for x in (log.dir / "log.jsonl").read_text().splitlines()]
    assert "novelty" not in recs[0]
    assert recs[1]["novelty"] == 0.9


def test_history_accumulates_for_the_sparkline(tmp_path):
    log = RunLogger(root=tmp_path)
    for g in range(4):
        log.log_generation({"gen": g, "fit_best": g / 10, "fit_mean": g / 20})
    h = log.history()
    assert h["fit_best"] == [0.0, 0.1, 0.2, 0.3]
    assert len(h["fit_mean"]) == 4
    log.close()


def test_unwritable_root_degrades_to_a_warning(tmp_path):
    """Evolution must not be blocked by a disk problem."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    log = RunLogger(root=blocker)
    assert log.enabled is False
    log.log_generation({"gen": 0})   # must not raise
    assert log.history()["fit_best"] == []
    log.close()


def test_save_frame_writes_a_png(tmp_path):
    log = RunLogger(root=tmp_path)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    log.save_frame(img, 10)
    assert (log.dir / "frames" / "gen_000010.png").is_file()
    log.close()
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_run_logger.py -v
```

Expected: FAIL, `No module named 'services.run_logger'`.

- [ ] **Step 3: Implement**

Create `services/run_logger.py`:

```python
"""Per-generation metrics for an automatic tournament run.

JSONL rather than CSV because the schema will grow (physics blocks, novelty
terms), and appending a field must not break older files or readers.

All values are FITNESS, higher is better. Never 'loss'.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

_HISTORY_KEYS = ("gen", "fit_best", "fit_mean", "sigma")


def new_run_id(now: _dt.datetime | None = None) -> str:
    return (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")


class RunLogger:
    def __init__(self, root="runs", run_id: str | None = None, config: dict | None = None):
        self.run_id = run_id or new_run_id()
        self.dir = Path(root) / self.run_id
        self.enabled = True
        self._fh = None
        self._history: dict[str, list] = {k: [] for k in _HISTORY_KEYS}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "config.json").write_text(json.dumps(config or {}, indent=2))
            self._fh = open(self.dir / "log.jsonl", "a", encoding="utf-8")
        except OSError as exc:
            self.enabled = False
            print(f"[RunLogger] logging disabled ({exc}); the run continues")

    def log_generation(self, rec: dict) -> None:
        if not self.enabled:
            return
        for k in _HISTORY_KEYS:
            if k in rec:
                self._history[k].append(rec[k])
        try:
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()  # a crash must lose at most one generation
        except OSError as exc:
            self.enabled = False
            print(f"[RunLogger] logging disabled mid-run ({exc})")

    def history(self) -> dict[str, list]:
        return {k: list(v) for k, v in self._history.items()}

    def save_frame(self, img: np.ndarray, gen: int) -> None:
        """Periodic best-tile PNG. A run folder of these assembles directly
        into a timelapse of the evolution."""
        if not self.enabled:
            return
        try:
            from PIL import Image

            d = self.dir / "frames"
            d.mkdir(exist_ok=True)
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(d / f"gen_{gen:06d}.png")
        except (OSError, ValueError) as exc:
            print(f"[RunLogger] frame save failed ({exc})")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
```

- [ ] **Step 4: Run to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_run_logger.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add services/run_logger.py tests/test_run_logger.py
git commit -m "feat: RunLogger - per-generation JSONL metrics

Flushed every generation so a crash loses at most one line. JSONL over CSV
because the schema will grow and a new field must not break older readers.
An unwritable runs/ directory degrades to a warning rather than blocking the
evolution.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: Checkpoints

**Files:**
- Create: `services/run_checkpoint.py`
- Test: `tests/test_run_checkpoint.py`

**Interfaces:**
- Consumes: `services.optimizers` (via caller-supplied state dicts).
- Produces:
  - `FORMAT_VERSION = 1`
  - `class CheckpointError(Exception)`
  - `save_checkpoint(path, state: dict) -> None`
  - `load_checkpoint(path, expect_signature: str | None = None) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_checkpoint.py`:

```python
import numpy as np
import pytest

from services.run_checkpoint import (
    FORMAT_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)


def _state():
    return {
        "genome_spec_signature": "brain:80",
        "generation": 12,
        "optimizer_name": "CMA-ES",
        "optimizer_state": {"dim": 80, "popsize": 16, "_mean": np.zeros(80)},
        "base_seed": 7,
        "prompt": "glowing coral",
        "distractors": ["a blank image", "random noise"],
        "settings": {"grid": 4, "steps_per_gen": 300, "sigma0": 0.5},
        "history": {"fit_best": [0.1, 0.3], "fit_mean": [0.05, 0.2]},
        "best_z": np.full(80, 0.25, dtype=np.float32),
        "best_fitness": 0.31,
    }


def test_roundtrip_restores_everything(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    got = load_checkpoint(p)
    assert got["generation"] == 12
    assert got["prompt"] == "glowing coral"
    assert got["optimizer_name"] == "CMA-ES"
    assert got["settings"]["grid"] == 4
    assert got["history"]["fit_best"] == [0.1, 0.3]
    assert got["distractors"] == ["a blank image", "random noise"]
    assert np.allclose(got["best_z"], 0.25)
    assert np.allclose(got["optimizer_state"]["_mean"], 0.0)
    assert got["optimizer_state"]["popsize"] == 16


def test_format_version_mismatch_is_specific(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    import json

    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    meta["format_version"] = FORMAT_VERSION + 99
    np.savez(p, meta=np.array(json.dumps(meta)), **arrays)

    with pytest.raises(CheckpointError, match="format_version"):
        load_checkpoint(p)


def test_signature_mismatch_is_specific(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    with pytest.raises(CheckpointError, match="brain:80"):
        load_checkpoint(p, expect_signature="brain:80,physics:12")


def test_matching_signature_passes(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    assert load_checkpoint(p, expect_signature="brain:80")["generation"] == 12


def test_write_is_atomic(tmp_path):
    """A crash mid-write must never destroy a good checkpoint."""
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    good = p.read_bytes()

    bad = _state()
    bad["optimizer_state"] = {"boom": object()}   # not serializable
    with pytest.raises(Exception):
        save_checkpoint(p, bad)

    assert p.read_bytes() == good
    assert not (tmp_path / "ck.npz.tmp").exists()


def test_no_pickled_objects_are_written(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    with np.load(p, allow_pickle=False) as z:   # raises if anything was pickled
        assert "meta" in z.files
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_run_checkpoint.py -v
```

Expected: FAIL, `No module named 'services.run_checkpoint'`.

- [ ] **Step 3: Implement**

Create `services/run_checkpoint.py`:

```python
"""Full run state save/load.

A checkpoint answers 'resume this exact search'. It carries optimizer state,
RNG, history and settings, and is inseparable from its grid size, because
cmaes.CMA fixes its population at construction.

For 'keep this creature', see services/genome_io.py instead.

Format is a single .npz: arrays at the top level, everything else as one JSON
string under 'meta'. Explicitly never a pickle - allow_pickle stays False on
load, so a checkpoint cannot execute code and is not tied to a library version.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1

_ARRAY_PREFIX = "arr__"
_OPT_PREFIX = "opt__"


class CheckpointError(Exception):
    pass


def _split(state: dict):
    """Separate numpy arrays (npz top level) from JSON-able scalars (meta)."""
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {"format_version": FORMAT_VERSION}

    for key in ("genome_spec_signature", "generation", "optimizer_name",
                "base_seed", "prompt", "distractors", "settings", "history",
                "best_fitness"):
        if key in state:
            meta[key] = state[key]

    if "best_z" in state:
        arrays[_ARRAY_PREFIX + "best_z"] = np.asarray(state["best_z"], dtype=np.float32)

    opt_meta = {}
    for k, v in (state.get("optimizer_state") or {}).items():
        if isinstance(v, np.ndarray):
            arrays[_OPT_PREFIX + k] = v
        elif isinstance(v, (int, float, str, bool, list, tuple)):
            opt_meta[k] = list(v) if isinstance(v, tuple) else v
        else:
            raise CheckpointError(
                f"optimizer_state[{k!r}] is {type(v).__name__}; "
                "checkpoints store explicit arrays and scalars, never pickles"
            )
    meta["optimizer_meta"] = opt_meta
    return arrays, meta


def save_checkpoint(path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = _split(state)          # raises before any file is touched
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(path, expect_signature: str | None = None) -> dict:
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"no checkpoint at {p}")

    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}

    got = int(meta.get("format_version", -1))
    if got != FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint format_version is {got}, this build expects {FORMAT_VERSION}"
        )

    sig = meta.get("genome_spec_signature")
    if expect_signature is not None and sig != expect_signature:
        raise CheckpointError(
            f"checkpoint genome is {sig!r}, this build expects {expect_signature!r}"
        )

    opt_state = dict(meta.get("optimizer_meta", {}))
    for k, v in arrays.items():
        if k.startswith(_OPT_PREFIX):
            opt_state[k[len(_OPT_PREFIX):]] = v

    out = {k: v for k, v in meta.items() if k not in ("optimizer_meta",)}
    out["optimizer_state"] = opt_state
    if _ARRAY_PREFIX + "best_z" in arrays:
        out["best_z"] = arrays[_ARRAY_PREFIX + "best_z"]
    return out
```

- [ ] **Step 4: Run to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_run_checkpoint.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add services/run_checkpoint.py tests/test_run_checkpoint.py
git commit -m "feat: atomic .npz checkpoints with explicit optimizer state

Never a pickle: allow_pickle stays False on load, so a checkpoint cannot
execute code and is not tied to the installed cmaes version. Writes go to .tmp
then os.replace, so a crash mid-write cannot destroy a good checkpoint.
format_version and genome_spec_signature mismatches fail loudly by name.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: AutoTournamentService — the generation state machine

**Files:**
- Create: `services/auto_tournament_service.py`
- Test: `tests/test_auto_tournament_service.py`

**Interfaces:**
- Consumes: `services.optimizers.make_optimizer`, `services.genome_spec.BRAIN_SPEC/decode`, `services.run_logger.RunLogger`, `services.cohort_tiling.cohorts_for/max_variants`.
- Produces:
  - `class Action(str, Enum)`: `NONE`, `WRITE_RULES`, `STEP`, `CAPTURE`, `SCORE`
  - `class Phase(str, Enum)`: `IDLE = "idle"`, `ROLLOUT = "rollout"`, `SCORE = "score"`, `PAUSED = "paused"`
  - `snapshot_steps(steps_per_gen: int, n: int) -> list[int]`
  - `class AutoTournamentService(tournament_service, scorer=None, logger=None, spec=BRAIN_SPEC, base_seed=1000)`
    - settings: `steps_per_gen`, `snapshots_per_gen`, `sim_steps_per_frame`, `sigma0`, `algorithm`, `autosave_every`, `tile_mutation_enabled`, `variants_per_tile`, `tile_mutation_strength`
    - read-only: `generation`, `step_in_gen`, `fitness`, `phase`, `optimizer`, `prompt`, `popsize`, `gen_seed`, `sigma`, `current_z`
    - methods: `configure(**kw)`, `set_prompt(text)`, `set_x0(z)`, `start(prompt=None)`, `pause()`, `reset()`, `abort_generation()`, `update() -> Action`, `submit_frames(crops)`, `score_and_tell() -> np.ndarray`, `checkpoint_state() -> dict`, `restore(state)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auto_tournament_service.py`:

```python
import numpy as np
import pytest

from services.auto_tournament_service import (
    Action,
    AutoTournamentService,
    Phase,
    snapshot_steps,
)
from services.tournament_service import TournamentService


class FakeScorer:
    """Scores each tile by its mean brightness, so tests control fitness."""

    def __init__(self):
        self.prompt = ""
        self.calls = 0

    def set_prompt(self, text, distractors=None):
        self.prompt = text

    def score(self, images):
        self.calls += 1
        return images.reshape(len(images), -1).mean(axis=1).astype(np.float32) / 255.0


def make(grid=2, **kw):
    ts = TournamentService(grid=grid)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=FakeScorer(), logger=None)
    svc.configure(steps_per_gen=100, snapshots_per_gen=2, sim_steps_per_frame=10, **kw)
    return svc, ts


def crops_for(svc, value=128):
    n = svc.tournament.tiles
    return np.full((n, 224, 224, 3), value, dtype=np.uint8)


def run_one_generation(svc, value=128):
    """Drive update() until one full generation completes."""
    seen = []
    for _ in range(500):
        a = svc.update()
        seen.append(a)
        if a is Action.CAPTURE:
            svc.submit_frames(crops_for(svc, value))
        elif a is Action.SCORE:
            svc.score_and_tell()
            return seen
    raise AssertionError("generation never completed")


def test_snapshot_steps_are_evenly_spaced_and_end_on_the_last_step():
    assert snapshot_steps(300, 4) == [75, 150, 225, 300]
    assert snapshot_steps(300, 1) == [300]
    assert snapshot_steps(100, 3) == [33, 66, 100]


def test_starts_idle_and_does_nothing():
    svc, _ = make()
    assert svc.phase is Phase.IDLE
    assert svc.update() is Action.NONE


def test_first_action_after_start_is_write_rules():
    svc, _ = make()
    svc.start("coral")
    assert svc.update() is Action.WRITE_RULES
    assert svc.phase is Phase.ROLLOUT


def test_full_generation_sequence():
    svc, _ = make()
    svc.start("coral")
    seen = run_one_generation(svc)
    assert seen[0] is Action.WRITE_RULES
    assert seen.count(Action.CAPTURE) == 2
    assert seen[-1] is Action.SCORE
    assert svc.generation == 1
    assert svc.fitness is not None and len(svc.fitness) == 4


def test_captures_land_on_the_snapshot_steps():
    svc, _ = make()
    svc.configure(steps_per_gen=100, snapshots_per_gen=4, sim_steps_per_frame=5)
    svc.start("coral")
    at = []
    for _ in range(500):
        a = svc.update()
        if a is Action.CAPTURE:
            at.append(svc.step_in_gen)
            svc.submit_frames(crops_for(svc))
        elif a is Action.SCORE:
            svc.score_and_tell()
            break
    assert at == snapshot_steps(100, 4)


@pytest.mark.parametrize("steps,snaps,per_frame", [(300, 4, 10), (50, 1, 1), (137, 3, 7)])
def test_snapshot_timing_holds_for_odd_configurations(steps, snaps, per_frame):
    svc, _ = make()
    svc.configure(steps_per_gen=steps, snapshots_per_gen=snaps, sim_steps_per_frame=per_frame)
    svc.start("coral")
    n = 0
    for _ in range(5000):
        a = svc.update()
        if a is Action.CAPTURE:
            n += 1
            svc.submit_frames(crops_for(svc))
        elif a is Action.SCORE:
            svc.score_and_tell()
            break
    assert n == snaps


def test_seed_is_shared_within_a_generation_and_changes_between():
    svc, _ = make()
    svc.start("coral")
    s0 = svc.gen_seed
    run_one_generation(svc)
    run_one_generation(svc)
    assert svc.gen_seed != s0


def test_elite_injection_ranks_selected_tiles_first():
    svc, ts = make(grid=2)
    svc.start("coral")
    for _ in range(500):
        a = svc.update()
        if a is Action.CAPTURE:
            imgs = crops_for(svc, 10)
            imgs[2] = 250        # tile 2 is genuinely brightest
            svc.submit_frames(imgs)
        elif a is Action.SCORE:
            ts.toggle_select(0)  # but the human picks tile 0
            svc.score_and_tell()
            break
    assert svc.fitness[0] == svc.fitness.max()
    assert ts.selected == set(), "selection is cleared after tell"


def test_nan_fitness_is_replaced_with_the_generation_minimum():
    svc, _ = make()
    svc.scorer.score = lambda imgs: np.array([np.nan, 0.2, 0.5, 0.1], dtype=np.float32)
    svc.start("coral")
    run_one_generation(svc)
    assert np.all(np.isfinite(svc.fitness))
    assert svc.fitness[0] == pytest.approx(0.1)


def test_abort_restarts_the_generation():
    svc, _ = make()
    svc.start("coral")
    svc.update()
    for _ in range(3):
        svc.update()
    g = svc.generation
    svc.abort_generation()
    assert svc.step_in_gen == 0
    assert svc.update() is Action.WRITE_RULES
    assert svc.generation == g


def test_pause_and_resume_preserve_optimizer_state():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    before = svc.optimizer.state_dict()["popsize"]
    svc.pause()
    assert svc.update() is Action.NONE
    svc.start("coral")
    assert svc.optimizer.state_dict()["popsize"] == before
    assert svc.generation == 1


def test_changing_the_prompt_does_not_restart_the_search():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    svc.set_prompt("a city map")
    assert svc.generation == 1
    assert svc.scorer.prompt == "a city map"


def test_reset_clears_the_optimizer():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    svc.reset()
    assert svc.generation == 0
    assert svc.phase is Phase.IDLE


def test_population_is_written_into_the_tournament_service():
    svc, ts = make(grid=2)
    svc.start("coral")
    svc.update()
    assert len(ts.population) == 4
    assert all(g.shape == (10, 8) for g in ts.population)
    assert ts.is_dirty()


def test_checkpoint_state_roundtrip():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    st = svc.checkpoint_state()
    assert st["generation"] == 1
    assert st["prompt"] == "coral"
    assert st["settings"]["grid"] == 2

    svc2, _ = make()
    svc2.restore(st)
    assert svc2.generation == 1
    assert svc2.prompt == "coral"
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_auto_tournament_service.py -v
```

Expected: FAIL, `No module named 'services.auto_tournament_service'`.

- [ ] **Step 3: Implement**

Create `services/auto_tournament_service.py`:

```python
"""The automatic tournament generation state machine.

Holds an Optimizer, a CLIPScorer and a RunLogger, all injected. Contains no GL
calls and no ImGui, so the whole loop is testable with fakes.

The rollout is spread across real application frames - update() returns one
action per call and never loops internally. That is the mechanism by which the
app stays responsive; it is not an optimization to be added later.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from services.genome_spec import BRAIN_SPEC, decode
from services.optimizers import make_optimizer


class Action(str, Enum):
    NONE = "none"
    WRITE_RULES = "write_rules"
    STEP = "step"
    CAPTURE = "capture"
    SCORE = "score"


class Phase(str, Enum):
    IDLE = "idle"
    ROLLOUT = "rollout"
    SCORE = "score"
    PAUSED = "paused"


def snapshot_steps(steps_per_gen: int, n: int) -> list[int]:
    """Evenly spaced step counts, always ending on the last step."""
    n = max(1, min(int(n), int(steps_per_gen)))
    return [int(round(steps_per_gen * (i + 1) / n)) for i in range(n)]


class AutoTournamentService:
    def __init__(self, tournament_service, scorer=None, logger=None,
                 spec=BRAIN_SPEC, base_seed: int = 1000):
        self.tournament = tournament_service
        self.scorer = scorer
        self.logger = logger
        self.spec = spec
        self.base_seed = int(base_seed)

        self.steps_per_gen = 300
        self.snapshots_per_gen = 4
        self.sim_steps_per_frame = 10
        self.sigma0 = 0.5
        self.algorithm = "CMA-ES"
        self.tile_mutation_enabled = False
        self.variants_per_tile = 4
        self.tile_mutation_strength = 0.1
        self.autosave_every = 10

        self.phase = Phase.IDLE
        self.generation = 0
        self.step_in_gen = 0
        self.fitness: np.ndarray | None = None
        self.prompt = ""
        self.optimizer = None

        self._z = None
        self._snaps: list[int] = []
        self._next_snap = 0
        self._buffer: list[np.ndarray] = []
        self._needs_write = False
        self._elapsed_gens_history: list[float] = []

    # ---- configuration -------------------------------------------------

    def configure(self, **kw) -> None:
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(f"unknown setting {k!r}")
            setattr(self, k, v)

    @property
    def popsize(self) -> int:
        return self.tournament.tiles

    @property
    def gen_seed(self) -> int:
        return self.base_seed + self.generation

    @property
    def current_z(self):
        """This generation's search vectors, or None before the first ask."""
        return self._z

    @property
    def sigma(self) -> float:
        return float(self.optimizer.sigma) if self.optimizer else self.sigma0

    def _ensure_optimizer(self, x0=None) -> None:
        if self.optimizer is None:
            self.optimizer = make_optimizer(
                self.algorithm, self.spec.dim, self.popsize,
                self.sigma0, self.base_seed, x0,
            )

    def set_prompt(self, text: str) -> None:
        """Changing the prompt keeps the optimizer's learned covariance and
        simply starts climbing a new landscape."""
        self.prompt = text
        if self.scorer is not None:
            self.scorer.set_prompt(text)

    def set_x0(self, z: np.ndarray) -> None:
        """Load a genome as the search starting point. Discards optimizer
        state; sigma, algorithm and grid come from the UI, not the file."""
        self.optimizer = None
        self._ensure_optimizer(np.asarray(z, dtype=np.float64))

    # ---- lifecycle -----------------------------------------------------

    def start(self, prompt: str | None = None) -> None:
        if prompt is not None:
            self.set_prompt(prompt)
        self._ensure_optimizer()
        if self.phase in (Phase.IDLE, Phase.PAUSED):
            if self.phase is Phase.IDLE:
                self._begin_generation()
            self.phase = Phase.ROLLOUT

    def pause(self) -> None:
        if self.phase is not Phase.IDLE:
            self.phase = Phase.PAUSED

    def reset(self) -> None:
        self.optimizer = None
        self.generation = 0
        self.step_in_gen = 0
        self.fitness = None
        self._z = None
        self._buffer.clear()
        self.phase = Phase.IDLE

    def abort_generation(self) -> None:
        """A partial rollout is not a valid fitness sample. Used on resize,
        shader reload, and grid change."""
        self.step_in_gen = 0
        self._buffer.clear()
        self._next_snap = 0
        self._needs_write = True

    def _begin_generation(self) -> None:
        self._ensure_optimizer()
        self._z = self.optimizer.ask(self.popsize)
        self.tournament.population = [decode(z) for z in self._z]
        self.tournament.mark_dirty()
        self._snaps = snapshot_steps(self.steps_per_gen, self.snapshots_per_gen)
        self._next_snap = 0
        self.step_in_gen = 0
        self._buffer.clear()
        self._needs_write = True

    # ---- per-frame driver ----------------------------------------------

    def update(self) -> Action:
        if self.phase in (Phase.IDLE, Phase.PAUSED):
            return Action.NONE

        if self._needs_write:
            self._needs_write = False
            return Action.WRITE_RULES

        if self._next_snap < len(self._snaps) and self.step_in_gen >= self._snaps[self._next_snap]:
            self._next_snap += 1
            return Action.CAPTURE

        if self._next_snap >= len(self._snaps):
            self.phase = Phase.SCORE
            return Action.SCORE

        self.step_in_gen += self.sim_steps_per_frame
        return Action.STEP

    def submit_frames(self, crops: np.ndarray) -> None:
        self._buffer.append(np.asarray(crops, dtype=np.uint8))

    # ---- scoring -------------------------------------------------------

    def score_and_tell(self) -> np.ndarray:
        n = self.popsize
        if self.scorer is None or not self._buffer:
            fit = np.zeros(n, dtype=np.float32)
        else:
            per_snap = [np.asarray(self.scorer.score(c), dtype=np.float32)
                        for c in self._buffer]
            fit = np.mean(np.stack(per_snap, axis=0), axis=0).astype(np.float32)

        bad = ~np.isfinite(fit)
        if bad.any():
            finite = fit[~bad]
            fit[bad] = float(finite.min()) if finite.size else 0.0

        # Elite injection: selected tiles are already in this population, so
        # this overwrites their fitness to force them to the top ranks.
        selected = sorted(self.tournament.selected)
        if selected:
            top = float(fit.max())
            for rank, tile in enumerate(selected):
                if 0 <= tile < n:
                    fit[tile] = top + 1e-3 * (len(selected) - rank)

        self.optimizer.tell(self._z, fit)
        self.fitness = fit
        self.generation += 1
        self.tournament.selected.clear()

        if self.logger is not None:
            self.logger.log_generation({
                "gen": self.generation,
                "prompt": self.prompt,
                "algorithm": self.algorithm,
                "sigma": self.sigma,
                "popsize": n,
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
                "elites_injected": len(selected),
                "tile_mutation": bool(self.tile_mutation_enabled),
                "nan_replaced": int(bad.sum()),
            })

        self._begin_generation()
        self.phase = Phase.ROLLOUT
        return fit

    # ---- checkpointing -------------------------------------------------

    def checkpoint_state(self) -> dict:
        best_z, best_f = (self.optimizer.best() if self.optimizer
                          else (np.zeros(self.spec.dim, np.float32), -np.inf))
        return {
            "genome_spec_signature": self.spec.signature(),
            "generation": self.generation,
            "optimizer_name": self.algorithm,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else {},
            "base_seed": self.base_seed,
            "prompt": self.prompt,
            "distractors": [],
            "settings": {
                "grid": self.tournament.grid,
                "steps_per_gen": self.steps_per_gen,
                "snapshots_per_gen": self.snapshots_per_gen,
                "sim_steps_per_frame": self.sim_steps_per_frame,
                "sigma0": self.sigma0,
                "autosave_every": self.autosave_every,
                "tile_mutation_enabled": self.tile_mutation_enabled,
                "variants_per_tile": self.variants_per_tile,
                "tile_mutation_strength": self.tile_mutation_strength,
            },
            "history": self.logger.history() if self.logger else {},
            "best_z": best_z,
            "best_fitness": float(best_f) if np.isfinite(best_f) else 0.0,
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
            variants_per_tile=int(s.get("variants_per_tile", 4)),
            tile_mutation_strength=float(s.get("tile_mutation_strength", 0.1)),
            algorithm=str(state["optimizer_name"]),
        )
        self.base_seed = int(state["base_seed"])
        self.generation = int(state["generation"])
        self.optimizer = None
        self._ensure_optimizer()
        if state.get("optimizer_state"):
            self.optimizer.load_state_dict(state["optimizer_state"])
        self.set_prompt(str(state["prompt"]))
        self.phase = Phase.PAUSED
```

- [ ] **Step 4: Run and iterate until green**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_auto_tournament_service.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add services/auto_tournament_service.py tests/test_auto_tournament_service.py
git commit -m "feat: AutoTournamentService generation state machine

Returns one action per frame and never loops internally, so the rollout is
spread across real frames and the app stays responsive by construction.

Tiles share a seed within a generation so genomes are compared fairly, and the
seed changes each generation so the population does not overfit one lucky
scatter. Elite injection overwrites selected tiles' fitness before tell, since
those tiles are already members of the population.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 12: State and command wiring

**Files:**
- Create: `state/auto_tournament_state.py`
- Modify: `state/__init__.py`, `state/ui_state.py`, `command_handler.py`
- Test: `tests/test_auto_tournament_state.py`

**Interfaces:**
- Consumes: `services.auto_tournament_service.AutoTournamentService`.
- Produces:
  - `AutoTournamentState` dataclass
  - `UIState.auto_tournament: AutoTournamentState`
  - `CommandHandler._handle_auto_tournament(ui_state)`, `CommandHandler._clear_auto_flags(ats)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auto_tournament_state.py`:

```python
from dataclasses import fields

from state import UIState
from state.auto_tournament_state import AutoTournamentState


def test_defaults_match_the_spec():
    s = AutoTournamentState()
    assert s.enabled is False
    assert s.running is False
    assert s.prompt == ""
    assert s.algorithm == "CMA-ES"
    assert s.grid == 4
    assert s.steps_per_gen == 300
    assert s.sim_steps_per_frame == 10
    assert s.snapshots_per_gen == 4
    assert s.sigma0 == 0.5
    assert s.autosave_every == 10
    assert s.tile_mutation_enabled is False
    assert s.variants_per_tile == 4
    assert s.tile_mutation_strength == 0.1


def test_all_one_shot_flags_default_false_or_empty():
    s = AutoTournamentState()
    assert s.start_requested is False
    assert s.pause_requested is False
    assert s.reset_requested is False
    assert s.prompt_changed is False
    assert s.grid_changed is False
    assert s.save_checkpoint_requested is False
    assert s.save_best_requested is False
    assert s.save_tile_requested == -1
    assert s.load_checkpoint_path == ""
    assert s.load_genome_path == ""
    assert s.download_model_requested is False


def test_ui_state_exposes_auto_tournament():
    assert any(f.name == "auto_tournament" for f in fields(UIState))
    assert isinstance(UIState().auto_tournament, AutoTournamentState)


def test_state_is_not_shared_between_instances():
    a, b = UIState(), UIState()
    a.auto_tournament.prompt = "coral"
    assert b.auto_tournament.prompt == ""
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_auto_tournament_state.py -v
```

Expected: FAIL, `No module named 'state.auto_tournament_state'`.

- [ ] **Step 3: Implement the state**

Create `state/auto_tournament_state.py`:

```python
"""UI state for the CLIP-guided automatic tournament.

One-shot request flags are set by the UI and cleared by the consuming side in
CommandHandler - never inside UI.get_state(), which returns the live object.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoTournamentState:
    # persistent settings
    enabled: bool = False
    running: bool = False
    prompt: str = ""
    algorithm: str = "CMA-ES"
    grid: int = 4
    steps_per_gen: int = 300
    sim_steps_per_frame: int = 10
    snapshots_per_gen: int = 4
    sigma0: float = 0.5
    autosave_every: int = 10
    tile_mutation_enabled: bool = False
    variants_per_tile: int = 4
    tile_mutation_strength: float = 0.1

    # one-shot request flags, cleared by CommandHandler
    start_requested: bool = False
    pause_requested: bool = False
    reset_requested: bool = False
    prompt_changed: bool = False
    grid_changed: bool = False
    save_checkpoint_requested: bool = False
    save_best_requested: bool = False
    save_tile_requested: int = -1
    load_checkpoint_path: str = ""
    load_genome_path: str = ""
    download_model_requested: bool = False
```

In `state/ui_state.py`, add the import and field:

```python
from .auto_tournament_state import AutoTournamentState
```

```python
    auto_tournament: AutoTournamentState = field(default_factory=AutoTournamentState)
```

In `state/__init__.py`, add `AutoTournamentState` to the imports and `__all__`.

- [ ] **Step 4: Run to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_auto_tournament_state.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Wire the CommandHandler**

In `command_handler.py`, extend `__init__` to accept and store `auto_service=None`:

```python
        self.auto_service = auto_service
```

Add, immediately after the `self._handle_tournament(ui_state)` call at line 151:

```python
        self._handle_auto_tournament(ui_state)
```

Add these methods to the class:

```python
    @staticmethod
    def _clear_auto_flags(ats):
        """Cleared here, not in UI.get_state(): get_state returns the live
        object, so clearing there would wipe flags before we read them."""
        ats.start_requested = False
        ats.pause_requested = False
        ats.reset_requested = False
        ats.prompt_changed = False
        ats.grid_changed = False
        ats.save_checkpoint_requested = False
        ats.save_best_requested = False
        ats.save_tile_requested = -1
        ats.load_checkpoint_path = ""
        ats.load_genome_path = ""
        ats.download_model_requested = False

    def _handle_auto_tournament(self, ui_state):
        svc = self.auto_service
        ats = ui_state.auto_tournament
        if svc is None:
            self._clear_auto_flags(ats)
            return

        if ats.grid_changed and self.tournament_service is not None:
            # Applied at a generation boundary; resets the optimizer because
            # cmaes.CMA fixes popsize at construction.
            self.tournament_service.set_grid(ats.grid)
            svc.reset()

        svc.configure(
            steps_per_gen=ats.steps_per_gen,
            snapshots_per_gen=ats.snapshots_per_gen,
            sim_steps_per_frame=ats.sim_steps_per_frame,
            sigma0=ats.sigma0,
            algorithm=ats.algorithm,
            autosave_every=ats.autosave_every,
            tile_mutation_enabled=ats.tile_mutation_enabled,
            variants_per_tile=ats.variants_per_tile,
            tile_mutation_strength=ats.tile_mutation_strength,
        )

        if ats.prompt_changed:
            svc.set_prompt(ats.prompt)
        if ats.reset_requested:
            svc.reset()
        if ats.pause_requested:
            svc.pause()
        if ats.start_requested and ats.prompt.strip():
            svc.start(ats.prompt)
        if ats.load_genome_path:
            self._load_auto_genome(svc, ats.load_genome_path)
        if ats.save_best_requested:
            self._save_auto_genome(svc, ui_state, tile=None)
        if ats.save_tile_requested >= 0:
            self._save_auto_genome(svc, ui_state, tile=ats.save_tile_requested)

        ats.running = svc.phase.value == "rollout"
        self._clear_auto_flags(ats)

    def _load_auto_genome(self, svc, path):
        from services.genome_io import import_genome

        try:
            z, clamped, meta = import_genome(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[auto] could not load genome: {exc}")
            return
        svc.set_x0(z)
        if clamped:
            print(f"[auto] loaded genome with {clamped}/80 coefficients clamped "
                  "to the searchable range")

    def _save_auto_genome(self, svc, ui_state, tile):
        from pathlib import Path

        from services.genome_io import export_genome
        from utilities.paths import get_user_physics_configs_dir

        if tile is None:
            z, _ = svc.optimizer.best() if svc.optimizer else (None, 0.0)
            name = f"evolved_best_gen{svc.generation:04d}.json"
        else:
            z = svc.current_z[tile] if svc.current_z is not None else None
            name = f"evolved_tile{tile}_gen{svc.generation:04d}.json"
        if z is None:
            return
        meta = {
            "generation": svc.generation,
            "prompt": svc.prompt,
            "algorithm": svc.algorithm,
            "evolved_at_canvas_px": int(getattr(self.sim, "canvas_shape", (1024, 1024))[0]),
            "evolved_at_tile_px": 224,
            "evolved_with_tile_mutation": bool(svc.tile_mutation_enabled),
            "mutation_strength": float(svc.tile_mutation_strength),
            "variants_per_tile": int(svc.variants_per_tile),
        }
        path = Path(get_user_physics_configs_dir()) / name
        export_genome(path, z, ui_state.sim, meta)
        print(f"[auto] saved {path}")
```

If `sim.canvas_shape` does not exist, substitute whatever attribute `sim.py` uses for the canvas dimensions — check near `sim.py:83` where `ctx.texture(canvas_shape, ...)` is called.

- [ ] **Step 6: Run the full suite**

```bash
./.venv/Scripts/python.exe -m pytest -m "not gpu" -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add state/auto_tournament_state.py state/ui_state.py state/__init__.py command_handler.py tests/test_auto_tournament_state.py
git commit -m "feat: AutoTournamentState and CommandHandler wiring

One-shot flags are cleared on the consuming side, not in UI.get_state(), which
returns the live state object - clearing there would wipe flags before the
handler read them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 13: Orchestrator wiring in main.py

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a running Auto mode, driven from `orchestrate_frame`.

- [ ] **Step 1: Construct the services**

In `main.py`, after `self.tournament_service = TournamentService()` at line 70, replace that line and add:

```python
        self.tournament_service = TournamentService(grid=4)
        self.auto_service = None          # built lazily, see _ensure_auto_service
        self.tile_capture = None
        self.clip_scorer = None
        self._auto_prev_aspect = None
```

Pass `auto_service` into the CommandHandler construction at line 93 — note it is set later, so pass the App and let the handler read it, or simply assign after construction:

```python
        self.command_handler.auto_service = None   # updated by _ensure_auto_service
```

Add near `self._tournament_was_enabled = False` at line 124:

```python
        self._auto_was_enabled = False
```

- [ ] **Step 2: Add lazy service construction**

Add this method to `App`:

```python
    def _ensure_auto_service(self):
        """Build the CLIP scorer, capture buffer and service on first use.

        Imports are deliberately lazy: onnxruntime and cmaes must not be
        imported at startup, and Auto mode must degrade to a message rather
        than crashing when they are absent.
        """
        if self.auto_service is not None:
            return True
        try:
            from services.auto_tournament_service import AutoTournamentService
            from services.clip_scorer import CLIPScorer
            from services.run_logger import RunLogger
            from services.tile_capture import TileCapture
            from tools.fetch_clip_onnx import MODEL_DIR, is_present
        except ImportError as exc:
            self.ui.auto_unavailable = f"missing package: {exc.name}"
            return False

        if not is_present(MODEL_DIR):
            self.ui.auto_unavailable = "model_missing"
            return False

        try:
            self.clip_scorer = CLIPScorer(MODEL_DIR)
        except Exception as exc:
            self.ui.auto_unavailable = f"could not load CLIP: {exc}"
            return False

        self.tile_capture = TileCapture(self.ctx, self.tournament_service.grid)
        self.auto_service = AutoTournamentService(
            self.tournament_service,
            scorer=self.clip_scorer,
            logger=RunLogger(config={"grid": self.tournament_service.grid}),
        )
        self.command_handler.auto_service = self.auto_service
        self.ui.auto_service = self.auto_service
        self.ui.auto_unavailable = ""
        return True
```

- [ ] **Step 3: Force a 1:1 canvas on the Auto enable edge**

In `orchestrate_frame`, immediately after the tournament conflict-prevention block at line 254-268, add:

```python
        # 5.1.6. Auto mode requires square tiles: tiles inherit the canvas
        # aspect ratio, and a 16:9 tile cannot be fitted to CLIP's square input
        # without distortion, padding or discarding content.
        auto = ui_state.auto_tournament
        if auto.enabled and not self._auto_was_enabled:
            self._auto_prev_aspect = ui_state.sim.canvas_aspect_ratio
            if ui_state.sim.canvas_aspect_ratio != "1:1":
                ui_state.sim.canvas_aspect_ratio = "1:1"
                ui_state.request_world_size_change = True
            self._ensure_auto_service()
        elif not auto.enabled and self._auto_was_enabled:
            if self._auto_prev_aspect and self._auto_prev_aspect != "1:1":
                ui_state.sim.canvas_aspect_ratio = self._auto_prev_aspect
                ui_state.request_world_size_change = True
            if self.auto_service is not None:
                self.auto_service.pause()
        self._auto_was_enabled = auto.enabled
```

If the one-shot flag for a world-size change is named differently, use whatever `_handle_world_size_change` reads at `command_handler.py:155`.

- [ ] **Step 4: Drive the loop**

Replace the `apply_tournament` call at line 233 with:

```python
        _auto = self.auto_service
        _mut = (
            _auto.tile_mutation_strength
            if (_auto is not None and _auto.tile_mutation_enabled)
            else 0.0
        )
        self.sim.apply_tournament(
            ui_state.tournament.enabled,
            grid=self.tournament_service.grid,
            mutation=_mut,
        )
        if _auto is not None and _auto.tile_mutation_enabled:
            from services.cohort_tiling import cohorts_for

            ui_state.sim.num_cohorts = cohorts_for(
                self.tournament_service.grid, _auto.variants_per_tile
            )
```

Then, immediately before the simulation step in section 5 (`SimulationRunner.step_and_assemble`), add:

```python
        auto_steps = 1
        if self.auto_service is not None and ui_state.auto_tournament.enabled:
            from services.auto_tournament_service import Action

            action = self.auto_service.update()
            if action is Action.WRITE_RULES:
                self.sim.write_tournament_rules(self.tournament_service.pack_rule_bytes())
                self.tournament_service.clear_dirty()
                self.sim.reset()
                auto_steps = 0
            elif action is Action.STEP:
                auto_steps = self.auto_service.sim_steps_per_frame
            elif action is Action.CAPTURE:
                self.tile_capture.resize(self.tournament_service.grid)
                self._last_crops = self.tile_capture.capture(
                    lambda fbo: self.frame_assembler.render_to(fbo)
                )
                self.auto_service.submit_frames(self._last_crops)
                auto_steps = 0
            elif action is Action.SCORE:
                fit = self.auto_service.score_and_tell()
                gen = self.auto_service.generation
                log = self.auto_service.logger
                if log is not None and self._last_crops is not None and gen % 10 == 0:
                    log.save_frame(self._last_crops[int(fit.argmax())], gen)
                every = self.auto_service.autosave_every
                if log is not None and every and gen % every == 0:
                    from services.run_checkpoint import save_checkpoint

                    save_checkpoint(
                        log.dir / f"checkpoint_gen{gen:06d}.npz",
                        self.auto_service.checkpoint_state(),
                    )
                auto_steps = 0
```

Initialise `self._last_crops = None` in `App.__init__` alongside the other new attributes from Step 1.

Then run the sim `auto_steps` times instead of once. If `SimulationRunner.step_and_assemble` does not take a step count, call it in a loop:

```python
        for _ in range(max(auto_steps, 0)):
            self.simulation_runner.step_and_assemble(...)   # existing arguments
```

and skip the call entirely when `auto_steps == 0`, still performing the frame assembly and camera render so the UI keeps drawing.

**`self.frame_assembler.render_to(fbo)` may not exist.** If it does not, add a small helper on `App` that binds `fbo`, sets the viewport to `fbo.size`, and issues the same draw the normal path uses to produce the assembled view — reusing `camera.generate_view_texture(tiling_mode=...)` and the frame assembler's existing vao/shader. The requirement is only that the offscreen FBO ends up holding the same image the user sees.

- [ ] **Step 5: Manual smoke test**

```bash
./.venv/Scripts/python.exe main.py
```

Confirm: the app starts, manual tournament still works, and enabling Auto mode either starts running or shows a clear unavailability message. No crash either way.

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: drive the auto tournament loop from orchestrate_frame

Services are built lazily on first use so onnxruntime and cmaes are never
imported at startup and Auto mode degrades to a message when absent.

Enabling Auto mode forces a 1:1 canvas and restores the previous ratio on exit:
tiles inherit the canvas aspect, and a 16:9 tile cannot be fitted to CLIP's
square input without distortion, padding or losing content.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 14: The Auto tab UI

**Files:**
- Create: `ui/auto_tournament_window.py`
- Modify: `ui/core.py`, `ui/tournament_window.py`

**Interfaces:**
- Consumes: `AutoTournamentState`, `AutoTournamentService`, `services.cohort_tiling.max_variants`.
- Produces: `AutoTournamentWindowMixin.render_auto_tournament_tab()`, mixed into `UI`.

- [ ] **Step 1: Write the mixin**

Create `ui/auto_tournament_window.py`:

```python
"""Auto (CLIP) tab of the tournament window. Passive: renders widgets and sets
state, runs no logic."""
from __future__ import annotations

from imgui_bundle import imgui

from services.cohort_tiling import cohorts_for, max_variants

ALGORITHM_NAMES = ["CMA-ES", "Sep-CMA-ES", "GA", "Random Search"]

COHORT_TOOLTIP = (
    "Particles are numbered in one long list. Both 'which tile' and 'which "
    "cohort' are just slices of that list, so they nest: with 64 cohorts across "
    "16 tiles, each tile contains 4 cohorts, and each cohort gets its own small "
    "tweak of that tile's genome.\n\n"
    "This is why cohorts must be a multiple of the tile count. If they were "
    "equal, each tile would contain exactly one cohort - one tweak applied to "
    "every particle - and the tile would still be uniform, just shifted. "
    "Cohorts are set for you while this is enabled."
)


class AutoTournamentWindowMixin:
    auto_service = None
    auto_unavailable = ""

    def render_auto_tournament_tab(self):
        ats = self.state.auto_tournament
        ats.enabled = True

        if self.auto_unavailable:
            self._render_auto_unavailable()
            return

        svc = self.auto_service
        gen = svc.generation if svc else 0
        sigma = svc.sigma if svc else ats.sigma0

        imgui.text(f"Generation {gen}")
        imgui.separator()

        changed, ats.prompt = imgui.input_text(
            "Goal", ats.prompt, imgui.InputTextFlags_.enter_returns_true
        )
        if changed:
            ats.prompt_changed = True
        imgui.same_line()
        if imgui.button("Set"):
            ats.prompt_changed = True

        idx = ALGORITHM_NAMES.index(ats.algorithm) if ats.algorithm in ALGORITHM_NAMES else 0
        ch, idx = imgui.combo("Algorithm", idx, ALGORITHM_NAMES)
        if ch:
            ats.algorithm = ALGORITHM_NAMES[idx]

        ch, g = imgui.slider_int("Grid", ats.grid, 2, 8)
        if ch and g != ats.grid:
            ats.grid = g
            ats.grid_changed = True
        tiles = ats.grid * ats.grid
        src_px = 1024 // ats.grid
        imgui.text_disabled(f"population {tiles}   source {src_px}px/tile")
        if src_px < 224:
            imgui.text_disabled(f"  (upscaled to 224 for CLIP - raise canvas res?)")
        if ats.grid == 2:
            imgui.text_disabled("  popsize 4 is small for 80-D CMA-ES")
        imgui.text_disabled("changing the grid resets the optimizer")

        imgui.separator()
        if ats.running:
            if imgui.button("Pause"):
                ats.pause_requested = True
        else:
            imgui.begin_disabled(not ats.prompt.strip())
            if imgui.button("Start"):
                ats.start_requested = True
            imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Reset"):
            ats.reset_requested = True

        imgui.separator()
        _, ats.steps_per_gen = imgui.slider_int("Steps per Gen", ats.steps_per_gen, 50, 2000)
        _, ats.sim_steps_per_frame = imgui.slider_int(
            "Sim Steps per Frame", ats.sim_steps_per_frame, 1, 50
        )
        _, ats.snapshots_per_gen = imgui.slider_int(
            "Snapshots per Gen", ats.snapshots_per_gen, 1, 8
        )
        _, ats.sigma0 = imgui.slider_float("Initial Sigma", ats.sigma0, 0.05, 1.5)

        imgui.separator()
        self._render_tile_mutation(ats, sigma)

        imgui.separator()
        self._render_metrics(svc)

        imgui.separator()
        if imgui.button("Save best genome"):
            ats.save_best_requested = True
        imgui.same_line()
        if imgui.button("Save checkpoint"):
            ats.save_checkpoint_requested = True
        _, ats.autosave_every = imgui.slider_int("Autosave every N gens", ats.autosave_every, 0, 100)

    def _render_auto_unavailable(self):
        ats = self.state.auto_tournament
        if self.auto_unavailable == "model_missing":
            imgui.text_wrapped("CLIP model weights are not downloaded.")
            if imgui.button("Download CLIP model (~330 MB)"):
                ats.download_model_requested = True
        else:
            imgui.text_wrapped(f"Auto mode unavailable: {self.auto_unavailable}")
            imgui.text_disabled(
                "pip install onnxruntime-directml tokenizers cmaes"
            )

    def _render_tile_mutation(self, ats, sigma):
        _, ats.tile_mutation_enabled = imgui.checkbox(
            "Per-tile Mutation", ats.tile_mutation_enabled
        )
        if not ats.tile_mutation_enabled:
            return

        kmax = max_variants(ats.grid)
        ats.variants_per_tile = min(ats.variants_per_tile, kmax)
        _, ats.variants_per_tile = imgui.slider_int(
            "Variants per Tile", ats.variants_per_tile, 1, kmax
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(COHORT_TOOLTIP)
        imgui.text_disabled(
            f"cohorts driven to {cohorts_for(ats.grid, ats.variants_per_tile)} "
            f"(max {kmax} variants at this grid)"
        )

        _, ats.tile_mutation_strength = imgui.slider_float(
            "Mutation Strength", ats.tile_mutation_strength, 0.0, 0.5
        )
        # Sigma shrinks as CMA-ES converges. Once it approaches the mutation
        # strength, the spread between tiles no longer exceeds the wobble inside
        # each tile and the search stalls with no error.
        ok = ats.tile_mutation_strength < sigma / 3.0
        imgui.same_line()
        if ok:
            imgui.text_disabled(f"(sigma {sigma:.3f} - ok)")
        else:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.6, 0.2, 1.0),
                f"(sigma {sigma:.3f} - too high, search may stall)",
            )

    def _render_metrics(self, svc):
        if svc is None or svc.logger is None:
            imgui.text_disabled("no metrics yet")
            return
        h = svc.logger.history()
        best = h.get("fit_best") or []
        mean = h.get("fit_mean") or []
        if not best:
            imgui.text_disabled("no generations completed yet")
            return
        imgui.text(f"best {max(best):.3f}   last {best[-1]:.3f}")
        imgui.plot_lines("fitness", best, graph_size=imgui.ImVec2(0, 60))
        if imgui.begin_table("autolog", 3, imgui.TableFlags_.borders):
            imgui.table_setup_column("gen")
            imgui.table_setup_column("best")
            imgui.table_setup_column("mean")
            imgui.table_headers_row()
            for i in range(max(0, len(best) - 50), len(best)):
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(str(i + 1))
                imgui.table_next_column(); imgui.text(f"{best[i]:.3f}")
                imgui.table_next_column()
                imgui.text(f"{mean[i]:.3f}" if i < len(mean) else "-")
            imgui.end_table()
```

`imgui.plot_lines` needs a float array; if the imgui_bundle binding rejects a Python list, wrap it: `np.asarray(best, dtype=np.float32)`.

- [ ] **Step 2: Add the tab selector**

In `ui/tournament_window.py`, wrap the existing body of `render_tournament_window` in a tab bar:

```python
    def render_tournament_window(self):
        # ... existing window begin / early-return guards stay as they are ...
        if imgui.begin_tab_bar("tournament_modes"):
            if imgui.begin_tab_item("Manual")[0]:
                self.state.auto_tournament.enabled = False
                self._render_manual_tournament()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Auto (CLIP)")[0]:
                self.render_auto_tournament_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()
```

Move the current body (the grid, Next Gen, Reset Gen, mutation controls) into a new method `_render_manual_tournament(self)` on the same mixin, unchanged apart from `grid = self.tournament_service.grid` from Task 4.

- [ ] **Step 3: Mix it into UI**

In `ui/core.py`, add the import next to the other mixin imports at line 29:

```python
from .auto_tournament_window import AutoTournamentWindowMixin
```

and add `AutoTournamentWindowMixin,` to the `class UI(...)` base list at line 39.

- [ ] **Step 4: Handle the download button**

In `command_handler.py`, inside `_handle_auto_tournament`, before the `svc is None` early return, add:

```python
        if ats.download_model_requested:
            import threading

            from tools.fetch_clip_onnx import fetch

            threading.Thread(target=fetch, daemon=True).start()
            print("[auto] downloading CLIP model in the background")
```

- [ ] **Step 5: Run the full suite and the app**

```bash
./.venv/Scripts/python.exe -m pytest -m "not gpu" -q
```

```bash
./.venv/Scripts/python.exe main.py
```

Confirm: the tournament window shows Manual and Auto (CLIP) tabs; Manual behaves exactly as before; Auto shows either controls or a clear unavailability message.

- [ ] **Step 6: Commit**

```bash
git add ui/auto_tournament_window.py ui/tournament_window.py ui/core.py command_handler.py
git commit -m "feat: Auto (CLIP) tab with sigma readout and cohort tooltip

Both affordances exist because the mechanics are invisible and fail silently:
the mutation slider is shown against the optimizer's live sigma, and the
variants slider explains why cohorts must be a multiple of the tile count.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 15: Capture sanity checks, sweep warning, abort-on-resize, remaining actions

These are the spec requirements not covered by Tasks 1–14: §7.6, §5.3.4, the
abort rows of §11.2, and the load/save actions §2 lists but Task 14 did not render.

**Files:**
- Create: `services/capture_health.py`
- Modify: `services/tile_capture.py`, `command_handler.py`, `ui/auto_tournament_window.py`
- Test: `tests/test_capture_health.py`

**Interfaces:**
- Produces:
  - `check_capture(crops) -> str | None` — returns a warning message, or `None` when healthy
  - `sweeping_parameters(sim_state) -> list[str]` — names of parameters with a non-zero spatial or cohort sweep

- [ ] **Step 1: Write the failing tests**

Create `tests/test_capture_health.py`:

```python
import numpy as np

from services.capture_health import check_capture, sweeping_parameters
from state import SimState


def test_healthy_capture_returns_none():
    rng = np.random.default_rng(0)
    crops = rng.integers(20, 200, (4, 224, 224, 3), dtype=np.uint8)
    assert check_capture(crops) is None


def test_all_zero_capture_is_reported():
    """An unbound or never-drawn FBO produces a perfectly flat fitness
    landscape, which is easy to mistake for 'CLIP has no signal'."""
    msg = check_capture(np.zeros((4, 224, 224, 3), dtype=np.uint8))
    assert msg is not None and "black" in msg.lower()


def test_blown_out_capture_is_reported():
    msg = check_capture(np.full((4, 224, 224, 3), 255, dtype=np.uint8))
    assert msg is not None and "blown" in msg.lower()


def test_a_clean_sim_state_has_no_sweeping_parameters():
    assert sweeping_parameters(SimState()) == []


def test_a_spatial_sweep_is_named():
    """Tiles partition position space, so any x/y/cohort sweep makes tiles
    run different physics and the comparison is confounded."""
    s = SimState()
    s.x_sweeps["SENSOR_DISTANCE"] = 0.8
    s.cohort_sweeps["TRAIL_DIFFUSION"] = -1.0
    got = sweeping_parameters(s)
    assert "SENSOR_DISTANCE" in got
    assert "TRAIL_DIFFUSION" in got


def test_mutation_scale_is_excluded():
    """Tournament mode owns Mutation Scale, so a sweep on it is neutralised
    rather than warned about."""
    s = SimState()
    s.x_sweeps["MUTATION_SCALE"] = 1.0
    assert sweeping_parameters(s) == []
```

- [ ] **Step 2: Run to verify it fails**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_capture_health.py -v
```

Expected: FAIL, `No module named 'services.capture_health'`.

- [ ] **Step 3: Implement**

Create `services/capture_health.py`:

```python
"""Cheap validity checks on the capture and the physics configuration.

Both failure modes here are silent: a black capture and a confounded physics
sweep each produce a fitness landscape that looks plausible and is meaningless.
"""
from __future__ import annotations

import numpy as np

# Tournament mode takes ownership of this one (it is zeroed in sim.py), so a
# sweep on it is neutralised rather than warned about.
_OWNED = {"MUTATION_SCALE"}


def check_capture(crops: np.ndarray) -> str | None:
    """Return a warning string, or None when the capture looks usable."""
    if crops.size == 0:
        return "Capture was empty - the offscreen framebuffer produced no pixels."
    mx = int(crops.max())
    if mx == 0:
        return (
            "Capture is entirely black. Fitness will be flat and meaningless - "
            "check that the offscreen framebuffer is being drawn into."
        )
    mean = float(crops.mean())
    if mean < 2.0:
        return f"Capture is nearly black (mean {mean:.1f}/255); fitness will be flat."
    if mean > 253.0:
        return f"Capture is blown out (mean {mean:.1f}/255); fitness will be flat."
    return None


def sweeping_parameters(sim_state) -> list[str]:
    """Parameters whose value differs across tiles.

    Tiles partition both position space and cohort index, so any parameter with
    a non-zero x, y or cohort sweep takes different values in different tiles.
    Neither human selection nor a CLIP score is then comparing genomes on equal
    terms - the comparison is confounded by position.
    """
    names: set[str] = set()
    for attr in ("x_sweeps", "y_sweeps", "cohort_sweeps"):
        for key, value in (getattr(sim_state, attr, None) or {}).items():
            if value and key not in _OWNED:
                names.add(key)
    return sorted(names)
```

- [ ] **Step 4: Run to verify it passes**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_capture_health.py -v
```

Expected: 6 passed. If `SimState` names the sweep dicts differently, check `state/sim_state.py` and use the real attribute names in `sweeping_parameters` — the shader reads them via `setting.x_sweep` / `y_sweep` / `cohort_sweep`.

- [ ] **Step 5: Report the capture check once per run**

In `command_handler.py`, add to `CommandHandler.__init__`:

```python
        self._auto_capture_warned = False
```

and add this method:

```python
    def report_capture_health(self, crops, ui_state):
        """Reported once per run rather than once per frame."""
        if self._auto_capture_warned:
            return
        from services.capture_health import check_capture

        msg = check_capture(crops)
        if msg:
            self._auto_capture_warned = True
            ui_state.auto_tournament_warning = msg
            print(f"[auto] {msg}")
```

Add `auto_tournament_warning: str = ""` to `AutoTournamentState` in `state/auto_tournament_state.py` (persistent, not a one-shot — do **not** add it to `_clear_auto_flags`), and reference it as `ats.auto_tournament_warning` rather than on `ui_state` directly:

```python
            ui_state.auto_tournament.auto_tournament_warning = msg
```

In `main.py`, in the `Action.CAPTURE` branch added in Task 13, call it right after the capture:

```python
                self.command_handler.report_capture_health(self._last_crops, ui_state)
```

Reset the flag when a run starts: in `_handle_auto_tournament`, inside the `if ats.start_requested ...` branch, add `self._auto_capture_warned = False`.

- [ ] **Step 6: Wire abort-on-resize and abort-on-shader-reload**

In `command_handler.py`, inside `_handle_world_size_change`, immediately after the existing `self.tournament_service.mark_dirty()` call, add:

```python
        # A partial rollout is not a valid fitness sample.
        if self.auto_service is not None:
            self.auto_service.abort_generation()
```

Find the shader hot-reload handler (the one bound to `V`; search `command_handler.py` for `reload`) and add the same two lines there. If the reload is handled in `main.py` rather than `command_handler.py`, add it there instead, guarded by `if self.auto_service is not None`.

- [ ] **Step 7: Render the remaining actions and the sweep warning**

In `ui/auto_tournament_window.py`, replace the final button block of `render_auto_tournament_tab` with:

```python
        imgui.separator()
        if imgui.button("Save best genome"):
            ats.save_best_requested = True
        imgui.same_line()
        if imgui.button("Save checkpoint"):
            ats.save_checkpoint_requested = True

        ch, self._auto_load_path = imgui.input_text(
            "Load path", getattr(self, "_auto_load_path", "")
        )
        imgui.same_line()
        if imgui.button("Load genome"):
            ats.load_genome_path = self._auto_load_path
        imgui.same_line()
        if imgui.button("Load checkpoint"):
            ats.load_checkpoint_path = self._auto_load_path
        imgui.text_disabled(
            "Load genome takes only the starting point - sigma, algorithm and "
            "grid stay as set here. Load checkpoint restores the whole search "
            "and forces the grid to the saved value."
        )

        if svc is not None and svc.logger is not None and svc.logger.enabled:
            if imgui.button("Open run folder"):
                import os
                import subprocess

                subprocess.Popen(["explorer", os.path.abspath(svc.logger.dir)])
            imgui.same_line()
            imgui.text_disabled(str(svc.logger.run_id))

        _, ats.autosave_every = imgui.slider_int(
            "Autosave every N gens", ats.autosave_every, 0, 100
        )
```

Add a per-tile save action to the manual grid in `ui/tournament_window.py`: inside the tile button loop, after the existing click handling, add a right-click handler:

```python
                if imgui.is_item_clicked(imgui.MouseButton_.right):
                    self.state.auto_tournament.save_tile_requested = tile
```

Add the warning banner at the top of `render_auto_tournament_tab`, immediately after `ats.enabled = True`:

```python
        if ats.auto_tournament_warning:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.4, 0.3, 1.0), ats.auto_tournament_warning
            )
            imgui.same_line()
            if imgui.button("Dismiss"):
                ats.auto_tournament_warning = ""

        from services.capture_health import sweeping_parameters

        sweeps = sweeping_parameters(self.state.sim)
        if sweeps:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.6, 0.2, 1.0),
                "Tiles are not comparable: " + ", ".join(sweeps),
            )
            imgui.text_disabled(
                "These parameters have a spatial or cohort sweep, so they take "
                "different values in different tiles. Fitness is confounded by "
                "position until they are cleared."
            )
```

- [ ] **Step 8: Run the full suite and the app**

```bash
./.venv/Scripts/python.exe -m pytest -m "not gpu" -q
```

```bash
./.venv/Scripts/python.exe main.py
```

Confirm: with a sweep set on any physics parameter, the Auto tab shows the "Tiles are not comparable" banner naming it; clearing the sweep removes the banner.

- [ ] **Step 9: Commit**

```bash
git add services/capture_health.py tests/test_capture_health.py command_handler.py main.py ui/auto_tournament_window.py ui/tournament_window.py state/auto_tournament_state.py
git commit -m "feat: capture sanity checks, sweep warning, and abort on resize

Three silent failure modes made visible:
- an all-black capture produces a flat landscape that reads as 'CLIP has no
  signal'; now reported once per run
- any physics parameter with an x/y/cohort sweep takes different values in
  different tiles, so fitness is confounded by position; the offending
  parameters are named rather than silently overridden
- a resize or shader reload mid-rollout leaves a partial, invalid fitness
  sample; the generation is now aborted and restarted

Adds the load-genome, load-checkpoint, save-tile and open-run-folder actions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 16: End-to-end verification and documentation

**Files:**
- Modify: `docs/testing_checklist.md`, `ARCHITECTURE.md`, `CLAUDE.md`
- Create: `docs/superpowers/plans/2026-08-04-clip-guided-tournament-execution-log.md`

- [ ] **Step 1: Run one real evolution**

```bash
./.venv/Scripts/python.exe main.py
```

Enable Tournament → Auto (CLIP). Type `glowing coral`. Press Start. Let it run 30 generations.

Verify:
- The canvas snapped to 1:1 on enable.
- Tiles visibly change each generation.
- `fit_best` in the sparkline trends upward over 30 generations. If it is flat, note it — that is the CLIP-signal risk from Task 3 reappearing at loop level.
- `runs/<id>/log.jsonl` has one line per generation.
- `runs/<id>/checkpoint_gen000010.npz` exists.
- Retyping the goal mid-run changes the target without resetting the generation counter.
- Clicking a tile makes it dominate the next generation.
- Disabling Auto mode restores the previous canvas aspect.

- [ ] **Step 2: Verify per-tile mutation**

Tick **Per-tile Mutation**. Confirm each tile becomes visibly heterogeneous rather than uniform, the cohort readout shows `cohorts driven to <k·T>`, and the sigma readout shows "ok" at strength 0.1.

Then set Mutation Strength to 0.5 and confirm the readout turns to the "too high" warning.

- [ ] **Step 3: Verify checkpoint and genome export**

Save a checkpoint, quit, relaunch, load it, confirm the generation counter and prompt resume.

Save the best genome, disable tournament mode entirely, and load the exported file through the normal config browser. Confirm it runs as a single full-resolution simulation.

- [ ] **Step 4: Update the testing checklist**

Append to `docs/testing_checklist.md`:

```markdown
## Auto (CLIP) Tournament

- [ ] Enabling Auto mode snaps the canvas to 1:1; disabling restores the previous ratio
- [ ] Start with an empty goal is disabled; with a goal it runs
- [ ] fit_best trends upward over ~30 generations on a concrete prompt
- [ ] Retyping the goal changes the target without resetting the generation counter
- [ ] Clicking tiles makes them dominate the next generation, then clears the selection
- [ ] Grid slider 2..8 works; changing it resets the optimizer and resizes the button grid
- [ ] Per-tile mutation makes tiles visibly heterogeneous; cohort readout matches k*T
- [ ] Sigma readout warns when mutation strength exceeds sigma/3
- [ ] runs/<id>/log.jsonl has one line per generation and survives a hard quit
- [ ] Checkpoint save -> quit -> load -> resume restores generation, prompt and settings
- [ ] Exported genome opens in the normal single-sim view at full resolution
- [ ] With onnxruntime uninstalled, Auto tab shows a message and Manual mode still works
```

- [ ] **Step 5: Update ARCHITECTURE.md and CLAUDE.md**

In `ARCHITECTURE.md`, add a short "Automatic Tournament (CLIP)" subsection describing the state machine, the injected units, and pointing at the spec.

In `CLAUDE.md`, add to Important Caveats:

```markdown
- **CLIP/evolution dependencies are optional and lazy** — `onnxruntime-directml`,
  `tokenizers` and `cmaes` must never be imported at startup. Auto tournament mode
  degrades to a message when they are absent; manual mode must keep working.
- **Tile and cohort indices both derive from the particle index**, so cohort count must be
  a multiple of the tile count. See `services/cohort_tiling.py`.
```

- [ ] **Step 6: Write the execution log**

Create `docs/superpowers/plans/2026-08-04-clip-guided-tournament-execution-log.md` recording, for each task: what was implemented, any deviation from this plan and why, and the actual outcome of the Task 3 gate with its numbers.

- [ ] **Step 7: Final full-suite run**

```bash
./.venv/Scripts/python.exe -m pytest -q
```

```bash
./.venv/Scripts/python.exe -m pytest -q -m gpu
```

Expected: all pass. Report the actual counts — do not claim success without the output.

- [ ] **Step 8: Commit**

```bash
git add docs/ ARCHITECTURE.md CLAUDE.md
git commit -m "docs: auto tournament testing checklist, architecture and execution log

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
