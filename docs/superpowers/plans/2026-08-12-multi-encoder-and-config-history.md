# Selectable Vision Encoders + Archive Config History — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the vision encoder a calibrated, selectable, per-archive-immutable choice across both automatic modes, and record every archive settings change with entries stamped by config version.

**Architecture:** A frozen `VisionModel` registry becomes the single home for everything model-specific (paths, preprocessing, tokenizer context, logit scales, separation default). `CLIPScorer` becomes `VisionScorer(model_key)` reading from it. Each archive pins one encoder in `encoder.json` at the archive root — above the brain-layout signature directory, so it is insulated from layout changes. A one-resident scorer slot follows whichever mode is scoring. Settings changes append to `settings_history.jsonl` and entries carry the version they were admitted under.

**Tech Stack:** Python 3.12, onnxruntime-directml, tokenizers, NumPy, PIL, imgui_bundle, pytest.

**Spec:** `docs/superpowers/specs/2026-08-12-multi-encoder-and-config-history-design.md`

## Global Constraints

**This is a SHARED working tree. Another Claude session is actively editing it right now.**

- **Never** run `git add -A`, `git add .`, `git stash`, `git checkout -- .`, `git reset --hard`, or `git clean`. Always `git add` explicit paths. Uncommitted work in this tree belongs to someone else.
- **Do not modify these eight files.** They are the concurrent session's in-flight variable-depth-MLP work: `services/brains/__init__.py`, `services/brains/mlp.py`, `shaders/brains/mlp.glsl`, `services/brain_preview.py`, `ui/brain_window.py`, `sim.py`, `utilities/gl_helpers.py`, `tests/test_brain_modalities_gpu.py`. If a task appears to require one, **stop and report** rather than editing.
- **Line numbers in this plan will drift.** Other sessions commit to `integration` between tasks. Every edit below is anchored by quoted source text — search for the text, do not trust a line number.
- **A full-suite failure may not be yours.** Routine verification runs only the test files named in the task. Before acting on any failure from `pytest -q`, run `git status --porcelain` and check whether the failing test's module is in the eight files above.
- Do not rebase, force-push, or `git pull --rebase`. Commit to `integration` and leave it.
- Run pytest as `.venv/Scripts/python.exe -m pytest` — bare `python` is 3.10 with no pytest.
- Windows: use forward slashes in paths; use `rm` not `del` in bash.
- `sim.py` is user-owned and is also in the other session's set. Nothing in this plan touches it.
- Comment/docstring rule (CLAUDE.md): state the rule, never the evidence. No percentages, timings, or rationale essays in code. Measured facts belong in CLAUDE.md or `docs/`.

---

## File Structure

**Created:**
- `services/vision_models.py` — the `VisionModel` dataclass and `REGISTRY`. Single home for model-specific constants. No I/O, no onnxruntime import.
- `services/vision_scorer.py` — `VisionScorer`, renamed from `CLIPScorer` and driven by a registry entry.
- `tools/fetch_models.py` — per-model downloads, stdlib only.
- `tools/archive_sample.py` — thumbnail sampling and embedding shared by the two measurement tools.
- `tools/calibrate_encoder.py` — Phase 0: measures the three calibrated fields.
- `services/settings_history.py` — diffing and appending settings versions.
- `tests/test_vision_models.py`, `tests/test_vision_scorer.py`, `tests/test_archive_encoder.py`, `tests/test_settings_history.py`.

**Deleted:** `services/clip_scorer.py`, `tools/fetch_clip_onnx.py` (replaced, via `git mv` where possible so history follows).

**Modified:** `services/expedition_fitness.py`, `services/imgep_driver.py`, `services/archive.py`, `services/archive_io.py`, `state/archive_state.py`, `ui/archive_window.py`, `ui/auto_tournament_window.py`, `ui/tournament_window.py`, `main.py`, `tools/hue_nuisance.py`, `CLAUDE.md`, `ui/README.md`.

---

## Task 1: Model registry

**Files:**
- Create: `services/vision_models.py`
- Test: `tests/test_vision_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VisionModel` frozen dataclass with fields `key, label, subdir, repo, files, mean, std, px, dim, context, text_logit_scale, image_logit_scale, default_min_separation`; `REGISTRY: dict[str, VisionModel]`; `get(key) -> VisionModel`; `DEFAULT_KEY = "clip-b32"`. The three scale/separation fields are `float | None` until Task 7 fills them.

- [ ] **Step 1: Write the failing test**

```python
"""The registry is the one home for what differs between encoders."""
import pytest

from services.vision_models import DEFAULT_KEY, REGISTRY, VisionModel, get


def test_the_default_is_the_encoder_every_existing_archive_used():
    assert DEFAULT_KEY == "clip-b32"
    assert DEFAULT_KEY in REGISTRY


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_entry_declares_its_fixed_facts(key):
    m = REGISTRY[key]
    assert m.key == key, "the dict key and the entry must agree"
    assert m.dim > 0 and m.px > 0 and m.context > 0
    assert len(m.mean) == 3 and len(m.std) == 3
    assert "vision_model_fp16.onnx" in m.files
    assert "text_model_fp16.onnx" in m.files
    assert "tokenizer.json" in m.files


def test_keys_are_stable_identifiers_not_paths():
    for key in REGISTRY:
        assert "/" not in key and "\\" not in key


def test_get_refuses_an_unknown_key_by_name():
    with pytest.raises(KeyError, match="nope"):
        get("nope")


def test_the_known_encoders_are_present():
    assert set(REGISTRY) == {"clip-b32", "clip-b16", "siglip2-b16", "clip-l14"}


def test_clip_b32_is_already_calibrated():
    """Its three values are the ones the app shipped with."""
    m = get("clip-b32")
    assert m.text_logit_scale == 100.0
    assert m.image_logit_scale == 30.0
    assert m.default_min_separation == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vision_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.vision_models'`

- [ ] **Step 3: Write the registry**

```python
"""What differs between one vision encoder and the next.

Imports nothing beyond the standard library, so it is safe to import from ui/
where onnxruntime must never be pulled in.

A key is written into archives on disk and can never be renamed - it is the
same class of fact as BrainLayout.signature().
"""
from __future__ import annotations

from dataclasses import dataclass

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
HALF = (0.5, 0.5, 0.5)

DEFAULT_KEY = "clip-b32"


@dataclass(frozen=True)
class VisionModel:
    key: str
    label: str
    subdir: str
    repo: str
    files: dict          # local filename -> path within the repo
    mean: tuple
    std: tuple
    px: int
    dim: int
    context: int
    # Calibrated per encoder; see tools/calibrate_encoder.py and CLAUDE.md.
    text_logit_scale: float | None = None
    image_logit_scale: float | None = None
    default_min_separation: float | None = None

    @property
    def calibrated(self) -> bool:
        return None not in (self.text_logit_scale, self.image_logit_scale,
                            self.default_min_separation)

    @property
    def separation_slider_max(self) -> float:
        """The Min Separation track. A model whose distances spread wider needs
        a longer track or its own default sits at the top of it."""
        return 2.5 * float(self.default_min_separation)


def _clip_files() -> dict:
    return {
        "vision_model_fp16.onnx": "onnx/vision_model_fp16.onnx",
        "text_model_fp16.onnx": "onnx/text_model_fp16.onnx",
        "tokenizer.json": "tokenizer.json",
    }


REGISTRY: dict[str, VisionModel] = {
    "clip-b32": VisionModel(
        key="clip-b32", label="CLIP ViT-B/32", subdir="clip-vit-b32",
        repo="https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main",
        files=_clip_files(), mean=CLIP_MEAN, std=CLIP_STD, px=224, dim=512,
        context=77,
        text_logit_scale=100.0, image_logit_scale=30.0,
        default_min_separation=0.02,
    ),
    "clip-b16": VisionModel(
        key="clip-b16", label="CLIP ViT-B/16", subdir="clip-vit-b16",
        repo="https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main",
        files=_clip_files(), mean=CLIP_MEAN, std=CLIP_STD, px=224, dim=512,
        context=77, text_logit_scale=100.0,
    ),
    "siglip2-b16": VisionModel(
        key="siglip2-b16", label="SigLIP 2 base/16", subdir="siglip2-b16-224",
        repo="https://huggingface.co/onnx-community/"
             "siglip2-base-patch16-224-ONNX/resolve/main",
        files=_clip_files(), mean=HALF, std=HALF, px=224, dim=768, context=64,
    ),
    "clip-l14": VisionModel(
        key="clip-l14", label="CLIP ViT-L/14", subdir="clip-vit-l14",
        repo="https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main",
        files=_clip_files(), mean=CLIP_MEAN, std=CLIP_STD, px=224, dim=768,
        context=77, text_logit_scale=100.0,
    ),
}


def get(key: str) -> VisionModel:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown vision model {key!r}; known: {', '.join(sorted(REGISTRY))}"
        ) from None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vision_models.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add services/vision_models.py tests/test_vision_models.py
git commit -m "feat: a registry for what differs between vision encoders"
```

---

## Task 2: `VisionScorer` — rename and drive from the registry

**Files:**
- Rename: `services/clip_scorer.py` → `services/vision_scorer.py` (use `git mv`)
- Modify: `services/vision_scorer.py`
- Test: `tests/test_vision_scorer.py`
- Modify: `tests/test_clip_scorer.py`, `tests/test_clip_real_model.py` (import path only)

**Interfaces:**
- Consumes: `services.vision_models.get`, `VisionModel`.
- Produces: `VisionScorer(model_key: str = "clip-b32", providers=None, n_views: int = 3, seed: int = 0)`, with a `model` property returning the `VisionModel`. All existing methods keep their names and signatures: `available`, `prompt`, `set_prompt`, `embed_text`, `embed`, `embed_mean`, `score`. Module still exports `preprocess`, `augment`, `DEFAULT_DISTRACTORS`.

- [ ] **Step 1: Move the file so history follows**

```bash
git mv services/clip_scorer.py services/vision_scorer.py
```

- [ ] **Step 2: Write the failing test**

```python
"""The scorer takes its constants from the registry, not from module globals."""
import numpy as np
import pytest

from services.vision_models import get
from services.vision_scorer import VisionScorer, preprocess


def _bare(model_key):
    """A scorer with no ONNX session - __new__ skips __init__, which is how the
    existing scorer tests build one."""
    s = VisionScorer.__new__(VisionScorer)
    s._model = get(model_key)
    return s


def test_the_scorer_exposes_its_model():
    assert _bare("siglip2-b16").model.dim == 768


def test_preprocess_uses_the_models_normalisation_not_clips():
    """SigLIP centres on 0.5/0.5, so a mid-grey image maps to zero for it and
    does not for CLIP."""
    grey = np.full((1, 224, 224, 3), 128, dtype=np.uint8)
    sig = preprocess(grey, get("siglip2-b16"))
    clip = preprocess(grey, get("clip-b32"))
    assert abs(float(sig.mean())) < 0.01
    assert abs(float(clip.mean())) > 0.05


def test_preprocess_returns_nchw_contiguous():
    out = preprocess(np.zeros((2, 224, 224, 3), np.uint8), get("clip-b32"))
    assert out.shape == (2, 3, 224, 224)
    assert out.flags["C_CONTIGUOUS"]


def test_output_preference_picks_the_pooled_embedding():
    class FakeOut:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    s = VisionScorer.__new__(VisionScorer)
    s._outputs = [FakeOut("last_hidden_state", [1, 197, 768]),
                  FakeOut("pooler_output", [1, 768])]
    assert s._pick_output() == "pooler_output"
    s._outputs = [FakeOut("image_embeds", [1, 512])]
    assert s._pick_output() == "image_embeds"


def test_an_unknown_model_key_is_refused():
    with pytest.raises(KeyError):
        VisionScorer("not-a-model")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vision_scorer.py -q`
Expected: FAIL — `ImportError: cannot import name 'VisionScorer'`

- [ ] **Step 4: Rewrite the module header and constants**

Replace the module docstring and the constants block. Search for `CLIP preprocessing constants, from preprocessor_config.json` and replace from there through the `_OFFSET_CHW` definition with:

```python
# Kept at module scope: DEFAULT_DISTRACTORS is the prompt-side rule, not a
# per-model fact.
DEFAULT_DISTRACTORS = [ ... unchanged ... ]


def _scale_offset(model):
    """The normalisation folded into one multiply-add, in NCHW, so it can be
    applied in place after the transpose."""
    mean = np.asarray(model.mean, dtype=np.float32)
    std = np.asarray(model.std, dtype=np.float32)
    return ((1.0 / 255.0 / std).reshape(1, 3, 1, 1).astype(np.float32),
            (-mean / std).reshape(1, 3, 1, 1).astype(np.float32))


def preprocess(crops, model, dtype=np.float32):
    """uint8 (B,px,px,3) -> (B,3,px,px) in `dtype`, C-contiguous.

    TRANSPOSE FIRST, while the data is still uint8 - this was the single
    largest cost in the scoring path.
    """
    scale, offset = _scale_offset(model)
    x = np.ascontiguousarray(crops.transpose(0, 3, 1, 2)).astype(np.float32)
    x *= scale
    x += offset
    return x.astype(dtype)
```

Delete `CLIP_MEAN`, `CLIP_STD`, `LOGIT_SCALE`, `CONTEXT_LENGTH`, `_SCALE_CHW`, `_OFFSET_CHW`.

- [ ] **Step 5: Rewrite `__init__` and add `_pick_output`**

Replace `class CLIPScorer:` with `class VisionScorer:` and its `__init__` with:

```python
    def __init__(self, model_key: str = DEFAULT_KEY, providers=None,
                 n_views: int = 3, seed: int = 0):
        self._available = False
        self._model = get(model_key)
        self._text_emb = None
        self._prompt = ""
        self._n_views = n_views
        self._rng = np.random.default_rng(seed)
        self._pool = None

        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path("models") / self._model.subdir
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        have = set(ort.get_available_providers())
        providers = [p for p in providers if p in have] or ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        # ORT_ENABLE_ALL crashes the fp16 vision model on the CPU provider.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3

        self._vision = ort.InferenceSession(
            str(d / "vision_model_fp16.onnx"), opts, providers=providers)
        self._text = ort.InferenceSession(
            str(d / "text_model_fp16.onnx"), opts, providers=providers)

        vin = self._vision.get_inputs()[0]
        self._vision_in = vin.name
        self._vision_dtype = _ORT_DTYPES.get(vin.type, np.float32)
        self._outputs = self._vision.get_outputs()
        self._vision_out = self._pick_output()
        self._text_in = self._text.get_inputs()[0].name
        self._text_out = self._text.get_outputs()[0].name

        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tokenizer.enable_truncation(self._model.context)
        self._tokenizer.enable_padding(length=self._model.context)

        self._available = True

    @property
    def model(self):
        return self._model

    def _pick_output(self) -> str:
        """The pooled image embedding. CLIP exports name it image_embeds,
        SigLIP's is pooler_output. Taking output 0 by position picks
        last_hidden_state on the SigLIP export."""
        names = {o.name for o in self._outputs}
        for pref in ("image_embeds", "pooler_output"):
            if pref in names:
                return pref
        for o in self._outputs:
            if len(o.shape) == 2:
                return o.name
        return self._outputs[0].name
```

Add `from services.vision_models import DEFAULT_KEY, get` to the imports.

- [ ] **Step 6: Thread the model through the remaining call sites**

In `_embed_images`, both `preprocess(...)` calls take the model:

```python
            batch = preprocess(crops, self._model, self._vision_dtype)
```
```python
            return ex.submit(preprocess, crops[s:s + c], self._model,
                             self._vision_dtype)
```

In `score()`, replace `logits = LOGIT_SCALE * (...)` with:

```python
        logits = self._model.text_logit_scale * (emb @ self._text_emb.T)
```

- [ ] **Step 7: Fix the two existing test modules' imports**

In `tests/test_clip_scorer.py` and `tests/test_clip_real_model.py`, replace every `from services.clip_scorer import` with `from services.vision_scorer import` and every `CLIPScorer` with `VisionScorer`. Where a test calls `preprocess(crops)`, pass the model: `preprocess(crops, get("clip-b32"))`.

- [ ] **Step 8: Run the scorer tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vision_scorer.py tests/test_clip_scorer.py tests/test_clip_real_model.py -q`
Expected: PASS, with `test_clip_real_model` skipping if the model is absent.

- [ ] **Step 9: Commit**

```bash
git add services/vision_scorer.py services/vision_models.py tests/test_vision_scorer.py tests/test_clip_scorer.py tests/test_clip_real_model.py
git commit -m "refactor: the scorer takes its constants from the model, not the module"
```

---

## Task 3: Per-model downloads

**Files:**
- Rename: `tools/fetch_clip_onnx.py` → `tools/fetch_models.py` (use `git mv`)
- Modify: `tools/fetch_models.py`
- Test: `tests/test_fetch_models.py`

**Interfaces:**
- Consumes: `services.vision_models.REGISTRY`, `get`.
- Produces: `is_present(model_key) -> bool`, `fetch(model_key, progress=None) -> None`, `model_dir(model_key) -> Path`, `missing(model_key) -> list[str]`.

- [ ] **Step 1: Move the file**

```bash
git mv tools/fetch_clip_onnx.py tools/fetch_models.py
```

- [ ] **Step 2: Write the failing test**

```python
"""Download bookkeeping, without touching the network."""
import pytest

from tools.fetch_models import is_present, missing, model_dir


def test_a_model_dir_is_named_for_its_subdir():
    assert model_dir("siglip2-b16").name == "siglip2-b16-224"


def test_an_empty_dir_is_not_present_and_lists_everything_missing(tmp_path,
                                                                 monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not is_present("clip-b32")
    assert len(missing("clip-b32")) == 3


def test_a_partial_download_is_not_present(tmp_path, monkeypatch):
    """A .part file must never read as a finished asset."""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "models" / "clip-vit-b32"
    d.mkdir(parents=True)
    (d / "vision_model_fp16.onnx").write_bytes(b"x")
    (d / "text_model_fp16.onnx.part").write_bytes(b"x")
    assert not is_present("clip-b32")
    assert "text_model_fp16.onnx" in missing("clip-b32")


def test_an_unknown_key_is_refused():
    with pytest.raises(KeyError):
        is_present("nope")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fetch_models.py -q`
Expected: FAIL — `ModuleNotFoundError` or `ImportError: cannot import name 'missing'`

- [ ] **Step 4: Rewrite the module**

```python
"""One-time download of a vision encoder's ONNX assets.

Stdlib only - this runs before onnxruntime is necessarily importable, and three
static URLs do not justify a huggingface_hub dependency.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from services.vision_models import get

MODELS_ROOT = "models"


def model_dir(model_key: str) -> Path:
    return Path(MODELS_ROOT) / get(model_key).subdir


def missing(model_key: str) -> list[str]:
    d = model_dir(model_key)
    return [name for name in get(model_key).files if not (d / name).is_file()]


def is_present(model_key: str) -> bool:
    """True only when every asset is fully downloaded."""
    return not missing(model_key)


def fetch(model_key: str, progress=None) -> None:
    """Download any missing assets.

    Writes to .part then renames, so an interrupted download can never be
    mistaken for a finished one.
    """
    model = get(model_key)
    d = model_dir(model_key)
    d.mkdir(parents=True, exist_ok=True)
    for name, remote in model.files.items():
        final = d / name
        if final.is_file():
            continue
        part = d / (name + ".part")
        if progress:
            progress(name, 0.0)
        with urllib.request.urlopen(f"{model.repo}/{remote}") as resp, \
                open(part, "wb") as out:
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
    import sys

    def _cli(name: str, frac: float) -> None:
        print(f"\r{name}: {frac * 100:5.1f}%", end="", flush=True)
        if frac >= 1.0:
            print()

    key = sys.argv[1] if len(sys.argv) > 1 else "clip-b32"
    fetch(key, progress=_cli)
    print(f"{key} ready in {model_dir(key)}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fetch_models.py -q`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add tools/fetch_models.py tests/test_fetch_models.py
git commit -m "feat: downloads are per model, driven by the registry"
```

---

## Task 4: Scales come off the model, not the module

**Files:**
- Modify: `services/expedition_fitness.py`, `services/imgep_driver.py`, `main.py`
- Test: `tests/test_expedition_fitness.py` (import fix only)

**Interfaces:**
- Consumes: `VisionScorer.model`.
- Produces: `expedition_fitness` no longer imports from the scorer module. `TEXT_LOGIT_SCALE` / `IMAGE_LOGIT_SCALE` remain as `clip-b32`'s values for the existing tests, but `ImgepDriver` reads `self.scorer.model.text_logit_scale` / `.image_logit_scale`.

- [ ] **Step 1: Break the import cycle in `expedition_fitness.py`**

Replace:

```python
from services.clip_scorer import LOGIT_SCALE

# CLIP's own learned logit_scale.exp(), and the right value for TEXT goals: the
# modality gap compresses text-image similarity into a narrow band, and 100 is
# the temperature trained to spread that band out.
TEXT_LOGIT_SCALE = LOGIT_SCALE  # 100.0

# Image-image similarity sits above 0.9, a different regime - 100 is far too
# sharp there and floors most of the population to zero. See CLAUDE.md.
IMAGE_LOGIT_SCALE = 30.0
```

with:

```python
from services.vision_models import get

# clip-b32's scales, kept as module constants because the calibration tests
# reason about this pair specifically. Runtime callers read the scales off
# the scorer's own model - every encoder has its own. See CLAUDE.md.
TEXT_LOGIT_SCALE = get("clip-b32").text_logit_scale
IMAGE_LOGIT_SCALE = get("clip-b32").image_logit_scale
```

- [ ] **Step 2: Point the driver at its scorer's model**

In `services/imgep_driver.py`, replace the import of `IMAGE_LOGIT_SCALE` and `TEXT_LOGIT_SCALE` with a pair of properties on `ImgepDriver`:

```python
    @property
    def _text_scale(self) -> float:
        return self.scorer.model.text_logit_scale

    @property
    def _image_scale(self) -> float:
        return self.scorer.model.image_logit_scale
```

Then replace every `logit_scale=IMAGE_LOGIT_SCALE` with `logit_scale=self._image_scale` and every `logit_scale=TEXT_LOGIT_SCALE` with `logit_scale=self._text_scale`. Find them with:

```bash
grep -n "LOGIT_SCALE" services/imgep_driver.py
```

- [ ] **Step 3: Update `main.py`'s service construction**

In `_ensure_auto_service`, replace:

```python
            from services.clip_scorer import CLIPScorer
```
with
```python
            from services.vision_scorer import VisionScorer
```

and

```python
            from tools.fetch_clip_onnx import MODEL_DIR, is_present
```
with
```python
            from tools.fetch_models import is_present
```

then replace:

```python
        if not is_present(MODEL_DIR):
            self.ui.auto_unavailable = "model_missing"
            return False

        try:
            self.clip_scorer = CLIPScorer(MODEL_DIR)
        except Exception as exc:
            self.ui.auto_unavailable = f"could not load CLIP: {exc}"
            return False
```

with:

```python
        key = self.ui.get_state().auto_tournament.model_key
        if not is_present(key):
            self.ui.auto_unavailable = "model_missing"
            return False

        if not self._ensure_scorer(key):
            return False
```

`_ensure_scorer` arrives in Task 10; for now add a stub beside it that Task 10 replaces:

```python
    def _ensure_scorer(self, model_key: str) -> bool:
        """One resident scorer. Replaced with the swapping version in Task 10."""
        from services.vision_scorer import VisionScorer

        if self.vision_scorer is not None and \
                self.vision_scorer.model.key == model_key:
            return True
        try:
            self.vision_scorer = VisionScorer(model_key)
        except Exception as exc:
            self.ui.auto_unavailable = f"could not load encoder: {exc}"
            return False
        return True
```

Rename the attribute: `self.clip_scorer` → `self.vision_scorer` everywhere in `main.py` (four sites; find with `grep -n clip_scorer main.py`). Add `model_key: str = "clip-b32"` to the auto-tournament state dataclass in `state/` (find it with `grep -rn "class AutoTournamentState" state/`).

- [ ] **Step 4: Run the affected tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_expedition_fitness.py tests/test_vision_scorer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/expedition_fitness.py services/imgep_driver.py main.py state/
git commit -m "refactor: logit scales come off the encoder, not a module constant"
```

---

## Task 5: Shared archive sampling

**Files:**
- Create: `tools/archive_sample.py`
- Modify: `tools/hue_nuisance.py`
- Test: `tests/test_archive_sample.py`

**Interfaces:**
- Consumes: `services.vision_models`, `utilities.paths.get_archives_root`.
- Produces: `recent_layouts(root, count) -> list[Path]`, `load_thumbs(layout_dir, n, px, rng) -> np.ndarray`, `nn_distance(emb, block=512) -> np.ndarray`, and `Encoder` (the timed ONNX wrapper, moved verbatim from `hue_nuisance`).

- [ ] **Step 1: Write the failing test**

```python
"""Sampling helpers shared by the two measurement tools."""
import numpy as np
import pytest
from PIL import Image

from tools.archive_sample import load_thumbs, nn_distance, recent_layouts


def _layout(root, archive, layout, n):
    d = root / archive / layout
    (d / "thumbs").mkdir(parents=True)
    (d / "index.jsonl").write_text("")
    for i in range(n):
        Image.fromarray(np.full((160, 160, 3), i * 7 % 256, np.uint8)).save(
            d / "thumbs" / f"{i:06d}.jpg")
    return d


def test_an_empty_sibling_layout_is_never_ranked(tmp_path):
    """Switching brain creates an empty sibling and touches it last, so mtime
    alone hands back a directory nothing was explored in."""
    _layout(tmp_path, "a", "fourier-n10", 3)
    empty = tmp_path / "a" / "gabor-n7"
    (empty / "thumbs").mkdir(parents=True)
    (empty / "index.jsonl").write_text("")
    got = recent_layouts(tmp_path, 3)
    assert [p.name for p in got] == ["fourier-n10"]


def test_thumbs_are_resized_to_the_models_input(tmp_path):
    d = _layout(tmp_path, "a", "fourier-n10", 4)
    out = load_thumbs(d, 4, 224, np.random.default_rng(0))
    assert out.shape == (4, 224, 224, 3)


def test_sampling_never_returns_more_than_asked(tmp_path):
    d = _layout(tmp_path, "a", "fourier-n10", 10)
    assert len(load_thumbs(d, 3, 224, np.random.default_rng(0))) == 3


def test_nn_distance_never_matches_a_row_with_itself():
    e = np.eye(4, dtype=np.float32)
    assert np.allclose(nn_distance(e), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_sample.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.archive_sample'`

- [ ] **Step 3: Move the helpers out of `hue_nuisance.py`**

Cut `recent_layouts`, `load_thumbs`, `nn_distance`, `Encoder` and `_ORT_DTYPES` from `tools/hue_nuisance.py` into `tools/archive_sample.py` unchanged, with this module docstring:

```python
"""Sampling an archive's thumbnails and embedding them, for the measurement
tools. Knows nothing about what is being measured.

Images are archive thumbnails upscaled to the encoder's input size. That
understates a fine-patch model's discrimination, but identically for every
model, so a comparison between them is fair.
"""
```

Change `Encoder.__init__` to take a `VisionModel` instead of loose `mean/std/px`:

```python
class Encoder:
    """One ONNX vision tower. Times only session.run."""

    def __init__(self, model, providers):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        self.model = model
        self.key = model.key
        self.px = model.px
        path = Path("models") / model.subdir / "vision_model_fp16.onnx"
        self.sess = ort.InferenceSession(str(path), opts, providers=providers)
        self.provider = self.sess.get_providers()[0]
        ...  # the rest unchanged, with mean/std read from self.model
```

- [ ] **Step 4: Point `hue_nuisance.py` at the helpers**

Replace its copies with:

```python
from tools.archive_sample import Encoder, load_thumbs, nn_distance, recent_layouts
from services.vision_models import REGISTRY, get
```

and delete its own `MODELS` list, `CLIP_MEAN`, `CLIP_STD`, `HALF` and `_ORT_DTYPES` — `build_encoders` now iterates `REGISTRY`.

- [ ] **Step 5: Run the tests and re-run the tool**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_sample.py -q`
Expected: PASS (4 tests)

Run: `.venv/Scripts/python.exe -m tools.hue_nuisance --entries 40 --only clip-b32`
Expected: same three archives discovered, a ratio near 1.0 at 74°, no traceback.

- [ ] **Step 6: Commit**

```bash
git add tools/archive_sample.py tools/hue_nuisance.py tests/test_archive_sample.py
git commit -m "refactor: archive sampling is shared by the measurement tools"
```

---

## Task 6: The calibration tool

**Files:**
- Create: `tools/calibrate_encoder.py`
- Test: `tests/test_calibrate_encoder.py`

**Interfaces:**
- Consumes: `tools.archive_sample`, `services.expedition_fitness.contrastive`, `services.vision_models`.
- Produces: `floored_fraction(descriptors, goal, references, scale) -> float`, `scale_matching(descriptors, goal, references, target_frac) -> float`, `retention(nn, threshold) -> float`, `threshold_matching(nn, target_frac) -> float`.

- [ ] **Step 1: Write the failing test**

```python
"""The two calibration criteria, on synthetic populations."""
import numpy as np

from tools.calibrate_encoder import (floored_fraction, retention,
                                     scale_matching, threshold_matching)


def _cone(rng, n=64, dim=64):
    base = rng.normal(size=dim).astype(np.float32)
    e = base + 0.15 * rng.normal(size=(n, dim)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_a_sharper_scale_floors_more_of_the_population():
    rng = np.random.default_rng(3)
    e = _cone(rng)
    c = e.mean(axis=0)
    c /= np.linalg.norm(c)
    goal = c + 3.0 * (e[0] - c)
    goal /= np.linalg.norm(goal)
    assert floored_fraction(e, goal, c[None, :], 100.0) > \
           floored_fraction(e, goal, c[None, :], 30.0)


def test_scale_matching_hits_the_target_fraction():
    rng = np.random.default_rng(5)
    e = _cone(rng)
    c = e.mean(axis=0)
    c /= np.linalg.norm(c)
    goal = c + 3.0 * (e[0] - c)
    goal /= np.linalg.norm(goal)
    s = scale_matching(e, goal, c[None, :], target_frac=0.014)
    assert abs(floored_fraction(e, goal, c[None, :], s) - 0.014) < 0.02


def test_retention_falls_as_the_threshold_rises():
    nn = np.linspace(0.0, 0.1, 200).astype(np.float32)
    assert retention(nn, 0.01) > retention(nn, 0.05)


def test_threshold_matching_recovers_a_known_threshold():
    nn = np.linspace(0.0, 0.1, 500).astype(np.float32)
    target = retention(nn, 0.02)
    assert abs(threshold_matching(nn, target) - 0.02) < 0.002
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_calibrate_encoder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.calibrate_encoder'`

- [ ] **Step 3: Write the tool**

```python
"""Calibrate one encoder's scales and separation default against real archives.

    python -m tools.calibrate_encoder --model siglip2-b16

Equates on BEHAVIOUR, not on a summary statistic: each value is chosen so it
reproduces on this population what clip-b32's value does. The criteria and the
numbers they produced live in CLAUDE.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from services.expedition_fitness import contrastive
from services.vision_models import REGISTRY, get
from tools.archive_sample import Encoder, load_thumbs, nn_distance, recent_layouts
from utilities.paths import get_archives_root

# clip-b32's behaviour, the target every other encoder is matched to.
REFERENCE_KEY = "clip-b32"


def floored_fraction(descriptors, goal, references, scale: float) -> float:
    """Fraction of the population whose contrastive fitness saturates to zero.

    A floored tile carries no information to a rank-based optimizer.
    """
    f = contrastive(np.asarray(descriptors)[None, ...], goal, references,
                    logit_scale=float(scale))
    return float(np.mean(f <= 1e-6))


def scale_matching(descriptors, goal, references, target_frac: float,
                   lo: float = 1.0, hi: float = 200.0) -> float:
    """The logit scale whose floored fraction is target_frac. Bisection, because
    floored fraction is monotone in the scale."""
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if floored_fraction(descriptors, goal, references, mid) > target_frac:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def retention(nn: np.ndarray, threshold: float) -> float:
    """Fraction of a population that a separation bar would admit."""
    return float(np.mean(np.asarray(nn) >= threshold))


def threshold_matching(nn: np.ndarray, target_frac: float,
                       lo: float = 0.0, hi: float = 0.5) -> float:
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if retention(nn, mid) < target_frac:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def latent_goal(e: np.ndarray):
    """The +3sd construction recorded in tests/test_expedition_fitness.py."""
    c = e.mean(axis=0)
    c /= np.linalg.norm(c)
    g = c + 3.0 * (e[0] - c)
    return g / np.linalg.norm(g), c[None, :]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="", help="blank means every registered")
    ap.add_argument("--entries", type=int, default=700)
    ap.add_argument("--archives", type=int, default=3)
    ap.add_argument("--out", default="encoder_calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import onnxruntime as ort
    have = set(ort.get_available_providers())
    providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                 if p in have] or ["CPUExecutionProvider"]

    keys = [args.model] if args.model else sorted(REGISTRY)
    if REFERENCE_KEY not in keys:
        keys.insert(0, REFERENCE_KEY)

    layouts = recent_layouts(get_archives_root(), args.archives)
    rng = np.random.default_rng(args.seed)
    samples = {p: load_thumbs(p, args.entries, get(REFERENCE_KEY).px, rng)
               for p in layouts}

    out = {}
    for key in keys:
        model = get(key)
        enc = Encoder(model, providers)
        per_archive = []
        for path, imgs in samples.items():
            e = enc.embed(imgs)
            goal, refs = latent_goal(e)
            per_archive.append({
                "archive": f"{path.parent.name}/{path.name}",
                "nn": nn_distance(e),
                "goal": goal, "refs": refs, "e": e,
            })
        out[key] = per_archive

    ref = out[REFERENCE_KEY]
    target_floor = float(np.mean([
        floored_fraction(a["e"], a["goal"], a["refs"],
                         get(REFERENCE_KEY).image_logit_scale)
        for a in ref]))
    target_keep = float(np.mean([
        retention(a["nn"], get(REFERENCE_KEY).default_min_separation)
        for a in ref]))
    print(f"reference: floored={target_floor:.4f}  retained={target_keep:.4f}")

    report = {"target_floored": target_floor, "target_retained": target_keep,
              "models": {}}
    for key, rows in out.items():
        scales = [scale_matching(a["e"], a["goal"], a["refs"], target_floor)
                  for a in rows]
        seps = [threshold_matching(a["nn"], target_keep) for a in rows]
        report["models"][key] = {
            "image_logit_scale": float(np.median(scales)),
            "default_min_separation": float(np.median(seps)),
            "per_archive_scale": [float(s) for s in scales],
            "per_archive_separation": [float(s) for s in seps],
        }
        print(f"{key:12s} image_logit_scale={np.median(scales):7.2f}  "
              f"default_min_separation={np.median(seps):.4f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_calibrate_encoder.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/calibrate_encoder.py tests/test_calibrate_encoder.py
git commit -m "feat: a tool that calibrates an encoder against clip-b32's behaviour"
```

---

## Task 7: Run the calibration and fill the registry

**Files:**
- Modify: `services/vision_models.py`, `tests/test_vision_models.py`, `CLAUDE.md`

**Interfaces:**
- Consumes: `tools/calibrate_encoder.py`.
- Produces: every `REGISTRY` entry has non-`None` `text_logit_scale`, `image_logit_scale`, `default_min_separation`.

- [ ] **Step 1: Download the text towers and tokenizers**

The vision towers for all four are already on disk; the text halves are not.

```bash
.venv/Scripts/python.exe -m tools.fetch_models clip-b16
.venv/Scripts/python.exe -m tools.fetch_models siglip2-b16
.venv/Scripts/python.exe -m tools.fetch_models clip-l14
```

Expected: each prints per-file progress to 100% and a "ready in models/..." line. SigLIP 2's `tokenizer.json` is 34.4 MB and is the slowest of its three.

- [ ] **Step 2: Run the calibration**

```bash
.venv/Scripts/python.exe -m tools.calibrate_encoder --entries 700
```

Expected: a `reference:` line, then one line per model, then `wrote encoder_calibration.json`. Runtime is roughly 15 minutes — `clip-l14` is 23x `clip-b32` per image.

- [ ] **Step 3: Sanity-check the output before trusting it**

`clip-b32`'s own reported values must come back as its shipped constants, because it is being matched against itself. If `image_logit_scale` is not within 1.0 of 30.0, or `default_min_separation` not within 0.001 of 0.02, the harness is wrong — **stop and report** rather than editing the registry.

- [ ] **Step 4: Fill the registry**

Copy each model's `image_logit_scale` (rounded to 1 decimal) and `default_min_separation` (rounded to 4 decimals) into its `VisionModel(...)` call in `services/vision_models.py`. For `siglip2-b16` also set `text_logit_scale` from the same run; the other three keep 100.0.

- [ ] **Step 5: Add the completeness test**

Append to `tests/test_vision_models.py`:

```python
@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_entry_is_calibrated(key):
    """A model shipping an uncalibrated scale would silently saturate the
    landscape. See CLAUDE.md."""
    assert REGISTRY[key].calibrated, f"{key} was never calibrated"


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_the_separation_slider_has_room_above_the_default(key):
    m = REGISTRY[key]
    assert m.separation_slider_max > m.default_min_separation


def test_clip_b32s_slider_range_is_unchanged():
    """The existing 0..0.05 track is what 2.5x has to reproduce."""
    assert get("clip-b32").separation_slider_max == pytest.approx(0.05)
```

- [ ] **Step 6: Record the numbers in CLAUDE.md**

Add to the "CLIP and the capture" section (which Task 13 renames), stating the rule and the criteria — the table of values, the two criteria, and `python -m tools.calibrate_encoder` as the re-measure command. No rationale essay in code comments; this is the home.

- [ ] **Step 7: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vision_models.py tests/test_expedition_fitness.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add services/vision_models.py tests/test_vision_models.py CLAUDE.md
git commit -m "feat: every encoder's scales and separation bar are measured"
```

---

## Task 8: `encoder.json` — the archive's pinned encoder

**Files:**
- Modify: `services/archive_io.py`
- Test: `tests/test_archive_encoder.py`

**Interfaces:**
- Consumes: `services.vision_models.DEFAULT_KEY`, `get`.
- Produces: on `ArchiveStore`: `encoder_path` property, `encoder` property (`str`, defaulting to `DEFAULT_KEY`), `save_encoder(model_key) -> bool` (refuses to overwrite).

- [ ] **Step 1: Write the failing test**

```python
"""An archive name identifies one embedding space, permanently."""
import json

import pytest

from services.archive_io import ArchiveStore
from services.vision_models import DEFAULT_KEY


def _store(tmp_path):
    return ArchiveStore(tmp_path / "arc")


def test_an_archive_with_no_file_reads_as_the_original_encoder(tmp_path):
    """Every archive that exists predates the choice and is clip-b32."""
    assert _store(tmp_path).encoder == DEFAULT_KEY


def test_the_encoder_is_written_once(tmp_path):
    s = _store(tmp_path)
    assert s.save_encoder("siglip2-b16") is True
    assert s.encoder == "siglip2-b16"


def test_a_second_write_is_refused_and_changes_nothing(tmp_path):
    """A second write would reinterpret every entry already filed here."""
    s = _store(tmp_path)
    s.save_encoder("siglip2-b16")
    assert s.save_encoder("clip-l14") is False
    assert s.encoder == "siglip2-b16"


def test_an_unknown_encoder_is_never_written(tmp_path):
    s = _store(tmp_path)
    assert s.save_encoder("not-a-model") is False
    assert not s.encoder_path.exists()


def test_a_corrupt_file_reads_as_the_default_rather_than_raising(tmp_path):
    """A disk problem must never stop the search."""
    s = _store(tmp_path)
    s.encoder_path.write_text("{ this is not json", encoding="utf-8")
    assert s.encoder == DEFAULT_KEY


def test_the_encoder_lives_beside_goals_not_under_the_layout(tmp_path):
    """One archive holds every brain layout, and they share an encoder."""
    s = _store(tmp_path)
    s.save_encoder("clip-b16")
    assert s.encoder_path.parent == s.goals_path.parent
    assert s.encoder_path.parent != s.root
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_encoder.py -q`
Expected: FAIL — `AttributeError: 'ArchiveStore' object has no attribute 'encoder'`

- [ ] **Step 3: Implement on `ArchiveStore`**

Add beside `settings_path` (search for `def settings_path`):

```python
    @property
    def encoder_path(self) -> Path:
        return self.base / "encoder.json"
```

and in the reading/writing sections:

```python
    def save_encoder(self, model_key: str) -> bool:
        """Pin this archive's encoder. -> whether a file is now on disk for it.

        Never overwrites: an archive name identifies ONE embedding space, and a
        second write would reinterpret every entry already filed under it.
        """
        from services.vision_models import REGISTRY

        if not self.enabled or model_key not in REGISTRY:
            return False
        if self.encoder_path.exists():
            return False
        try:
            tmp = self.encoder_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"encoder": model_key, "created": time.time()}),
                encoding="utf-8")
            os.replace(tmp, self.encoder_path)
            return True
        except OSError as exc:
            print(f"[Archive] encoder not pinned ({exc})")
            return False

    @property
    def encoder(self) -> str:
        """-> the pinned encoder key. A missing file means the encoder every
        archive written before this used."""
        from services.vision_models import DEFAULT_KEY, REGISTRY

        try:
            data = json.loads(self.encoder_path.read_text(encoding="utf-8"))
            key = str(data["encoder"])
        except (OSError, ValueError, KeyError, TypeError):
            return DEFAULT_KEY
        return key if key in REGISTRY else DEFAULT_KEY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_encoder.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add services/archive_io.py tests/test_archive_encoder.py
git commit -m "feat: an archive pins its encoder, once and permanently"
```

---

## Task 9: The archive refuses a mismatched encoder

**Files:**
- Modify: `services/archive.py`
- Test: `tests/test_archive_encoder.py` (append)

**Interfaces:**
- Consumes: `ArchiveStore.encoder`, `VisionModel.dim`.
- Produces: `Archive.__init__` gains `encoder: str = DEFAULT_KEY` and derives `dim` from it when `dim` is not given. `Archive.load_from_store()` returns `(ok: bool, message: str)`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_archive_takes_its_width_from_its_encoder():
    from services.archive import Archive

    assert Archive(encoder="siglip2-b16")._dim == 768
    assert Archive(encoder="clip-b32")._dim == 512


def test_loading_a_store_under_the_wrong_encoder_is_refused(tmp_path):
    """A 512-d and a 768-d vector describe the same tile in incompatible
    spaces, and nothing downstream would notice."""
    from services.archive import Archive

    s = ArchiveStore(tmp_path / "arc")
    s.save_encoder("siglip2-b16")
    a = Archive(store=s, encoder="clip-b32")
    ok, message = a.load_from_store()
    assert ok is False
    assert "siglip2-b16" in message and "clip-b32" in message


def test_loading_a_matching_store_succeeds(tmp_path):
    from services.archive import Archive

    s = ArchiveStore(tmp_path / "arc")
    s.save_encoder("clip-b16")
    ok, _ = Archive(store=s, encoder="clip-b16").load_from_store()
    assert ok is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_encoder.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'encoder'`

- [ ] **Step 3: Implement**

In `services/archive.py`, change the signature (search for `liveness_min: float = 0.002, dim: int = 512,`):

```python
                 liveness_min: float = 0.002, dim: int | None = None,
                 encoder: str = DEFAULT_KEY,
```

and in the body, before `self._dim = int(dim)`:

```python
        from services.vision_models import get

        self.encoder = str(encoder)
        if dim is None:
            dim = get(self.encoder).dim
```

Add at the top of `load_from_store`:

```python
        if self.store is not None and self.store.encoder != self.encoder:
            return False, (
                f"this archive was built with {self.store.encoder}; "
                f"the search is running {self.encoder}")
```

and make every other `return` in `load_from_store` return `(True, "")`.

- [ ] **Step 4: Fix the caller**

`main.py`'s `_build_archive_set` calls `load_from_store()`. Update it to unpack and surface the message:

```python
        ok, message = archive.load_from_store()
        if not ok:
            self.ui.archive_unavailable = message
            return False
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_encoder.py tests/test_archive_switch.py tests/test_archive_id_reuse.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/archive.py main.py tests/test_archive_encoder.py
git commit -m "feat: an archive refuses to load under the wrong encoder"
```

---

## Task 10: One resident scorer, following the active mode

**Files:**
- Modify: `main.py`
- Test: `tests/test_scorer_lifecycle.py`

**Interfaces:**
- Consumes: `VisionScorer`, `ArchiveStore.encoder`.
- Produces: `App._ensure_scorer(model_key) -> bool` replaces the Task 4 stub; `App.vision_scorer` holds at most one instance.

- [ ] **Step 1: Write the failing test**

```python
"""Switching encoders replaces the resident scorer rather than stacking them."""
import pytest

from main import App


class FakeScorer:
    built = 0

    def __init__(self, key):
        FakeScorer.built += 1
        self.model = type("M", (), {"key": key})()


class FakeApp:
    def __init__(self, scorer=None):
        self.vision_scorer = scorer
        self.ui = type("U", (), {"auto_unavailable": ""})()


def _ensure(app, key, monkeypatch):
    monkeypatch.setattr("services.vision_scorer.VisionScorer", FakeScorer)
    return App._ensure_scorer(app, key)


def test_the_same_key_reuses_the_resident_scorer(monkeypatch):
    FakeScorer.built = 0
    app = FakeApp(FakeScorer("clip-b32"))
    assert _ensure(app, "clip-b32", monkeypatch) is True
    assert FakeScorer.built == 1, "a matching key must not rebuild"


def test_a_different_key_replaces_it(monkeypatch):
    FakeScorer.built = 0
    app = FakeApp(FakeScorer("clip-b32"))
    assert _ensure(app, "siglip2-b16", monkeypatch) is True
    assert app.vision_scorer.model.key == "siglip2-b16"
    assert FakeScorer.built == 2


def test_a_failure_leaves_no_half_built_scorer(monkeypatch):
    def boom(key):
        raise RuntimeError("no such file")

    monkeypatch.setattr("services.vision_scorer.VisionScorer", boom)
    app = FakeApp()
    assert App._ensure_scorer(app, "clip-l14") is False
    assert app.vision_scorer is None
    assert "no such file" in app.ui.auto_unavailable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scorer_lifecycle.py -q`
Expected: FAIL on `test_a_failure_leaves_no_half_built_scorer` — the Task 4 stub assigns before it can raise.

- [ ] **Step 3: Replace the stub**

```python
    def _ensure_scorer(self, model_key: str) -> bool:
        """One resident scorer, replaced rather than stacked.

        Keeping every encoder loaded would cost about a gigabyte of weights for
        a switch that happens once per archive.
        """
        import services.vision_scorer as vs

        if (self.vision_scorer is not None
                and self.vision_scorer.model.key == model_key):
            return True
        try:
            built = vs.VisionScorer(model_key)
        except Exception as exc:
            self.ui.auto_unavailable = f"could not load encoder: {exc}"
            return False
        self.vision_scorer = built
        if self.auto_service is not None:
            self.auto_service.scorer = built
        return True
```

- [ ] **Step 4: Route Explore's archive through it**

In `_build_archive_set`, before constructing the `Archive`, adopt the store's encoder:

```python
        if not self._ensure_scorer(store.encoder):
            return False
        archive = Archive(store=store, encoder=store.encoder)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scorer_lifecycle.py -q`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_scorer_lifecycle.py
git commit -m "feat: the resident encoder follows whichever mode is scoring"
```

---

## Task 11: The settings history store

**Files:**
- Create: `services/settings_history.py`
- Modify: `services/archive_io.py`
- Test: `tests/test_settings_history.py`

**Interfaces:**
- Consumes: `state.archive_state.PERSISTED_FIELDS`.
- Produces: `diff(previous: dict, current: dict) -> dict` (field → `[old, new]`); on `ArchiveStore`: `history_path`, `append_history(row) -> None`, `load_history() -> list[dict]`, `latest_version() -> int`.

- [ ] **Step 1: Write the failing test**

```python
"""A settings change is recorded once, with enough context to place it."""
from services.archive_io import ArchiveStore
from services.settings_history import diff


def test_no_change_produces_no_diff():
    assert diff({"a": 1}, {"a": 1}) == {}


def test_a_change_records_old_and_new():
    assert diff({"a": 1}, {"a": 2}) == {"a": [1, 2]}


def test_a_new_field_records_none_as_its_old_value():
    assert diff({}, {"a": 2}) == {"a": [None, 2]}


def test_a_removed_field_is_not_a_change():
    """PERSISTED_FIELDS shrinking is a code change, not a user action."""
    assert diff({"a": 1}, {}) == {}


def test_version_zero_carries_the_whole_block(tmp_path):
    s = ArchiveStore(tmp_path / "arc")
    s.append_history({"v": 0, "ts": 1.0, "gen": 0, "entries": 0,
                      "full": {"min_separation": 0.02}})
    rows = s.load_history()
    assert rows[0]["v"] == 0 and "full" in rows[0]


def test_versions_increase_and_latest_reports_the_last(tmp_path):
    s = ArchiveStore(tmp_path / "arc")
    for v in range(3):
        s.append_history({"v": v, "ts": float(v), "gen": v, "entries": 0,
                          "changed": {}})
    assert s.latest_version() == 2


def test_an_empty_history_reports_version_minus_one(tmp_path):
    """Nothing written yet, so the next row is version 0."""
    assert ArchiveStore(tmp_path / "arc").latest_version() == -1


def test_a_torn_trailing_line_costs_one_row_not_the_file(tmp_path):
    s = ArchiveStore(tmp_path / "arc")
    s.append_history({"v": 0, "ts": 1.0, "gen": 0, "entries": 0, "full": {}})
    with open(s.history_path, "a", encoding="utf-8") as fh:
        fh.write('{"v": 1, "ts"')
    assert len(s.load_history()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_settings_history.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.settings_history'`

- [ ] **Step 3: Write `services/settings_history.py`**

```python
"""What changed in an archive's explore settings, and when.

settings.json is rewritten wholesale, so the value that admitted a given entry
is unrecoverable the moment a slider moves. This is the record that makes it
recoverable. It never restores anything - putting a setting back is the user
moving the slider.
"""
from __future__ import annotations


def diff(previous: dict, current: dict) -> dict:
    """-> {field: [old, new]} for every field whose value moved.

    A field absent from `current` is NOT a change: PERSISTED_FIELDS shrinking is
    a code change, and recording it would put a phantom row in every archive on
    the first run after it.
    """
    out = {}
    for key, new in current.items():
        old = previous.get(key)
        if key not in previous or old != new:
            out[key] = [old, new]
    return out
```

- [ ] **Step 4: Add the store methods**

Beside `settings_path` in `services/archive_io.py`:

```python
    @property
    def history_path(self) -> Path:
        return self.base / "settings_history.jsonl"

    def append_history(self, row: dict) -> None:
        if not self.enabled:
            return
        try:
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except (OSError, TypeError) as exc:
            print(f"[Archive] settings history not written ({exc})")

    def load_history(self) -> list[dict]:
        rows = []
        try:
            text = self.history_path.read_text(encoding="utf-8")
        except OSError:
            return rows
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue        # a torn trailing line is one lost row
        return rows

    def latest_version(self) -> int:
        """-> the highest version on disk, or -1 for an archive with none."""
        rows = self.load_history()
        return max((int(r.get("v", -1)) for r in rows), default=-1)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_settings_history.py -q`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add services/settings_history.py services/archive_io.py tests/test_settings_history.py
git commit -m "feat: an archive records what changed in its settings, and when"
```

---

## Task 12: Stamping entries with their config version

**Files:**
- Modify: `services/archive.py`, `main.py`
- Test: `tests/test_settings_history.py` (append)

**Interfaces:**
- Consumes: `ArchiveStore.append_history`, `latest_version`, `services.settings_history.diff`.
- Produces: `Archive.cfg_version: int`; `Archive.record_settings(current: dict, gen: int) -> int` returns the version in force after recording; `ArchiveEntry.cfg: int`; `index.jsonl` rows carry `"cfg"`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_first_record_writes_version_zero_with_the_full_block(tmp_path):
    from services.archive import Archive

    s = ArchiveStore(tmp_path / "arc")
    a = Archive(store=s)
    assert a.record_settings({"min_separation": 0.02}, gen=0) == 0
    rows = s.load_history()
    assert rows[0]["full"]["min_separation"] == 0.02


def test_an_unchanged_generation_writes_nothing(tmp_path):
    from services.archive import Archive

    s = ArchiveStore(tmp_path / "arc")
    a = Archive(store=s)
    a.record_settings({"min_separation": 0.02}, gen=0)
    assert a.record_settings({"min_separation": 0.02}, gen=1) == 0
    assert len(s.load_history()) == 1


def test_a_change_bumps_the_version_and_records_the_move(tmp_path):
    from services.archive import Archive

    s = ArchiveStore(tmp_path / "arc")
    a = Archive(store=s)
    a.record_settings({"min_separation": 0.02}, gen=0)
    assert a.record_settings({"min_separation": 0.03}, gen=7) == 1
    row = s.load_history()[1]
    assert row["changed"]["min_separation"] == [0.02, 0.03]
    assert row["gen"] == 7


def test_an_entry_written_without_a_version_reads_as_zero(tmp_path):
    """Every entry admitted before this existed."""
    from services.archive import ArchiveEntry

    assert ArchiveEntry(**_minimal_entry_kwargs()).cfg == 0
```

Replace `_minimal_entry_kwargs()` with the actual constructor arguments `ArchiveEntry` requires — read them off the dataclass at the top of `services/archive.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_settings_history.py -q`
Expected: FAIL — `AttributeError: 'Archive' object has no attribute 'record_settings'`

- [ ] **Step 3: Implement on `Archive`**

Add `cfg: int = 0` to the `ArchiveEntry` dataclass, and in `__init__`:

```python
        self.cfg_version = max(0, self.store.latest_version()) if self.store else 0
        self._last_settings: dict | None = None
```

and the method:

```python
    def record_settings(self, current: dict, gen: int) -> int:
        """Append a settings version if anything moved. -> the version in force.

        Called once per generation and once more when the archive closes, never
        per frame.
        """
        from services.settings_history import diff

        if self.store is None:
            return self.cfg_version
        data = dict(current)
        data["encoder"] = self.encoder
        if self._last_settings is None and self.store.latest_version() < 0:
            self.store.append_history({
                "v": 0, "ts": time.time(), "gen": int(gen),
                "entries": len(self.entries), "full": data})
            self._last_settings = data
            self.cfg_version = 0
            return 0
        if self._last_settings is None:
            rows = self.store.load_history()
            base = dict(rows[0].get("full", {})) if rows else {}
            for row in rows[1:]:
                for k, (_old, new) in row.get("changed", {}).items():
                    base[k] = new
            self._last_settings = base
        moved = diff(self._last_settings, data)
        if not moved:
            return self.cfg_version
        self.cfg_version = self.store.latest_version() + 1
        self.store.append_history({
            "v": self.cfg_version, "ts": time.time(), "gen": int(gen),
            "entries": len(self.entries), "changed": moved})
        self._last_settings = data
        return self.cfg_version
```

In `_add`, set `cfg=self.cfg_version` on the new `ArchiveEntry` and include `"cfg": self.cfg_version` in the `append_index` row. In the load path (search for `thumb=str(r.get("thumb", ""))`), add `cfg=int(r.get("cfg", 0))`.

- [ ] **Step 4: Call it once per generation and on close**

In `main.py`'s explore handling, after the generation completes and before `maybe_flush`, and in the archive-close path:

```python
            self.archive.record_settings(
                ui_state.archive.persisted_snapshot(), gen=self.archive.generation)
```

Add `persisted_snapshot()` to `ArchiveState` in `state/archive_state.py`:

```python
    def persisted_snapshot(self) -> dict:
        """The persisted fields, by value - the same allowlist settings.json
        uses, so the history and the settings file cannot disagree."""
        return {name: getattr(self, name) for name in PERSISTED_FIELDS}
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_settings_history.py tests/test_archive_encoder.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/archive.py state/archive_state.py main.py tests/test_settings_history.py
git commit -m "feat: entries record the settings version they were admitted under"
```

---

## Task 13: UI — encoder pickers, per-encoder separation, history table, rename

**Files:**
- Modify: `ui/archive_window.py`, `ui/auto_tournament_window.py`, `ui/tournament_window.py`, `ui/README.md`
- Test: `tests/test_archive_window_render.py` (append)

**Interfaces:**
- Consumes: `services.vision_models.REGISTRY`, `tools.fetch_models.is_present`, `ArchiveStore.load_history`.
- Produces: no new public API; UI state fields `ArchiveState.encoder_key`, `AutoTournamentState.model_key`, `ArchiveState.show_history`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_encoder_combo_is_disabled_once_entries_exist(ui_fixture):
    """The encoder is archive-level, so a populated fourier-n10 locks the
    choice for an empty gabor-n7 sibling."""
    ui_fixture.state.archive.archive_entry_count = 12
    render_archive_window(ui_fixture)
    assert ui_fixture.disabled_items("Encoder") is True


def test_the_encoder_combo_is_editable_while_the_archive_is_empty(ui_fixture):
    ui_fixture.state.archive.archive_entry_count = 0
    render_archive_window(ui_fixture)
    assert ui_fixture.disabled_items("Encoder") is False


def test_the_separation_slider_max_follows_the_encoder(ui_fixture):
    ui_fixture.state.archive.encoder_key = "clip-l14"
    render_archive_window(ui_fixture)
    lo, hi = ui_fixture.slider_range("Min Separation")
    assert hi == pytest.approx(get("clip-l14").separation_slider_max)


def test_no_label_still_says_clip(ui_fixture):
    """The scorer is not CLIP-only any more; a label that says so is a lie."""
    render_archive_window(ui_fixture)
    assert not any("CLIP" in s for s in ui_fixture.rendered_strings())
```

Match the existing fixture style in `tests/test_archive_window_render.py` — read it first and reuse its harness rather than inventing `ui_fixture`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_window_render.py -q`
Expected: FAIL on the new cases.

- [ ] **Step 3: Rename the user-facing strings**

- `ui/tournament_window.py`: `imgui.begin_tab_item("Auto (CLIP)")` → `imgui.begin_tab_item("Auto (Prompt)")`
- `ui/auto_tournament_window.py`: module docstring `"""Auto (CLIP) tab..."""` → `"""Auto (Prompt) tab..."""`; `"CLIP model weights are not downloaded."` → `"Encoder weights are not downloaded."`; `"Download CLIP model (~330 MB)"` → a per-model label built from the selected key; `"upscaled to 224 for CLIP"` → `"upscaled to 224 for the encoder"`
- `ui/archive_window.py`: `SEED_TOOLTIP` → `"Starts Auto (Prompt) mode's search from this genome."`; `"Explore mode needs the same CLIP model and packages as Auto mode. "` → `"...the same encoder and packages..."`; `imgui.slider_int("CLIP Views", ...)` → `imgui.slider_int("Encoder Views", ...)`
- `ui/README.md`: the `auto_tournament_window.py` row.

Leave `Config Clipboard` and the `ListClipper` comment alone — unrelated uses of the substring.

- [ ] **Step 4: Add the encoder combo and the per-encoder slider**

In `ui/archive_window.py`, beside the archive combo (use `##` suffixes — an ImGui widget's identity is its label, and a duplicate silently kills the loser):

```python
        keys = sorted(REGISTRY)
        locked = ast.archive_entry_count > 0
        if locked:
            imgui.begin_disabled()
        idx = keys.index(ast.encoder_key) if ast.encoder_key in keys else 0
        changed, idx = imgui.combo("Encoder##archive", idx,
                                   [REGISTRY[k].label for k in keys])
        if changed:
            ast.encoder_key = keys[idx]
            ast.min_separation = REGISTRY[ast.encoder_key].default_min_separation
        if locked:
            imgui.end_disabled()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Fixed once an archive holds entries - its vectors are in "
                    "this encoder's space.")
```

and replace the hardcoded separation slider:

```python
        model = REGISTRY[ast.encoder_key]
        _, ast.min_separation = imgui.slider_float(
            "Min Separation", ast.min_separation, 0.0,
            model.separation_slider_max, "%.4f")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Refuses anything this close to a stored entry. "
                f"{model.label} calibrates at {model.default_min_separation:.4f}.")
```

- [ ] **Step 5: Add the history table**

A collapsing section in the Archive tab, reading `store.load_history()` on open rather than per frame:

```python
        if imgui.collapsing_header("Config History##archive")[0]:
            if imgui.begin_table("cfg_history", 4):
                for name in ("Version", "Generation", "Entries", "Changed"):
                    imgui.table_setup_column(name)
                imgui.table_headers_row()
                for row in ast.history_rows:
                    imgui.table_next_row()
                    imgui.table_next_column(); imgui.text(str(row.get("v", 0)))
                    imgui.table_next_column(); imgui.text(str(row.get("gen", 0)))
                    imgui.table_next_column(); imgui.text(str(row.get("entries", 0)))
                    imgui.table_next_column()
                    imgui.text(", ".join(sorted(row.get("changed", {})))
                               or "initial settings")
                imgui.end_table()
```

Add `encoder_key: str = "clip-b32"`, `archive_entry_count: int = 0`, `history_rows: list = field(default_factory=list)` and `show_history: bool = False` to `ArchiveState`. Add `encoder_key` and `show_history` to `PERSISTED_FIELDS`; **do not** add `history_rows` or `archive_entry_count` — they are view buffers, and the allowlist exists to keep those out.

- [ ] **Step 6: Add the Auto tab's model combo**

In `ui/auto_tournament_window.py`, above the download button, a free combo bound to `AutoTournamentState.model_key`, with the download button labelled from `is_present(key)` and the model's own total size.

- [ ] **Step 7: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_window_render.py tests/test_label_widths.py -q`
Expected: PASS. `test_label_widths` matters — "Encoder Views" and "Min Separation" must fit `WIDEST_LABEL`.

- [ ] **Step 8: Commit**

```bash
git add ui/ tests/test_archive_window_render.py state/archive_state.py
git commit -m "feat: pick an encoder per archive, and per prompt run"
```

---

## Task 14: Documentation

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `ARCHITECTURE.md`

- [ ] **Step 1: Rename the CLAUDE.md section and add the encoder caveats**

Retitle `### CLIP and the capture` to `### The encoder and the capture`. Add caveats stating: the registry key is written to disk and can never be renamed; an archive pins one encoder permanently and a missing `encoder.json` means `clip-b32`; the scales and separation bar are calibrated per encoder by `python -m tools.calibrate_encoder` against `clip-b32`'s behaviour, with the two criteria named; the Min Separation slider's track is per-encoder because a wider-spread encoder would otherwise put its own default at the top of it.

Update the existing `min_separation (0.02, measured)` caveat to say the number is `clip-b32`'s and point at the registry.

- [ ] **Step 2: Update the user-facing docs**

`README.md`: the tournament walkthrough names the tab "Auto (Prompt)" and describes choosing an encoder when creating an archive. `ARCHITECTURE.md`: `services/clip_scorer.py` → `services/vision_scorer.py`, plus the new modules.

- [ ] **Step 3: Verify nothing still points at the old names**

```bash
grep -rn "clip_scorer\|CLIPScorer\|fetch_clip_onnx" --include=*.py --include=*.md .
```
Expected: hits only in `docs/superpowers/` history and the CLAUDE.md sentence describing the rename.

- [ ] **Step 4: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. **If anything fails, run `git status --porcelain` first** — a failure in `test_brain_modalities_gpu.py` or any module touching `services/brains/` belongs to the concurrent session, not to this plan.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md ARCHITECTURE.md
git commit -m "docs: the encoder is a per-archive choice, calibrated per model"
```

---

## Self-Review

**Spec coverage.** §1 registry → Task 1. §1a Phase 0 → Tasks 5, 6, 7. §2 scorer → Tasks 2, 4. §3 downloads → Task 3. §4 encoder pinning → Tasks 8, 9. §5 scorer lifecycle → Task 10. §6 config history → Tasks 11, 12. §7 per-encoder separation → Tasks 7 (values), 13 (slider and seeding). §8 testing → distributed. Migration → Tasks 8, 9, 12 all assert the missing-file/missing-field defaults. Non-goals are not implemented, as intended.

**Known gap, deliberate.** The spec's `text_logit_scale` calibration for SigLIP against text goals with `DEFAULT_DISTRACTORS` is measured in Task 7 but `tools/calibrate_encoder.py` as written sweeps the image construction only. Task 6's `scale_matching` is modality-agnostic — it takes whatever goal and references it is handed — so the text pass is the same call with the text goal's embedding and the distractor set. If Task 7 Step 2's output lacks a text scale for `siglip2-b16`, extend `main()` there rather than shipping 100.0 unexamined.

**Type consistency.** `_ensure_scorer` returns `bool` in Tasks 4, 10 and is called for its truth value in both. `load_from_store` returns `(bool, str)` from Task 9 and every caller unpacks it. `record_settings` returns `int` and is used as the version in Task 12. `separation_slider_max` is defined in Task 1 and consumed in Tasks 7 and 13. `persisted_snapshot()` is defined in Task 12 and used only there.
