# Selectable vision encoders, pinned per archive, with a config change log

**Date:** 2026-08-12
**Status:** design approved, not yet implemented

## Problem

The search has exactly one pair of eyes. `CLIPScorer` hardcodes CLIP ViT-B/32:
its preprocessing constants, its 77-token BPE tokenizer, its `logit_scale` of
100, and its 512-d output are module globals, and `models/clip-vit-b32` is the
only path anything looks at. Trying a different encoder means editing the
scorer and invalidating every archive.

Three encoders are now known to work on this machine through ONNX/DirectML, and
they are not interchangeable at runtime: an embedding is only comparable to
others from the same model, so an archive's stored vectors pin it to the
encoder that produced them.

Separately, an archive's explore settings are persisted but not *historied*.
`settings.json` is rewritten wholesale, so the value of `min_separation` that
admitted entry #4000 is unrecoverable the moment the slider moves.

This design makes the encoder a first-class, per-archive, immutable choice;
lets both automatic modes pick a model; and records every settings change with
enough context to say what any entry was admitted under.

## Motivating measurements

From `python -m tools.hue_nuisance --entries 700` on 2026-08-12, over the three
most recently worked archives (`debug11-mlp/mlp-n16-a0`,
`debug10-gabor/gabor-n7`, `debug09/fourier-n10`), 2100 entries per model,
DirectML on the discrete GPU.

| model | dim | ms/image | vs b32 | median NN distance |
|---|---|---|---|---|
| clip-b32 | 512 | 1.16 | 1.0x | 0.0405 |
| clip-b16 | 512 | 5.61 | 4.9x | 0.0375 |
| siglip2-b16 | 768 | 6.18 | 5.3x | 0.0483 |
| clip-l14 | 768 | 26.56 | 23x | 0.0892 |

Two consequences shape the design.

**Distances are not comparable across models.** Median nearest-neighbour
distance spans 2.4x, so `min_separation = 0.02` is a materially stricter bar
under L/14 than under B/32. Every threshold calibrated in one encoder's space
is wrong in another's.

**Cost is not comparable either.** L/14 is 23x B/32 per image, which is why the
model is a user choice rather than an upgrade: at grid 8 it is the difference
between a generation that scores in under a second and one that takes half a
minute.

Both figures come from 160px thumbnails upscaled to 224, which understates a
fine-patch model's discrimination. The comparison is fair because the handicap
is identical for every model, but the absolute NN distances are a floor.

## Scope

In scope: a model registry; a renamed, model-driven scorer; per-model
downloads; an immutable per-archive encoder with a load-time guard; a model
selector in both automatic modes; and an append-only settings change log with
entries stamped by config version.

Out of scope, deliberately:

- **Migrating an existing archive to a different encoder.** Re-embedding from
  thumbnails is feasible but mixes fidelities against natively-captured
  entries. Deferred; the encoder is immutable in this design.
- **Changing SigLIP's scoring formulation.** See Non-goals.
- **Re-deriving `IMAGE_LOGIT_SCALE` by experiment.** Estimates ship; the
  calibration procedure is named.

## 1. Model registry — `services/vision_models.py`

A frozen dataclass and a `REGISTRY` dict. The key is a stable identifier
written into archives on disk and **must never be renamed** — it is the same
class of fact as `BrainLayout.signature()`.

```python
@dataclass(frozen=True)
class VisionModel:
    key: str                     # "clip-b32" - written to encoder.json
    label: str                   # "CLIP ViT-B/32" - UI only
    subdir: str                  # models/<subdir>/
    repo: str                    # HuggingFace repo
    files: dict[str, str]        # local filename -> path within the repo
    mean: tuple; std: tuple
    px: int
    dim: int
    context: int
    text_logit_scale: float
    image_logit_scale: float
    default_min_separation: float
```

| key | repo | dim | ctx | mean/std | text scale | image scale | min_sep |
|---|---|---|---|---|---|---|---|
| `clip-b32` | `Xenova/clip-vit-base-patch32` | 512 | 77 | CLIP | 100 | 30 | 0.020 |
| `clip-b16` | `Xenova/clip-vit-base-patch16` | 512 | 77 | CLIP | 100 | 32 | 0.018 |
| `siglip2-b16` | `onnx-community/siglip2-base-patch16-224-ONNX` | 768 | 64 | 0.5/0.5 | 100 | 25 | 0.024 |
| `clip-l14` | `Xenova/clip-vit-large-patch14` | 768 | 77 | CLIP | 100 | 14 | 0.044 |

`default_min_separation` is `0.02` scaled by the model's median NN distance
relative to B/32's. `image_logit_scale` is `30` scaled inversely by the same
ratio — B/32's 30 was chosen because image-image similarity sits above 0.9, and
a model whose distances spread wider needs a softer scale to avoid saturating.
**These three are estimates derived from NN spread, not direct measurements.**
Confirm each with the existing saturation check in
`tests/test_expedition_fitness.py` — the one asserting that no tile of a
16-tile generation is lost to saturation — before trusting a search run on
them.

`siglip2-b16`'s text scale starts at CLIP's 100 because the softmax-over-
distractors formulation is retained (see Non-goals); SigLIP's own learned scale
belongs to a sigmoid objective this code does not use.

This module is the single home for constants currently duplicated across
`services/clip_scorer.py` module globals and `tools/hue_nuisance.py`.

## 2. `VisionScorer` — renamed and model-driven

`services/clip_scorer.py` → `services/vision_scorer.py`;
`CLIPScorer` → `VisionScorer(model_key, ...)`.

- Preprocessing scale/offset built from the model's mean/std, not module
  constants.
- `_pick_output()` prefers `image_embeds`, then `pooler_output`, then the first
  rank-2 output. Taking output 0 by position picks `last_hidden_state` on the
  SigLIP export.
- Tokenizer truncation and padding use `model.context` (64 for SigLIP 2).
- `score()` uses `model.text_logit_scale`.
- A `model` property so callers can reach the registry entry.

`TEXT_LOGIT_SCALE` and `IMAGE_LOGIT_SCALE` move out of
`services/expedition_fitness.py` onto the model. `contrastive()` keeps its
no-default-scale rule; `ImgepDriver` reads both off `self.scorer.model`.

## 3. Downloads — `tools/fetch_models.py`

Replaces `tools/fetch_clip_onnx.py`. `fetch(model_key, progress)` and
`is_present(model_key)` drive off the registry's `repo` and `files`. Keeps the
existing `.part`-then-`os.replace` discipline and stays stdlib-only, since it
runs before `onnxruntime` is necessarily importable.

Verified fp16 vision-tower sizes: b32 168 MB, b16 165 MB, siglip2 177 MB, l14
581 MB. Text towers and tokenizers add to each (SigLIP 2's tokenizer alone is
34.4 MB, against CLIP's 2.1 MB). The UI shows the per-model total from the
registry; the registry records each file's size so the download prompt does not
have to guess.

`main.py` currently imports `MODEL_DIR` and `is_present` from
`tools.fetch_clip_onnx`; both call sites move.

## 4. Per-archive encoder

`<archive>/encoder.json`, at the archive root beside `goals.json` and
`settings.json` — **archive-level, not layout-level**, so every brain layout
under one archive shares an encoder.

```json
{"encoder": "clip-b32", "created": 1786500000.0}
```

Written once when the archive is created. **Refuses to overwrite**, the same
discipline as `ArchiveStore.save_run_config`: an archive name identifies one
embedding space, and a second write would reinterpret every entry already
filed under it.

**A missing file means `clip-b32`.** Every archive that exists predates the
choice and is B/32, so no migration runs and every existing archive opens
unchanged.

`ArchiveStore` gains an `encoder` property. `Archive.load_from_store` compares
it against the live scorer's key and **refuses to load on mismatch**, surfacing
a notice rather than raising — the same "a disk problem must never stop the
search" rule the module already follows. `Archive`'s `dim` comes from the
model rather than defaulting to 512.

UI: an encoder combo in the Explore tab and in the New Archive dialog,
**disabled once the archive holds any entries under any layout**. The encoder
is archive-level, so a populated `fourier-n10` locks the choice for a
`gabor-n7` sibling that is still empty. Existing `archive_library._count_entries`
already sums across layout directories and answers this. Empty archives stay
editable; the first admission anywhere locks the choice.

## 5. Scorer lifecycle

One resident scorer. `App._ensure_scorer(model_key)` returns the existing
instance when the key matches and otherwise builds a replacement, dropping the
old sessions. Loading costs about 0.4–1.2 s; keeping all four resident would
cost roughly 1.1 GB of weights for a switch that happens once per archive.

Auto and Explore are already mutually exclusive, so the resident model follows
whichever mode is scoring: Auto from its own combo, Explore from the open
archive. `_ensure_auto_service` and `_ensure_archive_service` both route
through `_ensure_scorer`.

## 6. Config change history

`<archive>/settings_history.jsonl`, append-only, archive-level.

```json
{"v":0,"ts":1786500000.0,"gen":0,"entries":0,"full":{"encoder":"clip-b32","min_separation":0.02, "...":"every PERSISTED_FIELD"}}
{"v":1,"ts":1786512345.0,"gen":340,"entries":4812,"changed":{"min_separation":[0.02,0.03]}}
```

Version 0 carries the complete block so the timeline can be replayed from a
base; every later row is a true diff of `PERSISTED_FIELDS` plus `encoder`.
Diffed **once per generation**, not per frame, and once more when the archive
closes so a change made after the last generation is not lost.

`index.jsonl` rows gain `"cfg": <version>`. `Archive` holds the current version
and stamps admissions with it. **Rows without `cfg` read as version 0**, which
covers every existing entry.

A read-only history table in the Archive tab: version, wall time, generation,
entry count, and what changed.

Nothing here replays or restores a past config. The log is a record, not a
mechanism — restoring an old setting is the user moving the slider back.

## 7. One existing rule this breaks

CLAUDE.md records that a brand new archive inherits the settings currently on
screen, so that making one to try a variation keeps your working setup.

**Exception:** when the new archive's encoder differs from the resident one,
`min_separation` is seeded from the new model's `default_min_separation`
instead of inherited. Carrying 0.02 into an L/14 archive would be roughly half
the intended strictness. Every other field still inherits.

## Non-goals

- **SigLIP 2's sigmoid scoring is not adopted.** It offers a calibrated
  absolute match probability that could replace the distractor set, but
  `DEFAULT_DISTRACTORS` is load-bearing against the dead-canvas attractor, and
  changing the objective and the encoder together makes any regression
  unattributable. Worth a separate experiment.
- **No encoder migration.** Deferred, as agreed.
- **`n_views` is not revisited.** Whether B/16's finer patch stride buys enough
  translation invariance to drop from 3 views is a separate measurement, and
  today's numbers do not answer it.

## Testing

- Registry: every entry's key, dim, context and file map round-trip; keys are
  unique; no key collides with a directory name in use.
- `encoder.json`: written once; a second write is refused; a missing file reads
  as `clip-b32`; an unknown key is refused with a notice rather than a crash.
- `Archive.load_from_store` refuses a store whose encoder differs from the
  scorer's, and says which is which.
- History: a changed field produces exactly one row carrying old and new; an
  unchanged generation produces none; version 0 holds the full block; an
  admission stamps the current version; an `index.jsonl` row without `cfg`
  loads as version 0.
- New-archive seeding: `min_separation` inherits within one encoder and is
  reseeded across encoders (section 7).
- UI: encoder combo disabled when entries exist; no duplicate ImGui ids in the
  Archive tab (extends `test_archive_window_render.py::id_clashes`).
- Rename: no `CLIPScorer` or `clip_scorer` reference survives outside the spec
  and CLAUDE.md history.
- Tests needing a model that is not downloaded **skip**, matching
  `tests/test_clip_real_model.py`.

## Migration

None runs. `vectors.npz` arrays are unchanged, so `FORMAT_VERSION` stays at 1.
Existing archives gain no `encoder.json` and read as `clip-b32`; existing
`index.jsonl` rows gain no `cfg` and read as version 0; a first settings change
after this lands writes version 0 with the full block, then the diff.
