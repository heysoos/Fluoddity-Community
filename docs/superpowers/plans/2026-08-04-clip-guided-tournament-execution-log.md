# CLIP-Guided Automatic Tournament — Execution Log

Deviations from `2026-08-04-clip-guided-tournament.md` and their reasons.

## Task 1 — Dependencies and model fetch

**Done as planned.** `requirements.txt`, `.gitignore`, `tools/fetch_clip_onnx.py`,
4 tests passing. All three assets downloaded (176 MB / 127 MB / 2 MB).

`DmlExecutionProvider` is available on this machine, so the DirectML choice works
as intended — no CUDA toolkit, no PyTorch.

### Finding 1: the split exports are correct — no fallback needed

Spec §3.1 required verifying that `vision_model` / `text_model` emit *projected*
embeddings rather than `pooler_output`. They do:

```
vision: in 'pixel_values'          -> out 'image_embeds'
text:   in 'input_ids' (int64)     -> out 'text_embeds'
```

The combined `model_fp16.onnx` fallback is not needed. The text model takes
**only** `input_ids` — there is no `attention_mask` input.

### Finding 2: fp16 models take float32 input — plan was wrong

`pixel_values` is declared `tensor(float)`, not `tensor(float16)`. The fp16
export has fp16 *weights* but fp32 I/O.

The plan's `preprocess()` cast the batch to `np.float16`, which would have been
rejected or silently reinterpreted. **Deviation:** `preprocess()` takes a `dtype`
argument, and `CLIPScorer` passes the dtype declared by the session's own input
descriptor. That is correct for both this export and any future fp16-I/O one, so
the fix is not merely "use float32".

### Finding 3: ORT_ENABLE_ALL crashes on CPU — must pin ORT_ENABLE_BASIC

Loading the fp16 vision model with default session options fails:

```
FAIL : Exception during initialization: graph_utils.cc:30
GetIndexFromName ... Attempting to get index by a name which does not exist:
InsertedPrecisionFreeCast_/vision_model/encoder/layers.6/layer_norm1/Constant_output_0
for node: /vision_model/pre_layrnorm/Mul/SimplifiedLayerNormFusion/
```

Measured across providers and optimization levels:

| Provider | ORT_ENABLE_ALL | ORT_ENABLE_BASIC |
|---|---|---|
| DirectML | OK | OK |
| CPU | **FAIL** | OK |

This matters because spec §11.1 requires the CPU fallback to work when DirectML
is unavailable — with default options that path would crash rather than degrade.

**Deviation:** `CLIPScorer` sets
`SessionOptions.graph_optimization_level = ORT_ENABLE_BASIC` on both sessions.
Applied unconditionally rather than only on CPU, so the two providers run the
same graph and cannot diverge in behaviour.
