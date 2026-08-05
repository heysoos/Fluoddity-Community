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

## Task 2 — CLIPScorer

Implemented with the three Task 1 findings folded in. 11 unit tests + 4
gpu-marked real-model tests passing, including an explicit CPU-fallback test.

**Deviation:** `preprocess()` gained a `dtype` parameter (default float32),
driven by the session's declared input type, per Finding 2.

## Task 3 — CLIP signal gate

**GATE PASSED: 5/6 prompts** above the std 0.05 bar, on 32 real tournament
tiles captured headlessly across two generations.

| prompt | std | range |
|---|---|---|
| glowing coral | 0.190 | 0.008 – 0.741 |
| a spider web | 0.182 | 0.044 – 0.753 |
| a dense city map | 0.146 | 0.027 – 0.587 |
| tree branches | 0.083 | 0.004 – 0.299 |
| a swirling galaxy | 0.060 | 0.027 – 0.231 |
| flowing water | 0.039 | 0.004 – 0.146 (flat) |

Rankings inspected visually: coral → pink polyp-like clusters, tree branches →
radiating frond structures, spider web → a cyan reticulated network with real
junctions. All defensible. Losers were uniformly dense noise, consistently.

### Finding 4: a dead canvas was a degenerate attractor

The visual inspection caught what the std numbers hid — "flowing water" was won
by a near-black tile. Testing a pure black image directly, with the spec's
original six distractors:

| prompt | black scores | black outranks |
|---|---|---|
| flowing water | 0.366 | **31 of 32 real tiles** |
| tree branches | 0.352 | 30 of 32 |
| a spider web | 0.297 | 12 of 32 |

CMA-ES would have driven straight to an empty simulation. `"a blank image"` and
`"a solid color"` did not catch it.

**Deviation:** `DEFAULT_DISTRACTORS` gains three black-specific entries —
`"a black image"`, `"an empty black background"`, `"a dark empty scene"`.
Measured effect:

| | before | after |
|---|---|---|
| black, worst case | outranks 31/32 | outranks 6/32 |
| black on flowing water | 0.366 | 0.029 |
| coral std (real signal) | 0.189 | 0.194 |
| spider web std | 0.177 | 0.161 |

The degenerate attractor is removed at essentially no cost to real signal.
`tests/test_clip_real_model.py::test_black_canvas_is_not_a_degenerate_attractor`
pins this.

"flowing water" drops below the bar as a result. That is the honest outcome:
it was only passing because black was gaming it. The substrate does not produce
water-like imagery, and the prompt is a poor fit rather than the scorer being
broken.
