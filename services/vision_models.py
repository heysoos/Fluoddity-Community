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

# The Min Separation track, as a multiple of the encoder's calibrated default.
# A model whose distances spread wider needs a longer track, or its own default
# sits at the top of it. Reproduces the historical 0..0.05 for clip-b32.
SEPARATION_TRACK = 2.5


# eq=False keeps identity hashing: the entries are singletons, and the default
# frozen __hash__ would raise on the files dict.
@dataclass(frozen=True, eq=False)
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
    # Calibrated per encoder against clip-b32's BEHAVIOUR, never scaled off a
    # summary statistic - the median predicted SigLIP's separation bar badly.
    # clip-b32 keeps its own historical values, which the calibration
    # reproduces; every other model's are measured. The text scale carries a
    # margin below its flooring onset, the same margin clip-b32's trained 100
    # has. See CLAUDE.md and tools/calibrate_encoder.py.
    text_logit_scale: float | None = None
    image_logit_scale: float | None = None
    default_min_separation: float | None = None
    # One sentence on what this encoder sees differently. The relative cost is
    # measured; CLAUDE.md's table is the home for the figure itself.
    blurb: str = ""

    @property
    def calibrated(self) -> bool:
        return None not in (self.text_logit_scale, self.image_logit_scale,
                            self.default_min_separation)

    @property
    def separation_slider_max(self) -> float:
        return SEPARATION_TRACK * float(self.default_min_separation)


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
        blurb="The original, and by far the fastest: it reads the frame as 49 "
              "coarse patches, so it judges overall shape and colour rather "
              "than fine texture. Every archive made before the others "
              "existed is in this space.",
    ),
    "clip-b16": VisionModel(
        key="clip-b16", label="CLIP ViT-B/16", subdir="clip-vit-b16",
        repo="https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main",
        files=_clip_files(), mean=CLIP_MEAN, std=CLIP_STD, px=224, dim=512,
        context=77,
        text_logit_scale=170.3, image_logit_scale=33.1,
        default_min_separation=0.0195,
        blurb="The same CLIP over 16-pixel patches - four times as many - so "
              "lattices, filaments and fine structure survive that B/32 "
              "averages away. Roughly twice the cost per generation.",
    ),
    "siglip2-b16": VisionModel(
        key="siglip2-b16", label="SigLIP 2 base/16", subdir="siglip2-b16-224",
        repo="https://huggingface.co/onnx-community/"
             "siglip2-base-patch16-224-ONNX/resolve/main",
        files=_clip_files(), mean=HALF, std=HALF, px=224, dim=768, context=64,
        text_logit_scale=150.8, image_logit_scale=32.5,
        default_min_separation=0.0195,
        blurb="Newer training on a sigmoid loss and far more data, and the "
              "most colour-sensitive of the four. Strongest at what a picture "
              "literally contains. Around two and a half times the cost.",
    ),
    "clip-l14": VisionModel(
        key="clip-l14", label="CLIP ViT-L/14", subdir="clip-vit-l14",
        repo="https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main",
        files=_clip_files(), mean=CLIP_MEAN, std=CLIP_STD, px=224, dim=768,
        context=77,
        text_logit_scale=129.0, image_logit_scale=17.7,
        default_min_separation=0.0519,
        blurb="A much larger model over the finest patches, and the best on "
              "abstract or compositional prompts. It also spreads creatures "
              "further apart, so its separation bar is wider. Around eight "
              "times the cost.",
    ),
}


def get(key: str) -> VisionModel:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown vision model {key!r}; known: {', '.join(sorted(REGISTRY))}"
        ) from None
