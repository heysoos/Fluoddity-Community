"""Kill-switch: does CLIP produce a usable fitness landscape on trail patterns?

Usage:
    python -m tools.clip_signal_check <dir-of-pngs>

Bar to pass: the std-dev of scores across the sample must exceed 0.05 for at
least half the prompts. A flat landscape means no optimizer can climb it, and
we should fix the eval image before building the loop on top of it.

Files whose name starts with '_' are skipped, so contact sheets can live
alongside the tiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import ui  # noqa: F401  primes a pre-existing circular import (services -> ui -> services)
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
    paths = sorted(
        p for p in d.iterdir()
        if p.suffix.lower() == ".png" and not p.name.startswith("_")
    )
    if not paths:
        raise SystemExit(f"no PNGs in {d}")
    imgs = [
        np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR))
        for p in paths
    ]
    return np.stack(imgs).astype(np.uint8), [p.stem for p in paths]


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
        print(f"{prompt!r:22} std={s.std():.4f} "
              f"range=[{s.min():.3f},{s.max():.3f}] {'PASS' if ok else 'flat'}")
        print(f"    best : {names[order[0]]} ({s[order[0]]:.3f})  "
              f"{names[order[1]]} ({s[order[1]]:.3f})")
        print(f"    worst: {names[order[-1]]} ({s[order[-1]]:.3f})")

    verdict = passes >= len(PROMPTS) / 2
    print(f"\n{passes}/{len(PROMPTS)} prompts above std {STD_BAR} -> "
          f"{'GATE PASSED' if verdict else 'GATE FAILED'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
