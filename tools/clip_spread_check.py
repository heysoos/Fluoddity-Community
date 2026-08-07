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
