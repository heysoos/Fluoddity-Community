"""Calibrate one encoder's scales and separation bar against real archives.

    python -m tools.calibrate_encoder --entries 700

Equates on BEHAVIOUR, not on a summary statistic: each value is chosen so it
reproduces on this population what the reference encoder's value does. The
criteria and the numbers they produced live in CLAUDE.md.

Two criteria, both from procedures already recorded in the codebase:

  image_logit_scale     the scale whose FLOORED FRACTION under a +3sd latent
                        goal matches the reference's
  min_separation        the threshold that ADMITS the same fraction of a real
                        archive as the reference's does

Needs the vision tower of every model being calibrated; missing ones are
skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui -> services.config_saver. It resolves only when `ui` is imported first.
import ui  # noqa: F401,E402

from services.expedition_fitness import contrastive  # noqa: E402
from services.vision_models import DEFAULT_KEY, REGISTRY, get  # noqa: E402
from tools.archive_sample import (  # noqa: E402
    build_encoders,
    load_thumbs,
    nn_distance,
    recent_layouts,
)
from utilities.paths import get_archives_root  # noqa: E402

# Whose behaviour every other encoder is matched to.
REFERENCE_KEY = DEFAULT_KEY


def latent_goal(e: np.ndarray):
    """-> (goal, references) for the +3sd construction recorded in
    tests/test_expedition_fitness.py: a goal pushed away from the archive
    centroid, scored against that centroid."""
    c = e.mean(axis=0)
    c = c / max(1e-9, float(np.linalg.norm(c)))
    g = c + 3.0 * (e[0] - c)
    g = g / max(1e-9, float(np.linalg.norm(g)))
    return g.astype(np.float32), c[None, :].astype(np.float32)


def floored_fraction(descriptors, goal, references, scale: float) -> float:
    """Fraction of the population whose contrastive fitness saturates to zero.

    A floored tile is invisible to a rank-based optimizer, so this is what a
    logit scale actually costs.
    """
    f = contrastive(np.asarray(descriptors, dtype=np.float32)[None, ...],
                    goal, references, logit_scale=float(scale))
    return float(np.mean(f <= 1e-6))


def scale_matching(descriptors, goal, references, target_frac: float,
                   lo: float = 1.0, hi: float = 400.0) -> float:
    """The logit scale whose floored fraction is target_frac.

    Bisection: floored fraction is monotone increasing in the scale.
    """
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if floored_fraction(descriptors, goal, references, mid) > target_frac:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def retention(nn, threshold: float) -> float:
    """Fraction of a population a separation bar would admit."""
    return float(np.mean(np.asarray(nn) >= float(threshold)))


def threshold_matching(nn, target_frac: float,
                       lo: float = 0.0, hi: float = 0.5) -> float:
    """The separation bar that admits target_frac of this population.

    Bisection: retention is monotone decreasing in the threshold.
    """
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if retention(nn, mid) < target_frac:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="", help="blank means every registered")
    ap.add_argument("--entries", type=int, default=700)
    ap.add_argument("--archives", type=int, default=3)
    ap.add_argument("--out", default="encoder_calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keys = [args.model] if args.model else sorted(REGISTRY)
    if REFERENCE_KEY not in keys:
        keys.insert(0, REFERENCE_KEY)
    encoders = build_encoders(keys)
    if not any(e.key == REFERENCE_KEY for e in encoders):
        raise SystemExit(f"{REFERENCE_KEY} is the reference and is not present")

    layouts = recent_layouts(get_archives_root(), args.archives)
    if not layouts:
        raise SystemExit("no archive holds any thumbnails")

    rng = np.random.default_rng(args.seed)
    px = encoders[0].px
    samples = {p: load_thumbs(p, args.entries, px, rng) for p in layouts}
    samples = {p: v for p, v in samples.items() if len(v) >= 2}
    for p, v in samples.items():
        print(f"  {p.parent.name}/{p.name}: {len(v)} entries")

    rows: dict[str, list] = {}
    for enc in encoders:
        rows[enc.key] = []
        for path, imgs in samples.items():
            e = enc.embed(imgs)
            goal, refs = latent_goal(e)
            rows[enc.key].append({
                "archive": f"{path.parent.name}/{path.name}",
                "e": e, "goal": goal, "refs": refs, "nn": nn_distance(e),
            })

    ref_model = get(REFERENCE_KEY)
    ref = rows[REFERENCE_KEY]
    target_floor = float(np.mean([
        floored_fraction(a["e"], a["goal"], a["refs"],
                         ref_model.image_logit_scale) for a in ref]))
    target_keep = float(np.mean([
        retention(a["nn"], ref_model.default_min_separation) for a in ref]))
    print(f"\nreference {REFERENCE_KEY}: floored={target_floor:.4f}  "
          f"retained={target_keep:.4f}")

    report = {"reference": REFERENCE_KEY, "target_floored": target_floor,
              "target_retained": target_keep, "entries": args.entries,
              "models": {}}
    print(f"\n{'model':14s}{'image_logit_scale':>19s}{'min_separation':>17s}"
          f"{'ms/image':>11s}")
    for enc in encoders:
        scales = [scale_matching(a["e"], a["goal"], a["refs"], target_floor)
                  for a in rows[enc.key]]
        seps = [threshold_matching(a["nn"], target_keep) for a in rows[enc.key]]
        report["models"][enc.key] = {
            "image_logit_scale": float(np.median(scales)),
            "default_min_separation": float(np.median(seps)),
            "per_archive_scale": [float(s) for s in scales],
            "per_archive_separation": [float(s) for s in seps],
            "nn_median": [float(np.median(a["nn"])) for a in rows[enc.key]],
            "ms_per_image": enc.ms_per_image,
            "dim": int(enc.dim),
        }
        print(f"{enc.key:14s}{np.median(scales):19.2f}"
              f"{np.median(seps):17.4f}{enc.ms_per_image:11.2f}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("Check the reference reproduces its own shipped values before "
          "editing the registry.")


if __name__ == "__main__":
    main()
