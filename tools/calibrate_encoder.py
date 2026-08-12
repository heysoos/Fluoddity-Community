"""Calibrate one encoder's scales and separation bar against real archives.

    python -m tools.calibrate_encoder --entries 700

Equates on BEHAVIOUR, not on a summary statistic: each value is chosen so it
reproduces on this population what the reference encoder's value does. The
criteria and the numbers they produced live in CLAUDE.md.

Three criteria, the first two from procedures already recorded in the codebase:

  image_logit_scale     the scale whose FLOORED FRACTION under a +3sd latent
                        goal matches the reference's
  min_separation        the threshold that ADMITS the same fraction of a real
                        archive as the reference's does
  text_logit_scale      the sharpest scale at which no tile floors. WEAKER
                        than the other two: nothing floors at the reference's
                        own value, so the target does not constrain and this
                        reads the onset instead. It reproduces the reference's
                        trained temperature to about 11%.

Both towers of every model being calibrated must be downloaded; missing ones
are skipped. Read min_separation off a REAL sample size - at a few dozen
entries every stored entry clears the bar and the criterion saturates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui -> services.config_saver. It resolves only when `ui` is imported first.
import ui  # noqa: F401,E402

from services.expedition_fitness import contrastive  # noqa: E402
from services.vision_models import DEFAULT_KEY, REGISTRY, get  # noqa: E402
from services.vision_scorer import DEFAULT_DISTRACTORS, VisionScorer  # noqa: E402
from tools.archive_sample import (  # noqa: E402
    load_thumbs,
    nn_distance,
    recent_layouts,
)
from tools.fetch_models import is_present  # noqa: E402
from utilities.paths import get_archives_root  # noqa: E402

# Whose behaviour every other encoder is matched to.
REFERENCE_KEY = DEFAULT_KEY

# Text goals to calibrate the text scale over. Spread deliberately: a concrete
# object, an abstract quality, a colour and a phrase the archive holds nothing
# like, because a scale tuned on one prompt shape suits only that shape.
CALIBRATION_PROMPTS = (
    "a smiley face",
    "a network of neurons",
    "purple",
    "circuit board traces",
    "a photograph of a cat",
    "an underwater reef",
)


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


def _bisect_increasing(f, target: float, lo: float, hi: float) -> float:
    """The x where monotone-increasing f(x) crosses target.

    With target 0 this returns the LARGEST x that has not yet crossed, which is
    the useful reading: as sharp as the scale can go without losing anyone.
    """
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def scale_matching(descriptors, goal, references, target_frac: float,
                   lo: float = 1.0, hi: float = 400.0) -> float:
    """The logit scale whose floored fraction is target_frac.

    Bisection: floored fraction is monotone increasing in the scale.
    """
    return _bisect_increasing(
        lambda s: floored_fraction(descriptors, goal, references, s),
        target_frac, lo, hi)


def text_floored(rows, scale: float) -> float:
    """Mean floored fraction over every (archive, prompt) pair.

    A text goal's references are the distractor set rather than the archive
    centroid, so its scale is calibrated separately - the two modalities sit in
    different similarity regimes. See CLAUDE.md.
    """
    out = []
    for a in rows:
        for g in a["text_goals"]:
            out.append(floored_fraction(a["e"], g, a["distractors"], scale))
    return float(np.mean(out)) if out else 0.0


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
    usable = []
    for key in keys:
        if is_present(key):
            usable.append(key)
        else:
            print(f"[skip] {key}: assets not downloaded")
    if REFERENCE_KEY not in usable:
        raise SystemExit(f"{REFERENCE_KEY} is the reference and is not present")

    layouts = recent_layouts(get_archives_root(), args.archives)
    if not layouts:
        raise SystemExit("no archive holds any thumbnails")

    rng = np.random.default_rng(args.seed)
    px = get(REFERENCE_KEY).px
    samples = {p: load_thumbs(p, args.entries, px, rng) for p in layouts}
    samples = {p: v for p, v in samples.items() if len(v) >= 2}
    for p, v in samples.items():
        print(f"  {p.parent.name}/{p.name}: {len(v)} entries")

    rows: dict[str, list] = {}
    timing: dict[str, float] = {}
    for key in usable:
        # The real scorer, not a bare vision session: what is calibrated has to
        # be what runs. n_views=1 - augmentation is for a directed objective.
        scorer = VisionScorer(key, n_views=1)
        text_goals = scorer.embed_text(list(CALIBRATION_PROMPTS))
        distractors = scorer.embed_text(list(DEFAULT_DISTRACTORS))
        per, seen, t0 = [], 0, time.perf_counter()
        for path, imgs in samples.items():
            e = scorer.embed(imgs, n_views=1)
            seen += len(imgs)
            goal, refs = latent_goal(e)
            per.append({
                "archive": f"{path.parent.name}/{path.name}",
                "e": e, "goal": goal, "refs": refs, "nn": nn_distance(e),
                "text_goals": text_goals, "distractors": distractors,
            })
        timing[key] = 1000.0 * (time.perf_counter() - t0) / max(1, seen)
        rows[key] = per
        del scorer                      # one resident session at a time

    ref_model = get(REFERENCE_KEY)
    ref = rows[REFERENCE_KEY]
    target_floor = float(np.mean([
        floored_fraction(a["e"], a["goal"], a["refs"],
                         ref_model.image_logit_scale) for a in ref]))
    target_keep = float(np.mean([
        retention(a["nn"], ref_model.default_min_separation) for a in ref]))
    target_text = text_floored(ref, ref_model.text_logit_scale)
    print(f"\nreference {REFERENCE_KEY}: image floored={target_floor:.4f}  "
          f"text floored={target_text:.4f}  retained={target_keep:.4f}")

    report = {"reference": REFERENCE_KEY, "target_floored": target_floor,
              "target_text_floored": target_text,
              "target_retained": target_keep, "entries": args.entries,
              "prompts": list(CALIBRATION_PROMPTS), "models": {}}
    print(f"\n{'model':14s}{'image_scale':>13s}{'text_scale':>12s}"
          f"{'min_sep':>10s}{'ms/image':>10s}{'dim':>6s}")
    for key in usable:
        scales = [scale_matching(a["e"], a["goal"], a["refs"], target_floor)
                  for a in rows[key]]
        seps = [threshold_matching(a["nn"], target_keep) for a in rows[key]]
        tscale = _bisect_increasing(
            lambda s: text_floored(rows[key], s), target_text, 1.0, 400.0)
        report["models"][key] = {
            "image_logit_scale": float(np.median(scales)),
            "text_logit_scale": float(tscale),
            "default_min_separation": float(np.median(seps)),
            "per_archive_scale": [float(s) for s in scales],
            "per_archive_separation": [float(s) for s in seps],
            "nn_median": [float(np.median(a["nn"])) for a in rows[key]],
            "ms_per_image": timing[key],
            "dim": int(rows[key][0]["e"].shape[1]),
        }
        print(f"{key:14s}{np.median(scales):13.2f}{tscale:12.2f}"
              f"{np.median(seps):10.4f}{timing[key]:10.2f}"
              f"{rows[key][0]['e'].shape[1]:6d}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"GATE: {REFERENCE_KEY} must reproduce its own shipped values "
          f"(image {ref_model.image_logit_scale}, "
          f"sep {ref_model.default_min_separation}) before the registry is "
          f"edited.")


if __name__ == "__main__":
    main()
