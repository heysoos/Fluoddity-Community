"""Does a picture goal separate a real population, with and without distractors?

    python -m tools.measure_image_goal --entries 700
    python -m tools.measure_image_goal --refs photo.jpg other.png

Scores an archive's thumbnails - read straight off disk, never through an
ArchiveStore, which writes - against picture goals through the REAL scorer, so
what is measured is what runs. Two losses per goal:

  cosine       set_image_goal(distractors=False)
  contrastive  set_image_goal(distractors=True): softmax against black, white,
               grey and noise at the encoder's image scale

For each it reports the fraction floored to zero and saturated near one, how
many of 16 tiles a rank-based optimizer can tell apart, the two losses' rank
agreement, and where a black, white, grey and noise TILE lands among the
archive - the dead canvas the text path's distractors exist to reject.

Goals default to held-out archive thumbnails (the chase-a-picture case);
--refs scores real pictures instead. The numbers this produced live in
CLAUDE.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui -> services.config_saver. It resolves only when `ui` is imported first.
import ui  # noqa: F401,E402

from services.vision_models import DEFAULT_KEY, get  # noqa: E402
from services.vision_scorer import VisionScorer, load_goal_image  # noqa: E402
from tools.archive_sample import load_thumbs, recent_layouts  # noqa: E402
from tools.fetch_models import is_present  # noqa: E402
from utilities.paths import get_archives_root  # noqa: E402

DEAD_TILES = ("black", "white", "grey", "noise")
GRID_TILES = 16
DRAWS = 200


def dead_tiles(px: int) -> np.ndarray:
    """The dead canvases, as TILES. The noise is a different draw from the
    distractor's, or it would match that reference exactly."""
    flat = [np.full((px, px, 3), v, dtype=np.uint8) for v in (0, 255, 128)]
    noise = np.random.default_rng(1).integers(0, 256, (px, px, 3), dtype=np.uint8)
    return np.stack(flat + [noise], axis=0)


def distinct_per_grid(f: np.ndarray, rng, tiles: int = GRID_TILES,
                      draws: int = DRAWS) -> float:
    """Mean number of distinct float32 values among `tiles` random tiles - what
    a rank-based optimizer can actually order."""
    f = np.asarray(f, dtype=np.float32)
    if len(f) <= tiles:
        return float(len(np.unique(f)))
    return float(np.mean([len(np.unique(f[rng.choice(len(f), tiles, replace=False)]))
                          for _ in range(draws)]))


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_from_top(archive_scores: np.ndarray, score: float) -> float:
    """0.0 beats every archive tile, 1.0 loses to every one."""
    return float(np.mean(np.asarray(archive_scores) > score))


def measure(scorer, goal_img: np.ndarray, pop: np.ndarray, n_archive: int,
            seed: int) -> dict:
    """-> {loss: metrics} for one goal over pop, whose first n_archive rows
    are archive tiles and the rest the dead ones."""
    out = {}
    for name, distractors in (("cosine", False), ("contrastive", True)):
        # The same views for both losses, so they differ only in the formula.
        scorer._rng = np.random.default_rng(seed)
        scorer.set_image_goal(goal_img, distractors=distractors)
        t0 = time.perf_counter()
        f = scorer.score(pop)
        ms = 1000.0 * (time.perf_counter() - t0) / len(pop)
        arch, dead = f[:n_archive], f[n_archive:]
        rng = np.random.default_rng(seed)
        out[name] = {
            "f": f,
            "floored": float(np.mean(arch <= 1e-6)),
            "saturated": float(np.mean(arch >= 1.0 - 1e-3)),
            "p10": float(np.percentile(arch, 10)),
            "p50": float(np.percentile(arch, 50)),
            "p90": float(np.percentile(arch, 90)),
            "distinct16": distinct_per_grid(arch, rng),
            "dead": {k: rank_from_top(arch, v) for k, v in zip(DEAD_TILES, dead)},
            "ms": ms,
        }
    c, k = out["cosine"], out["contrastive"]
    a_c, a_k = c["f"][:n_archive], k["f"][:n_archive]
    top_c = set(np.argsort(-a_c)[:GRID_TILES])
    top_k = set(np.argsort(-a_k)[:GRID_TILES])
    out["agreement"] = {
        "spearman": spearman(a_c, a_k),
        "top16_overlap": len(top_c & top_k) / GRID_TILES,
    }
    return out


def print_goal(label: str, m: dict) -> None:
    print(f"\n== goal: {label}")
    print(f"  {'loss':12s} {'floored':>8s} {'saturated':>10s} {'p10':>8s} "
          f"{'p50':>8s} {'p90':>8s} {'distinct/16':>12s} "
          + " ".join(f"{d:>7s}" for d in DEAD_TILES) + f" {'ms/img':>7s}")
    for name in ("cosine", "contrastive"):
        r = m[name]
        print(f"  {name:12s} {r['floored']:8.3f} {r['saturated']:10.3f} "
              f"{r['p10']:8.4f} {r['p50']:8.4f} {r['p90']:8.4f} "
              f"{r['distinct16']:12.2f} "
              + " ".join(f"{r['dead'][d]:7.3f}" for d in DEAD_TILES)
              + f" {r['ms']:7.2f}")
    a = m["agreement"]
    print(f"  agreement: spearman {a['spearman']:.3f}   "
          f"top-16 overlap {a['top16_overlap']:.2f}")
    print("  (dead columns: rank from the top among archive tiles, "
          "0 = beats every real tile)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_KEY)
    ap.add_argument("--entries", type=int, default=700)
    ap.add_argument("--goals", type=int, default=4,
                    help="held-out thumbnails to use as goals when --refs is empty")
    ap.add_argument("--refs", nargs="*", default=[],
                    help="pictures to use as goals instead of thumbnails")
    ap.add_argument("--layout", default="",
                    help="archive/layout directory; default the most recent")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not is_present(args.model):
        raise SystemExit(f"{args.model}: assets not downloaded")
    px = get(args.model).px
    rng = np.random.default_rng(args.seed)

    if args.layout:
        layout = get_archives_root() / args.layout
    else:
        found = recent_layouts(get_archives_root(), 1)
        if not found:
            raise SystemExit("no archive holds any thumbnails")
        layout = found[0]
    thumbs = load_thumbs(layout, args.entries + args.goals, px, rng)
    print(f"[population] {layout.parent.name}/{layout.name}: {len(thumbs)} entries")
    if len(thumbs) < 2 + args.goals:
        raise SystemExit("too few thumbnails to measure anything")

    if args.refs:
        goals = [(Path(p).name, load_goal_image(p, px)) for p in args.refs]
        archive = thumbs
    else:
        held = rng.choice(len(thumbs), args.goals, replace=False)
        goals = [(f"thumb #{int(i)}", thumbs[i][None]) for i in held]
        archive = np.delete(thumbs, held, axis=0)
    pop = np.concatenate([archive, dead_tiles(px)], axis=0)

    scorer = VisionScorer(args.model)
    print(f"[encoder] {args.model}  image scale {scorer.model.image_logit_scale}"
          f"  views {scorer._n_views}")

    results = []
    for label, img in goals:
        m = measure(scorer, img, pop, len(archive), args.seed)
        print_goal(label, m)
        results.append(m)

    print("\n== mean over goals")
    for name in ("cosine", "contrastive"):
        rs = [m[name] for m in results]
        print(f"  {name:12s} floored {np.mean([r['floored'] for r in rs]):.3f}  "
              f"saturated {np.mean([r['saturated'] for r in rs]):.3f}  "
              f"distinct/16 {np.mean([r['distinct16'] for r in rs]):.2f}  "
              + "  ".join(f"{d} {np.mean([r['dead'][d] for r in rs]):.3f}"
                          for d in DEAD_TILES))
    print(f"  spearman {np.mean([m['agreement']['spearman'] for m in results]):.3f}"
          f"  top-16 overlap "
          f"{np.mean([m['agreement']['top16_overlap'] for m in results]):.2f}")


if __name__ == "__main__":
    main()
