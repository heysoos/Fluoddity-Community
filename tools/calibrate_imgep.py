"""Measurements for the constants spec 12.2 says must not be guessed.

    python -m tools.calibrate_imgep              # sigma_expand, offline
    python -m tools.calibrate_imgep --liveness   # liveness_min, needs GL + CLIP

sigma_expand: the decoded spread of a mutated population should be visibly
related to, but narrower than, a fresh random population. This compares the
per-coefficient std of genomes decoded from N(0, sigma0) against genomes
decoded from a single parent plus N(0, sigma_expand).

liveness_min: the floor exists to reject frozen canvases, so it has to sit
between what a frozen preset scores and what a lively one scores. Guessing it
is exactly the mistake this file exists to prevent - and it CANNOT be inferred
offline, because liveness is a property of rendered frames. The --liveness mode
therefore runs real presets through the real Camera at production settings.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import ui  # noqa: F401,E402  prime the services/ui import cycle

from services.genome import random_genome
from services.genome_spec import DIM, decode, encode

# What tournament mode forces, and therefore what CLIP is really handed.
from state.view_modes import CAMERA as VIEW_CAM_BRUSH  # one home: state/view_modes.py
WINDOW_PX = 1024


# ---- sigma_expand -------------------------------------------------------

def _population_spread(pop: np.ndarray) -> float:
    """Mean per-coefficient std ACROSS the population.

    Not pop.std(). The whole-array std folds in how much a single genome's 80
    coefficients differ from each other, which is a property of the parent
    rather than of the mutation, and it swamps everything - a sweep of
    sigma_expand comes back nearly flat. Reducing over axis 0 first asks the
    actual question: how far apart are the children?
    """
    return float(np.asarray(pop, dtype=np.float64).std(axis=0).mean())


def measure_expansion_spread(sigma0=0.5, sigma_expand=0.15, n=512, seed=0):
    rng = np.random.default_rng(seed)
    fresh = np.stack([random_genome(rng) for _ in range(n)])
    boot = np.stack([decode(z) for z in sigma0 * rng.normal(size=(n, DIM))])

    parent = random_genome(rng)
    pz, _ = encode(parent)
    kids = np.stack([decode(pz + sigma_expand * rng.normal(size=DIM))
                     for _ in range(n)])

    b, k = _population_spread(boot), _population_spread(kids)
    return {
        "sigma0": sigma0,
        "sigma_expand": sigma_expand,
        "spread_random_genome": _population_spread(fresh),
        "spread_bootstrap": b,
        "spread_expansion_children": k,
        "ratio_children_to_bootstrap": float(k / max(b, 1e-12)),
    }


def report_sigma(sweep):
    r = measure_expansion_spread()
    print("--- sigma_expand calibration ---")
    for k, v in r.items():
        print(f"{k:32s} {v:.4f}")
    print()
    print("Bar: bootstrap spread should be close to spread_random_genome, and")
    print("children should sit at roughly 0.2-0.4 of it - related but narrower.")
    print("If ratio_children_to_bootstrap is above ~0.6, lower sigma_expand;")
    print("below ~0.1, expansion cannot escape its parent and will stall.")
    if sweep:
        print()
        print("sigma_expand   ratio to bootstrap")
        for s in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60):
            got = measure_expansion_spread(sigma_expand=s)
            mark = "  <-- in band" if 0.2 <= got["ratio_children_to_bootstrap"] <= 0.4 else ""
            print(f"    {s:<10.2f} {got['ratio_children_to_bootstrap']:.4f}{mark}")
    return r


# ---- liveness_min -------------------------------------------------------

def measure_liveness(preset_names, steps, snapshots, seed=12345, limit=0):
    """Run each preset for `steps` and report its per-tile liveness.

    Through the real Camera, for the same reason tools/capture_presets.py is:
    the app renders PARTICLES over the trails, and liveness is measured on
    whatever CLIP is handed.
    """
    import glfw
    import moderngl

    from camera import Camera
    from services.vision_scorer import VisionScorer
    from services.config_saver import ConfigSaver
    from services.capture_blit import CaptureBlit
    from services.descriptor import liveness, stack_snapshots
    from services.tile_capture import TileCapture
    from sim import Sim
    from state import SimState
    from state.preferences_state import PreferencesState
    from utilities.paths import get_app_physics_configs_dir

    root = get_app_physics_configs_dir()
    paths = []
    for sub in ("Core", "Advanced"):
        for p in sorted((root / sub).glob("*.json")):
            if not preset_names or p.stem in preset_names:
                paths.append(p)
    if limit:
        # Every Nth, not the first N: the two subdirectories are alphabetical,
        # so a head slice would sample one corner of the library.
        paths = paths[:: max(1, len(paths) // int(limit))][: int(limit)]
    if not paths:
        print("no presets matched")
        return []

    if not glfw.init():
        print("glfw init failed")
        return []
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(WINDOW_PX, WINDOW_PX, "calibrate", None, None)
    glfw.make_context_current(window)
    ctx = moderngl.create_context()

    prefs = PreferencesState()
    sim = Sim(ctx, world_size=prefs.world_size, canvas_aspect_ratio="1:1",
              particle_density=prefs.particle_density)
    camera = Camera(ctx, sim, window)
    camera.cam_brush_mode = True
    camera.BRIGHTNESS = prefs.brightness
    camera.trail_overlay_strength = prefs.trail_overlay_strength
    cap = TileCapture(ctx, grid=1)
    blit = CaptureBlit(ctx)
    saver = ConfigSaver()
    scorer = VisionScorer()

    every = max(1, steps // snapshots)
    out = []
    print(f"{len(paths)} presets, {steps} steps, {snapshots} snapshots "
          f"(one every {every} steps)\n")

    for path in paths:
        config = saver.load_from_file(path)
        if config is None:
            continue
        state = SimState()
        rule = saver.apply_config(config, state)
        sim.apply_state(state)
        sim.apply_rule(rule)
        sim.reset_seed = float(seed)
        sim.reset()
        camera.watercolor_mode = state.watercolor_mode
        camera.ink_weight = state.ink_weight

        frames = []
        for step in range(1, steps + 1):
            sim.update(ctx)
            if step % every == 0 and len(frames) < snapshots:
                raw = camera.generate_view_texture()
                tex = camera.frame_assembler.assemble_frame(
                    raw, total_samples=1, current_sample_index=0,
                    view_mode=VIEW_CAM_BRUSH, screen_aspect=1.0,
                    brightness=camera.BRIGHTNESS, exposure=prefs.exposure,
                    ink_weight=state.ink_weight,
                    watercolor_mode=state.watercolor_mode,
                    camera_position=tuple(camera.position),
                    camera_zoom=camera.zoom,
                    canvas_resolution=sim.get_canvas_dimensions(),
                    tonemap_softness=prefs.tonemap_softness,
                    trail_overlay_strength=camera.trail_overlay_strength)
                if tex is None:
                    continue
                crops = cap.capture(
                    lambda fbo: blit.draw(tex, (0.0, 0.0), (1.0, 1.0)))
                frames.append(np.asarray(scorer.embed(crops, n_views=1),
                                         dtype=np.float32))
        if len(frames) < 2:
            continue
        liv = float(liveness(stack_snapshots(frames))[0])
        out.append((path.stem, liv))
        print(f"  {path.stem:34s} liveness {liv:.4f}")

    glfw.terminate()
    return out


def report_liveness(rows):
    if len(rows) < 2:
        print("\nnot enough presets measured to place a floor")
        return
    vals = np.array([v for _, v in rows], dtype=np.float32)
    order = sorted(rows, key=lambda r: r[1])
    print("\n--- liveness distribution ---")
    print(f"n           {len(vals)}")
    print(f"min         {vals.min():.4f}   ({order[0][0]})")
    print(f"p05         {np.percentile(vals, 5):.4f}")
    print(f"median      {np.median(vals):.4f}")
    print(f"p95         {np.percentile(vals, 95):.4f}")
    print(f"max         {vals.max():.4f}   ({order[-1][0]})")
    print("\nquietest five:")
    for name, v in order[:5]:
        print(f"    {name:34s} {v:.4f}")
    print("\nliveliest five:")
    for name, v in order[-5:]:
        print(f"    {name:34s} {v:.4f}")
    floor = float(np.percentile(vals, 5))
    print(f"\nA floor at the 5th percentile is {floor:.4f}.")
    print("Curated presets are all supposed to be alive, so the floor should")
    print("sit BELOW this - it exists to reject dead genomes produced during")
    print("search, not to second-guess the preset library. If the spread here")
    print("is narrow, liveness is not separating much at this snapshot count")
    print("and snapshots_per_gen should go up before the floor is tuned.")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--liveness", action="store_true",
                    help="measure liveness over presets (needs GL and CLIP)")
    ap.add_argument("--sweep", action="store_true",
                    help="also print a sigma_expand sweep")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--snapshots", type=int, default=6)
    ap.add_argument("--only", default="", help="comma-separated preset stems")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.liveness:
        report_sigma(args.sweep)
        print()
        print("liveness_min must be measured with rendered frames:")
        print("    python -m tools.calibrate_imgep --liveness")
        return 0

    names = {s.strip() for s in args.only.split(",") if s.strip()}
    rows = measure_liveness(names, args.steps, args.snapshots, limit=args.limit)
    report_liveness(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
