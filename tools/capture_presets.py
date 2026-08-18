"""Render every saved preset offscreen and save one 224px tile each.

Exists to answer the stage-0 question honestly (spec 12.1). Measuring CLIP's
spread over `runs/*/frames/` was misleading twice over: those images are best
tiles from prompt-driven searches, so they were pre-selected for CLIP
legibility, and they were produced by builds that predate the NaN and
cohort-colouring fixes. Presets are the neutral sample - the patterns a person
actually curated as distinct.

USES THE REAL Camera, through a hidden GLFW window, rather than reimplementing
the view pipeline. That is not fussiness. The first version of this tool fed
sim.can straight to FrameAssembler and reported 26% of presets as near-black,
which is nonsense - the app's view is generate_view_texture(), which renders
the PARTICLES with additive blending on top of the trails, and for a preset
whose trails are diffuse the particles are most of the visible image. Anything
that re-derives the view path will drift from it; borrowing Camera cannot.

    python -m tools.capture_presets out_dir --steps 2000

Then measure:

    python -m tools.clip_spread_check out_dir/*.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui.physics_params -> ui/__init__ -> ui.core -> services.config_saver. It
# resolves only when `ui` is imported first, which is what main.py happens to
# do. Prime that order, as tests/conftest.py does.
import ui  # noqa: F401,E402

# What tournament mode forces (main.py section 5.1.5), and therefore what CLIP
# is really handed: camera view, particles over trails.
from state.view_modes import CAMERA as VIEW_CAM_BRUSH  # one home: state/view_modes.py
WINDOW_PX = 1024


def find_presets(root: Path) -> list[Path]:
    out: list[Path] = []
    for sub in ("Core", "Advanced"):
        d = root / sub
        if d.is_dir():
            out.extend(sorted(d.glob("*.json")))
    return out


def capture_preset(ctx, sim, camera, cap, blit, config_saver, path, steps,
                   seed, prefs):
    """Load one preset, run it, and return a (224, 224, 3) uint8 crop."""
    from state import SimState

    config = config_saver.load_from_file(path)
    if config is None:
        return None

    state = SimState()
    rule = config_saver.apply_config(config, state)
    sim.apply_state(state)
    sim.apply_rule(rule)
    # Fixed per preset, so a re-run is comparable rather than a fresh scatter.
    sim.reset_seed = float(seed)
    sim.reset()

    for _ in range(int(steps)):
        sim.update(ctx)

    # The camera pass: particles rendered additively over the trail canvas.
    camera.watercolor_mode = state.watercolor_mode
    camera.ink_weight = state.ink_weight
    raw = camera.generate_view_texture()
    assembled = camera.frame_assembler.assemble_frame(
        raw, total_samples=1, current_sample_index=0,
        view_mode=VIEW_CAM_BRUSH,
        screen_aspect=1.0,
        brightness=camera.BRIGHTNESS,
        exposure=prefs.exposure,
        ink_weight=state.ink_weight,
        watercolor_mode=state.watercolor_mode,
        camera_position=tuple(camera.position),
        camera_zoom=camera.zoom,
        canvas_resolution=sim.get_canvas_dimensions(),
        tonemap_softness=prefs.tonemap_softness,
        trail_overlay_strength=camera.trail_overlay_strength,
    )
    if assembled is None:
        return None
    crops = cap.capture(lambda fbo: blit.draw(assembled, (0.0, 0.0), (1.0, 1.0)))
    return crops[0]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="0 = every preset")
    ap.add_argument("--only", default="", help="comma-separated preset stems")
    ap.add_argument("--density", type=float, default=None,
                    help="default: the app's particle_density preference")
    ap.add_argument("--world-size", type=float, default=None,
                    help="default: the app's world_size preference")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    import glfw
    import moderngl
    from PIL import Image

    from camera import Camera
    from services.capture_blit import CaptureBlit
    from services.config_saver import ConfigSaver
    from services.tile_capture import TileCapture
    from sim import Sim
    from state.preferences_state import PreferencesState
    from utilities.paths import get_app_physics_configs_dir

    presets = find_presets(get_app_physics_configs_dir())
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        presets = [p for p in presets if p.stem in want]
    if args.limit:
        presets = presets[: args.limit]
    if not presets:
        print("no presets found")
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # A hidden window, not a standalone context: Camera calls
    # glfw.get_framebuffer_size(self.window) in several places, and borrowing
    # the real Camera is what keeps this tool from drifting away from the view
    # the app actually shows.
    if not glfw.init():
        print("glfw init failed")
        return 2
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(WINDOW_PX, WINDOW_PX, "capture", None, None)
    if not window:
        glfw.terminate()
        print("could not create a hidden window")
        return 2
    glfw.make_context_current(window)
    ctx = moderngl.create_context()

    # The app's own preferences, not this module's opinion of them. A preset is
    # authored under these; rendering it under anything else measures a
    # different simulation.
    prefs = PreferencesState()
    if args.world_size is not None:
        prefs.world_size = args.world_size
    if args.density is not None:
        prefs.particle_density = args.density

    sim = Sim(ctx, world_size=prefs.world_size, canvas_aspect_ratio="1:1",
              particle_density=prefs.particle_density)
    camera = Camera(ctx, sim, window)
    camera.cam_brush_mode = True          # what view option 2 means
    camera.BRIGHTNESS = prefs.brightness
    camera.trail_overlay_strength = prefs.trail_overlay_strength
    cap = TileCapture(ctx, grid=1)
    blit = CaptureBlit(ctx)
    config_saver = ConfigSaver()

    print(f"{len(presets)} presets, {args.steps} steps each, "
          f"{sim.entity_count} particles, canvas {sim.can.size}")
    print(f"world_size {prefs.world_size}  density {prefs.particle_density}  "
          f"brightness {prefs.brightness}  tonemap {prefs.tonemap_softness}  "
          f"view cam_brush (particles over trails)")

    t0 = time.time()
    written = 0
    dark = 0
    for i, p in enumerate(presets):
        try:
            crop = capture_preset(ctx, sim, camera, cap, blit, config_saver,
                                  p, args.steps, args.seed, prefs)
        except Exception as exc:
            print(f"  [{i + 1}/{len(presets)}] {p.stem}: FAILED ({exc})")
            continue
        if crop is None:
            print(f"  [{i + 1}/{len(presets)}] {p.stem}: no frame")
            continue
        Image.fromarray(crop).save(out / f"{p.parent.name}_{p.stem}.png")
        written += 1
        mean, peak = float(crop.mean()), int(crop.max())
        note = ""
        if mean < 2.0:
            dark += 1
            note = "  <-- near black"
        print(f"  [{i + 1}/{len(presets)}] {p.stem:32s} "
              f"mean {mean:6.1f}  max {peak:3d}{note}")

    dt = time.time() - t0
    print(f"\nwrote {written}/{len(presets)} crops to {out} in {dt:.0f}s")
    print(f"near-black: {dark}/{written}")
    print(f"\nnow run:  python -m tools.clip_spread_check {out}/*.png")

    glfw.terminate()
    return 0 if written >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
