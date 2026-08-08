"""Render every saved preset headlessly and save one 224px tile each.

Exists to answer the stage-0 question honestly (spec 12.1). Measuring CLIP's
spread over `runs/*/frames/` was misleading twice over: those images are best
tiles from prompt-driven searches, so they were pre-selected for CLIP
legibility, and they were produced by builds that predate the NaN and
cohort-colouring fixes. Presets are the neutral sample - they are the patterns
a person actually curated as distinct.

Runs with no window. A standalone GL context plus FrameAssembler, TileCapture
and CaptureBlit is the same tonemap-and-readback path the tournament uses, so
these crops are what CLIP would really be handed.

    python -m tools.capture_presets out_dir --steps 2000

Then measure:

    python -m tools.clip_spread_check out_dir/*.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui.physics_params -> ui/__init__ -> ui.core -> services.config_saver. It
# resolves only when `ui` is imported first, which is what main.py happens to
# do. Prime that order, as tests/conftest.py does.
import ui  # noqa: F401,E402

# view_mode 0 is the plain canvas. Tournament capture uses the cam_brush view
# (2), but that needs the camera's window-sized brush target; the canvas view is
# the neutral choice for a spread measurement and is noted in the output.
VIEW_CANVAS = 0


def find_presets(root: Path) -> list[Path]:
    out: list[Path] = []
    for sub in ("Core", "Advanced"):
        d = root / sub
        if d.is_dir():
            out.extend(sorted(d.glob("*.json")))
    return out


def capture_preset(ctx, sim, assembler, cap, blit, config_saver, path, steps, seed):
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

    assembled = assembler.assemble_frame(
        sim.can, 1, 0,
        view_mode=VIEW_CANVAS,
        screen_aspect=1.0,
        brightness=1.0,
        canvas_resolution=sim.can.size,
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
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    import moderngl
    from PIL import Image

    from services.capture_blit import CaptureBlit
    from services.config_saver import ConfigSaver
    from services.tile_capture import TileCapture
    from sim import Sim
    from utilities.frame_assembler import FrameAssembler
    from utilities.paths import get_app_physics_configs_dir

    presets = find_presets(get_app_physics_configs_dir())
    if args.limit:
        presets = presets[: args.limit]
    if not presets:
        print("no presets found")
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        print(f"no GL context: {exc}")
        return 2

    sim = Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
              particle_density=args.density)
    assembler = FrameAssembler(ctx, sim.can)
    cap = TileCapture(ctx, grid=1)
    blit = CaptureBlit(ctx)
    config_saver = ConfigSaver()

    print(f"{len(presets)} presets, {args.steps} steps each, "
          f"{sim.entity_count} particles, canvas {sim.can.size}")
    t0 = time.time()
    written = 0
    for i, p in enumerate(presets):
        try:
            crop = capture_preset(ctx, sim, assembler, cap, blit, config_saver,
                                  p, args.steps, args.seed)
        except Exception as exc:
            print(f"  [{i + 1}/{len(presets)}] {p.stem}: FAILED ({exc})")
            continue
        if crop is None:
            print(f"  [{i + 1}/{len(presets)}] {p.stem}: no frame")
            continue
        Image.fromarray(crop).save(out / f"{p.parent.name}_{p.stem}.png")
        written += 1
        mean = float(crop.mean())
        note = "  <-- near black" if mean < 2.0 else ""
        print(f"  [{i + 1}/{len(presets)}] {p.stem:32s} mean {mean:6.1f}{note}")

    dt = time.time() - t0
    print(f"\nwrote {written}/{len(presets)} crops to {out} in {dt:.0f}s")
    print("view_mode = canvas (not the tournament's cam_brush view)")
    print(f"\nnow run:  python -m tools.clip_spread_check {out}/*.png")
    return 0 if written >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
