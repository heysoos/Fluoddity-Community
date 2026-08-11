"""How fast do particles actually move, and does V Max's slider reach them?

    python -m tools.measure_speed              # every preset
    python -m tools.measure_speed --limit 12   # every Nth, for a quick look
    python -m tools.measure_speed --only Zipper,Karst

V Max is a hard cap on |vel| per step. Its slider has to reach below the
slowest preset and sit above the fastest, or the control appears to do nothing
over most of its travel. Both ends are properties of the running sim, not of
anything derivable on paper, so this steps each preset and reads the entity
buffer directly - no Camera and no CLIP.

Density is taken from PreferencesState rather than passed in: it changes trail
intensity, which feeds back through the sensors into speed, and a measurement
taken at the wrong density picks the wrong range.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import ui  # noqa: F401,E402  prime the services/ui import cycle


def measure(preset_names, steps=400, seed=12345, limit=0):
    """-> [(stem, p50, p99, peak)] of |vel| in canvas units per step."""
    import moderngl

    from services.config_saver import ConfigSaver
    from sim import SIZE_OF_ENTITY_STRUCT, Sim
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
        # Every Nth, not the first N: the subdirectories are alphabetical, so a
        # head slice would sample one corner of the library.
        paths = paths[:: max(1, len(paths) // int(limit))][: int(limit)]
    if not paths:
        print("no presets matched")
        return []

    ctx = moderngl.create_standalone_context(require=430)
    prefs = PreferencesState()
    saver = ConfigSaver()

    print(f"{len(paths)} presets, {steps} steps, world_size={prefs.world_size}, "
          f"density={prefs.particle_density}\n")
    print(f"{'preset':30s} {'p50':>10s} {'p99':>10s} {'peak':>10s}")

    out = []
    for path in paths:
        config = saver.load_from_file(path)
        if config is None:
            continue
        state = SimState()
        rule = saver.apply_config(config, state)

        sim = Sim(ctx, world_size=prefs.world_size, canvas_aspect_ratio="1:1",
                  particle_density=prefs.particle_density)
        sim.apply_state(state)
        sim.apply_rule(rule)
        sim.reset_seed = float(seed)
        sim.reset()
        for _ in range(steps):
            sim.apply_state(state)
            sim.update(ctx)
        ctx.finish()

        raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
        vel = raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)[:, 2:4]
        mag = np.linalg.norm(vel.astype(np.float64), axis=1)
        mag = mag[np.isfinite(mag)]
        if not mag.size:
            continue
        p50, p99 = np.percentile(mag, [50, 99])
        out.append((path.stem, float(p50), float(p99), float(mag.max())))
        print(f"{path.stem:30s} {p50:10.6f} {p99:10.6f} {mag.max():10.6f}")

    ctx.release()
    return out


def report(rows):
    """V Max is stored before the shader's 1/sqrt(world_size), so the slider's
    units are |vel| * sqrt(world_size)."""
    if not rows:
        return
    from state.preferences_state import PreferencesState
    from ui.physics_params import PARAM_BY_NAME

    scale = float(np.sqrt(PreferencesState().world_size))
    typical = np.array([r[1] for r in rows]) * scale
    peak = np.array([r[3] for r in rows]) * scale
    pdef = PARAM_BY_NAME["V_MAX"]

    slowest, fastest = float(typical.min()), float(peak.max())
    print("\n--- speed in V Max units (|vel| * sqrt(world_size)) ---")
    print(f"n                    {len(rows)}")
    print(f"slowest preset p50   {slowest:.6f}   ({min(rows, key=lambda r: r[1])[0]})")
    print(f"median preset p50    {float(np.median(typical)):.6f}")
    print(f"fastest particle     {fastest:.6f}   ({max(rows, key=lambda r: r[3])[0]})")

    print(f"\nslider is 0..{pdef.default_max} with exponent {pdef.power_exponent}")
    for label, v in (("slowest", slowest), ("fastest", fastest)):
        t = (v / pdef.default_max) ** (1.0 / pdef.power_exponent)
        print(f"  {label:8s} sits at {t * 100:5.1f}% of the track")
    if fastest >= pdef.default_max:
        print("\n  FAIL: this range would brake the fastest preset on load")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400,
                    help="frames per preset; enough for a pattern to settle")
    ap.add_argument("--only", default="", help="comma-separated preset stems")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    names = {s.strip() for s in args.only.split(",") if s.strip()}
    report(measure(names, args.steps, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
