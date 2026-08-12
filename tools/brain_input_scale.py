"""What range does the brain's 4D input actually occupy?

Every modality's unit centres are spread over a range chosen by its module
(Gabor CENTER_SCALE, Lenia MU_SCALE). If that range does not match where the
input lives, most of the search space describes readings the particles never
produce - a Gabor envelope sized for +/-2 is near-constant over an input that
never leaves +/-0.1, which degenerates it into a scaled Fourier.

Measured exactly, not inferred: a throwaway copy of entity_update.glsl writes
vec4(ltap.xy, rtap.xy) - the literal argument to calculate_entity_behavior - into
the adopted-brain buffer at binding 2, and this reads it back. The shader is
restored afterwards. Nothing here reimplements the tap maths on the CPU, because
the sensor offsets depend on orientation, sensor angle, gain and world size and a
CPU copy of that would drift.

That buffer holds ONE brain in the app, so this widens it to a row per particle
for the duration of the run. Nothing in the app may do that - see the caveat on
the writeback's scope.

usage: brain_input_scale.py <tree-root> [steps]
"""
import os
import subprocess
import sys
from pathlib import Path

TREE = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400

os.chdir(TREE)
sys.path.insert(0, TREE)

import numpy as np
import ui  # noqa: F401
import moderngl

from sim import Sim
from state import SimState
from services.config_saver import ConfigSaver

SHADER = "shaders/entity_update.glsl"

# The call whose arguments ARE the brain's input.
ANCHOR = ("calculate_entity_behavior(ltap.xy,rtap.xy,orientation,brain_base,"
          "e.pos,cohort,force,strafe,col_params);")
PROBE = ANCHOR + """
    // --- brain_input_scale.py probe ---
    {
        uint pb = index * uint(BRAIN_LEN);
        particle_brains[pb + 0u] = ltap.x;
        particle_brains[pb + 1u] = ltap.y;
        particle_brains[pb + 2u] = rtap.x;
        particle_brains[pb + 3u] = rtap.y;
    }
"""


def widen_probe_buffer(ctx, sim):
    """Give binding 2 a row per particle, which the probe writes into."""
    want = sim.entity_count * sim.brain_layout.length * 4
    if sim.rule_buffer.size == want:
        return
    sim.rule_buffer.release()
    sim.rule_buffer = ctx.buffer(reserve=want)
    sim.rule_buffer.bind_to_storage_buffer(2)


def measure(ctx, sim, saver, name):
    path = Path(TREE) / "physics_configs" / "Core" / f"{name}.json"
    cfg = saver.load_from_file(path)
    if cfg is None:
        return None
    st = SimState()
    rule = saver.apply_config(cfg, st)
    sim.apply_state(st)
    sim.apply_rule(rule)
    widen_probe_buffer(ctx, sim)         # after any layout change apply_rule made
    sim.reset_seed = 0.0
    sim.reset()
    for _ in range(STEPS):
        sim.apply_state(st)
        sim.update(ctx)

    n = sim.entity_count
    stride = sim.brain_layout.length
    raw = np.frombuffer(sim.rule_buffer.read(), dtype=np.float32)
    x = raw[: n * stride].reshape(n, stride)[:, :4]
    x = x[np.isfinite(x).all(axis=1)]

    # A particle sitting on blank canvas reads exactly zero and says nothing
    # about the scale the brain needs to resolve. The lit population is what
    # the unit centres have to cover.
    mag = np.abs(x).max(axis=1)
    lit = mag[mag > 1e-6]
    if lit.size == 0:
        return None
    return {
        "lit_frac": float(lit.size) / float(mag.size),
        "p50": float(np.percentile(lit, 50)),
        "p90": float(np.percentile(lit, 90)),
        "p99": float(np.percentile(lit, 99)),
        "max": float(lit.max()),
    }


def main():
    src = Path(SHADER).read_text()
    if ANCHOR not in src:
        print(f"anchor not found in {SHADER} - the call was edited")
        return 1
    old = subprocess.run(["git", "show", f"HEAD:{SHADER}"],
                         capture_output=True, text=True).stdout
    if old.strip() and old != src:
        print("WARNING: shader differs from HEAD; restoring the WORKING copy")

    presets = sorted(p.stem for p in
                     (Path(TREE) / "physics_configs" / "Core").glob("*.json"))
    try:
        Path(SHADER).write_text(src.replace(ANCHOR, PROBE), newline="")
        ctx = moderngl.create_standalone_context(require=430)
        sim = Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
                  particle_density=0.2)
        saver = ConfigSaver()
        print(f"  {'preset':<14} {'lit':>7} {'p50':>9} {'p90':>9} "
              f"{'p99':>9} {'max':>10}")
        rows = []
        for name in presets:
            try:
                r = measure(ctx, sim, saver, name)
            except Exception as exc:
                print(f"  {name:<14} failed: {exc}")
                continue
            if r is None:
                continue
            rows.append(r)
            print(f"  {name:<14} {r['lit_frac']:6.1%} {r['p50']:9.4f} "
                  f"{r['p90']:9.4f} {r['p99']:9.4f} {r['max']:10.3f}")
        if rows:
            print(f"\n  {'MEDIAN OVER PRESETS':<14} "
                  f"{np.median([r['lit_frac'] for r in rows]):6.1%} "
                  f"{np.median([r['p50'] for r in rows]):9.4f} "
                  f"{np.median([r['p90'] for r in rows]):9.4f} "
                  f"{np.median([r['p99'] for r in rows]):9.4f} "
                  f"{np.median([r['max'] for r in rows]):10.3f}")
    finally:
        Path(SHADER).write_text(src, newline="")
        assert Path(SHADER).read_text() == src, "SHADER NOT RESTORED"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
