"""Edge pile-up under REAL presets, particles and trail together.

usage: edge3.py <tree-root> [steps]
"""
import os
import sys
from pathlib import Path

TREE = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600

os.chdir(TREE)
sys.path.insert(0, TREE)

import numpy as np
import ui  # noqa: F401
import moderngl

from sim import Sim
from state import SimState
from services.config_saver import ConfigSaver

from sim import SIZE_OF_ENTITY_STRUCT

# Never hardcoded: the entity struct has been resized once already,
# and a stale stride reads other fields as positions rather than failing.
STRIDE = SIZE_OF_ENTITY_STRUCT // 4

PRESETS = ["Adrift", "Bubbles", "Critters", "Growth", "LavaLamp",
           "RingOfFire", "Salt", "Streamers"]


def run(ctx, sim, saver, name):
    path = Path(TREE) / "physics_configs" / "Core" / f"{name}.json"
    cfg = saver.load_from_file(path)
    if cfg is None:
        return None
    st = SimState()
    rule = saver.apply_config(cfg, st)
    sim.apply_state(st)
    sim.apply_rule(rule)
    sim.reset_seed = 0.0
    sim.reset()
    for _ in range(STEPS):
        sim.apply_state(st)
        sim.update(ctx)

    ents = np.frombuffer(sim.entities.read(), dtype=np.float32)
    pos = ents.reshape(-1, STRIDE)[: sim.entity_count, 0:2]
    ca = 1.0
    half = np.array([np.sqrt(ca), 1.0 / np.sqrt(ca)])
    d = ((half - np.abs(pos)) / half).min(axis=1)

    tex = sim.can_textures[0]
    w, h = tex.size
    img = np.frombuffer(tex.read(), dtype=np.float32).reshape(h, w, 4)
    lum = np.abs(img[:, :, :3]).sum(axis=2)
    interior = lum[h // 4:3 * h // 4, w // 4:3 * w // 4].mean() + 1e-30
    border = max(lum[0:2].mean(), lum[-2:].mean(),
                 lum[:, 0:2].mean(), lum[:, -2:].mean()) / interior

    return {
        "wall_0.5%": float((d < 0.005).mean()) * 100.0,
        "wall_2%": float((d < 0.02).mean()) * 100.0,
        "mode": int(st.boundary_conditions),
        "border_lum": float(border),
        # Absolute, because the ratio alone cannot tell a darker border from a
        # brighter interior - and an "improvement" that is really the interior
        # washing out is not one.
        "interior_lum": float(interior),
    }


def main():
    ctx = moderngl.create_standalone_context(require=430)
    sim = Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
              particle_density=0.2)
    saver = ConfigSaver()
    print(f"{TREE}")
    print(f"  {'preset':<12} {'bc':<7} {'%<0.5% wall':>12} {'%<2% wall':>10} "
          f"{'border/interior':>16} {'interior':>12}")
    for name in PRESETS:
        r = run(ctx, sim, saver, name)
        if r is None:
            print(f"  {name:<12} (not found)")
            continue
        bc = {0: "bounce", 1: "reset", 2: "wrap"}[r["mode"]]
        print(f"  {name:<12} {bc:<7} {r['wall_0.5%']:11.4f}% "
              f"{r['wall_2%']:9.3f}% {r['border_lum']:15.3f}x "
              f"{r['interior_lum']:11.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
