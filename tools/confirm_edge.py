"""Is the bounce-mode edge artifact caused by the sampler wrapping?

Runtime-only experiment: flip the canvas textures from repeat to clamp and
re-measure. No source edit, so it cannot leave the tree dirty.

Under wrap mode get_can() already fracts uv itself, so clamping the sampler is
a no-op there - which the wrap presets confirm.
"""
import os
import sys
from pathlib import Path

TREE = r"C:\Users\Heysoos\Documents\Pycharm Projects\fluoddity\.claude\worktrees\brain-modalities"
os.chdir(TREE)
sys.path.insert(0, TREE)

import numpy as np
import ui  # noqa: F401
import moderngl

from sim import Sim
from state import SimState
from services.config_saver import ConfigSaver

STEPS = 600
CASES = ["LavaLamp", "Streamers", "Adrift", "Critters"]


def measure(ctx, sim, saver, name):
    cfg = saver.load_from_file(Path(TREE) / "physics_configs" / "Core" / f"{name}.json")
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
    pos = ents.reshape(-1, 12)[: sim.entity_count, 0:2]
    d = (1.0 - np.abs(pos)).min(axis=1)
    tex = sim.can_textures[0]
    w, h = tex.size
    lum = np.abs(np.frombuffer(tex.read(), dtype=np.float32)
                 .reshape(h, w, 4)[:, :, :3]).sum(axis=2)
    interior = lum[h // 4:3 * h // 4, w // 4:3 * w // 4].mean() + 1e-30
    border = max(lum[0:2].mean(), lum[-2:].mean(),
                 lum[:, 0:2].mean(), lum[:, -2:].mean()) / interior
    return float((d < 0.005).mean()) * 100, float(border), int(st.boundary_conditions)


def main():
    ctx = moderngl.create_standalone_context(require=430)
    sim = Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1", particle_density=0.2)
    saver = ConfigSaver()
    print(f"  {'preset':<11}{'bc':<8}{'REPEAT (shipped)':>24}{'CLAMP':>22}")
    print(f"  {'':<19}{'%at wall':>11}{'border':>13}{'%at wall':>11}{'border':>11}")
    for name in CASES:
        out = []
        for repeat in (True, False):
            for t in sim.can_textures:
                t.repeat_x = repeat
                t.repeat_y = repeat
            out.append(measure(ctx, sim, saver, name))
        (w1, b1, bc), (w2, b2, _) = out
        mode = {0: "bounce", 1: "reset", 2: "wrap"}[bc]
        print(f"  {name:<11}{mode:<8}{w1:10.3f}%{b1:12.2f}x"
              f"{w2:10.3f}%{b2:10.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
