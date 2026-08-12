"""What does a deeper MLP cost per sim step, and per Inspector redraw?

The depth-1 path streams one hidden unit at a time and never holds the hidden
vector. Any depth above 1 must materialise a layer's activations, so mlp.glsl
ping-pongs two float[MAX_MLP_WIDTH] locals - which is the cost this prices, and
what MAX_MLP_WIDTH and MAX_MLP_DEPTH are set from.

Two traps it respects, both paid for elsewhere in this codebase:

  - The sim does not reproduce itself run to run, so every timed block restores
    the entity buffer from ONE snapshot taken before the first of them. An
    uncontrolled run once scored a format 37% SLOWER than itself.
  - Two runs in separate processes measure the laptop's thermal state rather
    than the shader, so everything here stays in one process and the stacks are
    interleaved rather than run in blocks.

usage: python -m tools.measure_brain_depth [steps] [reps]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import ui  # noqa: F401  - imported first, or services/ hits a circular import
import moderngl

from services.brain_preview import BrainPreview
from services.brains import REGISTRY, generated_brains
from sim import Sim
from state import SimState
from utilities.gl_helpers import pack_brains

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# Depth 1 at the shipped default first: it is the baseline every other row is
# read against, and the path that must not regress.
STACKS = [
    [[16, 0]],
    [[48, 0]],
    [[8, 0], [8, 0]],
    [[8, 0], [8, 0], [8, 0]],
    [[8, 0]] * 4,
    [[8, 0]] * 6,
    [[8, 0]] * 8,
]


def stack_name(layers) -> str:
    return "[" + ",".join(str(w) for w, _ in layers) + "]"


def timed_steps(ctx, sim, st, snapshot, steps) -> float:
    """Milliseconds per step, from ONE restored starting state."""
    sim.entities.write(snapshot)
    sim.apply_state(st)
    sim.update(ctx)
    ctx.finish()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.apply_state(st)
        sim.update(ctx)
    ctx.finish()
    return (time.perf_counter() - t0) * 1000.0 / steps


def timed_preview(ctx, preview, layout, buf, reps=20) -> float:
    """Milliseconds per Inspector redraw. Per-unit tiles re-run the forward
    pass, so this grows with depth as well as with the unit count."""
    preview.render(layout, buf)
    ctx.finish()
    t0 = time.perf_counter()
    for _ in range(reps):
        preview.render(layout, buf)
    ctx.finish()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main() -> int:
    ctx = moderngl.create_standalone_context(require=430)
    sim = Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
              particle_density=0.5)
    preview = BrainPreview(ctx, tile=96)
    m = REGISTRY["mlp"]
    st = SimState()
    st.MUTATION_SCALE = 0.0

    layouts = []
    for layers in STACKS:
        try:
            layouts.append((layers, m.layout_from_settings({"layers": layers})))
        except ValueError as exc:
            print(f"  {stack_name(layers):<20} skipped: {exc}")

    # ONE snapshot, before any timed block. Every stack starts from these exact
    # particles, or what is measured is where the sim happened to wander.
    sim.reset_seed = 0.0
    sim.reset()
    for _ in range(50):
        sim.apply_state(st)
        sim.update(ctx)
    ctx.finish()
    snapshot = sim.entities.read()

    step_ms = {stack_name(ls): [] for ls, _ in layouts}
    for _ in range(REPS):
        for layers, layout in layouts:                # interleaved, not blocked
            sim.realloc_brain_buffers(layout)
            sim.apply_rule(m.random(np.random.default_rng(7), layout))
            step_ms[stack_name(layers)].append(
                timed_steps(ctx, sim, st, snapshot, STEPS))

    print(f"\n  {sim.entity_count} particles, {STEPS} steps x {REPS} reps\n")
    print(f"  {'stack':<20} {'floats':>7} {'units':>6} {'ms/step':>9} "
          f"{'vs [16]':>9} {'ms/draw':>9}")
    base = None
    for layers, layout in layouts:
        buf = ctx.buffer(pack_brains(
            generated_brains(layout, 1.0, 1), layout))
        draw = timed_preview(ctx, preview, layout, buf)
        buf.release()
        ms = float(np.median(step_ms[stack_name(layers)]))
        base = ms if base is None else base
        print(f"  {stack_name(layers):<20} {layout.length:7d} "
              f"{preview.unit_count(layout):6d} {ms:9.3f} "
              f"{ms / base:8.2f}x {draw:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
