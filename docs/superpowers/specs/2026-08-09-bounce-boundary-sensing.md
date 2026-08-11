# The canvas edge under BOUNCE

**Status:** fixed 2026-08-10. Beyond a wall there is no world: a sensor reads
zero there, and a diffusion tap contributes nothing. Not a regression — see the
three-tree measurement below.

## Symptom

A bright line of particles hugging the canvas border, worst on the left and
bottom edges, with the interior unaffected. Reported from a zoomed screenshot.

## It was not new

Same preset, same seed, 600 steps, `particle_density=0.2`, `LavaLamp`:

| tree | % of particles within 0.5% of the wall | border luminance / interior |
|---|---|---|
| `87d94f4` (pre brain rewrite, pre tile-boundary rewrite) | 4.83% | 10.72x |
| `tournament-mode` @ `1cf194f` | 4.85% | 10.61x |
| `worktree-brain-modalities` | 4.77% | 10.42x |

So it predated both the brain rewrite and `1cf194f`. Probe: `tools/edge3.py`.

## It was a BOUNCE-mode problem

Border/interior luminance over `physics_configs/Core`:

| boundary | presets | border |
|---|---|---|
| wrap | Adrift, Bubbles, Critters | 1.0 - 2.1x |
| bounce | LavaLamp, Streamers | 2.5 - 10.7x |

`get_can()` wrapped uv **only** in wrap mode; otherwise uv left [0,1] and fell
through to the sampler, which has `repeat_x/repeat_y = True` (`sim.py:132`). The
world reflected but the senses WRAPPED: a particle at the left wall steered on
the right edge of the world. `getCan()` in `canvas.frag` leaked identically —
measured, 15.5% of a blob on the left edge arrived at the right edge in 60 steps.

## The wrap was HIDING the artifact, not causing it

This is the finding, and it took three failed fixes to see. Every treatment that
gave the particle a locally accurate reading at the wall made the pile-up
**worse**:

| sensor treatment | LavaLamp border | Streamers border | Streamers % at wall |
|---|---|---|---|
| wrap (shipped) | 10.6x | 2.9x | 4.70% |
| clamp | **16.1x** | — | — |
| mirror (zero-flux) | 11.2x | 8.1x | **10.50%** |
| void | **9.5x** | 3.4x | **1.89%** |

The original diagnosis blamed the steering differential — clamping puts both of
a particle's sensors on the same texel, so it cannot steer. That reasoning is
**wrong here**, and mirroring disproves it: a mirror keeps the two sensors
perfectly distinct and still measured worse than the clamp-free wrap.

What matters is *what they read*. A particle heading into a wall has both
sensors past it. Clamped or mirrored, both read the bright trail piled at that
wall, and a trail-following particle reinforces what it reads — positive
feedback with its own deposit. Wrapping fed it decorrelated data from across the
world, which broke the loop by accident. Void reads empty, which is what is
actually there, and the feedback has nothing to work on.

## The trail leak was a SINK

Same trap on the diffusion side. A zero-flux mirror is the tidier boundary and
it is the wrong one: sealing the border pushed LavaLamp from 10.6x to **14.2x**,
because the leak it replaced had been draining the bright edge. An absorbing
edge keeps that drain, locally, without teleporting anything.

Mass is therefore deliberately **not** conserved at the border under bounce or
reset. `tests/test_canvas_boundary_gl.py` asserts the drain, and asserts it
reaches only as far as diffusion does.

## What shipped

`sense_off_world()` + `sense_uv()` in `entity_update.glsl` (used by `get_can`
and `get_field`), and the matching branch in `canvas.frag`'s `getCan`. Under
wrap both reduce to `texture(sam, fract(p))` — byte-identical to before, so the
three wrap presets cannot have changed.

The trailing half-texel clamp is not cosmetic: `texture()` is bilinear and the
sampler repeats, so an *in-range* coordinate nearer the seam than half a texel
still blends the opposite edge, and uv exactly 0 is a 50/50 blend with it.

Before/after over Core, 600 steps:

| preset | mode | border before | after | interior before | after |
|---|---|---|---|---|---|
| LavaLamp | bounce | 10.72x | **7.45x** | 0.00815 | 0.00812 |
| Streamers | bounce | 2.49x | **0.45x** | 0.00968 | 0.01184 |
| RingOfFire | bounce | 0.00x | 0.00x | 0.02371 | 0.02403 |
| Salt | reset | 0.00x | 0.00x | 0.02976 | 0.02973 |
| Adrift | wrap | 1.03x | 1.09x | 0.00837 | 0.00836 |
| Bubbles | wrap | 2.36x | 1.52x | 0.01199 | 0.01225 |
| Critters | wrap | 0.80x | 1.51x | 0.00814 | 0.01083 |
| Growth | wrap | 0.00x | 0.00x | 0.02653 | 0.02667 |

Interior luminance is flat, so the border genuinely darkened rather than the
interior washing out. The wrap rows move because the sim does not reproduce
run to run — repeated runs of the *unchanged* tree gave Bubbles 1.35/1.94/2.36x
and Critters 0.74/0.80/1.92x, which brackets both columns.

## Still open: the tournament tile disagrees

`confine_sample()` still CLAMPS a tile sensor under bounce, and
`tests/test_tile_isolation_gl.py` asserts it. That is the treatment measured
worst of all at the canvas level. A tile is supposed to be a small world getting
the world's own boundary condition (CLAUDE.md), so it should read void past its
seam too. Not changed here because it would alter tournament results, and the
optimizer's archives were built under the current behaviour.
