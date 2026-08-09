# Deferred: the canvas edge under BOUNCE

**Status:** diagnosed, not fixed. Deferred 2026-08-09 to finish the brain
modalities. Not a regression — see the three-tree measurement below.

## Symptom

A bright line of particles hugging the canvas border, worst on the left and
bottom edges, with the interior unaffected. Reported from a zoomed screenshot.

## It is not new

Same preset, same seed, 600 steps, `particle_density=0.2`, `LavaLamp`:

| tree | % of particles within 0.5% of the wall | border luminance / interior |
|---|---|---|
| `87d94f4` (pre brain rewrite, pre tile-boundary rewrite) | 4.83% | 10.72x |
| `tournament-mode` @ `1cf194f` | 4.85% | 10.61x |
| `worktree-brain-modalities` | 4.77% | 10.42x |

So it predates both the brain rewrite and `1cf194f`. Probe:
`scratchpad/edge3.py` (particles + trail, over eight Core presets).

## It is a BOUNCE-mode problem

Measured over the Core presets, border/interior luminance:

| boundary | presets | border |
|---|---|---|
| wrap | Adrift, Bubbles, Critters | 1.0 - 2.1x |
| bounce | LavaLamp, Streamers | 2.5 - 10.7x |

`get_can()` wraps uv **only** in wrap mode:

```glsl
vec2 uv = p / (2.0 * half_extent) + 0.5;
if(get_particle_boundary_conditions() == 2) uv = fract(uv);
return texture(canvas, uv);
```

Otherwise uv leaves [0,1] and falls through to the sampler, which has
`repeat_x/repeat_y = True` (`sim.py:132`). Under bounce the world REFLECTS but
the senses WRAP: a particle at the left wall steers on the right edge of the
world. The same inconsistency `1cf194f` fixed for tournament tiles, never fixed
for the canvas itself.

## The obvious fix is wrong — measured

Clamping the sampler to the edge roughly doubles the artifact:

| preset | shipped (repeat) | clamped |
|---|---|---|
| LavaLamp | 10.58x | **16.06x** |
| Streamers | 2.50x, 3.7% at wall | **12.78x, 14.1% at wall** |

Because clamping puts BOTH sensors on the same texel within `sample_dist` of the
wall, so the steering differential is identically zero — a dead band all round
the border, exactly as documented for tiles in CLAUDE.md. Do not reach for
`clamp()` here. Probe: `scratchpad/confirm_edge.py` (runtime-only, flips
`repeat_x/repeat_y` and re-measures; no source edit).

## What to do instead

MIRROR the sensor sample back inside on a reflective wall — a zero-flux
(Neumann) boundary, which is what bounce physically implies. Fold with a
triangle wave rather than clamping, so there is no dead band, and no teleport
across the world as wrapping gives. This parallels `confine_sample()`'s wrap
branch, which already exists for tiles and would want a third branch.

The trail diffusion in `getBlur` (`canvas.frag`) needs the matching treatment —
it currently wraps across the canvas border through the same sampler.

## Before changing it

This alters the look of every bounce-mode preset in the curated library,
LavaLamp most of all. Re-measure the whole of `physics_configs/Core` before and
after, and get the appearance change signed off — it is a judgement call, not a
correctness one.
