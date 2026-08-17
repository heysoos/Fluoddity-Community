# Field injection: a layer stack for texture into the fields

2026-08-17

Inject external and procedural texture — webcam, video, images, noise, custom
GLSL, the sim's own canvas — into Fluoddity's fields, and let those injections
be layered, blended and previewed the way a compositor allows. Replaces the
"Shader Driven Field" override path, which is the current single-writer version
of the same idea.

## What exists today

One RGBA32F texture at canvas resolution: `.xy` is the force field, `.zw` is
the strafe field. It is consumed in exactly two lines of
`shaders/entity_update.glsl` — force adds to `e.vel`, strafe adds straight to
`step_delta` — through `get_field()`, which handles the aspect correction and
the boundary mode.

Three writers, each of which owns the whole texture:

- the mouse brush (`shaders/field_drawing.frag`), additive, with brush modes,
  fill and erase
- a PNG read through a fixed polar-to-cartesian mapping
  (`FieldHandler.load_field_from_image`)
- an override `.frag` from `Documents/Fluoddity/` or `shaders/field_override/`,
  run fullscreen once per render frame with blending disabled

### The three observed defects, and the one cause

**"When turned on, I can't turn it off."** Unticking `shader_driven_field`
stops calling the shader; nothing clears the texture, so the last frame it
wrote persists indefinitely. Enabling the checkbox also force-switches the draw
target to canvas (`ui/advanced_drawing_window.py`), which relabels the Clear
button to "Clear Trails/Canvas" — so from that state there is no visible
control that clears the thing that was turned on.

**"Only the bottom half of the frame has texture."** Not a defect. `march.frag`
is a raymarched chain over a ground plane; rays that hit nothing leave the field
at zero. There is exactly one bundled example and it is idiosyncratic, so the
system cannot be told apart from the shader.

**"Black rectangle artifacts."** Not diagnosed. Reproduce before attributing.

The first is structural and the second is a symptom of the same structure: the
field is a **persistent painted texture with one writer slot and no ownership
model**. Whoever wrote last owns it until something explicitly erases it, and
nothing shows what any one contributor produced.

A fourth, less visible problem: the entire feature is gated on
`advanced_drawing_enabled`, which is only ever "is the Drawing Controls window
open." Closing the window silently stops the shader.

## Approach

An **injector stack**: an ordered list of layers, each binding a source to a
destination through a mapping and a blend mode. Photoshop layers, not a node
graph. A node editor was considered and rejected for now — it is an ImGui node
editor plus evaluation and caching machinery, for expressiveness that almost
never exceeds three nodes in a line. The layer record is a degenerate graph, so
growing into one later costs nothing today.

### The rule that dissolves the bug class

> **The bus is stateless per frame. Any source that needs memory owns its own
> buffer.**

Every destination texture is cleared and rebuilt from the stack every frame, so
turning a layer off removes its contribution by definition — there is nothing
left over to remove. Painting still works because the brush is not
special-cased: it is a source that owns a private accumulation buffer and
participates read-only, exactly as a webcam does. Persistence is a property of a
source, never of the bus.

## Architecture

| file | purpose |
|---|---|
| `state/field_stack.py` | `FieldLayer` / `FieldStack` dataclasses. No GL, no imgui. |
| `services/field_sources.py` | Source registry: descriptor, params, evaluation. |
| `services/shader_params.py` | `parse_shader_params(src)`, a pure function. |
| `utilities/field_bus.py` | Owns destination textures and scratch; runs the loop. |
| `ui/field_stack_window.py` | The layer list and the inspect panel. |
| `shaders/field/` | Built-in source shaders plus the one composite shader. |

`AdvancedDrawingProcessor`'s override path is removed; its brush becomes the
`brush` source. The `advanced_drawing_enabled` gate is removed — the bus runs
whenever the stack holds an enabled layer, independent of any window.

### Data model

```python
# state/field_stack.py
@dataclass
class FieldLayer:
    enabled: bool = True
    source: str = "noise"        # registry key
    params: dict = field(default_factory=dict)   # keyed by uniform name
    mapping: str = "curl"        # rg_direct | polar | gradient | curl | luminance
    destination: str = "force"   # force | strafe | canvas | spawn | param:NAME
    blend: str = "add"           # replace | add | multiply | max
    strength: float = 1.0
    error: str | None = None     # runtime only, never serialized
```

### Evaluation

Once per render frame, before the physics substeps — the slot
`process_override` occupies today. Multiple physics substeps see one field,
which is existing behaviour.

```
if not bus.dirty: return

for each destination texture D in use:
    clear D to zero
    for each enabled, error-free layer L targeting D, in stack order:
        tex = source(L).evaluate()          # returns a texture handle
        D.color_mask = channels_of(L.destination)
        apply_blend_state(L.blend)
        composite(tex, L.mapping, L.strength) -> D
```

Three properties carry the design.

**`evaluate()` returns a texture handle rather than filling a buffer.** GPU
sources render into a shared scratch texture and return it; image, video and
webcam sources return the texture they already own. No copies, and CPU-fed
sources cost no GPU pass at all. Scratch is safe to share because each layer's
composite consumes it immediately after its own evaluation.

**The four blend modes are native GL state, not shader branches.** `replace` is
blending disabled, `add` is `ONE, ONE`, `multiply` is `DST_COLOR, ZERO`, `max`
is `blend_equation = MAX`. There is exactly one composite shader in the system.

**`force` and `strafe` are channel pairs of one texture, written under
`color_mask`.** A force layer cannot touch the strafe channels, and
`get_field()` in `entity_update.glsl` is untouched by stage 1 — only the
producer changes.

**Source and mapping are separate passes on purpose.** Fusing them needs N×M
shader variants, and `gradient` and `curl` must sample the source's
*neighbours*, which a fused source shader cannot do — it only knows its own
fragment.

## Sources

A source has a descriptor (key, label, parameter list) and an `evaluate()`.
Stage 1 ships six:

| key | notes |
|---|---|
| `noise` | fbm / worley / curl. Scale, octaves, speed, domain warp. |
| `image` | A file. Fit: stretch / cover / contain / tile. |
| `gradient` | Linear or radial ramp. Absorbs today's Fill modes. |
| `shader` | A user `.frag`; sliders discovered from the file. |
| `brush` | The mouse. Owns the persistent accumulation buffer. |
| `feedback` | The sim's own canvas. The texture already exists, so it costs one composite pass and nothing else. |

`feedback` closes the loop — the trail field becomes its own forcing field. It
is also the source that can run away, which the composite's non-finite clamp
covers.

Stage 2 adds `video` and `webcam`. Both go through the ffmpeg subprocess pattern
already in `utilities/ffmpeg_recorder.py`, reading raw `rgb24` off stdout
instead of writing to stdin — a webcam is `-f dshow` on Windows. **No new Python
dependency**; ffmpeg is already resolved bundled-or-PATH by `find_ffmpeg()`.
Frames arrive on a worker thread and are uploaded on the main thread; a source
that has no new frame returns its last one, so a 30 fps camera under a 60 fps
render costs nothing extra.

### The `.frag` contract

Annotation is opt-in. **An unannotated shader is a valid shader** — `march.frag`
and anything already written keeps working untouched, with no UI and its
uniforms left at their defaults.

```glsl
#version 430
in  vec2 texcoord;
out vec4 fragColor;

// Always provided; declare only what you use.
uniform vec2  canvas_resolution;
uniform float time;
uniform int   frame_count;
uniform vec2  mouse, prev_mouse;

// Annotate to get a slider.
uniform float speed;    // 0..5 = 1.0       "Speed"
uniform int   octaves;  // 1..8 = 4         "Octaves"
uniform vec3  tint;     // color = 1,.5,0   "Tint"
uniform bool  invert;   // = false          "Invert"

void main() { fragColor = vec4(0.0); }
```

`parse_shader_params(src) -> list[ShaderParam]` reads `uniform <type> <name>;`
plus an optional trailing `// <range> = <default> "<label>"`. Supported kinds:
`float` and `int` (ranged), `bool`, and `vec2`/`vec3`/`vec4` as either a ranged
multi-slider or, with `color`, a colour picker. A uniform with no annotation
yields no `ShaderParam`. Values live in `FieldLayer.params` keyed by uniform
name, and reach the program through `tryset`, which already tolerates a uniform
the shader dropped.

Shader discovery keeps both existing locations — `Documents/Fluoddity/*.frag`
and the bundled directory — and adds `Documents/Fluoddity/shaders/` for new
work. Moving the existing location would strand files users already have.

## Mappings

RGBA in, whatever the destination needs out.

| mapping | out | notes |
|---|---|---|
| `rg_direct` | vec2 | Channels straight through. For hand-authored vector fields. |
| `polar` | vec2 | Hue as angle, value as magnitude. Today's PNG behaviour, preserved so existing `_fields.png` still means the same thing. |
| `gradient` | vec2 | ∇luminance. Particles flow up or down brightness. |
| `curl` | vec2 | ∇⊥luminance. Divergence-free, so particles circulate along contours instead of piling into extrema. Usually the better one for a camera feed. |
| `luminance` | float | For the scalar destinations. |

Each takes an invert and a scale. `gradient` and `curl` are 4-tap.

## Destinations

| key | texture | consumer | stage |
|---|---|---|---|
| `force` | existing RGBA32F `.xy` | `get_field()` in `entity_update.glsl` | 1 |
| `strafe` | existing RGBA32F `.zw` | same | 1 |
| `canvas` | RG32F | an extra pass in `can_update` | 3 |
| `spawn` | R32F | `reset()` in `entity_update.glsl` | 3 |
| `param:NAME` | packed | `calculate_setting()` | 4 |

`spawn` is sampled by rejection inside `reset()` with a fixed candidate count,
so the population takes the shape of the input without a CDF precompute.

`param:NAME` is the invasive one and carries its own gate; see Performance.

## Preview and inspect

Row thumbnails are 96px, filled by a cheap downscale during evaluation and only
while the window is open.

**Hover** peeks: after the same delay as `_delayed_tooltip`, a ~256px preview
floats beside the row. It is read-only — it borrows no rule and pushes nothing
onto the rule stack — so the hover-preview hazards documented in CLAUDE.md do
not apply, and it must not become a path that does.

**Click** pins an Inspect panel with three views:

- `source` — the raw RGBA the source produced.
- `mapped` — what the mapping made of it. For the vector mappings this reuses
  the existing arrow renderer (`shaders/arrow_debug.frag`) rather than adding a
  second one.
- `destination so far` — the composite up to and including this layer, so a
  layer stomping the one beneath it is visible.

Cost: nothing when closed. Image, video and webcam sources are free even open,
being a bare `imgui.image()` of a texture that already exists. Only one layer is
ever inspected at a time, so a procedural source costs one extra pass into a
dedicated inspect texture.

## Error handling

Three classes, three visible behaviours, none of them global. This is the direct
answer to a failing shader currently being indistinguishable from one that does
nothing.

1. **Won't compile.** The layer keeps its last-good program, `layer.error` holds
   the GLSL log, the row draws red with the log inline and a re-compile button,
   and the layer contributes nothing. The rest of the stack keeps running.
2. **File missing.** Image, video or shader path gone. Same treatment.
3. **Non-finite output.** The composite clamps NaN and Inf to zero per fragment,
   so no source author has to remember it and a runaway `feedback` layer cannot
   reach the particles.

`V` recompiles every shader in the stack; one that now builds clears its own
error. A failure raises a notice through `ui/notices.py` — a message that only
reaches the console reads as a no-op, the same reasoning already applied to
saves.

## Performance

The requirement is close to zero slowdown. Three of the four levers are
structural rather than tuning.

**Zero layers costs zero.** Nothing allocated, nothing dispatched, no texture
bound — the same lazy-init discipline as today.

**Static layers cost zero in steady state.** The bus carries a dirty flag and
rebuilds only when something changed: an animated uniform advanced, a video
source got a frame, the brush moved, a slider moved. A loaded image or a fixed
noise field costs one rebuild and then nothing.

**Animated layers cost 1–2 fullscreen passes each, per render frame, not per
physics substep.** Image, video and webcam are 1 — their texture exists, so
there is no source pass. Procedural sources are 2. Against today's override path
which already runs 1, the architecture's own overhead is one composite pass per
layer.

Order of magnitude, as arithmetic rather than measurement: RGBA32F at 647² is
6.7 MB per full read or write, so a composite pass moves roughly 20 MB counting
a 4-tap curl and a masked write — tens of microseconds against a 16.7 ms frame.
At world size 4.0 the canvas is 2048² and the same pass is about 10× that.

**The bus runs at its own resolution, defaulting to half canvas.** A forcing
field is smooth and does not need per-texel canvas resolution the way the trail
canvas does. Half res is 4× cheaper and visually identical for anything but a
camera feed wanted sharp. Settings are 1/1, 1/2, 1/4. This is the lever that
keeps the feature free at large world sizes.

Two consequences. The destination texture's filter becomes `LINEAR`, where the
field texture is `NEAREST` today — at half res, nearest sampling is visibly
blocky. And nothing else has to change: `get_field()` derives its aspect
correction from `textureSize(field_texture, 0)`, so a uniformly scaled texture
carries the same ratio and reads correctly without knowing the bus resolution
exists.

The footer states `<n> passes · <res> · <idle|rebuilding>`, so the guarantee is
on screen rather than promised, and a layer needlessly animating is visible.

### The consumer-side risk

Everything above is producer side, and the producer is cheap because it runs
once per render frame at a controlled resolution. Stages 3 and 4 add
*consumers*, which run per particle per physics substep.

- `canvas` is fine: another pass on a texture already being written.
- `spawn` is nearly free: it only runs inside `reset()`.
- **`param:NAME` is the one to be careful about.** `calculate_setting()` is
  called roughly ten times per particle per step; a texture fetch there is ten
  fetches × hundreds of thousands of particles × N substeps.

So stage 4 carries a rule: the texture path is `#define`-gated and compiles out
entirely when no `param:*` layer exists — the discipline `MAX_MLP_WIDTH` already
uses to stop a feature nobody enabled from taxing every brain in the build. And
it does not ship without a number.

`python -m tools.measure_field_bus` follows the existing convention, with the
trap the other measurement tools document: swap stacks inside **one** process
against **one** entity snapshot, or what is measured is the laptop's thermal
state rather than the feature.

RGBA16F for the destination would halve all of the above. It is a measurement,
not an assumption — fp16 already failed once for the trail canvas, where density
reaches 47 against an fp16 spacing of 0.031.

## Persistence

The stack is a `field_stack` block in the physics config JSON: a list of layer
dicts, travelling with a preset the way `brain_settings` does. Image, video and
shader sources store a path; a moved file yields the missing-file error row
rather than a failed load.

**A config with a `_fields.png` and no `field_stack` synthesises a single
`brush` layer holding that texture.** Every preset already on disk opens
unchanged and no migration runs, the same discipline by which a missing
`encoder.json` means `clip-b32`. `FieldHandler`'s save, load, preview and
clipboard machinery is retargeted from the global field to the brush source's
buffer and otherwise keeps working.

## Tournament interaction

**The bus is disabled while a tournament grid is running.** A correctness
requirement, not a preference: tiles are isolated small worlds, so a field
injected across the whole canvas is shared by every tile. Under Auto or Explore
the optimizer would be scoring the camera feed rather than the genome, and the
entries it admitted would be unreproducible without that exact frame of video.
Same discipline that already forces `color_by_cohort` off in tournament mode.
This belongs in CLAUDE.md as a caveat, not only in the code.

## Staging

Each stage is independently shippable.

**Stage 1 — the bus, force and strafe.** Data model, registry, composite,
dirty flag, bus resolution, layer list, inspect panel, error rows. Sources:
`noise`, `image`, `gradient`, `shader`, `brush`, `feedback`. Removes the
override path. *Done when* a stack of three layers composites correctly, a
disabled layer contributes exactly zero, a broken shader shows one red row while
the others keep rendering, and an existing preset with a `_fields.png` opens
with its brush layer intact.

**Stage 2 — video and webcam.** An ffmpeg subprocess feeding a texture on a
worker thread. Pure addition; no bus changes. *Done when* a camera drives a
force field at full render rate and closing the layer terminates the process.

**Stage 3 — the `canvas` and `spawn` destinations.** New consumers: an inject
pass in `can_update`, and rejection sampling in `reset()`. *Done when* particles
demonstrably react to an injected image through their own rules rather than
being pushed by it.

**Stage 4 — `param:NAME`.** Spatial physics parameters, `#define`-gated, behind
a measured cost gate. Touches `calculate_setting()`, which is synchronised
across `entity_update.glsl`, `canvas.frag` and `sim.py`; the eleven settings do
not fit one RGBA texture, so the packing is designed in this stage rather than
assumed now.

## Testing

**Pure Python.** `parse_shader_params()` against a table of declaration strings,
including unannotated uniforms that must yield no `ShaderParam`; stack round-trip
through JSON; a legacy `_fields.png` with no `field_stack` synthesising exactly
one brush layer.

**Headless GL**, the `tests/test_tile_isolation_gl.py` pattern. Each blend mode
produces the arithmetic it claims. A `force` layer leaves the strafe channels
bit-identical. **A disabled layer contributes exactly zero** — the regression
test for the reported defect. A source that fails to compile leaves the
destination unchanged rather than black. A NaN from a source reaches the
destination as zero.

**UI render**, the `tests/test_archive_window_render.py` pattern, observing both
traps it documents: size the host window taller than the panel, and assert on
the labels a real frame draws rather than on where the call sits.

**Measurement.** `tools/measure_field_bus.py`, per Performance.

## Out of scope

- A node graph. The layer record is a degenerate graph; revisit if linear
  ordering proves limiting in practice.
- Audio as a source or a modulator. It fits the uniform contract cleanly and is
  a natural follow-on, but it is not this spec.
- Migrating the bus into tournament mode. Per-tile injection is a different
  design and would have to answer reproducibility first.
- Making the injected field part of the searchable genome.
