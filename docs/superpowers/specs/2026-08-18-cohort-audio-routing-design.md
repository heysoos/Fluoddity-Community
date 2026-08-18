# Routing audio bands to cohorts, painted per mapping

Date: 2026-08-18

Status: EXPERIMENTAL. This feature is on trial and is built to be removed
without disturbing anything it sits beside. See "How to remove this" at the
end, which is as much a part of the design as the rest.

## The problem

An audio mapping modulates the whole swarm. `modulate()` returns one float per
physics parameter, `AudioRuntime.update()` puts it in a `replace()`d `SimState`
and `sim.apply_state` pushes it as that parameter's `slider_value`. Every
particle reads the same number, so a rig can make the field pulse but cannot
make PART of the field pulse.

Cohorts are the natural handle for that. `get_cohort(index)` already slices the
particle buffer into up to 144 groups and every particle knows which one it is
in, but nothing outside a sweep's linear ramp ever varies a parameter by
cohort - and a sweep is a ramp, not a selection.

What is missing is the ability to say: bass drives Axial Force on this quarter
of the swarm, hats drive Strafe Power on that quarter, at the same time.

## What this adds

1. A per-mapping COHORT MASK, painted by dragging across a strip of cells.
2. A per-cohort modulation path on the GPU, so a masked mapping's contribution
   reaches only the cohorts it names.

The mask is a property of one mapping, alongside its shaper. Nothing about the
band, the target or the rig as a whole is scoped; two mappings on the same
target may carry different masks.

The case that shape exists for: bass ADDS to Sensor Gain over the first half of
the cohorts while highs SUBTRACT from Sensor Gain over the second half. That is
one Sensor Gain row with two bands bound, and two drawer tabs carrying a mode
and a mask each. Nothing about it is a special case - for a cohort in the first
half only the bass row contributes to `A`, in the second half only the hi row
does, and `gain` stays 1 because neither is a multiply row.

Masks may OVERLAP, and where they do the contributions sum, as they already do
for two unmasked rows on one target. A cohort in NO mask gets `gain 1,
offset 0` and sits at the slider value.

## 1. The mask

144 bools per mapping - `MAX_COHORTS` from `services/cohort_tiling.py`, the
finest anyone can ever paint - held as a mask over the NORMALISED cohort axis
rather than over cohort indices.

- A live cohort `c` of `n` reads slot `floor(c / n * 144)`.
- Painting cell `i` of `n` writes every slot in
  `[floor(i/n*144), floor((i+1)/n*144))`, which is one slot at n=144 and
  eighteen at n=8.

So the mask RESCALES when the cohort count moves: a mask covering the first
quarter at 64 cohorts still covers the first quarter at 128. Changing the count
writes nothing at all - it is a pure re-read - so it cannot lose anything. The
one lossy direction is real and inherent: painting at a coarse count writes wide
spans, so returning to a fine count gives blocks rather than the detail that was
there before.

All-on is the default and means "every cohort", which is what an unmasked
mapping and every rig saved before this feature both come to.

An EMPTY mask means the mapping is idle, not that it covers everything. A row
painted blank should stop, and "All" is one click away.

## 2. The affine reduction

The modulation chain is affine in the base value, so a cohort's whole
contribution collapses into one multiply and one add. For a target with
per-target strength `Sp` and global strength `Sg`, over the mappings whose mask
covers cohort `c`:

```
A = sum over add/subtract rows of  sign * s * depth * span
M = product over multiply rows of  (1 + s * depth)
S = Sp * Sg

gain[c]   = 1 + S * (M - 1)
offset[c] = S * A * M
```

`s` is the post-shaper signal, `span` is `hi - lo` off the live slider range.
None of these depend on the base, which is what makes the reduction exact
rather than an approximation - and it is why multiply mode works under a sweep,
where the base differs per particle and the CPU cannot know it.

The shader then applies, after `calculate_setting`:

```
v = v * gain[c] + offset[c];
v = clamp(v, lo, hi);
```

`lo`/`hi` are the parameter's hard limits where it declares them and its slider
range otherwise - the same bounds `modulate()` already clamps to.

Because this lands ON TOP of `calculate_setting`'s result rather than replacing
`slider_value`, a masked mapping reaches parameters a sweep has made deaf. That
is a capability the global path does not have and deliberately does not gain
here; see section 6.

## 3. Where each piece lands

### Which parameters can carry a mask

Not all of them, and the panel must say so. A parameter is maskable only if it
reaches a `calculate_setting` call site inside `entity_update.glsl`, where a
cohort exists to index with. Ten do:

```
SENSOR_GAIN  SENSOR_ANGLE  SENSOR_DISTANCE  MUTATION_SCALE  GLOBAL_FORCE_MULT
DRAG  AXIAL_FORCE  LATERAL_FORCE  STRAFE_POWER  HAZARD_RATE
```

Three modulation targets are left out and cannot be masked at any cost:
`TRAIL_PERSISTENCE` and `TRAIL_DIFFUSION` are evaluated in `canvas.frag`, which
has no cohort - the trail is a property of a texel, not of a particle - and
`TIME_SCALE` is a `plain_uniform`, one float for the whole canvas. `V_MAX` is
absent for a different reason: it is `off_at_max`, so `physics_targets()`
already excludes it from modulation entirely.

Their drawer tabs draw no strip and say why, rather than offering a control that
does nothing - which is the "declared but never read" defect CLAUDE.md names
twice.

New `services/cohort_audio.py` owns everything conceptual:

```python
COHORT_AUDIO_PARAMS: tuple[str, ...]   # row order, one home
MASK_SLOTS = 144

def slot_of(cohort: int, n: int) -> int
def paint_span(cell: int, n: int) -> tuple[int, int]
def covers(mask, cohort: int, n: int) -> bool
def build_arrays(mappings, targets, signals, states, strengths,
                 global_strength, dt, deaf, n_cohorts,
                 rate_scale, held) -> tuple[np.ndarray, bool]
```

`build_arrays` returns an `(len(COHORT_AUDIO_PARAMS), 144, 2)` float32 array of
gain and offset, and a flag saying whether anything is masked at all - which is
what `COHORT_AUDIO_ACTIVE` carries, so an unmasked rig costs one branch. It calls
the same shaper primitives `modulate()` calls; `modulate()` itself is not
touched.

New `shaders/cohort_audio.glsl`, prepended to `entity_update.glsl` by
`shader_prepend` the way the brain header already is. It declares the SSBO at
binding 5 (0, 2, 3 and 4 are taken), the `CA_*` row constants, and:

```glsl
float cohort_audio(float v, int row, float cohort) {
    if (!COHORT_AUDIO_ACTIVE) return v;
    ...
}
```

New `ui/cohort_strip.py` holding the paint widget and nothing else.

Changed, minimally:

- `services/audio_mapping.py` - one optional field on `Mapping`.
- `services/audio_runtime.py` - one call to `build_arrays`, parked on
  `self.cohort_audio`. NOT an extra return value: `update()` is unpacked into a
  pair at its one call site, so widening the tuple would touch `main.py` in a
  way that has to be undone rather than deleted.
- `main.py` - one line, `sim.set_cohort_audio(self.audio_runtime.cohort_audio)`
  after `apply_state`.
- `state/audio_in_state.py` - the mask in `_mapping_to_dict` and its loader.
- `ui/audio_reactive_window.py` - one call into the strip from the drawer tab,
  and a marker on the signal chip.
- `sim.py` - a buffer reserve, `bind_to_storage_buffer(5)`, and a write. It goes
  OUTSIDE the frame-constant uniform block, beside the tournament uniforms: the
  arrays move every frame while a rig is playing, and the cached block is
  skipped whenever the program has not changed.
- `shaders/entity_update.glsl` - `cohort_audio(...)` wrapped around each of the
  `calculate_setting` call sites named in `COHORT_AUDIO_PARAMS`.

`calculate_setting` itself is NOT modified. It must stay character-identical
across `entity_update.glsl`, `canvas.frag` and `sim.py`, and `canvas.frag` has
no cohort to index with.

The row order is the one thing shared between Python and GLSL that neither can
check at runtime, exactly the hazard `MULTI_LOAD_CONFIG_SIZE` names. A test
parses the `CA_*` defines out of the shader and asserts they match
`COHORT_AUDIO_PARAMS` index for index.

## 4. The strip

One cell per live cohort, drawn from the mask through `slot_of`. At 144 it is a
barcode; that is accepted.

- Press a cell to start painting. The direction is set by that first cell: press
  a lit one and you erase, press a dark one and you draw.
- Drag to continue. Release anywhere ends it.
- Shift-click fills from the last pressed cell.
- Buttons: All, None, Invert, Every 2nd, Every 3rd, Every 4th. The stride
  presets are what interleave two bands through the field rather than clumping
  them into opposite corners.
- A readout below: how many cohorts of how many, or "this row is idle".

It lives in the drawer's per-signal tab, above the shaper controls, because the
mask is a per-mapping attribute exactly as the shaper is. A target row can carry
six mappings, so no single strip summarises one; the bound signal's chip gets a
marker instead, saying that something on this target is scoped without saying
what.

Each cell is an `invisible_button` inside the row's `push_id`, so no two cells
share an ID. The drag carries across frames in private `_snake_case` UI state.

## 5. Persistence

The mask is written to the rig only when something is painted out, so an
unmasked row adds nothing to the file and every existing rig loads as
all-cohorts with no migration. `PERSISTED_FIELDS` is unchanged - the mask rides
inside `mappings`, which is already there.

The mask takes part in `Mapping.__eq__` like every other field, so
`_save_last_rig`'s value diff sees a mask edit as the change it is.

## 6. What this deliberately does not do

- **The global path is untouched.** A parameter with no masked mapping goes
  through `modulate()` and `slider_value` exactly as it does today, and
  `deaf_targets()` still applies to it. Folding both into one per-cohort path
  would be tidier and would let audio reach swept parameters everywhere - but it
  would rewrite the path every existing rig depends on, which is the wrong trade
  for a feature on trial.
- **Brain mappings are out.** Brain decode scales are one set of numbers per
  brain slot, and per-cohort brains exist only when no rule is loaded. That is a
  separate feature.
- **The tournament is out.** `AudioRuntime.update()` already returns early when
  Auto or Explore is running, because modulating physics mid-comparison moves
  what is being compared.
- **`hue_sensitivity` is not a target,** so this cannot be used to paint colour
  onto cohorts. `color_by_cohort` is how you see the groups while aiming.

## 7. Testing

- `tests/test_cohort_audio.py` - the mask maths. Rescaling holds proportions
  across a count change; painting at a coarse count then re-reading at a fine
  one is blocky but not shifted; an empty mask is idle and a full mask is
  identity.
- The affine reduction is checked against `modulate()` itself: for a mask
  covering every cohort, `gain*base + offset` must equal what `modulate()`
  returns for that base, over add, subtract and multiply rows and a range of
  strengths. That is the property that keeps the two paths from drifting.
- The shader/Python row order test described in section 3.
- A GPU test that runs the sim with a mask covering half the cohorts and asserts
  the other half did not move.
- The strip needs a test that DRIVES THE MOUSE - press, move, release - in the
  manner of `tests/test_map_wheel.py`. A test that asserts the cells were drawn
  passes just as happily when the drag does nothing.

## 8. How to remove this

In the order that leaves the app runnable at every step:

1. Delete the strip call and the chip marker in `ui/audio_reactive_window.py`;
   delete `ui/cohort_strip.py`.
2. Delete the `build_arrays` call and `self.cohort_audio` in
   `services/audio_runtime.py`, and the one line in `main.py`.
3. Delete the buffer reserve, the bind and the write in `sim.py`.
4. Delete the `cohort_audio(...)` wrappers in `shaders/entity_update.glsl`, the
   prepend, and `shaders/cohort_audio.glsl`.
5. Delete the field on `Mapping` and its two lines in `state/audio_in_state.py`.
   Rigs carrying a mask key still load: `_mapping_from_dict` reads named keys
   one at a time and never sees the rest.
6. Delete `services/cohort_audio.py` and `tests/test_cohort_audio.py`.

Nothing that survives has been rewritten, so nothing that survives needs
retesting. While the feature is in, this file is its home for measured facts; if
it stays, the one-paragraph rule goes to CLAUDE.md and this document keeps the
evidence.
