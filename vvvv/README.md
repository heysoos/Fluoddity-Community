# Fluoddity as a vvvv source

Fluoddity is a Python/ModernGL GPU particle system. This folder runs it as a
source inside **vvvv beta 42 / DX11**: Spout for textures both ways, OSC for
parameters, `ShellExecute` for start and stop.

Spout is a shared GPU texture — no encode, no readback, and both processes must
be on the same adapter. The bridge needs `SpoutGL` and `python-osc`
(`requirements.txt`); Fluoddity itself runs without them.

```
  vvvv beta 42 / DX11                     Fluoddity
  -------------------                     ---------
  FluoSpout: Spout Receiver   <- Spout -  --spout-out fluoddity
  FluoSpout: Spout Sender     -  Spout -> --spout-in vvvv_fluo
  FluoSpout: ShellExecute x2  ---------->  Fluoddity.bat / Kill.bat

  FluoMod:   8x OSCsend :5011 -- OSC --->  --osc-port 5011
  FluoMod:   OSCgetSeq  :5012 <- OSC ---   --osc-return-port 5012
```

## Files

| File | |
|---|---|
| `Fluo (VJ).v4p` | host patch. Open this first: both modules, previews, a test source, a Spout watchdog |
| `FluoSpout.v4p` | textures and lifecycle |
| `FluoMod.v4p` | all OSC: tempo, modulation, palette, mask, field, structure, presets |
| `OSCgetSeq (Devices).v4p` | latching OSC receive (section 9) |
| `OSCsend (Devices).v4p`, `OSCget (Devices).v4p`, `Spout/` | dependencies, bundled so the patch opens on a bare vvvv |
| `Fluoddity.bat`, `Kill.bat` | launcher and stopper |

---

## 1. Launcher

    Fluoddity.bat [width] [height] [fluoddity-dir] [extra args...]
    Fluoddity.bat 1920 1080
    Fluoddity.bat 1280 720 C:/code/Fluoddity-Community --offscreen

`FluoSpout` fills all three in: width and height from `Send Res`, the directory
from the patch's own location on disk. If `Run` opens a console that cannot find
`main.py`, set `FLUO_DIR` at the top of the bat — it is the fallback when the
third argument is empty.

The `pushd` inside is required. Fluoddity loads shaders by relative path
(`read_shader('shaders/camera.vert')`), so it must run with the repo root as its
working directory.

**Resolution is window size.** Render targets follow the window framebuffer.
`--offscreen` hides the window and disables vsync; without it the ImGui sliders
stay usable on a control monitor.

`Kill.bat` sends `/fluoddity/cmd quit` first and force-kills only what is left,
matching on the command line rather than on `python.exe`. The graceful step
matters: a force-kill skips `releaseSender()` and leaves a stale name in Spout's
registry, after which the next run is silently renamed `fluoddity_1`.

---

## 2. `FluoSpout.v4p` — textures and lifecycle

Sends no OSC.

**In:** `Texture` (node), `Send Res` (2), `Offscreen`, `Run`, `Kill`,
`Force Update`, `Receiver Name` (`fluoddity`), `Sender Name` (`vvvv_fluo`).
**Out:** `Texture` (node), `Handle`, `Res`.

`Run` and `Kill` are bangs. `Handle` is the receiver's raw handle: zero means
nothing is on the other end, and it is what the watchdog watches.

**`Send Res` sizes both ends** — the `vvvv_fluo` sender, and the `width height`
pair on Fluoddity's command line. Re-`Run` for a new size to reach Fluoddity;
the sender follows immediately.

Bat paths are not pins. `ShellExecute` gets the plain relative filename and the
repo location is computed from where the patch sits.

---

## 3. `Fluo (VJ).v4p` — host patch

One `FluoSpout`, one `FluoMod`, previews of both directions, and an animated
quad through `Blur (DX11.TextureFX)` as a stand-in source until you patch your
own into `Texture In`.

| | |
|---|---|
| `Enable` | `TogEdge` bangs `Run` up, `Kill` down. App lifetime follows the toggle |
| `Texture In` | your source into `FluoSpout.Texture`, via `AvoidNIL (DX11.Texture)` |
| `Texture Out` | the particles |
| `Preset`, `FPS`, `Presets` | through to `FluoMod` (section 9) |

**Spout watchdog.** `Handle` → `Count (Value)` → `EQ`, gated by `Enable`; while
enabled with no handle, an `LFO` at 0.23 s pulses `Force Update` through a
`FrameDelay`. A receiver that missed its sender — app still booting, or
restarted under a running patch — re-latches within a few frames instead of
staying black.

The two modules are independent and do not talk to each other. Take either one
into your own patch; give both the same `OSC Port`.

---

## 4. OSC map

Generated from Fluoddity's registry (`ui/physics_params.py`), so ranges match
the sliders and a parameter added there appears automatically.

Every parameter answers to `/fluoddity/<name>` (absolute) and
`/fluoddity/n/<name>` (0..1, mapped through its range). Groups also take a
comma-separated string on one address:

| Address | Slices | Applies to |
|---|---|---|
| `/fluoddity/params`, `/fluoddity/n/params` | 12 | SimState |
| `/fluoddity/structure` | 5 | SimState |
| `/fluoddity/rule_seed` | 1 | SimState |
| `/fluoddity/palette` | 6 | preferences |
| `/fluoddity/mask` | 9 | preferences |
| `/fluoddity/field` | 5 | preferences |
| `/fluoddity/mosh` | 11 | preferences |
| `/fluoddity/clock/tick`, `/bang`, `/auto`, `/audio` | 1 each | mod matrix |
| `/fluoddity/mod/base`, `/depth`, `/source`, `/period`, `/lo`, `/hi` | 12 each | mod matrix |
| `/fluoddity/preset` | 1 | action (section 9) |
| `/fluoddity/presets/refresh` | bang | action (section 9) |
| `/fluoddity/cmd` | string | `quit` |

Return traffic on 5012: `/fluoddity/fps`, `/fluoddity/presets/count`,
`/fluoddity/presets/names`, `/fluoddity/preset/name`.

A per-parameter address also accepts the whole group's payload. `Add (String
Spectral)` at its default `Bin Size` of -1 concatenates a spread into one
comma-separated string, and `OSCsend` then repeats that string on every address
in its Address spread — so each parameter arrives holding the entire group.
Each address reads the slice at its own index, which is the pairing the patch
meant. Setting `Bin Size` to 1 instead, so each address carries its own single
value, works identically. A payload that is not a number at all is reported once
per address and then ignored.

The extra groups are **not** appended to `params`. The physics registry is what
might gain a parameter later, and appending would shift every slice index below
it in the patch. Appending *within* a group is safe — the bulk path zips specs
against however many values arrive. Never insert in the middle.

### Physics parameters

Slice order of `params`, and of `FluoMod`'s `Depth` / `Source` / `Period` /
`Lo` / `Hi`:

| # | Name | Range | |
|---|---|---|---|
| 0 | `sensor_gain` | 0 .. 10 | |
| 1 | `sensor_angle` | -1 .. 1 | hard-limited |
| 2 | `sensor_distance` | 0 .. 3 | |
| 3 | `mutation_scale` | 0 .. 1 | |
| 4 | `global_force_mult` | 0 .. 2 | |
| 5 | `drag` | -1 .. 1 | hard-limited |
| 6 | `axial_force` | -1 .. 1 | |
| 7 | `lateral_force` | -1 .. 1 | |
| 8 | `strafe_power` | 0 .. 0.5 | |
| 9 | `trail_persistence` | 0 .. 1 | hard-limited |
| 10 | `trail_diffusion` | 0 .. 1 | hard-limited |
| 11 | `hazard_rate` | 0 .. 0.05 | hard-limited, power-scaled t³ |

Values outside a hard limit are clamped; garbage payloads are ignored.

**The ImGui sliders do not track OSC-driven values.** OSC is injected after
`ui.get_state()`, which sidesteps slider clamping and `ParameterLockService`, so
the sliders stay usable as a manual override.

**`FluoMod` does not send `params`.** The modulation matrix uses each
parameter's current value in Fluoddity as its base, unless a `/mod/base` row
sets one. Dial the sound of it in Fluoddity, let vvvv move it. Add a `params`
send of your own if you want vvvv to own the values instead.

**`OSCsend` spreads its `Address` pin.** It wraps `OSCEncoder (Network)`, whose
`Spread as Bundle` pin is wired to the module's `Bundle` toggle. With
`Bundle = 1` one node sends every address at once as an OSC bundle; python-osc
unpacks and dispatches each message separately.

**Hazard.** Driving several parameters to their extremes at once can NaN the
simulation — it goes black and stays black until reset. High `mutation_scale`
with high `sensor_gain` is the reliable way to find it. The per-parameter
`Lo` / `Hi` clamps exist for this.

---

## 5. Sending a vvvv texture into the particles

Two halves, and both are needed; the vvvv half alone does nothing visible.

**vvvv:** connect a texture to `Texture In`. It leaves as `vvvv_fluo`.

**Fluoddity:** Extras → Advanced Drawing, then Shader Driven Field, then pick
`spout.frag`. All three are reachable over OSC from `FluoMod` (section 8), which
is the easier route.

| Mode | Behaviour | Good for |
|---|---|---|
| 0 channels | `r,g` → force xy, `b,a` → strafe zw, each `x*2-1` | flow maps, normal maps, signed noise |
| 1 luminance gradient | particles climb toward bright regions | photographic and soft imagery |
| 2 gradient perpendicular | particles circulate along contours | hard-edged graphics, text, bars |

Hard edges make huge gradients and nothing in between, so mode 1 slams particles
into the bright areas and they stall there. Mode 2 keeps them moving along the
edges. Start `field scale` at 0.5–1.0.

The shader samples across the canvas UV range, so a 16:9 source is squashed into
a 1:1 canvas. Usually fine for a force field. With no sender the field is exactly
zero, so it is safe to leave armed.

To see the field itself, switch Fluoddity's view mode to Force Field or Strafe
Field.

---

## 6. Datamosh — the particles move the picture

The inverse of section 5. There the incoming texture drives the particles; here
the particles drive the incoming texture, dragging its pixels along their own
motion. Nothing is drawn over the feed — Fluoddity outputs the vvvv frame
processed. So in the rig this replaces a composite: send the texture in, take
the moshed version out, and do not blend the particles over it downstream.

The flow field is the velocity map the simulation already keeps. `brush.frag`
splats each particle's velocity into the brush texture, and the canvas is that
same quantity accumulated under Trail Persistence and Trail Diffusion. Both are
RG32F velocity vectors (section 12), so the mode costs two small fullscreen
passes and changes nothing about the physics.

| What the particles do | What the pixels do |
|---|---|
| local velocity direction | direction of the shift, rotated by `swirl` |
| speed × local density | shift amplitude, normalized against the frame's own average |
| trail persistence / diffusion | how smooth and how long-lived the smear is |

Fluoddity: Extras → Advanced Drawing → Datamosh → Enabled. Over OSC it is
`/fluoddity/mosh`, or `/fluoddity/mosh_enabled` on its own.

| # | Parameter | Range | Notes |
|---|---|---|---|
| 0 | `mosh_source` | 0..2 | 0 Spout feed · 1 particle frame · 2 feed + particles, smeared in together |
| 1 | `mosh_amount` | 0..0.25 | ceiling on the shift per displayed frame, as a fraction of the frame |
| 2 | `mosh_contrast` | 0..4 | how selective the response is. 1 = average activity moves at half strength |
| 3 | `mosh_scale` | 0..8 | stroke size, as a blur of the flow field |
| 4 | `mosh_flow_mix` | 0..1 | 0 trail map (dense, long smears) · 1 this frame's splat (sparse, sharp) |
| 5 | `mosh_swirl` | -1..1 | rotates the shift off the flow, ±90° at ±1 — push becomes vortex |
| 6 | `mosh_refresh` | 0..1 | live source returning per frame. 1 = no accumulation, 0 = full melt |
| 7 | `mosh_block` | 0..64 | macroblock size in px for the flow lookup, 0 = off |
| 8 | `mosh_chroma` | 0..1 | per-channel displacement spread — colour fringing on the fast tears |
| 9 | `mosh_ink` | 0..1 | crisp particles added back on top, after the mosh |
| 10 | `mosh_enabled` | 0/1 | master switch, last so a shorter bulk send cannot toggle it |

**`mosh_contrast` and `mosh_scale` are the shape of the effect.** The flow
magnitude is normalized against the frame's own average before the response
curve sees it — the mean is read straight off the top of the flow field's mip
chain, which costs nothing — so neither knob needs recalibrating when the world
size, the particle count or the config changes.

`contrast` is `x^c / (1 + x^c)`, which passes through half strength at average
activity for every setting. It changes how *selective* the effect is, not how
strong:

- **below 1** the curve flattens: quiet and busy regions shift by similar
  amounts and the whole frame drifts together in broad strokes.
- **1** is neutral.
- **above 1** it sharpens toward a threshold — only the busiest streaks move at
  all, and everything else stays crisp.

`scale` is a mip level of the flow field, i.e. a blur of the field itself.
Averaging vectors lets opposing directions cancel, so raising it strips out the
individual streaks and leaves the large-scale coherent motion: the picture moves
in regions rather than tracing filaments. It softens the shift as well, so
`amount` usually wants to come up with it. `contrast` high with `scale` at 3–5
gives selective macro-brushes; `contrast` low with `scale` at 0 gives a fine
all-over shimmer.

`mosh_refresh` is the character knob, and it trades against `amount`. Both the
feedback buffer and the live frame are sampled at the shifted coordinate, so the
whole range warps and refresh only decides how much of the shift is inherited
from previous frames:

- **1** — a plain displacement map on the live frame, no memory. One frame's
  worth of shift is small, so this is where `amount` wants to be high (0.05–0.2).
- **~0.05–0.15** — artefacts survive ten to twenty frames. The classic melt;
  keep `amount` low, around 0.02, and let it accumulate.
- **0** — nothing resets and the image is consumed entirely. **Reseed** refills
  the buffer from the live source.

`mosh_block` is what makes it read as a codec artefact rather than a
displacement map: whole blocks shift together instead of the field flowing
smoothly. 8–32 px is the usable range.

The output is the size of the incoming texture, so it goes back out at the
resolution it arrived at, and the pass ignores camera pan and zoom to keep it
pixel-aligned with the feed. The flow field is fitted over it with a **cover**
crop rather than stretched, so displacement angles are not skewed — but a canvas
aspect matching the feed still uses the whole field. Set Canvas Aspect Ratio to
16:9 for a 16:9 source.

With no sender connected, the feed modes fall back to the particle frame rather
than moshing black — the particles then smear themselves.

---

## 7. `FluoMod.v4p` — tempo, palette, mask

### Pins

| Pin | Slices | |
|---|---|---|
| `Enable` | 1 | gates `Push` on the palette, mask, seed and preset senders |
| `Tick` | 1 | seconds per beat. Also drives the internal `LFO` that makes `clock/bang` |
| `Audio` | 1 | 0..1 envelope for modulation source 7 |
| `Depth` | 12 | modulation amount, 0..1 |
| `Source` | 12 | modulator per parameter |
| `Period` | 12 | cycle length in beats |
| `Lo` / `Hi` | 12 | normalized clamp per parameter |
| `Send` | 1 | bang: push the clock and modulation bundle once |
| `Color` | 1 | palette colour, split to hue/sat/value inside |
| `Palette Shift` | 1 | hue offset added to `Color`, in turns |
| `Palette Spread` | 1 | band half-width in turns, 0..0.5 |
| `Palette Stops` | 1 | 0/1 continuous, 2+ quantizes |
| `Palette Mix` | 1 | 0 = the simulation's own hues, 1 = fully colour-driven |
| `Mask` | 9 | absolute values, order below |
| `Mask Range` | 9 | full-scale of each `Mask` slice |
| `cohorts` | 1 | part of `structure`, on its own pin |
| `Send Structure` | 1 | bang: fire the structure send |
| `Rule Seed` | 1 | 0..1, own address and own sender |
| `lum . grad` | 1 | field mode (section 8) |
| `field scale` / `strafe scale` | 1 each | field strengths (section 8) |
| `Preset` | 1 | preset index (section 9) |
| `OSC Port` | 1 | 5011 |

**Out:** `Preset` (name of what is loaded), `Presets` (how many), `FPS`.

### Senders

Eight `OSCsend` nodes, not one:

| Sender | Carries | Fires on |
|---|---|---|
| bundle | `clock/tick`, `/bang`, `/auto`, `/audio` (1 each) + `mod/depth`, `/source`, `/period`, `/lo`, `/hi` (12 each) | the `Send` bang, or its `Push` toggle to stream |
| palette | `/fluoddity/palette` | `Push`, gated by `Enable` |
| mask | `/fluoddity/mask` | `Push`, gated by `Enable` |
| seed | `/fluoddity/rule_seed` | `Push`, gated by `Enable` |
| preset | `/fluoddity/preset` | `Push`, gated by `Enable` |
| field | `/fluoddity/field` | `Push`, own toggle, on |
| structure | `/fluoddity/structure` | `Change` on the values, `OnOpen (VVVV)`, or `Send Structure` |
| refresh | `/fluoddity/presets/refresh` | `LFO`, every 5 s |

Streaming and fired groups cannot share a node: a fired sender transmits its
whole message, so a seed sharing with structure re-sends the cohort count on
every reseed, and a seed sharing with the palette visibly collapses the colour.

The bundle's `Add (String Spectral)` takes a computed `Bin Size` spread —
`Select (Value)` builds `1,1,1,1` and `12,12,12,12,12`, `Cons` joins them — so
one concatenation is cut into nine payloads. **Change the address list and you
must change the bin sizes with it.**

### Modulation

Fluoddity evaluates the matrix once per rendered frame:
`base + auto * depth * (mod - 0.5) * 2`, clamped to `lo`/`hi`, then through the
parameter's range. OSC arrives in vvvv-mainloop steps; evaluating on the Python
side is what makes a stepped fader a smooth sweep. The clock free-runs from
`Tick` and snaps to zero on each `clock/bang`.

**`clock/auto` is the master gate and it is not wired to anything.** It sits at
0, so `Depth` does nothing until you set it. First thing to check when
modulation seems dead.

**`clock/bang` only re-syncs while it keeps arriving.** With the bundle on the
`Send` bang it goes out once and the clock free-runs. Turn the bundle's `Push`
on to stay locked.

| # | Source | | # | Source |
|---|---|---|---|---|
| 0 | Off | | 4 | Random hold |
| 1 | Saw | | 5 | Random smooth |
| 2 | Sine | | 6 | Beat envelope (decay) |
| 3 | Triangle | | 7 | Audio, from the `Audio` pin |

Random hold seeds from `hash(floor(beats/period), index)`, so it is
deterministic and cannot drift.

A starting point:

| Parameter | Depth | Source | Period | |
|---|---|---|---|---|
| `hazard_rate` | 0.6 | 6 beat envelope | 1 | re-emission on the beat, the nearest thing to an emitter |
| `sensor_gain` | 0.4 | 2 sine | 8 | structure breathing over two bars |
| `global_force_mult` | 0.3 | 2 sine | 4 | overall agitation |
| `sensor_angle` | 0.5 | 4 random hold | 4 | a new shape on the bar |
| `trail_persistence` | 0.2 | 3 triangle | 16 | slow trail-length drift |

### Palette

`Color` goes through `AvoidNIL (Color)` and `HSV (Color Split)`; `Palette Shift`
is added to the hue. Each particle's own hue still picks its position *within*
the band, so cohorts stay distinguishable.

Slice order: `mix, hue, sat, value, spread, stops`.

`mix` is how far particle hue is pulled toward the band; `spread` is how wide
the band is. **`mix = 1` with `spread = 0` collapses the frame to one flat
colour** — a zero-width band, not a bug. For "my hue with the particles' own
variety", keep `mix` at 1 and open `spread` to 0.1–0.2.

A narrow band is also what stops overlaps washing out. Colour is deposited
additively, so a full-spectrum field averages toward white wherever particles
pile up; inside a narrow band the overlaps are neighbouring hues.

`spread 0.5` with `stops 2` gives a clean complementary pair.

### Mask

Weights how much particles do in each region, from the incoming `vvvv_fluo`
texture and/or a built-in vignette. **A bias, not a stencil.**

Slice order, with `Mask Range` full-scale in brackets: `ink` (1),
`force/strafe` (1), `pull grad` (2), `ceiling` (1), `gamma` (4), `blur` (8),
`vignette` (1), `vignette softness` (1), `source` (3).
`source`: 0 feed, 1 vignette, 2 max of both, 3 feed × vignette.

The mask reads the incoming texture whether or not `spout.frag` is the selected
field override. The two are independent: **field decides where particles go,
mask decides how much they do there.**

`ink` first. Particle alpha gates trail *deposition* as well as display, so a
quiet region lays down weaker trails, weakens sensor attraction there, and draws
particles back toward the busy areas on their own. The confinement is emergent;
`pull grad` is a nudge on top.

Raise `blur` on hard-edged sources — enough blur turns shapes into regions
rather than outlines to trace.

Expect the frame to dim as `ink` rises: less total ink. Compensate with
Brightness / Exposure, or raise the `ceiling`.

Palette and mask also exist in Fluoddity's UI (Preferences → Palette, Drawing
Controls → Activity Mask) and persist in `preferences.config`. Both live in
**preferences, not the physics config** — SimState's appearance fields are
overwritten on config load, and a preset recall must not change the colour the
whole show is running in.

### Structure

Slice order: `cohorts, initial, boundary, colour-by-cohort, hue sensitivity`.
Fired, not streamed — streaming would overwrite a just-loaded physics config
every frame. `rule_seed` has its own address and sender so a reseed does not
re-send the cohort count.

`rule_seed` is re-read by the shader every frame, so changing it swaps the whole
rule set instantly with no reallocation. It is the most dramatic
change-on-the-downbeat control the simulation has. Feed the pin from an `ERand`
on `Tick`, or a `Random (Value)` banged on the bar; the sender pushes on change.
Note `ERand` moves on about half its periods by design, so a 4-beat period gives
a new pattern roughly every other bar.

**Cohorts do two jobs.** With `colour-by-cohort` on, hue is
`hash(floor(cohort))`, so one cohort means one colour — and one cohort is also
what makes a reseed replace the entire behaviour instead of reshuffling an
ensemble. Turn `colour-by-cohort` off and hue moves onto
`hue_sensitivity × behaviour`, so a single cohort still shows a full spectrum.
Measured: 1 cohort with colour-by-cohort off gives both a strong reseed response
(0.211) and a wide hue spread (0.338); raising cohorts to 64 for colour variety
halves the reseed response (0.110). Decouple with the switch, not the count.

---

## 8. Field

`/fluoddity/field`, 5 slices:

| # | Slice | | From |
|---|---|---|---|
| 0 | `spout_field_mode` | 0 channels, 1 luminance gradient, 2 perpendicular | `lum . grad` |
| 1 | `spout_field_scale` | force, 0..4 | `field scale` |
| 2 | `spout_strafe_scale` | strafe, 0..4 | `strafe scale` |
| 3 | `shader_driven_field` | arm switch | constant 1 |
| 4 | `advanced_drawing_enabled` | arm switch | constant 1 |

`lum . grad` is a toggle, so it reaches modes 0 and 1. It passes through an
`Add (Value)` — raise that node's second input to 1 for mode 2, or replace the
toggle with a value pin.

**The arm switches are on this address deliberately.** Closing Fluoddity's
Drawing Controls window sets `advanced_drawing_enabled` false and silently stops
the incoming texture driving anything — indistinguishable from a dead Spout
link, a wrong share name or a crashed app. Held at 1, the field re-arms every
frame and a stray click cannot switch it off.

---

## 9. Presets

131 physics configs, flattened by `services/preset_index.py` into one numbered
list: **Core, then Advanced, then Custom, alphabetical within each.** Custom is
last because it is the only category that grows during a session, so saving a
config appends instead of renumbering everything the patch refers to.

| To Fluoddity | | |
|---|---|---|
| `/fluoddity/preset` | integer | load that index, edge-detected. Out-of-range is ignored, never clamped |
| `/fluoddity/presets/refresh` | bang | rescan and re-publish. Triggered by message *arrival*, so a constant would fire once — hence the `Random (Value)` behind it |

| Back to vvvv (5012) | |
|---|---|
| `/fluoddity/presets/count` | how many |
| `/fluoddity/presets/names` | the whole list, comma-separated, ~1.1 kB for 131 |
| `/fluoddity/preset/name` | what just loaded, whichever side loaded it |
| `/fluoddity/fps` | liveness |

One `OSCgetSeq (Devices)` takes all four with its `Address` pin spread and
`Port` 5012. Names come off the **`Message`** (String) pin; `Output` carries the
numbers — `CAR` for the count, `CDR`'s last slice for the fps. The list is
re-published every 5 s (`PRESET_REPUBLISH_SECONDS`), so a receiver that appears
later still gets it.

### One datagram per frame

**vvvv's `OSCget (Devices)` takes one datagram per frame off its
`UDP (Network Server)` node.** Messages arriving back-to-back inside a frame are
reduced to the first; the rest are dropped silently. It is not a size limit —
two-byte probes are dropped the same way, and raising `Buffer Size` changes
nothing.

Both halves of the fix ship here:

- **Python:** `publish_presets()` queues, `_pump_publish_queue()` emits exactly
  one message per frame.
- **vvvv:** `OSCgetSeq (Devices).v4p` latches decoded values with `S+H (String)`,
  so a value received on one frame is still readable later instead of existing
  for a single frame.

The vvvv → Fluoddity direction is unaffected — python-osc handles bursts — but
the clock and modulation rows are bundled anyway, which makes them one datagram.

### Watercolor

Selecting a preset over OSC keeps the watercolor mode that is on screen, exactly
as Fluoddity's own Load menu does ("Preserve current watercolor mode"). Seven of
the shipped presets store `watercolor_mode: true` — **8** Curls, **12**
HungryHungryHippos, **18** Searching, **64** Hatchets, **65** Inky, **68**
Jellies, **128** Zebra — and without this they would flip the frame to a white
ground mid-show. Watercolor stays a mode you own with `V`.

---

## 10. When nothing appears

### Sender and receiver on different GPUs

On a hybrid laptop (NVIDIA plus Intel iGPU), **a D3D11 shared handle cannot be
opened across adapters.** Spout's registry — name, handle, dimensions — lives in
plain shared memory and is adapter-agnostic, so it reads back perfectly while
`FromSharedTexture` yields nil: plausible handle, right dimensions, `Info` shows
`-1` and Format `Unknown`, preview black, no error anywhere.

Fluoddity logs the adapter it opened on at startup. Force both onto the discrete
GPU in the NVIDIA control panel or Windows Graphics Settings.

### Stale sender name

A force-killed Fluoddity leaves its name in Spout's registry, and the next run
is silently renamed `fluoddity_1`, which a receiver watching `fluoddity` never
sees. `Kill.bat` avoids it by asking the app to quit first; Fluoddity prints a
warning when it happens anyway.

### Then

1. `fluoddity` missing from the sender list means the app is not publishing —
   check the console for shader-compile errors, usually a wrong working
   directory.
2. Bang `Force Update` to re-latch the receiver. `Fluo (VJ).v4p` does it for you.
3. Image upside down: add `--no-invert`. OpenGL renders bottom-up, DX11 expects
   top-down, so the flip is on by default.

### When OSC has no effect

1. **`clock/auto` at 0** gates all modulation off. Most likely answer.
2. `Enable` gates `Push` on palette, mask, seed and preset.
3. `OSC Port` must match `--osc-port` (5011).
4. For modulation: `Depth` above zero, `Source` not 0, and the bundle actually
   sent — bang `Send` or turn its `Push` on.
5. `Push` sends on *change*. A static payload sent before the app was listening
   never arrives and nothing resends it. Nudge a control or toggle `Enable`.

**The failure mode to know is a silent slice shift.** The bundle is one
concatenated string cut by a fixed `Bin Size` spread. Anything contributing
fewer slices than expected — a nil input, a spread that shrank — does not error;
it shifts every address after it, and Fluoddity decodes each value into the
wrong parameter while the patch carries on sending. Keep `AvoidNIL` or a
`Default` behind anything that can go nil. An `Add (String Spectral)`'s `Slices`
output compared against the sum of its bin sizes catches it immediately:
`1+1+1+1 + 12×5` = **64** for the nine addresses here.

---

## 11. Performance

240k particles at the default `world_size = 0.40`, under a `#version 450`
compute shader, sharing one GPU with everything else vvvv is doing. Measure
early. Levers, cheapest first: `Send Res`, `world_size`, `speedmult`.

Measured alone on an RTX 3080 Ti at that setting: **~10.4 ms/frame, 96 fps.**
Spout across the process boundary at 1280×720: ~158 fps. The mask and datamosh
passes allocate nothing until switched on and free their buffers when switched
off. Datamosh is one fullscreen pass at the feed's resolution, run once per
displayed frame rather than once per motion-blur sample.

---

## 12. What the bridge adds inside Fluoddity

| New | |
|---|---|
| `bridge/` | OSC, Spout in/out, arg parsing, the modulation matrix |
| `services/preset_index.py` | the stable numbered preset list |
| `utilities/particle_mask.py`, `shaders/particle_mask.frag` | the mask pass, FBO at canvas/4, outputs `vec4(activity, weight, gradient.xy)` |
| `utilities/datamosh.py`, `shaders/datamosh.frag`, `shaders/datamosh_flow.frag` | the datamosh passes: a mipped flow field, then ping-pong RGBA16F at the feed's resolution |
| `state/render_params.py` | frozen `PaletteParams` / `MaskParams` / `MoshParams` |

Modified: `entity_update.glsl` (palette hue band, `get_mask()` mirroring
`get_field()`'s UV maths), `preferences_state.py`, `ui_state.py`, `main.py`,
`sim.py`, `simulation_runner.py`, the Palette, Activity Mask and Datamosh UI,
and `requirements.txt`.

Everything defaults to a no-op: with nothing sent, the output is unchanged.

### Canvas and brush are RG32F

`CANVAS_COMPONENTS = 2` in `sim.py`. The physics only needs the velocity vector
— the sensor taps read `.xy`, and `canvas.frag`'s draw and fill modes write
`.xy`. The `.z`/`.w` channels were `brush.frag`'s `vec4(vel, .01, 1)`,
accumulated kernel weight at two scales, so `w` was always exactly `100*z`: two
channels carrying one quantity that nothing read.

Measured at 240k particles / 863×485 canvas: **10.94 → 10.44 ms median, ~5%**
(91 → 96 fps), stable across four runs against a 1.4% control arm. It shrinks to
3.5% at `world_size` 1.0, so the bottleneck at higher settings is particle
compute, not canvas bandwidth. Set `CANVAS_COMPONENTS = 4` to revert.

**Caveat:** `frame_assembly.frag`'s emboss height is `length(.xy)` rather than a
density channel. Dense regions with opposing trail directions cancel and emboss
flat, so Emboss Intensity may need re-dialling. 2 of the 131 presets use emboss.

---

## 13. Tests

The bridge is pure Python and tested. The suites live in the development repo,
not here; run them after touching the palette, mask, modulation or preset path.

| Suite | |
|---|---|
| `test_osc.py` | address forms, clamping, garbage payloads, bundle dispatch |
| `test_injection.py` | OSC lands in `SimState` for all 12 physics parameters |
| `test_spout_field.py` | `spout.frag` drives the field as specified |
| `test_mod_matrix.py` | 8 sources, bang re-sync, zero depth is exact passthrough, clamps |
| `test_mask_palette.py` | GPU readback of the palette band and mask weighting |
| `test_vvvv_wire.py` | a frame of patch output, addresses read out of the `.v4p` |
| `test_live_app.py` | boots the real app offscreen, ~200 frames, reads the texture back |

`test_live_app.py` snapshots and restores `preferences.config` via `atexit`,
because it drives the real `App` and `App.cleanup()` saves preferences.
