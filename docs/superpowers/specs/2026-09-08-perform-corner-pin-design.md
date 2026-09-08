# Perform mode: corner pinning for a projector that is not square to the wall

Date: 2026-09-08

## The problem

A projector is rarely square to the surface it points at. Off to one side, or
tilted up from a table, it turns the rectangle it is given into a trapezoid on
the wall. Perform mode currently has no answer to this: `PerformWindow.draw`
letterboxes the frame with `fit_rect` and blits it to a screen-aligned quad, so
the only shape it can put on a wall is the one the projector's own optics give
it.

The transformation that fixes this is a **homography** — a projective transform,
the general case of what projectors call keystone correction. Dragging four
corners is **corner pinning**; the map it defines is the unique homography
taking the source rectangle to that quadrilateral.

## What this adds

A calibration mode. Four draggable corners on the laptop, the warped picture and
its guides on the projector, and the corner positions remembered per display.

An uncalibrated setup renders exactly what it renders today.

## 1. The shape of it

Four pieces, each with one job.

**`services/corner_pin.py`** — pure geometry. Solves and inverts the homography,
supplies the default corners, and answers whether a quad is still usable. No GL,
no imgui, no state. This is where the maths is tested.

**`shaders/perform.vert` / `shaders/perform.frag`** — the projector's own display
shader. A full-framebuffer pass that samples through the inverse homography and
draws the calibration guides.

**`state/perform_state.py`** — the live working copy: whether calibration is on,
which corner is held, where the four corners are.

**`ui/perform_window.py`** — the proxy canvas the corners are dragged on.

`services/perform_view.py` is **not touched**. It already publishes an identity
`DisplayFrame` at the projector's own shape; the warp is a property of the
display, not of the render.

## 2. Why the warp is a fragment pass and not a moved quad

Two ways to put a texture into an arbitrary quadrilateral.

Move the quad's four vertices and carry a third texcoord component `q`, so the
perspective divide the rasteriser already performs does the work. Four vertices,
no matrix.

Or draw one full-framebuffer quad, compute the inverse homography on the CPU,
and per fragment map the display point back into source space:

```glsl
vec3 p = inv_h * vec3(uv, 1.0);
vec2 src = p.xy / p.z;
```

**The fragment pass is chosen.** The solve becomes a pure function testable with
no GL context at all, which is what makes section 7 possible; the shader is three
lines; and "outside the quad is black" falls out of a bounds check on `src`
rather than needing a stencil or a discard rule per edge.

What it costs is that the shader now covers the whole framebuffer rather than the
letterboxed rect — about two million single-tap fragments at 1080p. The projector
already redoes the entire particle pass once per motion-blur sample, measured at
~1.5 ms each; a full-screen texture tap does not register against that. Do not
re-litigate this on performance grounds without measuring both.

## 3. The corners, and what they replace

Four points in **normalized display space**: `(0,0)`–`(1,1)` over the whole
projector framebuffer, ordered TL, TR, BR, BL.

**`v = 0` is the BOTTOM.** These are GL texture coordinates, the same convention
`canvas_view_rect` uses, because they are consumed by a fragment shader. ImGui
draws top-down, so the proxy canvas in section 6 flips `v` when it converts a
handle position to a corner and back. Naming the corners TL and BR is a claim
about the wall, not about which end of the range they sit at: TL is
`(0, 1)`. A stored calibration is in this space, so the flip lives in the widget
and nowhere else.

The flip cancels itself for a symmetric quad, so a centred rectangle validates
any orientation bug you like — the tests in section 7 use an ASYMMETRIC
trapezoid for that reason, the same discipline the capture-crop tests follow.

**The corners are the authority and the letterbox is their default.** `fit_rect`
is not removed and not composed with — it supplies `default_corners` and nothing
else. One code path, so a calibrated and an uncalibrated projector differ only in
the numbers in the matrix.

The map is `H`, taking the source unit square to those corners. Writing
`H = [[a,b,c],[d,e,f],[g,h,1]]`, each corner contributes two rows:

```
a*u + b*v + c - g*u*x - h*v*x = x
d*u + e*v + f - g*u*y - h*v*y = y
```

Eight equations, eight unknowns, one `numpy.linalg.solve` on an 8x8. The shader
is given `inv_h = inverse(H)`. It is recomputed only when a corner or the
framebuffer size moves.

**A degenerate quad is refused, never applied.** Dragging a corner across a
diagonal folds the map; a collapsed edge makes it singular. `inverse_homography`
returns `None` when the solve is non-finite or ill-conditioned, and the caller
keeps the last good matrix and raises a notice. `is_convex` — the cross products
of consecutive edges agreeing in sign — is what the drag itself is guarded on, so
the refusal is the backstop rather than the normal path.

## 4. What the projector shows while calibrating

The live warped image, with guides over it, behind a `show_guides` uniform that
is zero unless calibration is on.

**The grid is drawn in SOURCE space and warped by the same matrix.** That is what
makes it useful: a line that is straight in the source arrives on the wall bent
exactly as the picture is, so a grid that looks straight against the wall *is*
the alignment. A grid drawn in display space would stay straight whatever the
warp did and could not report anything.

Corner markers draw in display space at the four destination corners, with the
one named by `held_corner` highlighted — so the corner being dragged on the
laptop identifies itself on the wall.

## 5. State and persistence

`PerformState` gains the session-only working copy, none of it persisted, the
same rule `enabled` already follows:

- `calibrating: bool` — calibration mode is on
- `held_corner: int` — which corner the pointer has, `-1` for none. A readout.
- `corners` — the four points, or `None` for "use the default"

`PreferencesState` gains `perform_calibrations: dict[str, list[list[float]]]`,
keyed by **`device_key`** — the display identity that survives rearranging
monitors, which the display picker already uses. Keying by name would give a
laptop panel and a projector one entry between them, which is the defect
`MonitorInfo.key` exists to prevent.

`main.py` loads the entry for the monitor the window actually landed on and
writes back when a corner moves, mirroring the `perform_monitor` /
`_perform_requested` pair it sits beside. A display with no entry gets the
default corners; nothing is written for a projector nobody calibrated.

**The new field must be classified `NOT_UNDOABLE`** — it is the physical
alignment of an external display, not the look of the simulation, the same
reasoning `perform_monitor` carries. `tests/test_undo_fields.py` derives its
cases from `__dataclass_fields__`, so an unclassified field fails the suite.

## 6. Controls

A calibration section in the Perform Mode panel: a "Calibrate" checkbox, a proxy
canvas showing the display at its true aspect with the quad and four draggable
handles, a "Reset corners" button, and a readout of the held corner.

**Not inside a collapsing header.** A control the user has to find cannot live in
a folded section; the encoder combo that shipped inside a default-closed header
and was unreachable is the precedent.

The section is disabled unless perform mode is actually running — there is no
display to calibrate against otherwise, and a control that cannot do what it
offers is worse than no control.

It follows the panel rules the rest of the app follows: `push_settings_width()`,
`wrap_row()` for button rows, and a distinct `scope` on its notices so two
"Dismiss" buttons cannot collide on one ImGui id.

## 7. Verification

**`tests/test_corner_pin.py`** — pure and exact, no GL. Identity corners give an
identity map; a hand-computed point lands where it should under an ASYMMETRIC
trapezoid; `H` and `inv_h` round-trip; a folded quad returns `None`; a collapsed
edge returns `None`; `default_corners` reproduces `fit_rect`; and one test pins
the orientation outright — raising only the TL corner must move the picture at
the TOP of the display, never the bottom.

Every quad in these tests is asymmetric. A symmetric one cancels the `v` flip
and passes under either convention.

**A new GL test drives the real shader.** Unlike `tests/test_perform_view_gl.py`
this one **can assert exactly**: fed a synthetic source texture instead of the
particle pass there is no additive splat race, so a known trapezoid moving a
known pixel to a known place is a precise check rather than a tolerance against a
measured noise floor. It must use a synthetic texture for that reason.

**`tests/test_perform_window_render.py`** gains the calibration controls,
asserting on the labels a real frame draws rather than on where the calls sit —
and the host window must be sized tall enough to contain them, or they draw no
vertices and the assertion is a coin flip.

**`tools/drive_perform.py`** is extended to drive calibration end to end. A
render test drives a bare mixin and a GL test drives a bare window; neither runs
the assembled app, and a window body naming a missing attribute passes the whole
suite and crashes on the first frame.

## 8. Deliberately not in scope

Edge blending and soft-edge masks. Multiple projectors. Mesh warping or any
non-planar surface. Lens and barrel distortion. Named venue presets. Per-corner
numeric entry.

A flat wall at any angle is exactly a homography, and every item on that list is
a different feature with its own design.
