# Perform mode: the picture on a projector, the controls on the laptop

Date: 2026-08-27

## The problem

There is one window. It carries the simulation and every ImGui panel on top of
it, so showing the work on a projector means showing the controls with it — and
hiding the controls means flying blind. `main.py` creates exactly one
`glfw.create_window` and `Camera.render()` draws into `ctx.screen`; nothing in
the codebase enumerates monitors or opens a second surface.

## What this adds

A second, borderless, input-less window filling a chosen monitor, showing
exactly what the laptop's canvas shows and nothing else. The sim runs once. The
laptop keeps every panel it has now.

## 1. The shape of it

Three pieces, each with one job.

**`services/perform_window.py` — `PerformWindow`.** Owns the second GLFW window
and its `moderngl.Context`. Knows nothing about `Sim`, `UI` or `App`; it is
handed a frame and draws it.

**`camera.py` — publishes what it drew.** One frozen dataclass recorded at the
end of `render()`. This is the whole interface between Camera and
`PerformWindow`, and Camera stays unaware perform mode exists.

**`state/perform_state.py` — `PerformState`.** The live flag, the remembered
monitor, and a notice string.

## 2. Why a second GL context, and why textures cross it

A second window needs a second `moderngl.Context`: OpenGL shares textures,
buffers and programs between contexts created with `share`, but **not** VAOs or
FBOs, which is what moderngl's `Context` is mostly made of. So the perform
window compiles its own copy of the display shader and builds its own VAO, and
reaches the already-rendered frame through `ctx.external_texture(glo, size,
components, samples, dtype)`.

Verified on this machine before the design was written: a second window created
with `glfw.create_window(w, h, title, None, main_window)`, a second
`moderngl.create_context()`, and an `external_texture` wrapper around a texture
written in the first context read back its exact bytes. The main context was
still usable afterwards.

The external wrapper is rebuilt when — and only when — the incoming texture's
`glo` or size changes. `FrameAssembler` recreates its accumulation texture
whenever `total_samples` or the canvas size moves, so the wrapper cannot be
made once and kept.

## 3. What the projector shows

**Exactly the laptop's canvas rectangle, letterboxed to fit, with no ImGui.**

`Camera.render()` chooses between two different textures — the accumulated one
while the sim runs, a freshly assembled one while it is paused — so the perform
window must mirror *the texture the camera just drew*, never
`camera.assembled_texture`. Mirroring the latter freezes the projector on a
stale frame every time you pause, which is a normal thing to do mid-show.

So `Camera.render()` ends by recording:

```python
@dataclass(frozen=True)
class DisplayFrame:
    texture: moderngl.Texture
    cam_pos: tuple[float, float]
    cam_zoom: float
    tex_size: tuple[int, int]
    window_size: tuple[int, int]   # the MAIN framebuffer's size
```

as `self.last_display`. `cam_pos` and `cam_zoom` are the values actually sent to
the shader, so `cam_brush_mode`'s override of them to `(0, 0)` and `1` is
already folded in and needs no second implementation.

`PerformWindow.draw(frame)` replays those uniforms verbatim, passing
`window_size` = the **main** framebuffer size rather than its own. The display
shader therefore composes the identical image the laptop composed, and the only
thing the perform window contributes is where that image lands: the viewport is
set to `fit_rect(main_aspect, perform_size)`, centred, with the rest of the
screen cleared black.

`fit_rect(src_aspect, dst_size) -> (x, y, w, h)` is the letterbox, and it is
pure arithmetic with no GL in it, which is what makes it testable:

- `src_aspect >= dst_aspect` → full width, bars top and bottom.
- `src_aspect < dst_aspect` → full height, bars left and right.
- Equal aspects → the full destination, exactly, with no off-by-one rounding
  bar. This case is asserted, because a one-pixel bar on a matched projector is
  the kind of thing nobody notices until it is on a wall.

## 4. The overlays

Two overlays are baked into the rendered frame rather than drawn by ImGui: the
parameter-sweep reticle and the draw-brush circle. Both are added inside
`frame_assembly.frag`'s `final_sample` block, additively, into the same
accumulation texture the display path reads — so there is no clean copy of the
frame to hand the projector without either splitting display from accumulation
or assembling twice.

**Perform mode turns both overlays off, on both screens, for as long as it is
on.** No shader change, no second assembly, no extra GPU work. The cost is that
the brush circle is unavailable on the laptop while performing.

Suppression happens at the two points in `orchestrate_frame` where the values
are *derived*, not at the call sites that consume them: `sweep_reticle_visible`
is forced false beside the existing `is_recording or screenshot_in_progress`
test, and `draw_trail_mode` is forced false where `_render_camera_view` derives
it. Both the direct assembly call and `SimulationRunner`'s motion-blur one then
get the same answer from one place. Two call sites deriving this independently
is exactly how they come to disagree.

## 5. The frame loop

`orchestrate_frame` gains an enable-edge block beside the existing auto and
explore ones, before `process_commands`:

- rising edge → `perform_window.open(chosen_monitor)`
- falling edge → `perform_window.close()`
- open, but `monitor_name` changed → close and reopen

The projector draw does **not** happen inside `orchestrate_frame`. It happens in
`run()`, after `glfw.swap_buffers(self.window)`:

```python
while not glfw.window_should_close(self.window):
    glfw.poll_events()
    self.orchestrate_frame()
    glfw.swap_buffers(self.window)
    self.perform_window.draw(self.camera.last_display)
```

`draw()` returns immediately when the window is closed or the frame is `None`,
so the call is unconditional and costs a branch when perform mode is off.
`last_display` is `None` until the first `Camera.render()`, which is why the
frame is checked rather than assumed.

Two reasons for the placement, and both are load-bearing.

**The swap is the flush.** A texture written in one context is only safely
readable in another after the writer has flushed; `SwapBuffers` performs an
implicit flush. Drawing after it removes the need for a per-frame
`ctx.finish()`, which would block the CPU until the GPU went idle.

**The vsync wait lands where we want it.** `swap_interval` is a property of the
current context, so `open()` sets 1 on the perform context and 0 on the main
one; `close()` restores 1 on the main one. The frame loop therefore paces off
the projector, which is the display that must not tear, and the laptop UI runs
free. Waiting on both would throttle the sim to the slower of the two panels.

The price is that the projector is one frame behind the laptop — about 16 ms at
60 Hz. It is not perceptible in a room, and it is what buys the two properties
above.

## 6. Controls

**Extras > Perform Mode Panel** — a checkbox writing
`preferences.show_perform_window`, alongside Config Clipboard and the rest.

**The panel** (`ui/perform_window.py`, `PerformWindowMixin`) holds:

- A monitor combo, enumerated fresh each frame from `glfw.get_monitors()` so
  plugging a projector in mid-session simply works. Each row reads
  `name — WxH @ Hz` with `(primary)` where it applies.
- A Start / Stop button writing `state.perform.enabled`.
- A status line: which monitor is in use, or which is remembered but absent.
- A notice banner via `ui/notices.render_banner`, `scope="perform"` — the ImGui
  duplicate-id rule.

The panel pushes `layout.push_settings_width()` and wraps its button row with
`layout.wrap_row()`, so a narrow window does not clip the labels off the right
edge.

**F11** toggles `state.perform.enabled` using the remembered monitor. It goes in
`default_keyboard_controls.json` as `toggle_perform_mode`, resolved through
`KeybindingManager.get_key` — `F11` is already in `KEY_NAME_TO_GLFW` and is
bound to nothing.

## 7. What persists

`PerformState.enabled` is **not** persisted. Opening the app must not light up a
projector, the same rule `ArchiveState.enabled` already follows for not
resuming a search.

Two fields join `PreferencesState`:

- `perform_monitor: str = ""` — the remembered monitor's name.
  `NOT_UNDOABLE`, reason: *names an external display, not the look of the
  simulation.*
- `show_perform_window: bool = False` — caught automatically by the existing
  `show_` prefix rule in `_VISIBILITY_PREFIXES`.

`tests/test_undo_fields.py` derives its cases from `__dataclass_fields__`, so
both fail until classified. That is the guard working, not a chore.

## 8. When it goes wrong

**The perform window never takes keyboard focus** — created with
`glfw.FOCUS_ON_SHOW` false and never focused afterwards. Everything below
depends on it.

- *The projector is unplugged mid-show.* The window survives and Windows drops
  it wherever it likes, usually over the laptop screen. Because it never has
  focus, F11 still reaches the main window and closes it, and clicking the main
  window raises it above. Without the focus rule, a borderless input-less
  window covering the UI would be unclosable.
- *Only one display is connected.* Opening is allowed, on the primary. This is
  how the feature is tested at a desk, and the focus rule is what makes it
  safe.
- *The remembered monitor is absent at toggle time.* Fall back to the first
  non-primary monitor; failing that, the primary. A notice names what happened.
  The remembered name is left alone, so re-plugging and hitting F11 resumes.
- *Window or context creation fails.* Notice, `enabled` back to false, nothing
  else disturbed. `open()` leaves no half-built state behind: the GLFW window
  is destroyed if the context or shader stage raises.

`App.cleanup()` gains a guarded `self._step("perform", lambda: ...)` closing the
window, placed before `glfw.terminate`. It takes a lambda for the reason
`field_bus.cleanup` does: the attribute lookup must happen inside `_step`'s
guard, or a missing attribute raises outside it and skips every step below —
including the archive flush.

## 9. Verification

**`tests/test_perform_window.py`** — pure logic, no GL:

- `fit_rect` at wider, taller and exactly-matching aspects, asserting the
  matched case is an identity with no rounding bar.
- Monitor selection: remembered name present, remembered name absent with a
  secondary available, remembered name absent with only a primary.
- The overlay-suppression predicate is true while performing and false
  otherwise, including while recording (where it was already true).

**`tests/test_perform_window_render.py`** — panel smoke test in the existing
style: the host window is sized taller than the panel so nothing is clipped out
of the vertex buffer, `imgui.ini` is suppressed by `tests/conftest.py`, and the
assertions are on the labels a real frame draws plus the id-clash check.

**`tools/drive_perform.py`** — opens a bare GLFW window, a `Camera`-shaped stub
producing a `DisplayFrame`, and a real `PerformWindow`, and runs the two-context
path for a few hundred frames across a monitor change and a close/reopen cycle.
It deliberately does **not** boot the real `App`: a headless test cannot open a
second window at all, and an App-based driver would write the user's
`audio_rig.json` and `preferences.config` on exit.

**`docs/testing_checklist.md`** gains a manual pass — the only place the actual
projector, the actual unplug, and the actual refresh-rate pacing can be
checked.

## 10. Deliberately not in scope

- **Independent camera framing per output.** The projector mirrors; it does not
  get its own pan, zoom, view mode or exposure.
- **Blackout, fade and freeze.** Standard show controls, and cheaper to build
  now than later, but not asked for. Noted here so the omission is a decision
  rather than an oversight.
- **Input on the perform window.** None at all. Every control stays on the
  laptop.
- **Exclusive fullscreen.** Borderless filling the monitor, which does not
  renegotiate a projector's video mode.
- **Splitting display from accumulation in `FrameAssembler`.** It would let the
  overlays stay live on the laptop while the projector runs clean, and would
  incidentally stop the reticle bleeding into the long-exposure trail. Rejected
  as scope for this feature, not as an idea.
