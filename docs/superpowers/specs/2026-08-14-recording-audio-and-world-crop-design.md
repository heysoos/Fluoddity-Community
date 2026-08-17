# Recording: Audio Track and World-Space Crop — Design

**Date:** 2026-08-14
**Status:** Design approved, ready to implement
**Branch context:** `worktree-audio-reactive`

## Goal

Two changes to the video recorder, both asked for together because they land in
the same path:

1. A recording can carry the audio that drove it, muxed into the mp4.
2. A recording frames the world, not the window — no black bars where the
   canvas does not reach.

## Decisions

| Decision | Choice |
|---|---|
| Crop source | `camera.assembled_view_rect`, already stored with each texture |
| Crop mechanism | `services/capture_blit.CaptureBlit` into a fixed-size FBO |
| Crop tracking | Re-derived per frame; intersected with the texture |
| Output size | Chosen once at record start, even-dimensioned |
| Resolution loss | Accepted — crop only, no second render pass |
| Audio source | The stream already feeding the analyser |
| No audio | Records silently; not an error |
| Audio transport | Raw `f32le` sidecar, muxed at close |
| Video clock | Derived from audio duration, applied with `-itsscale` |
| Physics rate | `speedmult` override skipped while recording with audio |
| Silent recording | Unchanged — fixed 50 fps, no mux |

## Background: the two render paths

`Camera.generate_view_texture()` returns one of two different things, and the
distinction decides everything about the crop:

| View option | Texture handed to the recorder | Camera baked in? |
|---|---|---|
| Canvas, Brush, Force/Strafe field (0, 1, 4, 5) | `sim.view_tex`, at canvas resolution | No |
| Camera, Tiled, Particles+Trails (2, 3, 6) | `cam_brush_target`, at framebuffer size | Yes |

`FrameAssembler` sizes itself from its input, so the assembled texture inherits
whichever shape arrived. The black bars exist only in the second row: the
window-shaped buffer holds the canvas letterboxed inside it.

**Zooming out in the camera views destroys resolution, and no crop can undo
that.** The world is rasterised into however many pixels the zoom leaves it, so
a crop returns exactly those pixels and no more. Recovering full resolution
would need the recording to own its framebuffer and camera — a second particle
raster per frame — which was considered and rejected as too expensive for the
benefit. The crop is the whole fix; framing near fit-to-window is the user's
side of the bargain.

## Part 1 — The crop

### Data flow

```
assembled_tex ──► CaptureBlit(src_lo, src_hi) ──► record_fbo (fixed size)
                                                       │
                                        save_frame_gpu(supersample) ──► ffmpeg
```

`CaptureBlit` already maps an arbitrary source rect onto a full target quad,
which is this operation exactly. It costs one fullscreen quad per frame.

### Rules

- **The rect comes from `camera.assembled_view_rect`, never recomputed.** It is
  assigned with the texture it describes. Recomputing at crop time would crop
  one frame's pixels with a later frame's camera — the bug the field exists to
  prevent.
- **The rect is intersected with the full texture.** Zoomed in, the canvas
  overflows the view, the intersection is the whole texture and the blit is an
  identity. Zoomed out, it tightens onto the canvas and the bars disappear.
- **Output size is fixed at record start** — the crop's pixel size then, rounded
  to even for H.264. The encoder rejects a mid-stream dimension change, so a
  later zoom rescales into that fixed target rather than resizing the file.
- **In the raw texture views the rect is the whole texture**, so the blit is an
  identity and those recordings are unchanged.

### Consequence

Because the rect is re-derived per frame, panning while zoomed out does not read
as a pan in the video — the world stays framed. This follows from "always trim
to the world"; the alternative, freezing the rect at record start, lets the bars
back in as soon as you pan.

### Bonus

Output dimensions stop tracking the window, so a mid-recording window resize no
longer trips the "settings changed, starting a new file" split in `VidSaver`.
The canvas aspect keeps the framing correct; a resize only changes sampling
resolution.

## Part 2 — The audio track

### The tap

`AudioCapture._on_block` already receives interleaved float32 bytes and
currently mono-mixes them for analysis and discards the rest. It gains one
optional sink:

```
_on_block(in_data) ──┬──► analyser  (unchanged)
                     └──► tap(in_data)   # raw bytes, verbatim
```

Bytes are written verbatim — original channel count, original sample rate, no
conversion in Python. ffmpeg reads them back with `-f f32le -ar <rate>
-ac <channels>`.

The tap slot follows the precedent `self._analyzer` already sets: a plain
attribute, read once into a local by the callback, cleared *before* teardown in
`stop()`. It sits inside the existing catch-all, so a failing sink cannot kill
the audio thread.

A buffered write of one block is far cheaper than the 2048-point FFT and mel
matmul that thread already runs per block, so no queue or worker is needed.

### Ownership

File IO lives in a new `services/audio_track.py` (`AudioTrackWriter`), not in
`audio_capture.py`: the capture owns the device, the writer owns the file.
`main.py` wires them at start and stop, because components do not talk to each
other directly.

### The clock

The framerate is **derived from the audio**, not measured with a timer:

```
audio_seconds = bytes / (4 * channels * sample_rate)
measured_fps  = frame_count / audio_seconds
```

The tap starts and stops with the recorder, so both streams cover the same wall
interval by construction. Dividing frames by audio seconds gives precisely the
framerate that makes them equal in length — sync is arithmetic rather than
measurement, and it rides the sound card's clock instead of a frame loop's.

Applied at close during a stream copy:

```
ffmpeg -itsscale <nominal/measured> -i temp.mp4 -f f32le -ar R -ac C -i temp.pcm \
       -c:v copy -c:a aac -shortest out.mp4
```

`-itsscale` is an input option and retimes the video stream without re-encoding.
The correct framerate is not knowable until the recording ends, and `-framerate`
is fixed when the encoder process starts — which is the reason for a sidecar and
a mux pass rather than a live second pipe. Any live-mux design must guess the
framerate up front and drift.

### What this buys

Hitting real time stops being a correctness requirement. Whatever rate the
machine achieves, the framerate reports it honestly and the audio still lines
up. The physics rate therefore becomes a taste decision:

- With audio recording on, the `speedmult` override in `main.py` is skipped, so
  the rate follows the user's slider. Smooth video, little motion blur — blur
  samples *are* physics sub-steps, one knob.
- Raising the slider trades fps for blur. Sync is unaffected.

The `motion_blur` and `blur_quality` overrides are untouched; they are look
preferences, not rate.

### No audio, no clock

If capture is off or the device failed, there is no track, no `-itsscale` and no
mux. Silent recording keeps its fixed 50 fps and its current behaviour exactly.

## Part 3 — Surface and failures

### UI

One checkbox, `Record Audio`, in the Recording section of Preferences beside
`max_frames`, `supersample_k` and the motion blur controls, backed by
`record_audio: bool = False` on `PreferencesState`. Off by default.

On completion a notice reports the output path, the measured fps and whether an
audio track landed. A save that only prints to the console reads as a no-op.

### Failure paths

| Failure | Behaviour |
|---|---|
| ffmpeg missing | Existing `find_ffmpeg()` message; the mux reuses it |
| Audio off or device failed at start | Silent recording, notice, 50 fps |
| Device dies mid-take | Audio short, so derived fps would be too high. Wall-clock elapsed is kept as a cross-check; on disagreement beyond tolerance, fall back to wall clock and warn |
| Mux fails | Temp mp4 renamed to the final path, warning notice — the take survives |
| Disk full during tap | Caught by the audio thread's existing handler, tap disabled, `last_error` set, video continues |

Stereo 48 kHz f32 is about 384 KB/s, so a five-minute take buffers roughly
115 MB of temp PCM, deleted on success.

## Testing

Pure geometry and arithmetic get real tests. The GL blit reuses a shader that is
already covered and is not re-tested.

- Crop rect: intersection and clamping, zoomed in (identity) and zoomed out
  (tightens). **Evaluated with the camera panned, never centred** — a centred
  camera cancels the `v`-flip between `canvas_view_rect` and `tex_to_screen` and
  will validate any orientation bug.
- Output size chosen once, even-dimensioned, stable while the crop rect moves.
- Identity crop in the raw canvas, brush and field views.
- `AudioTrackWriter` byte-to-duration accounting.
- fps derived from frame count and byte count.
- The wall-clock cross-check fires on a truncated audio stream.
- Mux command asserted as argv; no subprocess in the suite.
- A raising sink does not propagate: `process()` still returns a snapshot and
  `last_error` is set.
- Silent recording unchanged: no `-itsscale`, no audio input, 50 fps.

Manual items go into `docs/testing_checklist.md` — sync is confirmed by watching
a take with a clear beat against its visuals.

## Out of scope

- A second render pass for zoom-independent full-resolution recording.
- A separate device selector for the soundtrack.
- Per-frame variable frame timing; `-itsscale` corrects the average rate, and
  local stutter is left as stutter.
