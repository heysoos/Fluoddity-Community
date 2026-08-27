"""Perform mode: a second, borderless, input-less window on another display.

The window shares the main GL context, so the simulation runs once and the
projector costs one fullscreen quad. OpenGL shares textures between contexts
created with `share` but NOT VAOs or FBOs, which is most of what a moderngl
Context is - so this owns a second Context and reaches the already-rendered
frame through `external_texture`.
"""
from __future__ import annotations

from dataclasses import dataclass

import glfw
import moderngl

from utilities.gl_helpers import read_shader, tryset


@dataclass(frozen=True)
class MonitorInfo:
    """One connected display, as GLFW reports it."""

    name: str
    width: int
    height: int
    refresh: int
    x: int
    y: int
    is_primary: bool
    handle: object = None

    def label(self) -> str:
        primary = " (primary)" if self.is_primary else ""
        return f"{self.name} - {self.width}x{self.height} @ {self.refresh}Hz{primary}"


def list_monitors() -> list[MonitorInfo]:
    """Every connected display. Enumerated fresh, so plugging one in works."""
    primary = glfw.get_primary_monitor()
    out = []
    for handle in glfw.get_monitors() or []:
        mode = glfw.get_video_mode(handle)
        if mode is None:
            continue
        name = glfw.get_monitor_name(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        x, y = glfw.get_monitor_pos(handle)
        out.append(MonitorInfo(
            name=name,
            width=mode.size.width,
            height=mode.size.height,
            refresh=mode.refresh_rate,
            x=x, y=y,
            is_primary=bool(handle == primary),
            handle=handle,
        ))
    return out


def choose_monitor(monitors: list[MonitorInfo],
                   remembered: str) -> tuple[MonitorInfo | None, str]:
    """Pick where to perform, and say so when it is not what was asked for.

    Falls back to the first non-primary display, then to the primary. The
    remembered name is never rewritten here, so re-plugging that display and
    toggling again resumes on it.
    """
    if not monitors:
        return None, "No displays reported by GLFW."
    for m in monitors:
        if m.name == remembered:
            return m, ""
    secondary = [m for m in monitors if not m.is_primary]
    if remembered and secondary:
        return secondary[0], f"'{remembered}' is not connected - using {secondary[0].name}."
    if remembered:
        return monitors[0], f"'{remembered}' is not connected - using the primary display."
    if secondary:
        return secondary[0], ""
    return monitors[0], "Only one display is connected - performing on it."


def fit_rect(src_aspect: float, dst_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Letterbox a source aspect into a destination, centred, as (x, y, w, h).

    Matching aspects return the destination exactly - a one-pixel bar is not
    noticed until it is on a wall.
    """
    dst_w, dst_h = dst_size
    if dst_w <= 0 or dst_h <= 0 or src_aspect <= 0:
        return (0, 0, max(dst_w, 0), max(dst_h, 0))
    if src_aspect >= dst_w / dst_h:
        w = dst_w
        h = min(dst_h, int(round(dst_w / src_aspect)))
    else:
        h = dst_h
        w = min(dst_w, int(round(dst_h * src_aspect)))
    return ((dst_w - w) // 2, (dst_h - h) // 2, w, h)


def overlays_hidden(*, performing: bool, recording: bool,
                    screenshotting: bool) -> bool:
    """The one home for every reason to hide the on-canvas overlays.

    The sweep reticle and the draw-brush circle are baked into the assembled
    frame by frame_assembly.frag, so there is no clean copy to hand a projector
    - perform mode turns them off on both screens instead.
    """
    return bool(performing or recording or screenshotting)


class PerformWindow:
    """The second window. Closed until opened; a no-op while closed."""

    def __init__(self, main_window):
        self._main_window = main_window
        self._window = None
        self._ctx = None
        self._program = None
        self._vao = None
        self._vbo = None
        self._ibo = None
        self._external = None
        self._external_key = None
        self.monitor_name = ""

    @property
    def is_open(self) -> bool:
        return self._window is not None

    def open(self, monitor: MonitorInfo) -> None:
        """Fill `monitor` with a borderless window that never takes focus.

        Focus is what makes an unplug recoverable: Windows drops the surviving
        window over the laptop screen, and the toggle key only still reaches
        the main window because this one never took the keyboard.
        """
        if self.is_open:
            self.close()

        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
        try:
            window = glfw.create_window(
                monitor.width, monitor.height, "Fluoddity - Perform",
                None, self._main_window)
        finally:
            glfw.default_window_hints()
        if not window:
            raise RuntimeError("could not create the perform window")

        try:
            glfw.set_window_pos(window, monitor.x, monitor.y)
            glfw.make_context_current(window)
            glfw.swap_interval(1)          # the show is the thing that must not tear
            self._ctx = moderngl.create_context()
            self._build_program()
            self._window = window
            self.monitor_name = monitor.name
        except Exception:
            # Leave nothing half-built: a failed open must not strand a window.
            glfw.make_context_current(self._main_window)
            glfw.destroy_window(window)
            self._ctx = None
            self._program = self._vao = self._vbo = self._ibo = None
            raise
        finally:
            glfw.make_context_current(self._main_window)

        glfw.swap_interval(0)              # the laptop runs free; see draw()

    def _build_program(self) -> None:
        """A recompile of the camera's own display shader in this context."""
        import numpy as np

        self._program = self._ctx.program(
            vertex_shader=read_shader('shaders/camera.vert'),
            fragment_shader=read_shader('shaders/camera.frag'),
        )
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        self._vbo = self._ctx.buffer(vertices.tobytes())
        self._ibo = self._ctx.buffer(indices.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program, [(self._vbo, '2f 2f', 'in_position', 'in_texcoord')],
            self._ibo)

    def _wrap(self, texture) -> moderngl.Texture:
        """External wrapper for a texture owned by the main context.

        Rebuilt when the texture's name or size moves - FrameAssembler drops
        and recreates its accumulation texture whenever the sample count or the
        canvas size changes.
        """
        key = (texture.glo, texture.size, texture.components, texture.dtype)
        if key != self._external_key:
            self._external = self._ctx.external_texture(
                texture.glo, texture.size, texture.components, 0, texture.dtype)
            self._external.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._external_key = key
        return self._external

    def draw(self, frame, on_drawn=None) -> None:
        """Mirror `frame`, letterboxed. A no-op while closed or before a frame.

        Call this AFTER the main window's swap: SwapBuffers performs an
        implicit flush, which is what makes a texture written in the main
        context safe to read here without a per-frame ctx.finish().

        `on_drawn(ctx, rect)` runs after the render and BEFORE the swap, which
        is the only moment what was drawn is readable - after a swap the back
        buffer holds the other frame. tools/drive_perform.py is its only
        caller; the app passes nothing.
        """
        if self._window is None or frame is None or frame.texture is None:
            return
        if glfw.window_should_close(self._window):
            return

        glfw.make_context_current(self._window)
        try:
            fb_w, fb_h = glfw.get_framebuffer_size(self._window)
            if fb_w <= 0 or fb_h <= 0:
                return
            src_w, src_h = frame.window_size
            src_aspect = (src_w / src_h) if src_h > 0 else 1.0
            x, y, w, h = fit_rect(src_aspect, (fb_w, fb_h))

            self._ctx.screen.use()
            self._ctx.viewport = (0, 0, fb_w, fb_h)
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            if w <= 0 or h <= 0:
                return
            self._ctx.viewport = (x, y, w, h)

            tex = self._wrap(frame.texture)
            # The main framebuffer's size, not this window's: the shader must
            # compose the identical image the laptop composed, and the only
            # thing this window contributes is where it lands.
            tryset(self._program, 'cam_pos', frame.cam_pos)
            tryset(self._program, 'cam_zoom', frame.cam_zoom)
            tryset(self._program, 'tex_size', frame.tex_size)
            tryset(self._program, 'window_size', frame.window_size)
            tex.use(location=0)
            tryset(self._program, 'view_tex', 0)
            self._vao.render()
            if on_drawn is not None:
                on_drawn(self._ctx, (x, y, w, h))
            glfw.swap_buffers(self._window)
        finally:
            glfw.make_context_current(self._main_window)

    def close(self) -> None:
        """Idempotent. Restores vsync on the main window."""
        window, self._window = self._window, None
        self.monitor_name = ""
        if window is None:
            return
        glfw.make_context_current(window)
        for obj in (self._vao, self._vbo, self._ibo, self._external, self._program):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._vao = self._vbo = self._ibo = self._external = self._program = None
        self._external_key = None
        self._ctx = None
        glfw.make_context_current(self._main_window)
        glfw.destroy_window(window)
        glfw.swap_interval(1)
