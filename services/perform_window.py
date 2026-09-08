"""Perform mode: a second, borderless, input-less window on another display.

The window shares the main GL context, so the simulation runs once and the
projector costs one fullscreen quad. OpenGL shares textures between contexts
created with `share` but NOT VAOs or FBOs, which is most of what a moderngl
Context is - so this owns a second Context and reaches the already-rendered
frame through `external_texture`.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import glfw
import moderngl

from utilities.gl_helpers import read_shader, tryset


@dataclass(frozen=True)
class MonitorInfo:
    """One connected display, as GLFW reports it.

    A NAME IS NOT AN IDENTITY. Windows reports displays attached to some
    drivers as "Generic PnP Monitor" - byte-identical for a laptop panel and a
    projector both - so a list keyed by name has rows that snap back to the
    first match the moment they are clicked, and Start opens the wrong
    display. `key` is the identity; `name` is only ever shown.
    """

    name: str
    width: int
    height: int
    refresh: int
    x: int
    y: int
    is_primary: bool
    phys_mm: tuple = (0, 0)
    dup_index: int = 0        # 0 unless another display shares this name
    handle: object = None

    @property
    def key(self) -> str:
        """What a preference stores. Never shown to anyone."""
        pw, ph = self.phys_mm
        return (f"{self.name}|{self.width}x{self.height}"
                f"|{pw}x{ph}|{self.x},{self.y}")

    @property
    def device_key(self) -> str:
        """The DEVICE, without where it currently sits.

        Rearranging displays in Windows moves every position, which must not
        make a remembered display unrecognisable.
        """
        pw, ph = self.phys_mm
        return f"{self.name}|{self.width}x{self.height}|{pw}x{ph}"

    def label(self) -> str:
        """A row a person can tell from the one above it.

        The suffix is what separates two identically named displays; the
        position is what maps them onto the arrangement Windows itself shows.
        """
        dup = f" #{self.dup_index}" if self.dup_index else ""
        where = "primary" if self.is_primary else f"at {self.x},{self.y}"
        return (f"{self.name}{dup} - {self.width}x{self.height} "
                f"@ {self.refresh}Hz ({where})")


def _same_handle(a, b) -> bool:
    """GLFW handles are ctypes pointers: `==` is False even at one address."""
    try:
        return ctypes.addressof(a.contents) == ctypes.addressof(b.contents)
    except Exception:
        return a is b


def list_monitors() -> list[MonitorInfo]:
    """Every connected display. Enumerated fresh, so plugging one in works."""
    primary = glfw.get_primary_monitor()
    raw = []
    for handle in glfw.get_monitors() or []:
        mode = glfw.get_video_mode(handle)
        if mode is None:
            continue
        name = glfw.get_monitor_name(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        x, y = glfw.get_monitor_pos(handle)
        try:
            phys = tuple(glfw.get_monitor_physical_size(handle))
        except Exception:
            phys = (0, 0)
        raw.append(dict(
            name=name, width=mode.size.width, height=mode.size.height,
            refresh=mode.refresh_rate, x=x, y=y, phys_mm=phys,
            is_primary=bool(primary is not None
                            and _same_handle(handle, primary)),
            handle=handle,
        ))

    counts = {}
    for r in raw:
        counts[r["name"]] = counts.get(r["name"], 0) + 1
    seen = {}
    out = []
    for r in raw:
        dup = 0
        if counts[r["name"]] > 1:
            seen[r["name"]] = seen.get(r["name"], 0) + 1
            dup = seen[r["name"]]
        out.append(MonitorInfo(dup_index=dup, **r))
    return out


def choose_monitor(monitors: list[MonitorInfo],
                   remembered: str) -> tuple[MonitorInfo | None, str]:
    """Pick where to perform, and say so when it is not what was asked for.

    Matched in three passes, loosest last: the exact key, then the DEVICE
    (which survives rearranging the displays), then the bare name - which is
    what a preference written before displays had a key holds, so an existing
    setup keeps working. Falls back to the first non-primary display, then to
    the primary. `remembered` is never rewritten here, so re-plugging the
    display that was asked for resumes on it.
    """
    if not monitors:
        return None, "No displays reported by GLFW."
    if remembered:
        # A key's last field is the position, so dropping it is the same
        # display somewhere else. Comparing a stored FULL key against a
        # candidate's device key never matches and silently skips this pass.
        device = remembered.rsplit("|", 1)[0]
        for match in (lambda m: m.key == remembered,
                      lambda m: m.device_key == device,
                      lambda m: m.name == remembered):
            found = [m for m in monitors if match(m)]
            if found:
                return found[0], ""
    secondary = [m for m in monitors if not m.is_primary]
    if remembered and secondary:
        return secondary[0], (f"That display is not connected - using "
                              f"{secondary[0].label()}.")
    if remembered:
        return monitors[0], ("That display is not connected - using the "
                             "primary one.")
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


def corner_rect(corners, fb_size) -> tuple:
    """The pixel bounding box of a corner quad, as (x, y, w, h).

    Reproduces `fit_rect` exactly for the corners `default_corners` derives
    from it, which is what lets tools/drive_perform.py keep checking an
    uncalibrated projector against the letterbox it should still be drawing.
    """
    fb_w, fb_h = fb_size
    xs = [c[0] * fb_w for c in corners]
    ys = [c[1] * fb_h for c in corners]
    x0, x1 = round(min(xs)), round(max(xs))
    y0, y1 = round(min(ys)), round(max(ys))
    return (x0, y0, x1 - x0, y1 - y0)


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
        # The warp, cached on (corners, framebuffer size). A degenerate quad
        # keeps the last good matrix rather than putting NaN on a wall.
        self._inv_key = None
        self._inv_corners = None
        self._inv_h = None
        self.warp_ok = True
        self.monitor_key = ""
        self.monitor_label = ""

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
            glfw.swap_interval(1)          # the show must not tear
            self._ctx = moderngl.create_context()
            self._build_program()
            self._window = window
            self.monitor_key = monitor.key
            self.monitor_label = monitor.label()
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
        """The projector's own display shader, compiled in this context.

        Not the camera's: this one covers the whole framebuffer and samples
        through a homography, so a skewed projector can be squared to a wall.
        """
        import numpy as np

        self._program = self._ctx.program(
            vertex_shader=read_shader('shaders/perform.vert'),
            fragment_shader=read_shader('shaders/perform.frag'),
        )
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        self._vbo = self._ctx.buffer(vertices.tobytes())
        self._ibo = self._ctx.buffer(indices.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program, [(self._vbo, '2f', 'in_position')], self._ibo)

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

    def resolve_corners(self, corners, src_aspect, fb_size):
        """The corners to draw with, and the display-to-source matrix.

        `corners` of None means the letterbox - `fit_rect` is still the one
        authority on where an uncalibrated picture lands, and this only
        normalises it into the same four points a calibrated one uses.

        A folded or collapsed quad keeps the LAST GOOD matrix rather than
        putting NaN on a wall, and says so through `warp_ok`. Cached on the
        corners and the framebuffer size, so a static calibration costs one
        dict comparison a frame rather than an 8x8 solve.
        """
        from services import corner_pin

        default = corner_pin.default_corners(src_aspect, fb_size)
        wanted = default if corners is None else tuple(
            (float(c[0]), float(c[1])) for c in corners)

        key = (wanted, tuple(fb_size))
        if key == self._inv_key:
            self.warp_ok = True
            return self._inv_corners, self._inv_h

        inv = corner_pin.inverse_homography(wanted)
        if inv is None:
            self.warp_ok = False
            if self._inv_h is not None:
                return self._inv_corners, self._inv_h
            wanted = default
            inv = corner_pin.inverse_homography(default)
            key = (wanted, tuple(fb_size))
        else:
            self.warp_ok = True

        self._inv_key, self._inv_corners, self._inv_h = key, wanted, inv
        return wanted, inv

    def draw(self, frame, corners=None, guides: bool = False,
             held_corner: int = -1, on_drawn=None) -> None:
        """Warp `frame` onto the projector. A no-op while closed.

        `corners` are four points in display space, TL TR BR BL, v = 1 at the
        top; None is the letterbox. `guides` draws the calibration grid and
        the corner markers, `held_corner` highlights one of them.

        Call this AFTER the main window's swap: SwapBuffers performs an
        implicit flush, which is what makes a texture written in the main
        context safe to read here without a per-frame ctx.finish().

        `on_drawn(ctx, rect)` runs after the render and BEFORE the swap, which
        is the only moment what was drawn is readable - after a swap the back
        buffer holds the other frame. `rect` is where the PICTURE landed, not
        the viewport, which is now always the whole framebuffer.
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
            quad, inv = self.resolve_corners(corners, src_aspect, (fb_w, fb_h))
            if inv is None:
                return

            self._ctx.screen.use()
            self._ctx.viewport = (0, 0, fb_w, fb_h)
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)

            tex = self._wrap(frame.texture)
            tex.use(location=0)
            tryset(self._program, 'view_tex', 0)
            self._write_matrix(inv)
            self._write_corners(quad)
            tryset(self._program, 'fb_size', (float(fb_w), float(fb_h)))
            tryset(self._program, 'show_guides', 1 if guides else 0)
            tryset(self._program, 'held_corner', int(held_corner))
            self._vao.render()
            if on_drawn is not None:
                on_drawn(self._ctx, corner_rect(quad, (fb_w, fb_h)))
            glfw.swap_buffers(self._window)
        finally:
            glfw.make_context_current(self._main_window)

    def _write_matrix(self, inv) -> None:
        """GLSL reads a mat3 COLUMN-major; the solve is row-major."""
        import numpy as np

        uniform = self._program.get('inv_h', None)
        if uniform is not None:
            uniform.write(np.ascontiguousarray(inv.T, dtype='f4').tobytes())

    def _write_corners(self, quad) -> None:
        import numpy as np

        uniform = self._program.get('corners', None)
        if uniform is not None:
            uniform.write(np.ascontiguousarray(quad, dtype='f4').tobytes())

    def close(self) -> None:
        """Idempotent. Restores vsync on the main window."""
        window, self._window = self._window, None
        self.monitor_key = ""
        self.monitor_label = ""
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
        self._inv_key = self._inv_corners = self._inv_h = None
        self.warp_ok = True
        self._ctx = None
        glfw.make_context_current(self._main_window)
        glfw.destroy_window(window)
        glfw.swap_interval(1)
