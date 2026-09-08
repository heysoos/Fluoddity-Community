"""Perform Mode panel: pick a display, start, stop, and square it to the wall."""
from imgui_bundle import imgui

from services import corner_pin
from services.perform_window import list_monitors
from . import layout, notices

CANVAS_MAX_W = 280.0      # the proxy display, in points
GRAB_PX = 14.0            # how close the pointer must be to pick a handle
HANDLE_R = 5.0

_QUAD_RGBA = (0.25, 0.75, 1.0, 1.0)
_HELD_RGBA = (1.0, 0.85, 0.15, 1.0)
_FRAME_RGBA = (0.45, 0.45, 0.45, 1.0)
_BACK_RGBA = (0.08, 0.08, 0.10, 1.0)


def _col(rgba):
    return imgui.get_color_u32(imgui.ImVec4(*rgba))


class PerformWindowMixin:
    """Mixin for the Perform Mode panel. Combined into UI via inheritance."""

    # Which handle this gesture grabbed, None for a drag that caught nothing.
    # A click far from every corner must not teleport the nearest one.
    _calib_drag = None

    def render_perform_window(self):
        """Draw the panel. Passive - it writes state, it opens no windows."""
        layout.constrain_panel()
        expanded, opened = imgui.begin("Perform Mode", True)
        if not opened:
            self.state.preferences.show_perform_window = False
            imgui.end()
            return
        if not expanded:
            imgui.end()
            return

        perform = self.state.perform
        prefs = self.state.preferences

        layout.text_disabled_wrapped(
            "Mirrors the canvas on another display with no UI on top. "
            f"{self.keybindings.get_key_display_name('toggle_perform_mode')} toggles it.")
        imgui.separator()

        monitors = list_monitors()
        selected = None
        layout.push_settings_width()
        if monitors:
            labels = [m.label() for m in monitors]
            # By KEY, never by name: Windows reports a laptop panel and a
            # projector under one byte-identical name, and a name lookup
            # snaps the second row back onto the first.
            current = next((i for i, m in enumerate(monitors)
                            if m.key == prefs.perform_monitor), -1)
            if current < 0:
                current = next((i for i, m in enumerate(monitors)
                                if m.device_key == prefs.perform_monitor), -1)
            if current < 0:
                current = 0
            changed, picked = imgui.combo("Display", current, labels)
            if changed:
                prefs.perform_monitor = monitors[picked].key
                current = picked
            selected = monitors[current]
        else:
            imgui.text_disabled("No displays reported.")
        imgui.pop_item_width()

        # The sweep reticle and the draw brush circle are baked into the
        # assembled frame, so there is no clean copy to send. See CLAUDE.md.
        layout.text_disabled_wrapped(
            "While performing, the sweep reticle and draw brush circle are "
            "hidden on both screens.")

        imgui.separator()
        start_label = "Stop" if perform.enabled else "Start"
        if imgui.button(f"{start_label}##perform"):
            perform.enabled = not perform.enabled

        if perform.enabled and perform.active_monitor:
            imgui.text_colored(imgui.ImVec4(*notices.OK),
                               f"Showing on {perform.active_monitor}")
        elif perform.enabled:
            imgui.text_colored(imgui.ImVec4(*notices.WARN), "Starting...")
        else:
            imgui.text_disabled("Not performing.")

        self._render_calibration(perform, selected)

        notices.render_banner(perform, "notice", notices.WARN, scope="perform")
        imgui.end()

    # --- calibration ------------------------------------------------------

    def _render_calibration(self, perform, monitor):
        """Drag four corners so a skewed projector comes out square.

        Not inside a collapsing header: a control the user has to find cannot
        live in a folded section.
        """
        imgui.separator()
        imgui.text("Corner Calibration")
        layout.text_disabled_wrapped(
            "Drag the corners to match the picture to the wall. The projector "
            "shows a grid while this is on.")

        changed, want = imgui.checkbox("Calibrate##perform", perform.calibrating)
        if changed:
            perform.calibrating = want
            if not want:
                perform.held_corner = -1
                self._calib_drag = None

        if not perform.enabled:
            layout.text_disabled_wrapped("Start performing to calibrate.")
            return
        if monitor is None:
            layout.text_disabled_wrapped("No display to calibrate.")
            return

        corners = self._calibration_canvas(perform, monitor)

        right = layout.row_right_edge()
        if imgui.button("Reset Corners##perform"):
            perform.reset_corners_requested = True
            perform.held_corner = -1
            self._calib_drag = None
        layout.wrap_row(right, layout.button_width("Corner: none"))
        held = perform.held_corner
        if 0 <= held < 4:
            name = corner_pin.CORNER_NAMES[held]
            imgui.text(f"Corner: {name} "
                       f"{corners[held][0]:.3f}, {corners[held][1]:.3f}")
        elif perform.corners is None:
            imgui.text_disabled("Corner: none (uncalibrated)")
        else:
            imgui.text_disabled("Corner: none")

    def _calibration_canvas(self, perform, monitor):
        """The proxy display. Returns the corners it drew.

        ImGui is top-down and a corner is a GL coordinate, so `v` is flipped
        here and nowhere else - a stored calibration is always GL-side up.
        """
        size = (monitor.width, monitor.height)
        aspect = monitor.width / max(monitor.height, 1)
        corners = perform.corners
        if corners is None:
            # The perform view renders at the display's own shape, so the
            # source aspect IS the display's and this is the whole screen.
            corners = corner_pin.default_corners(aspect, size)

        avail = imgui.get_content_region_avail().x
        w = max(140.0, min(CANVAS_MAX_W, avail))
        h = max(90.0, w / max(aspect, 0.01))
        origin = imgui.get_cursor_screen_pos()
        imgui.invisible_button("##perform_calib", imgui.ImVec2(w, h))
        x0, y0 = origin.x, origin.y

        def to_screen(c):
            return imgui.ImVec2(x0 + c[0] * w, y0 + (1.0 - c[1]) * h)

        def to_corner(pos):
            u = min(max((pos.x - x0) / w, 0.0), 1.0)
            v = min(max(1.0 - (pos.y - y0) / h, 0.0), 1.0)
            return (u, v)

        mouse = imgui.get_io().mouse_pos
        if imgui.is_item_activated():
            idx = corner_pin.nearest_corner(to_corner(mouse), corners)
            at = to_screen(corners[idx])
            near = ((at.x - mouse.x) ** 2 + (at.y - mouse.y) ** 2
                    <= GRAB_PX ** 2)
            self._calib_drag = idx if near else None
        if imgui.is_item_deactivated():
            self._calib_drag = None

        if self._calib_drag is not None and imgui.is_item_active():
            moved = list(corners)
            moved[self._calib_drag] = to_corner(mouse)
            # A fold is refused at the gesture, so the shader never sees one.
            if corner_pin.is_convex(tuple(moved)):
                corners = tuple(moved)
                perform.corners = corners
        perform.held_corner = (self._calib_drag
                               if self._calib_drag is not None else -1)

        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(origin, imgui.ImVec2(x0 + w, y0 + h),
                             _col(_BACK_RGBA))
        draw.add_rect(origin, imgui.ImVec2(x0 + w, y0 + h), _col(_FRAME_RGBA))
        pts = [to_screen(c) for c in corners]
        for i in range(4):
            draw.add_line(pts[i], pts[(i + 1) % 4], _col(_QUAD_RGBA), 2.0)
        for i, p in enumerate(pts):
            held = i == perform.held_corner
            draw.add_circle_filled(
                p, HANDLE_R + (2.0 if held else 0.0),
                _col(_HELD_RGBA if held else _QUAD_RGBA))
        return corners
