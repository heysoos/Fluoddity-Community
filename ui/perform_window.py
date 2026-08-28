"""Perform Mode panel: pick a display, start, stop."""
from imgui_bundle import imgui

from services.perform_window import list_monitors
from . import layout, notices


class PerformWindowMixin:
    """Mixin for the Perform Mode panel. Combined into UI via inheritance."""

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

        notices.render_banner(perform, "notice", notices.WARN, scope="perform")
        imgui.end()
