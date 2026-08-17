"""The undo history panel: browse steps, preview one, jump to it."""
from imgui_bundle import imgui

from . import layout, notices

REDO_COLOUR = imgui.ImVec4(0.55, 0.55, 0.55, 1.0)


class UndoWindowMixin:
    """Combined into UI via multiple inheritance."""

    def render_undo_window(self):
        """Steps newest-first. The orchestrator pushes undo_steps each frame."""
        expanded, opened = imgui.begin("Undo History", True)
        if not opened:
            self.state.preferences.show_undo_window = False
            self.state.undo_preview_index = -1
            imgui.end()
            return

        notices.render_banner(self.state, "undo_notice", notices.DIM, "undo")

        steps = getattr(self, "undo_steps", [])
        if not steps:
            imgui.text_disabled("No steps yet")
            self.state.undo_preview_index = -1
            imgui.end()
            return

        layout.push_settings_width()
        hovered = -1
        for index, label, is_redo in reversed(steps):
            if is_redo:
                imgui.push_style_color(imgui.Col_.text, REDO_COLOUR)
            marker = ">" if index == self.undo_cursor else " "
            text = f"{marker} {label}##undo_{index}"
            clicked, _ = imgui.selectable(text, index == self.undo_cursor)
            if is_redo:
                imgui.pop_style_color()

            recorder = getattr(self, "_undo_row_labels", None)
            if recorder is not None:
                recorder.append(label)

            if imgui.is_item_hovered():
                hovered = index
            if clicked:
                self.state.undo_jump_index = index
        imgui.pop_item_width()

        # Continuous, never a one-shot: written every frame, including the
        # frame nothing is hovered, so a closed panel cannot strand a preview.
        self.state.undo_preview_index = hovered
        imgui.end()
