"""Tournament mode UI: 4x4 tile selector + breeding controls."""
from imgui_bundle import imgui


def ui_row_order(grid: int = 4) -> list[int]:
    """Canvas tile-rows (ty) in the order ImGui should draw them, top row first.

    Tile 0 is the BOTTOM-left of the canvas because entity-space y grows upward,
    but ImGui draws the first row it is given at the TOP of the window. Handing it
    the rows in descending ty makes the button panel match the canvas on screen.
    """
    return list(reversed(range(grid)))


class TournamentWindowMixin:
    """Renders the interactive tournament window. Combined into UI via multiple inheritance."""

    def render_tournament_window(self):
        state = self.state.tournament
        if not state.enabled:
            return

        svc = getattr(self, "tournament_service", None)
        selected = svc.selected if svc is not None else set()

        expanded, opened = imgui.begin("Tournament - EXPERIMENTAL", True)
        if not opened:
            state.enabled = False
            imgui.end()
            return

        imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
                           "Click tiles (on canvas or below) to select, then breed")
        imgui.separator()

        # 4x4 selectable grid mirroring the on-canvas tiles.
        # Tile 0 is the BOTTOM-left of the canvas (entity space y grows upward), but
        # ImGui draws the first row it is given at the TOP. Walk rows top-down in
        # canvas terms (ty = 3 first) so this panel matches what you see on screen.
        grid = 4
        for ty in ui_row_order(grid):
            for tx in range(grid):
                tile = ty * grid + tx
                is_sel = tile in selected
                if is_sel:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.7, 0.3, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.8, 0.4, 1.0))
                if imgui.button(f"{tile}", imgui.ImVec2(40, 40)):
                    state.clicked_tile = tile
                if is_sel:
                    imgui.pop_style_color(2)
                if tx < grid - 1:
                    imgui.same_line()

        imgui.separator()

        _, state.mutation_strength = imgui.slider_float(
            "Mutation strength", state.mutation_strength, 0.0, 0.5)
        _, state.inject_randoms = imgui.slider_int(
            "Inject randoms", state.inject_randoms, 0, 4)
        _, state.crossover_enabled = imgui.checkbox(
            "Crossover", state.crossover_enabled)

        imgui.separator()

        if imgui.button("Next Generation"):
            state.next_gen_requested = True
        imgui.same_line()
        if imgui.button("Undo"):
            state.undo_requested = True

        if imgui.button("Reset Population"):
            state.reset_requested = True
        imgui.same_line()
        if imgui.button("Save Selected"):
            state.save_requested = True

        imgui.text(f"Selected: {sorted(selected)}")
        imgui.end()
