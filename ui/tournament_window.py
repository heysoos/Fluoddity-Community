"""Tournament mode UI: 4x4 tile selector + breeding controls."""
from imgui_bundle import imgui

from services import save_targets
from ui import layout
from ui.notices import OK, render_banner


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
            # A closed window must leave its sub-modes off. Only the tab that
            # is drawn sets its own mode, so with the window shut nothing would
            # ever clear them and the app would keep applying Auto/Explore's
            # overrides - forced grid, square tiles, no motion blur - to what
            # the user sees as an ordinary single simulation.
            self.state.auto_tournament.enabled = False
            self.state.archive.enabled = False
            return

        svc = getattr(self, "tournament_service", None)
        selected = svc.selected if svc is not None else set()

        # Wide enough that a slider label is never clipped, tall enough that
        # the transport buttons are not below the fold.
        imgui.set_next_window_size(imgui.ImVec2(520, 700),
                                   imgui.Cond_.first_use_ever)
        layout.constrain_panel()
        expanded, opened = imgui.begin("Tournament - EXPERIMENTAL", True)
        if not opened:
            state.enabled = False
            imgui.end()
            return

        if imgui.begin_tab_bar("tournament_modes"):
            if imgui.begin_tab_item("Manual")[0]:
                self.state.auto_tournament.enabled = False
                self.state.archive.enabled = False
                self._render_manual_tournament(state, selected)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Auto (Prompt)")[0]:
                self.state.archive.enabled = False
                self.render_auto_tournament_tab()
                imgui.end_tab_item()
            # Exactly one tab item is ever selected, and each branch disables
            # the modes it is not - so no else-branch is needed to turn Explore
            # off, and adding one would only fire in states a tab bar cannot
            # reach.
            if imgui.begin_tab_item("Explore (IMGEP)")[0]:
                self.state.auto_tournament.enabled = False
                self.render_explore_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.end()

    def _render_manual_tournament(self, state, selected):
        layout.text_colored_wrapped(
            (0.6, 0.6, 0.6, 1.0),
            "Click tiles (on canvas or below) to select, then breed")
        imgui.separator()

        # Tile 0 is the BOTTOM-left of the canvas but ImGui draws its first row
        # at the TOP, so walk rows in descending ty. See ui_row_order.
        grid = self.tournament_service.grid
        for ty in ui_row_order(grid):
            for tx in range(grid):
                tile = ty * grid + tx
                is_sel = tile in selected
                if is_sel:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.7, 0.3, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.8, 0.4, 1.0))
                if imgui.button(f"{tile}", imgui.ImVec2(40, 40)):
                    state.clicked_tile = tile
                if imgui.is_item_clicked(imgui.MouseButton_.right):
                    # Right-click is Auto mode's "save this tile"; in Explore
                    # the orchestrator redirects the same flag to 'chase this'.
                    self.state.auto_tournament.save_tile_requested = tile
                if is_sel:
                    imgui.pop_style_color(2)
                if tx < grid - 1:
                    imgui.same_line()

        imgui.separator()

        layout.push_settings_width()
        _, state.mutation_strength = imgui.slider_float(
            "Mutation strength", state.mutation_strength, 0.0, 0.5)
        _, state.inject_randoms = imgui.slider_int(
            "Inject randoms", state.inject_randoms, 0, 4)
        imgui.pop_item_width()
        _, state.crossover_enabled = imgui.checkbox(
            "Crossover", state.crossover_enabled)

        imgui.separator()

        right = layout.row_right_edge()
        if imgui.button("Next Generation"):
            state.next_gen_requested = True
        layout.wrap_row(right, layout.button_width("Undo"))
        if imgui.button("Undo"):
            state.undo_requested = True

        right = layout.row_right_edge()
        if imgui.button("Reset Population"):
            state.reset_requested = True
        layout.wrap_row(right, layout.button_width("Save Selected..."))
        # A disabled item does not receive hover, so the reason is plain text
        # rather than a tooltip that would never appear.
        imgui.begin_disabled(not selected)
        if imgui.button("Save Selected..."):
            self.open_save_popup(save_targets.TOURNAMENT_TILE,
                                 tiles=sorted(selected))
        imgui.end_disabled()
        if not selected:
            layout.wrap_row(right, imgui.calc_text_size("(select tiles first)").x)
            imgui.text_disabled("(select tiles first)")

        imgui.text_wrapped(f"Selected: {sorted(selected)}")
        render_banner(state, "notice", OK, scope="tournament")
