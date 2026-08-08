"""Explore (IMGEP) tab and the archive browser.

Passive: renders widgets and sets one-shot flags, runs no logic. Every value
displayed comes from driver.status() or archive.stats(), so the UI cannot
disagree with the search about what is happening.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

_BAD = (1.0, 0.4, 0.3, 1.0)
_WARN = (1.0, 0.6, 0.2, 1.0)
_OK = (0.4, 0.9, 0.5, 1.0)
_DIM = (0.6, 0.6, 0.6, 1.0)

GOAL_TOOLTIP = (
    "Expeditions alternate between two goal sources.\n\n"
    "LATENT goals extrapolate past the archive's frontier, away from its "
    "centroid: 'keep going in the direction that already looks unlike "
    "everything else'. No text involved, so nothing is lost in translation.\n\n"
    "TEXT goals are this list, cycled in order. E&E has a language model write "
    "these from archive thumbnails; here you write them, which is strictly more "
    "controllable and needs no network.\n\n"
    "Leave the list empty and every expedition is latent."
)

ALPHA_TOOLTIP = (
    "How strongly parent selection favours novel archive entries: p ~ NOV^alpha.\n\n"
    "alpha = 4 is the tuned value from Expedition & Expansion. alpha = 0 samples "
    "parents uniformly, which with Expansion Between = 0 reduces the whole search "
    "to random archive mutation - E&E's own baseline, reachable here without a "
    "code change."
)


class ArchiveWindowMixin:
    archive_service = None      # AutoTournamentService driving Explore mode
    archive_driver = None       # ImgepDriver
    archive_obj = None          # Archive
    archive_goals = None        # GoalList
    archive_unavailable = ""
    archive_projection = None
    archive_goal_point = None

    # ---- the tab -------------------------------------------------------

    def render_explore_tab(self):
        ast = self.state.archive
        ast.enabled = True

        if ast.warning:
            imgui.text_colored(imgui.ImVec4(*_BAD), ast.warning)
            imgui.same_line()
            if imgui.button("Dismiss##explore"):
                ast.warning = ""

        if self.archive_unavailable:
            imgui.text_colored(imgui.ImVec4(*_WARN), self.archive_unavailable)
            imgui.text_wrapped(
                "Explore mode needs the same CLIP model and packages as Auto mode. "
                "Manual mode is unaffected.")
            return

        self._render_explore_status(ast)
        imgui.separator()
        self._render_explore_transport(ast)
        imgui.separator()
        self._render_goal_list(ast)
        imgui.separator()
        self._render_rollout_controls(
            ast, grid_note="changing the grid ends any expedition in flight")
        imgui.separator()
        self._render_exploration_settings(ast)
        imgui.separator()
        self._render_expedition_settings(ast)
        imgui.separator()
        if imgui.button("Open Archive Browser"):
            ast.show_browser = True

    def _render_explore_status(self, ast):
        d = self.archive_driver
        if d is None:
            imgui.text_colored(imgui.ImVec4(*_DIM), "Not started")
            return
        st = d.status()
        imgui.text(f"Regime: {st['regime']}")
        goal = st.get("goal") or "-"
        imgui.text(f"Goal: {goal}")
        imgui.text(f"Archive: {st['archive_size']}   "
                   f"threshold {st['threshold']:.3f}   "
                   f"admitting {100.0 * st['admission_rate']:.0f}%")
        if st.get("blocked_by_pins"):
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "Archive is full and entirely pinned - nothing new can be added.")
        # spec 10: a persistently zero admission rate is a broken capture or a
        # dead preset, not a hard search.
        if st["archive_size"] > 0 and st["admission_rate"] <= 0.0:
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "Nothing has been admitted recently - check the preset is alive, "
                "the capture is not black, and Liveness Floor is not too high.")

    def _render_explore_transport(self, ast):
        if imgui.button("Start##explore"):
            ast.start_requested = True
        imgui.same_line()
        if imgui.button("Pause##explore"):
            ast.pause_requested = True
        imgui.same_line()
        if imgui.button("Reset Search##explore"):
            ast.reset_requested = True
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(*_DIM), "(Reset keeps the archive)")
        self._render_cycle_estimate(ast)

    def _render_cycle_estimate(self, ast):
        """One expansion+expedition cycle in wall-clock, at the measured
        716 sim steps/s. expedition_gens=350 (the paper's value) is ~16 minutes
        on a single goal, which is worth seeing before choosing it."""
        per_gen = ast.steps_per_gen / 716.0
        cycle = (ast.expansion_between + ast.expedition_gens) * per_gen
        imgui.text_colored(
            imgui.ImVec4(*_DIM),
            f"~{per_gen:.1f}s per generation, ~{cycle / 60.0:.1f} min per cycle")

    def _render_goal_list(self, ast):
        imgui.text("Goals")
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.set_tooltip(GOAL_TOOLTIP)

        goals = self.archive_goals
        if goals is not None:
            for i, item in enumerate(list(goals.items)):
                changed, enabled = imgui.checkbox(f"##goal_on{i}", item["enabled"])
                if changed:
                    item["enabled"] = enabled
                imgui.same_line()
                if imgui.button(f"^##goal_up{i}"):
                    ast.move_goal_index, ast.move_goal_delta = i, -1
                imgui.same_line()
                if imgui.button(f"v##goal_dn{i}"):
                    ast.move_goal_index, ast.move_goal_delta = i, +1
                imgui.same_line()
                if imgui.button(f"x##goal_rm{i}"):
                    ast.remove_goal_index = i
                imgui.same_line()
                imgui.text(item["text"])

        _, ast.new_goal_text = imgui.input_text("##new_goal", ast.new_goal_text)
        imgui.same_line()
        if imgui.button("Add Goal") and ast.new_goal_text.strip():
            ast.add_goal_requested = True

        order = ["round_robin", "least_matched"]
        idx = order.index(ast.goal_order) if ast.goal_order in order else 0
        ch, idx = imgui.combo("Goal Order", idx, ["Round robin", "Least matched"])
        if ch:
            ast.goal_order = order[idx]

    def _render_exploration_settings(self, ast):
        _, ast.sigma_expand = imgui.slider_float(
            "Expansion Sigma", ast.sigma_expand, 0.01, 1.0)
        _, ast.alpha = imgui.slider_float("Novelty Exponent", ast.alpha, 0.0, 8.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(ALPHA_TOOLTIP)
        _, ast.k = imgui.slider_int("Neighbours (k)", ast.k, 1, 50)
        _, ast.seed_n = imgui.slider_int("Seed Entries", ast.seed_n, 64, 2048)
        _, ast.sigma0 = imgui.slider_float("Bootstrap Sigma", ast.sigma0, 0.05, 1.5)
        _, ast.liveness_min = imgui.slider_float(
            "Liveness Floor", ast.liveness_min, 0.0, 0.5)
        _, ast.target_rate = imgui.slider_float(
            "Target Admission Rate", ast.target_rate, 0.01, 1.0)
        _, ast.refresh_per_gen = imgui.slider_int(
            "Novelty Refresh / Gen", ast.refresh_per_gen, 0, 512)
        _, ast.capacity = imgui.slider_int("Capacity", ast.capacity, 1000, 100000)

    def _render_expedition_settings(self, ast):
        _, ast.expansion_between = imgui.slider_int(
            "Expansion Between", ast.expansion_between, 0, 500)
        if imgui.is_item_hovered():
            imgui.set_tooltip("0 disables expeditions entirely - pure novelty "
                              "search, which is E&E's own ablation.")
        _, ast.expedition_gens = imgui.slider_int(
            "Expedition Gens", ast.expedition_gens, 5, 400)
        _, ast.expedition_sigma = imgui.slider_float(
            "Expedition Sigma", ast.expedition_sigma, 0.01, 1.0)
        _, ast.latent_share = imgui.slider_float(
            "Latent Goal Share", ast.latent_share, 0.0, 1.0)
        _, ast.beta = imgui.slider_float("Extrapolation (beta)", ast.beta, 0.0, 2.0)

    # ---- the browser ---------------------------------------------------

    def render_archive_window(self):
        ast = self.state.archive
        if not ast.show_browser:
            return
        # Without an explicit size ImGui auto-fits this window to something
        # smaller than its own content: the 360px gallery child gets clipped and
        # the Export / Seed / Delete row lands entirely below the fold, drawing
        # nothing at all. first_use_ever, so a window the user has resized keeps
        # their size.
        imgui.set_next_window_size(imgui.ImVec2(760, 620),
                                   imgui.Cond_.first_use_ever)
        expanded, opened = imgui.begin("Archive", True)
        if not opened:
            ast.show_browser = False
            imgui.end()
            return
        arc = self.archive_obj
        if arc is None:
            imgui.text_colored(imgui.ImVec4(*_DIM), "No archive yet.")
            imgui.end()
            return

        st = arc.stats()
        imgui.text(f"{st['size']} entries   {st['n_pinned']} pinned   "
                   f"threshold {st['threshold']:.3f}   "
                   f"admitting {100.0 * st['admission_rate']:.0f}%")
        imgui.separator()

        if imgui.begin_tab_bar("archive_views"):
            if imgui.begin_tab_item("Gallery")[0]:
                self._render_gallery(ast, arc)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Map")[0]:
                self._render_map(ast, arc)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.end()

    def _sorted_entries(self, ast, arc):
        entries = list(enumerate(arc.entries))
        if ast.pinned_only:
            entries = [(i, e) for i, e in entries if e.pinned]
        key = {"novelty": lambda p: -p[1].novelty,
               "liveness": lambda p: -p[1].liveness,
               "recency": lambda p: -p[1].ts}.get(ast.sort_by,
                                                  lambda p: -p[1].novelty)
        return sorted(entries, key=key)

    def _render_gallery(self, ast, arc):
        modes = ["novelty", "recency", "liveness"]
        idx = modes.index(ast.sort_by) if ast.sort_by in modes else 0
        ch, idx = imgui.combo("Sort", idx, ["Novelty", "Recency", "Liveness"])
        if ch:
            ast.sort_by = modes[idx]
        imgui.same_line()
        _, ast.pinned_only = imgui.checkbox("Pinned only", ast.pinned_only)

        cache = getattr(self, "thumb_cache", None)
        per_row = 6
        imgui.begin_child("gallery", imgui.ImVec2(0, 360))
        for n, (i, e) in enumerate(self._sorted_entries(ast, arc)[:240]):
            tex = cache.get(e.thumb) if cache is not None else None
            if tex is not None:
                imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(96, 96))
            else:
                imgui.button(f"#{e.id}", imgui.ImVec2(96, 96))
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"#{e.id}  {e.source}\nnovelty {e.novelty:.3f}\n"
                    f"liveness {e.liveness:.3f}\ngoal: {e.goal or '-'}")
            if imgui.is_item_clicked():
                ast.selected_entry_id = e.id
            if n % per_row != per_row - 1:
                imgui.same_line()
        imgui.end_child()

        if ast.selected_entry_id >= 0:
            imgui.separator()
            imgui.text(f"Selected #{ast.selected_entry_id}")
            if imgui.button("Export as config"):
                ast.export_entry_id = ast.selected_entry_id
            imgui.same_line()
            if imgui.button("Seed a run from here"):
                ast.seed_entry_id = ast.selected_entry_id
            imgui.same_line()
            if imgui.button("Delete"):
                ast.delete_entry_id = ast.selected_entry_id

    _MAP_COLORS = {
        "bootstrap": imgui.IM_COL32(120, 120, 130, 200),
        "expansion": imgui.IM_COL32(90, 200, 120, 220),
        "expedition": imgui.IM_COL32(255, 170, 60, 230),
        "pin": imgui.IM_COL32(90, 170, 255, 255),
    }

    def _render_map(self, ast, arc):
        proj = getattr(self, "archive_projection", None)
        # proj.fitted, not just proj: an unfitted Projection transforms
        # everything to the origin, which would stack the whole archive in one
        # corner and read as a broken map rather than an unbuilt one.
        if proj is None or not proj.fitted or len(arc) < 3:
            imgui.text_colored(imgui.ImVec4(*_DIM),
                               "Not enough entries to project yet.")
            return
        if imgui.button("Refit projection"):
            ast.refit_projection_requested = True
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(*_DIM),
                           "PCA of the CLIP embeddings. The map is a view - "
                           "novelty is always measured in the full 512-d space.")

        pts = proj.transform(arc.embeddings)
        if not len(pts):
            return
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)

        size = imgui.ImVec2(imgui.get_content_region_avail().x, 320)
        origin = imgui.get_cursor_screen_pos()
        imgui.invisible_button("map_canvas", size)
        hovering_canvas = imgui.is_item_hovered()
        clicked_canvas = imgui.is_item_clicked()
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(origin,
                             imgui.ImVec2(origin.x + size.x, origin.y + size.y),
                             imgui.IM_COL32(20, 20, 24, 255))

        mouse = imgui.get_mouse_pos()
        hovered, best_d = None, 1e9
        for i, e in enumerate(arc.entries):
            u = (pts[i] - lo) / span
            x = origin.x + 8.0 + float(u[0]) * (size.x - 16.0)
            y = origin.y + 8.0 + (1.0 - float(u[1])) * (size.y - 16.0)
            key = "pin" if e.pinned else e.source
            draw.add_circle_filled(
                imgui.ImVec2(x, y), 3.0,
                self._MAP_COLORS.get(key, self._MAP_COLORS["expansion"]))
            d = abs(mouse.x - x) + abs(mouse.y - y)
            if d < best_d:
                hovered, best_d = e, d

        goal_pt = getattr(self, "archive_goal_point", None)
        if goal_pt is not None:
            u = (proj.transform(np.asarray(goal_pt)[None])[0] - lo) / span
            gx = origin.x + 8.0 + float(u[0]) * (size.x - 16.0)
            gy = origin.y + 8.0 + (1.0 - float(u[1])) * (size.y - 16.0)
            draw.add_circle(imgui.ImVec2(gx, gy), 7.0,
                            imgui.IM_COL32(255, 90, 90, 255), 0, 2.0)

        if hovered is not None and best_d < 12.0 and hovering_canvas:
            imgui.set_tooltip(f"#{hovered.id}  {hovered.source}\n"
                              f"novelty {hovered.novelty:.3f}\n"
                              f"goal: {hovered.goal or '-'}")
            if clicked_canvas:
                ast.selected_entry_id = hovered.id
