"""Explore (IMGEP) tab and the archive browser.

Passive: renders widgets and sets one-shot flags, runs no logic. Every value
displayed comes from driver.status() or archive.stats(), so the UI cannot
disagree with the search about what is happening.
"""
from __future__ import annotations

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
        """Body arrives in Task 8; the early return is the permanent guard."""
        ast = self.state.archive
        if not ast.show_browser:
            return
        expanded, opened = imgui.begin("Archive", True)
        if not opened:
            ast.show_browser = False
            imgui.end()
            return
        imgui.text_colored(imgui.ImVec4(*_DIM), "No archive yet.")
        imgui.end()
