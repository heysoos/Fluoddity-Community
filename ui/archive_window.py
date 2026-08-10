"""Explore (IMGEP) tab and the archive browser.

Passive: renders widgets and sets one-shot flags, runs no logic. Every value
displayed comes from driver.status() or archive.stats(), so the UI cannot
disagree with the search about what is happening.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services import save_targets
from services.archive_library import safe_name
from ui.notices import BAD as _BAD
from ui.notices import DIM as _DIM
from ui.notices import OK as _OK
from ui.notices import WARN as _WARN
from ui.notices import render_banner

# One sentence each. A tooltip wider than the window is not read, it is
# dismissed - the long-form reasoning lives in the module docstrings.
GOAL_TOOLTIP = "Text goals for expeditions, cycled in order; leave empty for latent goals only."

ALPHA_TOOLTIP = "How strongly parent choice favours novel entries (p ~ novelty^alpha); 0 is uniform."

EXPORT_TOOLTIP = "Asks for a name, then saves this entry to your configs folder, openable from File > Load."

SEED_TOOLTIP = "Starts Auto (CLIP) mode's search from this genome; Explore mode picks its own parents."


def archive_row_model(ast) -> dict:
    """Everything the archive row draws, as data.

    Pure, so the rules that matter - which archive is selected, whether Delete
    is allowed - are assertable without an ImGui context.
    """
    names = [a["name"] for a in ast.archive_list] or [ast.archive_name]
    try:
        index = names.index(ast.archive_name)
    except ValueError:
        # The active folder was deleted outside the app. Pointing the combo at
        # a stale index would name somebody else's archive.
        index = 0
    labels = [f"{a['name']}  ({a['entries']} entries)" for a in ast.archive_list]
    if not labels:
        labels = list(names)

    active = next((a for a in ast.archive_list if a["name"] == ast.archive_name),
                  None)
    summary = (f"{active['entries']} entries · {active['size_mb']:.1f} MB"
               if active else "not loaded yet")
    return {
        "names": names,
        "labels": labels,
        "index": index,
        "entries": active["entries"] if active else 0,
        "delete_enabled": len(ast.archive_list) > 1,
        "summary": summary,
    }


def new_archive_status(ast) -> dict:
    """Whether the typed name can become an archive, and what to say if not."""
    typed = ast.new_archive_name
    safe = safe_name(typed)
    taken = any(a["name"] == safe for a in ast.archive_list)
    if taken:
        hint = f"An archive named '{safe}' already exists."
    elif safe and safe != typed.strip():
        hint = f"will be saved as: {safe}"
    else:
        hint = ""
    return {"safe": safe, "taken": taken,
            "can_create": bool(safe) and not taken, "hint": hint}


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

        # Wrapped, and Dismiss on its OWN line. These carry OS error strings -
        # "Could not empty 'default': [WinError 5] Access is denied: ..." with
        # two full paths in it - and same_line() after text that long pushed
        # the button off the right edge of the panel, so the banner could not
        # be dismissed at all.
        self._render_banner(ast, "warning", _BAD)
        self._render_banner(ast, "notice", _OK)

        if self.archive_unavailable:
            imgui.text_colored(imgui.ImVec4(*_WARN), self.archive_unavailable)
            imgui.text_wrapped(
                "Explore mode needs the same CLIP model and packages as Auto mode. "
                "Manual mode is unaffected.")
            return

        self._render_archive_row(ast)
        imgui.separator()
        # Above the fold, always: which archive, what the search is doing, and
        # the buttons that change it. Everything below is a setting you adjust
        # occasionally, so it folds away - the tab was one unbroken column of
        # twenty sliders and you had to scroll past all of them to reach the
        # browser button.
        self._render_explore_status(ast)
        imgui.separator()
        self._render_explore_transport(ast)
        imgui.separator()

        if imgui.collapsing_header("Goals", imgui.TreeNodeFlags_.default_open):
            self._render_goal_list(ast)
        if imgui.collapsing_header("Rollout"):
            self._render_rollout_controls(
                ast, grid_note="changing the grid ends any expedition in flight")
            self._render_view_setting(ast)
        if imgui.collapsing_header("Exploration"):
            self._render_exploration_settings(ast)
        # "Admission", not "Archive": an ImGui widget's identity IS its label,
        # and the archive combo at the top of this tab is already called
        # "Archive". Two visible items with one ID is a hard error - ImGui put
        # up its conflict dialog and the combo stopped responding to clicks
        # entirely. It is also the better name: this section is what the
        # archive KEEPS, which the combo is not.
        if imgui.collapsing_header("Admission"):
            self._render_archive_settings(ast)
        if imgui.collapsing_header("Expeditions"):
            self._render_expedition_settings(ast)

        imgui.separator()
        if imgui.button("Open Archive Browser"):
            ast.show_browser = True

    def _render_banner(self, ast, field, colour):
        """A dismissable message that may be arbitrarily long."""
        render_banner(ast, field, colour, scope="explore")

    def _render_archive_row(self, ast):
        """Which archive is active is an experimental variable, so it sits at
        the top of the tab rather than in a menu."""
        m = archive_row_model(ast)
        imgui.set_next_item_width(240)
        changed, idx = imgui.combo("Archive", m["index"], m["labels"])
        if changed and 0 <= idx < len(m["names"]) and m["names"][idx] != ast.archive_name:
            ast.switch_archive_name = m["names"][idx]
        imgui.same_line()
        if imgui.button("New##archive"):
            ast.new_archive_name = ""
            imgui.open_popup("New archive")
        imgui.same_line()
        if imgui.button("Empty##archive"):
            imgui.open_popup("Empty archive")
        imgui.same_line()
        imgui.begin_disabled(not m["delete_enabled"])
        if imgui.button("Delete##archive"):
            ast.confirm_delete_text = ""
            imgui.open_popup("Delete archive")
        imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Refresh##archive"):
            ast.refresh_archive_list_requested = True

        imgui.text_colored(imgui.ImVec4(*_DIM), m["summary"])
        self._render_archive_modals(ast)

    def _render_archive_modals(self, ast):
        flags = imgui.WindowFlags_.always_auto_resize
        if imgui.begin_popup_modal("New archive", flags=flags)[0]:
            imgui.text("Name for the new archive:")
            imgui.set_next_item_width(280)
            _, ast.new_archive_name = imgui.input_text(
                "##new_archive", ast.new_archive_name)
            st = new_archive_status(ast)
            if st["hint"]:
                colour = _WARN if st["taken"] else _DIM
                imgui.text_colored(imgui.ImVec4(*colour), st["hint"])
            imgui.separator()
            imgui.begin_disabled(not st["can_create"])
            if imgui.button("Create", imgui.ImVec2(120, 0)):
                ast.new_archive_requested = True
                imgui.close_current_popup()
            imgui.end_disabled()
            imgui.same_line()
            if imgui.button("Cancel##new_archive", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

        if imgui.begin_popup_modal("Empty archive", flags=flags)[0]:
            m = archive_row_model(ast)
            imgui.text_wrapped(
                f"Empty '{ast.archive_name}'? Its {m['entries']} entries move "
                f"to '{ast.archive_name}.cleared-<time>' and can be restored "
                f"by renaming that folder back. Nothing is deleted.")
            imgui.separator()
            if imgui.button("Empty it", imgui.ImVec2(120, 0)):
                ast.clear_archive_requested = True
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel##empty_archive", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

        if imgui.begin_popup_modal("Delete archive", flags=flags)[0]:
            imgui.text_wrapped(
                f"Permanently delete '{ast.archive_name}' - every entry, "
                f"thumbnail and goal in it. This cannot be undone.")
            imgui.text("Type the archive name to confirm:")
            imgui.set_next_item_width(280)
            _, ast.confirm_delete_text = imgui.input_text(
                "##confirm_delete", ast.confirm_delete_text)
            imgui.separator()
            imgui.begin_disabled(ast.confirm_delete_text != ast.archive_name)
            if imgui.button("Delete forever", imgui.ImVec2(140, 0)):
                ast.delete_archive_requested = True
                imgui.close_current_popup()
            imgui.end_disabled()
            imgui.same_line()
            if imgui.button("Cancel##delete_archive", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

    def _render_running_light(self, ast, running: bool):
        """Whether the search is actually stepping, as a colour.

        "Regime: expansion" reads identically whether the search is running or
        paused, so the tab looked the same either way and there was no way to
        tell a stalled run from a stopped one.
        """
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        r = imgui.get_text_line_height() * 0.32
        c = imgui.ImVec2(p.x + r + 2.0, p.y + imgui.get_text_line_height() * 0.5)
        colour = _OK if running else _DIM
        dl.add_circle_filled(c, r, imgui.get_color_u32(imgui.ImVec4(*colour)), 16)
        imgui.dummy(imgui.ImVec2(2.0 * r + 8.0, imgui.get_text_line_height()))
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(*colour),
                           "Exploring" if running else "Stopped")

    def _render_phase_bar(self, ph):
        """How far through the current phase, as a bar.

        `total` of 0 means the phase has no finish line - expeditions switched
        off - and drawing a full or empty bar there would both be lies.
        """
        imgui.text(ph["label"])
        if ph["total"] <= 0:
            imgui.text_colored(imgui.ImVec4(*_DIM), ph["note"] or "no end point")
            return
        frac = max(0.0, min(1.0, ph["done"] / float(ph["total"])))
        imgui.progress_bar(
            frac, imgui.ImVec2(-1.0, 0.0),
            f"{ph['done']} / {ph['total']} {ph['unit']}")
        if ph["note"]:
            imgui.text_colored(imgui.ImVec4(*_DIM), ph["note"])

    def _render_explore_status(self, ast):
        d = self.archive_driver
        if d is None:
            self._render_running_light(ast, False)
            imgui.text_colored(imgui.ImVec4(*_DIM), "Not started")
            return
        st = d.status()
        self._render_running_light(ast, bool(ast.running))
        # .get, like every other read here: the panel must render against
        # whatever the driver chooses to report rather than requiring it.
        phase = st.get("phase")
        if phase:
            self._render_phase_bar(phase)
        else:
            imgui.text(f"Regime: {st['regime']}")
            imgui.text(f"Goal: {st.get('goal') or '-'}")
        # Only while there is something to abandon - a permanently dead button
        # would be noise on a panel this dense.
        if st["regime"] == "expedition":
            if imgui.button("Cancel##expedition"):
                ast.cancel_expedition_requested = True
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Abandon this goal; the next expedition is a full "
                    "Expansion Between interval away.")
        imgui.text(f"Archive: {st['archive_size']} / {st['capacity']}   "
                   f"admitting {100.0 * st['admission_rate']:.0f}%   "
                   f"evicted {st['n_evicted']}")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "A low rate is normal with Min Separation on - it means the "
                "archive already holds those patterns.")
        if st.get("last_tiles"):
            imgui.text_colored(
                imgui.ImVec4(*_DIM),
                f"Last generation: kept {st.get('last_admitted', 0)} of "
                f"{st['last_tiles']} tiles   "
                f"({st.get('n_rejected_close', 0)} too close, "
                f"{st.get('n_rejected_dead', 0)} dead, all-time)")
        if st.get("n_summits") or st.get("n_records"):
            imgui.text_colored(
                imgui.ImVec4(*_OK),
                f"Kept for matching: {st.get('n_summits', 0)} summits, "
                f"{st.get('n_records', 0)} goal records")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Summits beat their own expedition's best; records beat "
                    "the whole archive's best for one of your goals.")
        self._render_explore_traces(d)
        # The seed pool actually used, so the band is legible rather than a
        # pair of numbers with no visible effect.
        if st.get("seed_ess"):
            imgui.text_colored(
                imgui.ImVec4(*_DIM),
                f"Seed pool: {st['seed_ess']:.0f} entries "
                f"(alpha {st['seed_alpha']:.1f})")
        if st.get("blocked_by_pins"):
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "Archive is over capacity and entirely pinned - nothing can "
                "be evicted, so it will keep growing.")
        # A persistently zero admission rate now means the capture is broken or
        # the preset is dead. It can no longer mean "the search is hard": there
        # is no novelty gate left for a hard search to fail.
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

    _SIZE_COLOR = (0.45, 0.70, 1.00, 1.0)      # blue, how much
    _NOV_COLOR = (0.85, 0.55, 1.00, 1.0)       # violet, how different
    _BEST_COLOR = (0.35, 0.85, 0.45, 1.0)      # green, climbing
    _MEAN_COLOR = (0.45, 0.55, 0.50, 1.0)      # grey-green, the population

    def _render_explore_traces(self, d):
        """Two plots: what the archive is doing, and what the goal chase is.

        Archive size alone cannot say whether exploration is working - it only
        ever goes up. Size together with mean novelty can: both climbing is new
        territory, size climbing while novelty falls is filling in ground the
        archive already covers, and that is the state the separation rule
        exists to prevent.

        The expedition plot answers the other question, which was previously
        unanswerable from the UI at all: is this goal chase still climbing, or
        did it converge thirty generations ago and start handing the archive
        sixty-four copies of the same tile?
        """
        t = getattr(d, "trace", None)
        if not t or not t["gen"]:
            imgui.text_colored(imgui.ImVec4(*_DIM), "no generations yet")
            return

        if imgui.tree_node_ex("Archive diversity",
                              imgui.TreeNodeFlags_.default_open):
            self._render_series(
                [(t["archive_size"], self._SIZE_COLOR, "size"),
                 (t["mean_novelty"], self._NOV_COLOR, "mean novelty")],
                height=70.0)
            imgui.tree_pop()

        ex = d.expedition_trace()
        if imgui.tree_node_ex("Expedition fitness",
                              imgui.TreeNodeFlags_.default_open):
            if not ex["gens"]:
                imgui.text_colored(imgui.ImVec4(*_DIM),
                                   "no expedition has run yet")
            else:
                self._render_series(
                    [(ex["mean"], self._MEAN_COLOR, "mean"),
                     (ex["best"], self._BEST_COLOR, "best")],
                    height=70.0)
                imgui.text_colored(
                    imgui.ImVec4(*_DIM),
                    f"{ex['gens']} generations - flat means converged, and a "
                    f"converged expedition keeps proposing the same tile")
            imgui.tree_pop()

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

        # enter_returns_true so a list can be typed straight through without
        # reaching for the mouse between every entry.
        submitted, ast.new_goal_text = imgui.input_text(
            "##new_goal", ast.new_goal_text,
            flags=imgui.InputTextFlags_.enter_returns_true)
        entered = submitted and bool(ast.new_goal_text.strip())
        if entered:
            # Enter defocuses the box by default, which would make the second
            # goal need a click after all. -1 is the item just submitted.
            imgui.set_keyboard_focus_here(-1)
        imgui.same_line()
        if (imgui.button("Add Goal") and ast.new_goal_text.strip()) or entered:
            ast.add_goal_requested = True

        order = ["round_robin", "least_matched"]
        idx = order.index(ast.goal_order) if ast.goal_order in order else 0
        ch, idx = imgui.combo("Goal Order", idx, ["Round robin", "Least matched"])
        if ch:
            ast.goal_order = order[idx]

    def _render_view_setting(self, ast):
        _, ast.n_views = imgui.slider_int("CLIP Views", ast.n_views, 1, 8)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Random sub-crops averaged into each tile's embedding; more is "
                "steadier but costs CLIP time.")
        if ast.n_views <= 1:
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "1 view is the raw frame - a 16px shift then reads as a "
                "different creature")

    def _render_exploration_settings(self, ast):
        """How the search MOVES: where children come from and how far."""
        _, ast.sigma_expand = imgui.slider_float(
            "Expansion Sigma", ast.sigma_expand, 0.01, 1.0)
        _, ast.alpha = imgui.slider_float("Novelty Exponent", ast.alpha, 0.0, 8.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(ALPHA_TOOLTIP)
        _, ast.k = imgui.slider_int("Neighbours (k)", ast.k, 1, 50)
        _, ast.seed_n = imgui.slider_int("Seed Entries", ast.seed_n, 64, 2048)
        _, ast.sigma0 = imgui.slider_float("Bootstrap Sigma", ast.sigma0, 0.05, 1.5)

    def _render_archive_settings(self, ast):
        """What the archive KEEPS: the admission gates and the retention cap.

        Split from the exploration settings because they answer a different
        question. These four decide what survives; the others decide where the
        search looks next.
        """
        # 0-0.1, not 0-0.5: measured preset liveness tops out at 0.079, so a
        # 0.5 range would bury the entire useful span in the leftmost sixth of
        # the slider.
        _, ast.liveness_min = imgui.slider_float(
            "Liveness Floor", ast.liveness_min, 0.0, 0.1, "%.4f")
        # 0-0.05, because the measurement says the whole useful span is there:
        # at 0.05 every real archive keeps under 6% of what it holds now.
        _, ast.min_separation = imgui.slider_float(
            "Min Separation", ast.min_separation, 0.0, 0.05, "%.4f")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Refuses anything this close to an entry already stored; 0 "
                "stores everything.")
        if ast.min_separation <= 0.0:
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "off - a converging expedition will store every tile it makes")
        _, ast.capacity = imgui.slider_int("Capacity", ast.capacity, 1000, 100000)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The long-run cap: over capacity, the least novel entries are "
                "evicted.")
        _, ast.refresh_sweep_gens = imgui.slider_int(
            "Novelty Sweep (gens)", ast.refresh_sweep_gens, 1, 100)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Generations until every entry has been re-scored, i.e. how "
                "stale novelty may get.")

    def _render_expedition_settings(self, ast):
        _, ast.expansion_between = imgui.slider_int(
            "Expansion Between", ast.expansion_between, 0, 500)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Expansion generations between expeditions; 0 disables "
                "expeditions entirely.")
        _, ast.expedition_gens = imgui.slider_int(
            "Expedition Gens", ast.expedition_gens, 5, 400)
        _, ast.expedition_sigma = imgui.slider_float(
            "Expedition Sigma", ast.expedition_sigma, 0.01, 1.0)
        _, ast.novelty_share = imgui.slider_float(
            "Novelty Goal Share", ast.novelty_share, 0.0, 1.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Expeditions with no target that climb novelty itself; always "
                "well posed, unlike a point that may not be reachable.")
        _, ast.latent_share = imgui.slider_float(
            "Latent Goal Share", ast.latent_share, 0.0, 1.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Expeditions toward a point extrapolated past the archive's "
                "frontier; your text goals take whatever these two leave.")
        left = 1.0 - min(1.0, ast.novelty_share + ast.latent_share)
        imgui.text_colored(
            imgui.ImVec4(*_DIM),
            f"goals: {100 * min(1.0, ast.novelty_share):.0f}% novelty, "
            f"{100 * min(1.0, max(0.0, 1.0 - ast.novelty_share), ast.latent_share):.0f}% latent, "
            f"{100 * left:.0f}% text")
        _, ast.seed_ess_min = imgui.slider_float(
            "Seed Pool Min", ast.seed_ess_min, 1.0, 128.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Fewest entries in the running as an expedition's starting "
                "point, so a repeated goal does not retrace one trajectory.")
        _, ast.seed_ess_max = imgui.slider_float(
            "Seed Pool Max", ast.seed_ess_max, 16.0, 4096.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Most entries in the running, so a diffuse goal does not stop "
                "influencing the starting point as the archive grows.")
        # "Extrapolation (beta)" was here. The latent goal no longer
        # extrapolates away from the centroid - it extrapolates in the archive's
        # principal subspace, in whitened units, and no value of beta made the
        # old construction work. See services/goal_source.py.

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
        # The name is in the title, not the body: a screenshot of the gallery
        # should say which archive it came from.
        expanded, opened = imgui.begin(f"Archive - {ast.archive_name}###archive",
                                       True)
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
        imgui.text(f"{st['size']} / {st['capacity']} entries   "
                   f"{st['n_pinned']} pinned   "
                   f"{st['n_evicted']} evicted")
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
        """The gallery's display order, cached against the archive revision.

        Sorting every entry to show 240 of them costs 1.5 ms at 4808 entries
        and 8.0 ms at the 20000 capacity - per frame, for an order that only
        changes when the archive does.
        """
        key = (ast.sort_by, ast.pinned_only, getattr(arc, "revision", None))
        hit = getattr(self, "_sort_cache", None)
        if hit is not None and hit[0] is arc and hit[1] == key:
            return hit[2]

        entries = list(enumerate(arc.entries))
        if ast.pinned_only:
            entries = [(i, e) for i, e in entries if e.pinned]
        keyfn = {"novelty": lambda p: -p[1].novelty,
                 "liveness": lambda p: -p[1].liveness,
                 "recency": lambda p: -p[1].ts}.get(ast.sort_by,
                                                    lambda p: -p[1].novelty)
        out = sorted(entries, key=keyfn)
        self._sort_cache = (arc, key, out)
        return out

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
            if imgui.button("Save as config..."):
                self.open_save_popup(save_targets.ARCHIVE_ENTRY,
                                     arg=ast.selected_entry_id)
            if imgui.is_item_hovered():
                imgui.set_tooltip(EXPORT_TOOLTIP)
            imgui.same_line()
            if imgui.button("Seed a run from here"):
                ast.seed_entry_id = ast.selected_entry_id
            if imgui.is_item_hovered():
                imgui.set_tooltip(SEED_TOOLTIP)
            imgui.same_line()
            if imgui.button("Delete"):
                ast.delete_entry_id = ast.selected_entry_id
            if imgui.is_item_hovered():
                imgui.set_tooltip("Remove this entry and its thumbnail.")

    _MAP_COLORS = {
        "bootstrap": imgui.IM_COL32(120, 120, 130, 200),
        "expansion": imgui.IM_COL32(90, 200, 120, 220),
        "expedition": imgui.IM_COL32(255, 170, 60, 230),
        # A tile that beat its expedition's best. Brighter than the expedition
        # colour it sits among, because the whole point is to be able to find
        # these afterwards - they are the results of the goal chases.
        "summit": imgui.IM_COL32(255, 240, 130, 255),
        # The archive's best-ever match for one of the text goals. Its own
        # colour rather than the summit's: a record can be set in any regime,
        # usually while chasing a completely different goal.
        "record": imgui.IM_COL32(255, 120, 200, 255),
        "pin": imgui.IM_COL32(90, 170, 255, 255),
    }

    _MAP_PAD = 8.0
    _MAP_ZOOM_MIN = 1.0
    _MAP_ZOOM_MAX = 200.0
    # Screen pixels of movement that turn a click into a drag. Without it a pan
    # also selects whatever dot the press happened to land on.
    _MAP_CLICK_SLOP = 4.0

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
        if imgui.is_item_hovered():
            imgui.set_tooltip("Recompute the 2-D PCA over the current archive.")
        imgui.same_line()
        if imgui.button("Home##map"):
            self._map_home(ast)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Reset zoom and recentre.")
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(*_DIM),
                           f"{ast.map_zoom:.1f}x - scroll to zoom, drag to pan")

        cached = self._map_points(arc, proj)
        if cached is None:
            return
        unit, lo, span, colors = cached

        size = imgui.ImVec2(imgui.get_content_region_avail().x, 320)
        origin = imgui.get_cursor_screen_pos()
        imgui.invisible_button("map_canvas", size)
        hovering = imgui.is_item_hovered()
        clicked = self._map_interact(ast, origin, size, hovering)

        draw = imgui.get_window_draw_list()
        far = imgui.ImVec2(origin.x + size.x, origin.y + size.y)
        draw.add_rect_filled(origin, far, imgui.IM_COL32(20, 20, 24, 255))
        # Zooming moves points outside the canvas; without a clip they would be
        # drawn over the rest of the tab.
        draw.push_clip_rect(origin, far, True)

        xs, ys = self._map_to_screen(ast, unit, origin, size)
        # Only what is actually on the canvas: at 200x almost nothing is, and a
        # draw call per archive entry per frame is the cost otherwise.
        on = ((xs >= origin.x) & (xs <= far.x) & (ys >= origin.y) & (ys <= far.y))
        sel = np.flatnonzero(on)
        # .tolist() first: indexing a numpy array with a Python int inside the
        # loop builds a scalar object per access, which costs more than the
        # draw call it feeds.
        px, py = xs[sel].tolist(), ys[sel].tolist()
        pc = colors[sel].tolist()
        for x, y, c in zip(px, py, pc):
            draw.add_circle_filled(imgui.ImVec2(x, y), 3.0, c)

        best_i, best_d = -1, 1e9
        if hovering and len(sel):
            mouse = imgui.get_mouse_pos()
            d = np.abs(xs[sel] - mouse.x) + np.abs(ys[sel] - mouse.y)
            j = int(np.argmin(d))
            best_i, best_d = int(sel[j]), float(d[j])

        goal_pt = getattr(self, "archive_goal_point", None)
        if goal_pt is not None:
            gu = (proj.transform(np.asarray(goal_pt)[None]) - lo) / span
            gx, gy = self._map_to_screen(ast, gu, origin, size)
            draw.add_circle(imgui.ImVec2(float(gx[0]), float(gy[0])), 7.0,
                            imgui.IM_COL32(255, 90, 90, 255), 0, 2.0)
        draw.pop_clip_rect()

        if hovering and best_i >= 0 and best_d < 12.0:
            self._map_hover_card(arc.entries[best_i])
            if clicked:
                ast.selected_entry_id = arc.entries[best_i].id

        self._render_map_selection(ast, arc)

    def _map_points(self, arc, proj):
        """-> (unit-square positions, lo, span, per-entry colours), or None.

        Cached against (archive revision, projection version), because neither
        operand of the projection changes between generations while the map is
        redrawn 60 times a second. Measured on the real archives: the transform
        alone is 3.8 ms at 4808 entries, 5.8 ms at 8002 and 20 ms at the 20000
        capacity - most of what the browser cost, and none of it new work.

        Colours are resolved here too: the pin/source lookup is a dict hit and
        a conditional per entry, which belongs with the rest of the per-entry
        work rather than in the draw loop.
        """
        key = (getattr(arc, "revision", None), getattr(proj, "version", None),
               len(arc))
        hit = getattr(self, "_map_cache", None)
        if hit is not None and hit[0] is arc and hit[1] is proj and hit[2] == key:
            return hit[3]

        pts = proj.transform(arc.embeddings)
        if not len(pts):
            return None
        lo = pts.min(axis=0)
        span = np.maximum(pts.max(axis=0) - lo, 1e-6)
        unit = (pts - lo) / span                    # whole archive in [0, 1]^2
        fallback = self._MAP_COLORS["expansion"]
        colors = np.array(
            [self._MAP_COLORS.get("pin" if e.pinned else e.source, fallback)
             for e in arc.entries], dtype=np.int64)

        out = (unit, lo, span, colors)
        self._map_cache = (arc, proj, key, out)
        return out

    @staticmethod
    def _map_home(ast) -> None:
        ast.map_zoom = 1.0
        ast.map_center_x = 0.5
        ast.map_center_y = 0.5

    def _map_to_screen(self, ast, unit, origin, size):
        """Normalised projection coords -> screen x, y arrays.

        y is flipped: the projection's second component points up, screen y
        points down.
        """
        pad = self._MAP_PAD
        w, h = size.x - 2.0 * pad, size.y - 2.0 * pad
        vx = (unit[:, 0] - ast.map_center_x) * ast.map_zoom + 0.5
        vy = (unit[:, 1] - ast.map_center_y) * ast.map_zoom + 0.5
        return origin.x + pad + vx * w, origin.y + pad + (1.0 - vy) * h

    def _map_interact(self, ast, origin, size, hovering) -> bool:
        """Wheel zoom and drag pan. -> was this a click rather than a drag?"""
        pad = self._MAP_PAD
        w, h = max(size.x - 2.0 * pad, 1.0), max(size.y - 2.0 * pad, 1.0)
        io = imgui.get_io()

        if hovering and io.mouse_wheel:
            m = imgui.get_mouse_pos()
            vx = (m.x - origin.x - pad) / w
            vy = 1.0 - (m.y - origin.y - pad) / h
            # The point under the cursor before the zoom...
            ux = (vx - 0.5) / ast.map_zoom + ast.map_center_x
            uy = (vy - 0.5) / ast.map_zoom + ast.map_center_y
            ast.map_zoom = float(np.clip(
                ast.map_zoom * (1.15 ** float(io.mouse_wheel)),
                self._MAP_ZOOM_MIN, self._MAP_ZOOM_MAX))
            # ...must still be under it afterwards, or zooming walks away from
            # whatever you were looking at.
            ast.map_center_x = ux - (vx - 0.5) / ast.map_zoom
            ast.map_center_y = uy - (vy - 0.5) / ast.map_zoom

        if imgui.is_item_active():
            d = io.mouse_delta
            self._map_drag_px = getattr(self, "_map_drag_px", 0.0) + \
                abs(d.x) + abs(d.y)
            ast.map_center_x -= (d.x / w) / ast.map_zoom
            ast.map_center_y += (d.y / h) / ast.map_zoom
            return False

        was_drag = getattr(self, "_map_drag_px", 0.0) > self._MAP_CLICK_SLOP
        released = imgui.is_item_deactivated()
        if released:
            self._map_drag_px = 0.0
        return bool(released and not was_drag)

    def _map_hover_card(self, entry) -> None:
        """The picture, on hover. A dot's position is not what it IS."""
        cache = getattr(self, "thumb_cache", None)
        tex = cache.get(entry.thumb) if cache is not None else None
        imgui.begin_tooltip()
        if tex is not None:
            imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(160, 160))
        imgui.text(f"#{entry.id}  {entry.source}"
                   f"{'  pinned' if entry.pinned else ''}")
        imgui.text(f"novelty {entry.novelty:.3f}   "
                   f"liveness {entry.liveness:.3f}")
        imgui.text(f"goal: {entry.goal or '-'}")
        imgui.end_tooltip()

    def _render_map_selection(self, ast, arc):
        """The picked dot's actual image.

        A dot is a position in a projection; without the picture beside it the
        map says where something sits but never what it is - which is most of
        why you would click it."""
        entry = next((e for e in arc.entries if e.id == ast.selected_entry_id),
                     None)
        if entry is None:
            imgui.text_colored(imgui.ImVec4(*_DIM),
                               "Click a point to see what it is.")
            return

        imgui.separator()
        cache = getattr(self, "thumb_cache", None)
        tex = cache.get(entry.thumb) if cache is not None else None
        if tex is not None:
            imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(128, 128))
        else:
            imgui.button(f"#{entry.id}##mapsel", imgui.ImVec2(128, 128))
        imgui.same_line()
        imgui.begin_group()
        imgui.text(f"#{entry.id}   {entry.source}"
                   f"{'   pinned' if entry.pinned else ''}")
        imgui.text(f"novelty {entry.novelty:.3f}   liveness {entry.liveness:.3f}")
        imgui.text(f"goal: {entry.goal or '-'}")
        imgui.text_disabled(f"gen {entry.gen}   tile {entry.tile}   {entry.spec}")
        if imgui.button("Save as config...##map"):
            self.open_save_popup(save_targets.ARCHIVE_ENTRY, arg=entry.id)
        imgui.same_line()
        if imgui.button("Delete##map"):
            ast.delete_entry_id = entry.id
        imgui.end_group()
