"""Explore (IMGEP) tab and the archive browser.

Passive: renders widgets and sets one-shot flags, runs no logic. Every value
displayed comes from driver.status() or archive.stats(), so the UI cannot
disagree with the search about what is happening.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services import map_view, save_targets
from services.archive_library import safe_name
from services.gallery_sort import (DEFAULT_SORT, GALLERY_SORTS,
                                   sort_entries)
from services.map_layout import ENGINE_LABELS, ENGINES
from ui import hints, layout
from ui.notices import BAD as _BAD
from ui.notices import DIM as _DIM
from ui.notices import OK as _OK
from ui.notices import WARN as _WARN
from ui.notices import render_banner

# One sentence each. See docs/imgep.md for what any of them actually do.
GOAL_TOOLTIP = "Text goals for expeditions, cycled in order."

ALPHA_TOOLTIP = "How strongly parent choice favours novel entries."

EXPORT_TOOLTIP = "Saves this entry to your configs folder, under a name you pick."

SEED_TOOLTIP = "Starts Auto (Prompt) mode's search from this genome."


# The gallery's tile size, in pixels. 160 is what a thumbnail holds
# (services/archive_io.THUMB_PX); past it there is nothing more to see.
GALLERY_SIZE_MIN = 16
GALLERY_SIZE_MAX = 160
# At and below this the gallery is a detail list instead of a grid. The grid
# therefore has a FLOOR, and that is what bounds how many thumbnails one frame
# can ask for - see ThumbCache.reserve.
GALLERY_LIST_MAX = 48
GALLERY_H = 360.0
GALLERY_TABLE = "gallery_list"
# The map atlas's thumbnail size, in pixels. The cell size IS the thumbnail
# size, which is what bounds the count by the viewport.
ATLAS_PX_MIN = 16
ATLAS_PX_MAX = 64
# The map decodes its own thumbnails SMALL, through JPEG's DCT scaling. A cell
# is never drawn above ATLAS_PX_MAX, so a texture at ATLAS_TEX_PX is always at
# least the drawn size - and at an eighth of the memory of the stored 160px
# one, which is what lets every cell have a picture rather than the most novel
# few.
ATLAS_TEX_PX = 80
ATLAS_CACHE_CAPACITY = 2048
# The most pictures the atlas will draw at once, and the headroom left over
# the top of them. The visible cell count is set by the size slider and the
# canvas alone, so it must be BOUNDED: a working set larger than the cache
# evicts its own cells and re-decodes them for as long as the map is open.
# The headroom is for the hover card and the selection panel, which call
# get() on entries that are not cell winners.
ATLAS_HEADROOM = 256
ATLAS_BUDGET = ATLAS_CACHE_CAPACITY - ATLAS_HEADROOM
# Thumbnails DECODED per frame. A JPEG decode is about a
# millisecond, and a zoom changes every cell's winner at once.
# Decoded per frame. A drafted decode is a third of a millisecond, and the
# atlas is double-buffered, so this paces the fill of a level the user cannot
# see yet rather than the one on screen.
ATLAS_NEW_PER_FRAME = 24
# (column label, the sort mode its header selects; "" for a column that does
# not sort). The modes are GALLERY_SORTS keys, so a header and the combo can
# never mean different things.
GALLERY_COLUMNS = (
    ("Thumb", ""),
    ("#", "id"),
    ("Source", "source"),
    ("Brain", "brain"),
    ("Novelty", "novelty"),
    ("Goal", "goal"),
)


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

        # Wrapped, and Dismiss on its OWN line: these carry OS error strings
        # with full paths, and same_line() pushes the button off the edge.
        self._render_banner(ast, "warning", _BAD)
        self._render_banner(ast, "notice", _OK)

        if self.archive_unavailable == "model_missing":
            self._render_missing_encoder(ast)
            return
        if self.archive_unavailable:
            imgui.text_colored(imgui.ImVec4(*_WARN), self.archive_unavailable)
            imgui.text_wrapped(
                "Explore mode needs the same encoder and packages as Auto mode. "
                "Manual mode is unaffected.")
            return

        self._render_archive_row(ast)
        imgui.separator()
        # Above the fold: which archive, what the search is doing, and the
        # buttons that change it. Settings fold away below.
        self._render_explore_status(ast)
        imgui.separator()
        self._render_explore_transport(ast)
        imgui.separator()
        # Above the fold with the transport: it decides what the search is FOR,
        # and a folded header's body does not run at all.
        self._render_physics_search(ast)
        imgui.separator()

        layout.push_settings_width()
        if imgui.collapsing_header("Goals", imgui.TreeNodeFlags_.default_open):
            self._render_goal_list(ast)
        if imgui.collapsing_header("Rollout"):
            self._render_rollout_controls(
                ast, grid_note="changing the grid ends any expedition in flight")
            self._render_view_setting(ast)
        if imgui.collapsing_header("Exploration"):
            self._render_exploration_settings(ast)
        # "Admission", not "Archive": a widget's identity IS its label, and the
        # archive combo at the top of this tab already owns "Archive".
        if imgui.collapsing_header("Admission"):
            self._render_archive_settings(ast)
        if imgui.collapsing_header("Expeditions"):
            self._render_expedition_settings(ast)
        self._render_config_history(ast)
        imgui.pop_item_width()

        imgui.separator()
        if imgui.button("Open Archive Browser"):
            ast.show_browser = True

    def _render_banner(self, ast, field, colour, scope="explore"):
        """A dismissable message that may be arbitrarily long.

        Distinct scopes for the tab and the browser: both can be on screen at
        once, and two Dismiss buttons sharing an ImGui id kills one of them.
        """
        render_banner(ast, field, colour, scope=scope)

    def _render_missing_encoder(self, ast):
        """This archive's vectors are in an encoder that is not downloaded.

        The archive row comes too, because the other way out is an archive in
        a space that IS on disk, and the rest of the tab cannot be drawn.
        """
        from services.vision_models import get

        try:
            label = get(ast.encoder_key).label
        except KeyError:
            label = ast.encoder_key
        imgui.text_colored(imgui.ImVec4(*_WARN),
                           f"{label} is not downloaded.")
        imgui.text_wrapped(
            "This archive's entries were embedded by it, and no other encoder "
            "can read them. Manual mode is unaffected.")
        if imgui.button(f"Download {ast.encoder_key}"):
            ast.download_model_requested = True
        imgui.separator()
        self._render_archive_row(ast)

    def _render_archive_row(self, ast, show_summary=True):
        """Which archive is active is an experimental variable, so it sits at
        the top of the tab rather than in a menu.

        Drawn in the browser window too - it is a whole window about one
        archive, and picking which one had no control in it. Safe to draw
        twice: a widget's ID includes its window, so the two combos do not
        collide. `show_summary` is off there because the browser prints its own
        entry count immediately below.
        """
        m = archive_row_model(ast)
        right = layout.row_right_edge()
        layout.push_settings_width()
        changed, idx = imgui.combo("Archive", m["index"], m["labels"])
        imgui.pop_item_width()
        if changed and 0 <= idx < len(m["names"]) and m["names"][idx] != ast.archive_name:
            ast.switch_archive_name = m["names"][idx]

        # Wrapped rather than one fixed row: five items do not fit a narrow
        # panel, and a clipped Delete button cannot be clicked.
        layout.wrap_row(right, layout.button_width("New"))
        if imgui.button("New##archive"):
            ast.new_archive_name = ""
            imgui.open_popup("New archive")
        layout.wrap_row(right, layout.button_width("Empty"))
        if imgui.button("Empty##archive"):
            imgui.open_popup("Empty archive")
        layout.wrap_row(right, layout.button_width("Delete"))
        imgui.begin_disabled(not m["delete_enabled"])
        if imgui.button("Delete##archive"):
            ast.confirm_delete_text = ""
            imgui.open_popup("Delete archive")
        imgui.end_disabled()
        layout.wrap_row(right, layout.button_width("Refresh"))
        if imgui.button("Refresh##archive"):
            ast.refresh_archive_list_requested = True

        self._render_encoder_combo(ast)
        if show_summary:
            layout.text_colored_wrapped(_DIM, m["summary"])
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
            self._render_new_archive_encoder(ast)
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
        """Whether the search is actually stepping, as a colour. The regime
        line alone reads the same running or paused."""
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
        # .get, like every other read here: the panel renders against whatever
        # the driver reports rather than requiring it.
        phase = st.get("phase")
        if phase:
            self._render_phase_bar(phase)
        else:
            imgui.text(f"Regime: {st['regime']}")
            imgui.text_wrapped(f"Goal: {st.get('goal') or '-'}")
        if st["regime"] == "expedition":
            if imgui.button("Cancel##expedition"):
                ast.cancel_expedition_requested = True
            hints.tip("Abandon this goal.")
        imgui.text_wrapped(f"Archive: {st['archive_size']} / {st['capacity']}   "
                           f"admitting {100.0 * st['admission_rate']:.0f}%   "
                           f"evicted {st['n_evicted']}")
        hints.tip("A low rate means the archive already holds "
                  "those patterns.")
        if st.get("last_tiles"):
            layout.text_disabled_wrapped(
                f"Last generation: kept {st.get('last_admitted', 0)} of "
                f"{st['last_tiles']} tiles   "
                f"({st.get('n_rejected_close', 0)} too close, "
                f"{st.get('n_rejected_dead', 0)} dead, all-time)")
        if st.get("n_summits") or st.get("n_records"):
            layout.text_colored_wrapped(
                _OK,
                f"Kept for matching: {st.get('n_summits', 0)} summits, "
                f"{st.get('n_records', 0)} goal records")
            hints.tip(
                "Summits beat their own expedition's best; records beat "
                "the archive's best for one of your goals.")
        self._render_explore_traces(d)
        if st.get("seed_ess"):
            layout.text_disabled_wrapped(
                f"Seed pool: {st['seed_ess']:.0f} entries "
                f"(alpha {st['seed_alpha']:.1f})")
        if st.get("seed_phys_clipped"):
            # Re-encoding an archived phenotype is the one lossy step in the
            # search, and nothing else on screen would show it happened.
            from services.physics_genome import PHYSICS_DIM
            layout.text_colored_wrapped(
                _WARN,
                f"Seed: the preset could not reach "
                f"{st['seed_phys_clipped']} of {PHYSICS_DIM} physics genes - "
                f"the chase started from the nearest creature to it.")
        if st.get("blocked_by_pins"):
            layout.text_colored_wrapped(
                _WARN,
                "Archive is over capacity and entirely pinned - nothing can "
                "be evicted, so it will keep growing.")
        if st["archive_size"] > 0 and st["admission_rate"] <= 0.0:
            layout.text_colored_wrapped(
                _WARN,
                "Nothing has been admitted recently - check the preset is alive, "
                "the capture is not black, and Liveness Floor is not too high.")

    def _render_explore_transport(self, ast):
        right = layout.row_right_edge()
        if imgui.button("Start##explore"):
            ast.start_requested = True
        layout.wrap_row(right, layout.button_width("Pause"))
        if imgui.button("Pause##explore"):
            ast.pause_requested = True
        layout.wrap_row(right, layout.button_width("Reset Search"))
        if imgui.button("Reset Search##explore"):
            ast.reset_requested = True
        note = "(Reset keeps the archive)"
        layout.wrap_row(right, imgui.calc_text_size(note).x)
        imgui.text_colored(imgui.ImVec4(*_DIM), note)
        self._render_cycle_estimate(ast)

    _SIZE_COLOR = (0.45, 0.70, 1.00, 1.0)      # blue, how much
    _NOV_COLOR = (0.85, 0.55, 1.00, 1.0)       # violet, how different
    _BEST_COLOR = (0.35, 0.85, 0.45, 1.0)      # green, climbing
    _MEAN_COLOR = (0.45, 0.55, 0.50, 1.0)      # grey-green, the population

    def _render_explore_traces(self, d):
        """Two plots: what the archive is doing, and what the goal chase is.

        Size climbing while novelty falls means the archive is filling in
        ground it already covers; a flat expedition trace means it converged.
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
                layout.text_disabled_wrapped(
                    f"{ex['gens']} generations - flat means converged")
            imgui.tree_pop()

    _STEPS_PER_SECOND = 716.0

    def _render_cycle_estimate(self, ast):
        """One expansion+expedition cycle in wall-clock, so the cost of a long
        expedition is visible before choosing it."""
        per_gen = ast.steps_per_gen / self._STEPS_PER_SECOND
        cycle = (ast.expansion_between + ast.expedition_gens) * per_gen
        layout.text_colored_wrapped(
            _DIM,
            f"~{per_gen:.1f}s per generation, ~{cycle / 60.0:.1f} min per cycle")

    def _render_goal_list(self, ast):
        imgui.text("Goals")
        imgui.same_line()
        imgui.text_disabled("(?)")
        hints.tip(GOAL_TOOLTIP)

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
        imgui.set_next_item_width(-layout.button_width("Add Goal")
                                  - imgui.get_style().item_spacing.x)
        submitted, ast.new_goal_text = imgui.input_text(
            "##new_goal", ast.new_goal_text,
            flags=imgui.InputTextFlags_.enter_returns_true)
        entered = submitted and bool(ast.new_goal_text.strip())
        if entered:
            # Enter defocuses the box by default. -1 is the item just submitted.
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
        _, ast.n_views = imgui.slider_int("Encoder Views", ast.n_views, 1, 8)
        hints.tip("Sub-crops averaged into each tile's embedding.")
        if ast.n_views <= 1:
            layout.text_colored_wrapped(
                _WARN,
                "1 view is the raw frame - a small shift then reads as a "
                "different creature")

    def _render_exploration_settings(self, ast):
        """How the search MOVES: where children come from and how far."""
        _, ast.sigma_expand = imgui.slider_float(
            "Expansion Sigma", ast.sigma_expand, 0.01, 1.0)
        _, ast.alpha = imgui.slider_float("Novelty Exponent", ast.alpha, 0.0, 8.0)
        hints.tip(ALPHA_TOOLTIP)
        _, ast.k = imgui.slider_int("Neighbours (k)", ast.k, 1, 50)
        _, ast.seed_n = imgui.slider_int("Seed Entries", ast.seed_n, 64, 2048)
        _, ast.sigma0 = imgui.slider_float("Bootstrap Sigma", ast.sigma0, 0.05, 1.5)

    def _render_encoder_combo(self, ast):
        """Which encoder this archive's vectors are in - a READOUT.

        Beside the archive it names, because it is part of which archive this
        is. Always disabled: the choice is made once, in the New Archive modal,
        and after that the stored vectors are in that space and no other.
        """
        from ui.encoder_widgets import encoder_readout

        layout.push_settings_width()
        encoder_readout("Encoder", ast.encoder_key)
        imgui.pop_item_width()

    def _render_new_archive_encoder(self, ast):
        """The one place an encoder is CHOSEN.

        Here rather than in the settings panel because it cannot be changed
        afterwards - offering it beside things that can would read as another
        slider.
        """
        from services.vision_models import REGISTRY
        from ui.encoder_widgets import encoder_combo

        imgui.set_next_item_width(280)
        _, ast.new_archive_encoder = encoder_combo("##new_archive_encoder",
                                                   ast.new_archive_encoder)
        model = REGISTRY[ast.new_archive_encoder]
        layout.text_disabled_wrapped(
            "Encoder, fixed for the life of this archive. Hover an option for "
            "what it sees differently.")
        layout.text_disabled_wrapped(model.blurb)

    def _render_config_history(self, ast):
        """When each setting changed, and what an entry was admitted under.

        Read off disk when the section is opened, not per frame: the file grows
        with the run and ImGui redraws this tab every frame.
        """
        opened = bool(imgui.collapsing_header("Config History"))
        if opened and not ast.show_history:
            ast.request_history_reload = True
        ast.show_history = opened
        if not opened:
            return
        if not ast.history_rows:
            imgui.text_disabled("No settings change recorded yet.")
            return
        if imgui.begin_table("cfg_history", 4):
            for name in ("Version", "Gen", "Entries", "Changed"):
                imgui.table_setup_column(name)
            imgui.table_headers_row()
            for row in ast.history_rows:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(str(row.get("v", 0)))
                imgui.table_next_column()
                imgui.text(str(row.get("gen", 0)))
                imgui.table_next_column()
                imgui.text(str(row.get("entries", 0)))
                imgui.table_next_column()
                imgui.text(", ".join(sorted(row.get("changed", {})))
                           or "initial settings")
            imgui.end_table()

    def _render_archive_settings(self, ast):
        """What the archive KEEPS: the admission gates and the retention cap."""
        from services.vision_models import REGISTRY, get

        # The slider ranges are narrow on purpose; see CLAUDE.md.
        _, ast.liveness_min = imgui.slider_float(
            "Liveness Floor", ast.liveness_min, 0.0, 0.1, "%.4f")
        # The track is PER ENCODER: one whose distances spread wider puts its
        # own default at the top of a fixed track. See CLAUDE.md.
        model = REGISTRY.get(ast.encoder_key) or get("clip-b32")
        _, ast.min_separation = imgui.slider_float(
            "Min Separation", ast.min_separation, 0.0,
            model.separation_slider_max, "%.4f")
        # Naming the calibrated value matters: 0.02 and 0.05 look wildly
        # different and mean the same thing under different encoders.
        hints.tip(
            f"Refuses anything this close to a stored entry. "
            f"{model.label} calibrates at "
            f"{model.default_min_separation:.4f}.")
        if ast.min_separation <= 0.0:
            layout.text_colored_wrapped(
                _WARN,
                "off - a converging expedition will store every tile it makes")
        _, ast.capacity = imgui.slider_int("Capacity", ast.capacity, 1000, 100000)
        hints.tip("Over this, the least novel entries are evicted.")
        _, ast.refresh_sweep_gens = imgui.slider_int(
            "Novelty Sweep (gens)", ast.refresh_sweep_gens, 1, 100)
        hints.tip("Generations to re-score every entry.")

    def _render_expedition_settings(self, ast):
        _, ast.expansion_between = imgui.slider_int(
            "Expansion Between", ast.expansion_between, 0, 500)
        hints.tip("Expansion generations between expeditions; "
                  "0 disables expeditions.")
        _, ast.expedition_gens = imgui.slider_int(
            "Expedition Gens", ast.expedition_gens, 5, 400)
        _, ast.expedition_sigma = imgui.slider_float(
            "Expedition Sigma", ast.expedition_sigma, 0.01, 1.0)
        _, ast.novelty_share = imgui.slider_float(
            "Novelty Goal Share", ast.novelty_share, 0.0, 1.0)
        hints.tip("Expeditions that climb novelty with no target.")
        _, ast.latent_share = imgui.slider_float(
            "Latent Goal Share", ast.latent_share, 0.0, 1.0)
        hints.tip("Expeditions toward a point past the archive's "
                  "frontier.")
        left = 1.0 - min(1.0, ast.novelty_share + ast.latent_share)
        layout.text_colored_wrapped(
            _DIM,
            f"goals: {100 * min(1.0, ast.novelty_share):.0f}% novelty, "
            f"{100 * min(1.0, max(0.0, 1.0 - ast.novelty_share), ast.latent_share):.0f}% latent, "
            f"{100 * left:.0f}% text")
        _, ast.seed_ess_min = imgui.slider_float(
            "Seed Pool Min", ast.seed_ess_min, 1.0, 128.0)
        hints.tip("Fewest entries in the running as a starting point.")
        _, ast.seed_ess_max = imgui.slider_float(
            "Seed Pool Max", ast.seed_ess_max, 16.0, 4096.0)
        hints.tip("Most entries in the running as a starting point.")

    # ---- the browser ---------------------------------------------------

    def render_archive_window(self):
        ast = self.state.archive
        # The pointer is over nothing until a widget claims it this frame. Set
        # before every early return below: a closed or empty browser must not
        # leave a preview running with no way to end it.
        ast.preview_entry_id = -1
        if not ast.show_browser:
            return
        # Without an explicit size ImGui auto-fits smaller than its own content
        # and the button row below the gallery draws nothing at all.
        # first_use_ever, so a window the user has resized keeps their size.
        imgui.set_next_window_size(imgui.ImVec2(760, 620),
                                   imgui.Cond_.first_use_ever)
        layout.constrain_panel(layout.MIN_PANEL_WIDTH, 320.0)
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

        # The picker belongs here as much as on the tab: this is a whole
        # window about one archive, and opened from Extras the tab may never
        # be on screen to switch from.
        self._render_archive_row(ast, show_summary=False)
        self._render_banner(ast, "warning", _BAD, scope="browser")
        self._render_banner(ast, "notice", _OK, scope="browser")
        st = arc.stats()
        imgui.text_wrapped(f"{st['size']} / {st['capacity']} entries   "
                           f"{st['n_pinned']} pinned   "
                           f"{st['n_evicted']} evicted")
        self._render_mixed_note(arc, st)
        self._render_live_preview_toggle(ast)
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

    def _render_mixed_note(self, arc, st) -> None:
        """How much of a MIXED archive the running brain can actually use.

        Silent when there is only one layout, which is every archive that has
        never had its brain changed. When there are several, the count that
        matters is not the size: parents and seeds come from the running
        brain's own entries, so a full-looking archive can still be
        bootstrapping and nothing else on screen says why.
        """
        if len(st.get("layouts", ())) < 2:
            return
        native, size = int(st["native"]), int(st["size"])
        # Wrapped: imgui.text_colored does not, and this line is long enough to
        # run off the edge of any panel narrow enough to sit beside the canvas.
        layout.text_colored_wrapped(
            _DIM,
            f"{len(st['layouts'])} brains here; {native} of {size} are "
            f"{arc.signature} - the rest browse and rank but cannot be bred from")
        hints.tip(
            "Novelty, admission and the map pool across every brain.\n"
            "Parents and seeds come from the running brain's entries only,\n"
            "because another brain's genome is a different creature under\n"
            "this one's decode. Switch brain to work on the others.\n\n"
            + "\n".join(f"  {s}" for s in st["layouts"]))

    def _render_live_preview_toggle(self, ast):
        """Run the hovered entry in the live sim, like hovering File > Load.

        Unavailable under a grid: the canvas is many simulations there, so
        there is no single sim for one entry to run in. Clicking an entry of
        another brain still switches to it.
        """
        busy = bool(self.state.tournament.enabled)
        imgui.begin_disabled(busy)
        _, ast.live_preview = imgui.checkbox("Live preview", ast.live_preview)
        imgui.end_disabled()
        if not busy and imgui.is_item_hovered():
            hints.tip(
                "Hover an entry to run it; click to keep it.")
        if busy:
            imgui.same_line()
            imgui.text_disabled("(no single sim under a grid - "
                                "clicking still switches brain)")
        elif ast.live_preview:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*_OK),
                               "hover to try, click to keep")

    def _sorted_entries(self, ast, arc):
        """The gallery's display order, cached against the archive revision.

        The order only changes when the archive does, so it must not be
        recomputed every frame.
        """
        key = (ast.sort_by, ast.sort_desc, ast.pinned_only,
               getattr(arc, "revision", None))
        hit = getattr(self, "_sort_cache", None)
        if hit is not None and hit[0] is arc and hit[1] == key:
            return hit[2]

        out = sort_entries(arc, ast.sort_by, ast.sort_desc, ast.pinned_only)
        self._sort_cache = (arc, key, out)
        return out

    def _render_gallery(self, ast, arc):
        self._render_gallery_toolbar(ast)
        if ast.thumb_size <= GALLERY_LIST_MAX:
            self._render_gallery_list(ast, arc)
        else:
            self._render_gallery_grid(ast, arc)
        self._render_gallery_selection(ast, arc)

    # ---- the toolbar ---------------------------------------------------

    def _render_gallery_toolbar(self, ast):
        # The size row. The slider is ##-labelled, so it owes nothing to
        # WIDEST_LABEL; the icons are what say which end is which.
        if self._size_icon("##gallery_list_icon", "list"):
            ast.thumb_size = GALLERY_SIZE_MIN
        hints.tip("Smallest size: a list with the details.")
        imgui.same_line()
        # Off the frame height, so the track scales with the font rather than
        # with whatever the widest label in the panel happens to be.
        imgui.set_next_item_width(imgui.get_frame_height() * 8.0)
        ch, size = imgui.slider_int("##thumb_size", int(ast.thumb_size),
                                    GALLERY_SIZE_MIN, GALLERY_SIZE_MAX, "%d px")
        if ch:
            ast.thumb_size = int(size)
        imgui.same_line()
        if self._size_icon("##gallery_grid_icon", "grid"):
            ast.thumb_size = GALLERY_SIZE_MAX
        hints.tip("Largest size, which is what a thumbnail holds.")

        # The sort row. The combo takes the width its label leaves, so the
        # direction and the filter start a row of their own.
        modes = list(GALLERY_SORTS)
        idx = (modes.index(ast.sort_by) if ast.sort_by in modes
               else modes.index(DEFAULT_SORT))
        layout.push_settings_width("Sort")
        ch, idx = imgui.combo("Sort", idx,
                              [GALLERY_SORTS[m].label for m in modes])
        imgui.pop_item_width()
        if ch:
            ast.sort_by = modes[idx]

        right = layout.row_right_edge()
        if imgui.arrow_button("##sort_dir",
                              imgui.Dir.down if ast.sort_desc else imgui.Dir.up):
            ast.sort_desc = not ast.sort_desc
        hints.tip("Reverse the order.")
        layout.wrap_row(right, layout.button_width("Pinned only"))
        _, ast.pinned_only = imgui.checkbox("Pinned only", ast.pinned_only)

    @staticmethod
    def _size_icon(str_id: str, kind: str) -> bool:
        """A drawn glyph that snaps the size slider to its end.

        There is no icon font, so both are laid out in the draw list.
        """
        h = imgui.get_frame_height()
        p = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(str_id, imgui.ImVec2(h, h))
        col = (imgui.IM_COL32(225, 225, 232, 255) if imgui.is_item_hovered()
               else imgui.IM_COL32(150, 150, 158, 255))
        dl = imgui.get_window_draw_list()
        pad = round(h * 0.26)
        span = h - 2 * pad
        x0, y0 = p.x + pad, p.y + pad
        if kind == "list":
            bar = max(1.0, round(span * 0.18))
            for k in range(3):
                y = y0 + k * (span - bar) * 0.5
                dl.add_rect_filled(imgui.ImVec2(x0, y),
                                   imgui.ImVec2(x0 + span, y + bar), col)
        else:
            cell = max(1.0, round((span - max(1.0, span * 0.2)) * 0.5))
            gap = span - 2 * cell
            for cx in (0, 1):
                for cy in (0, 1):
                    ax, ay = x0 + cx * (cell + gap), y0 + cy * (cell + gap)
                    dl.add_rect_filled(imgui.ImVec2(ax, ay),
                                       imgui.ImVec2(ax + cell, ay + cell), col)
        return clicked

    # ---- the two views -------------------------------------------------

    @staticmethod
    def _peek_thumb(cache, key):
        """A look at the cache that does not count as a use.

        Tolerant of a cache without `peek`, the same way `_reserve_thumbs` is
        tolerant of one without `reserve`: a test double need not grow a
        method to keep the window drawing.
        """
        fn = getattr(cache, "peek", None)
        return fn(key) if fn is not None else cache.get(key)

    @staticmethod
    def _reserve_thumbs(cache, n: int) -> None:
        """Hold room for a whole frame's thumbnails before asking for any.

        A frame that touches more than the cache holds evicts every texture and
        re-decodes the visible set on the next one.
        """
        fn = getattr(cache, "reserve", None)
        if fn is not None:
            fn(int(n))

    def _render_gallery_grid(self, ast, arc):
        cache = getattr(self, "thumb_cache", None)
        size = float(ast.thumb_size)
        imgui.begin_child("gallery", imgui.ImVec2(0, GALLERY_H))
        # Rows fit the window rather than a fixed six, so a narrow panel wraps
        # instead of clipping the right-hand thumbnails.
        style = imgui.get_style()
        step = size + style.item_spacing.x
        per_row = max(1, int(imgui.get_content_region_avail().x / step))
        items = self._sorted_entries(ast, arc)
        row_h = size + style.item_spacing.y
        self._reserve_thumbs(cache, per_row * (int(GALLERY_H / row_h) + 2))
        # CLIPPED to the rows on screen, rather than capped. This used to draw
        # the first 240 and stop: an archive of 1121 scrolled to a fifth of
        # itself and simply ended, with nothing on screen saying so.
        n_rows = (len(items) + per_row - 1) // per_row
        clipper = imgui.ListClipper()
        clipper.begin(n_rows, row_h)
        while clipper.step():
            for r in range(clipper.display_start, clipper.display_end):
                for c in range(per_row):
                    n = r * per_row + c
                    if n >= len(items):
                        break
                    if c:
                        imgui.same_line()
                    i, e = items[n]
                    tex = (cache.get(arc.thumb_key(i))
                           if cache is not None else None)
                    if tex is not None:
                        imgui.image(imgui.ImTextureRef(tex.glo),
                                    imgui.ImVec2(size, size))
                    else:
                        imgui.button(f"#{e.id}##g{i}", imgui.ImVec2(size, size))
                    if imgui.is_item_hovered():
                        # card, not tip: sweeping the gallery to see what each
                        # entry is IS the feature, so it cannot wait for a delay.
                        hints.card(
                            f"#{e.id}  {e.source}\nnovelty {e.novelty:.3f}\n"
                            f"liveness {e.liveness:.3f}\ngoal: {e.goal or '-'}")
                        ast.preview_entry_id = i
                    if imgui.is_item_clicked():
                        ast.selected_entry_id = i
                        if ast.live_preview:
                            ast.load_entry_id = i
        clipper.end()
        imgui.end_child()

    def _render_gallery_list(self, ast, arc):
        """The small end of the slider: one row per entry, with the details.

        No tooltip - the columns carry what the grid's tooltip says.
        """
        cache = getattr(self, "thumb_cache", None)
        size = float(ast.thumb_size)
        flags = (imgui.TableFlags_.hideable | imgui.TableFlags_.reorderable
                 | imgui.TableFlags_.resizable | imgui.TableFlags_.sortable
                 | imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y
                 | imgui.TableFlags_.borders_inner_v)
        if not imgui.begin_table(GALLERY_TABLE, len(GALLERY_COLUMNS), flags,
                                 imgui.ImVec2(0, GALLERY_H)):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        for label, mode in GALLERY_COLUMNS:
            if mode:
                imgui.table_setup_column(label)
            else:
                # no_resize is what makes the width FOLLOW the slider: a
                # resizable column's width is a table setting restored from
                # imgui.ini, so the width set up here would only ever apply to
                # a table that had none, and the pictures grew inside a column
                # that did not. The handle is no loss - the slider is the
                # control, and one the next frame overrules is not.
                imgui.table_setup_column(
                    label, imgui.TableColumnFlags_.no_sort
                    | imgui.TableColumnFlags_.no_resize
                    | imgui.TableColumnFlags_.width_fixed, size)
        imgui.table_headers_row()
        self._read_sort_specs(ast)

        items = self._sorted_entries(ast, arc)
        row_h = max(size, imgui.get_frame_height())
        self._reserve_thumbs(cache, int(GALLERY_H / row_h) + 4)
        clipper = imgui.ListClipper()
        clipper.begin(len(items), row_h)
        while clipper.step():
            for n in range(clipper.display_start, clipper.display_end):
                i, e = items[n]
                imgui.table_next_row(0, row_h)
                imgui.table_next_column()
                imgui.push_id(i)
                # The whole ROW is the hit target, and the selection is drawn.
                pos = imgui.get_cursor_pos()
                changed, _ = imgui.selectable(
                    "##row", i == ast.selected_entry_id,
                    imgui.SelectableFlags_.span_all_columns
                    | imgui.SelectableFlags_.allow_overlap,
                    imgui.ImVec2(0.0, row_h))
                # Read before the thumbnail is drawn over it.
                if imgui.is_item_hovered():
                    ast.preview_entry_id = i
                if changed:
                    ast.selected_entry_id = i
                    if ast.live_preview:
                        ast.load_entry_id = i
                imgui.set_cursor_pos(pos)
                tex = (cache.get(arc.thumb_key(i))
                       if cache is not None else None)
                if tex is not None:
                    imgui.image(imgui.ImTextureRef(tex.glo),
                                imgui.ImVec2(size, size))
                else:
                    imgui.dummy(imgui.ImVec2(size, size))
                for text in (f"#{e.id}", e.source, arc.layout_at(i),
                             f"{e.novelty:.3f}", e.goal or "-"):
                    imgui.table_next_column()
                    imgui.text(text)
                imgui.pop_id()
        clipper.end()
        imgui.end_table()

    @staticmethod
    def _read_sort_specs(ast) -> None:
        """A header click writes the same sort_by the combo does.

        One field behind both, so switching view never reorders anything.
        """
        specs = imgui.table_get_sort_specs()
        if specs is None or not specs.specs_dirty:
            return
        if specs.specs_count > 0:
            col = specs.get_specs(0)
            n = int(col.column_index)
            if 0 <= n < len(GALLERY_COLUMNS) and GALLERY_COLUMNS[n][1]:
                ast.sort_by = GALLERY_COLUMNS[n][1]
                ast.sort_desc = (col.sort_direction
                                 == imgui.SortDirection.descending)
        specs.specs_dirty = False

    # ---- what a click selected -----------------------------------------

    def _render_gallery_selection(self, ast, arc):
        if not (0 <= ast.selected_entry_id < len(arc.entries)):
            return
        sel = arc.entries[ast.selected_entry_id]
        imgui.separator()
        # The ENTRY's id, not the row: the row is how the app addresses it,
        # the id is what the archive calls it.
        imgui.text(f"Selected #{sel.id}")
        # The brain this creature was AUTHORED under, named on every selection
        # rather than only on a foreign one: an archive pools layouts, so
        # "which brain is this?" is a question about any entry. Only on a click
        # - the hover preview writes a row per pointer position.
        sig = arc.layout_at(ast.selected_entry_id)
        if arc.is_native(ast.selected_entry_id):
            imgui.text_colored(imgui.ImVec4(*_DIM), f"{sig} brain")
        else:
            # Hovering borrows this brain; only a click keeps it, and with the
            # toggle off nothing runs at all.
            imgui.text_colored(
                imgui.ImVec4(*_DIM),
                f"{sig} brain - click to switch to it" if ast.live_preview
                else f"{sig} brain - turn on Live preview to run it")
        right = layout.row_right_edge()
        if imgui.button("Save as config..."):
            self.open_save_popup(save_targets.ARCHIVE_ENTRY,
                                 arg=ast.selected_entry_id)
        hints.tip(EXPORT_TOOLTIP)
        layout.wrap_row(right, layout.button_width("Seed a run from here"))
        if imgui.button("Seed a run from here"):
            ast.seed_entry_id = ast.selected_entry_id
        hints.tip(SEED_TOOLTIP)
        layout.wrap_row(right, layout.button_width("Delete"))
        if imgui.button("Delete"):
            ast.delete_entry_id = ast.selected_entry_id
        hints.tip("Remove this entry and its thumbnail.")

    # Summits are brighter than the expeditions they sit among, and records get
    # their own hue because a record can be set in any regime.
    _MAP_COLORS = {
        "bootstrap": imgui.IM_COL32(120, 120, 130, 200),
        "expansion": imgui.IM_COL32(90, 200, 120, 220),
        "expedition": imgui.IM_COL32(255, 170, 60, 230),
        "summit": imgui.IM_COL32(255, 240, 130, 255),
        "record": imgui.IM_COL32(255, 120, 200, 255),
        "pin": imgui.IM_COL32(90, 170, 255, 255),
    }

    _MAP_PAD = 8.0
    _MAP_H = 320.0
    _MAP_ZOOM_MIN = 1.0
    _MAP_ZOOM_MAX = 200.0
    # Screen pixels of movement that turn a click into a drag. Without it a pan
    # also selects whatever dot the press happened to land on.
    _MAP_CLICK_SLOP = 4.0

    def _render_map(self, ast, arc):
        proj = getattr(self, "map_layout_service", None)
        # proj.fitted, not just proj: an unfitted Projection transforms
        # everything to the origin, which reads as a broken map rather than an
        # unbuilt one.
        if proj is None or not proj.fitted or len(arc) < 3:
            imgui.text_colored(imgui.ImVec4(*_DIM),
                               "Not enough entries to project yet.")
            return

        self._render_map_toolbar(ast)
        self._render_map_controls(ast, arc)
        self._render_atlas_controls(ast)

        pts = self._map_points(arc, proj, ast)
        if pts is None:
            imgui.text_colored(imgui.ImVec4(*_DIM),
                               "No entries match this filter.")
            return

        size = imgui.ImVec2(max(64.0, imgui.get_content_region_avail().x),
                            self._MAP_H)
        # A child so the canvas clips, and no_scrollbar so it draws none. The
        # WHEEL is claimed in _draw_map, not here: no_scroll_with_mouse is
        # ImGui's "give the parent a chance to scroll" flag, so a child cannot
        # absorb the wheel by asking for it.
        imgui.push_style_var(imgui.StyleVar_.window_padding,
                             imgui.ImVec2(0.0, 0.0))
        imgui.begin_child("map_canvas", size, imgui.ChildFlags_.none,
                          imgui.WindowFlags_.no_scrollbar
                          | imgui.WindowFlags_.no_scroll_with_mouse)
        imgui.pop_style_var()
        row, clicked = self._draw_map(ast, arc, proj, pts, size)
        imgui.end_child()

        if row >= 0:
            self._map_hover_card(arc, row)
            ast.preview_entry_id = row
            if clicked:
                ast.selected_entry_id = row
                if ast.live_preview:
                    ast.load_entry_id = row

        self._render_map_legend(ast, len(pts.idx), len(arc))
        self._render_map_selection(ast, arc)

    def _render_map_toolbar(self, ast):
        svc = getattr(self, "map_layout_service", None)
        right = layout.row_right_edge()

        # Which engine lays the map out. UMAP is optional, so it is offered
        # only when it is there - a control that cannot do what it offers is
        # worse than no control.
        engines = [e for e in ENGINES if e == "pca" or self._umap_available()]
        cur = ast.map_layout if ast.map_layout in engines else "pca"
        imgui.set_next_item_width(self._MAP_COMBO_W)
        changed, i = imgui.combo("Layout##map", engines.index(cur),
                                 [ENGINE_LABELS[e] for e in engines])
        if changed:
            ast.map_layout = engines[i]
        hints.tip(
            "PCA is instant and linear; UMAP separates clusters and is"
            "\nfitted in the background."
            if len(engines) > 1 else
            "UMAP needs umap-learn installed.")

        layout.wrap_row(right, layout.button_width("Relayout"))
        busy = bool(svc is not None and svc.fitting)
        imgui.begin_disabled(busy)
        if imgui.button("Relayout"):
            ast.refit_projection_requested = True
        imgui.end_disabled()
        hints.tip("Lay the map out again over the whole archive.")

        if svc is not None:
            status = svc.status()
            layout.wrap_row(right, imgui.calc_text_size(status).x)
            imgui.text_colored(imgui.ImVec4(*(_BAD if svc.error else _DIM)),
                               status)

        hint = f"{ast.map_zoom:.1f}x - scroll to zoom, drag to pan"
        layout.wrap_row(right, imgui.calc_text_size(hint).x)
        imgui.text_colored(imgui.ImVec4(*_DIM), hint)

    @staticmethod
    def _umap_available() -> bool:
        from services.map_layout import UmapLayout
        return UmapLayout.available()

    _HOME_PX = 22.0
    _HOME_PAD = 6.0

    def _render_map_home(self, ast, draw, origin, size) -> bool:
        """A recentre icon in the canvas's own top-right corner.

        Drawn AFTER the canvas hit-target, which is what gives it the click:
        the canvas declares set_next_item_allow_overlap so a later widget over
        it takes priority. -> whether the pointer is on it, so the map can stop
        treating that as a hover on the dots underneath.

        Drawn rather than lettered - the default font carries no icon set, and
        a word at this size clips to nonsense.
        """
        s = self._HOME_PX
        x = origin.x + size.x - s - self._HOME_PAD
        y = origin.y + self._HOME_PAD
        imgui.set_cursor_screen_pos(imgui.ImVec2(x, y))
        if imgui.invisible_button("maphome", imgui.ImVec2(s, s)):
            self._map_home(ast)
        hovered = imgui.is_item_hovered()
        if hovered:
            hints.tip("Reset zoom and recentre.")

        bg = (imgui.IM_COL32(70, 80, 100, 235) if hovered
              else imgui.IM_COL32(45, 50, 62, 170))
        fg = (imgui.IM_COL32(240, 245, 255, 255) if hovered
              else imgui.IM_COL32(175, 185, 205, 230))
        draw.add_rect_filled(imgui.ImVec2(x, y), imgui.ImVec2(x + s, y + s),
                             bg, 4.0)
        # A target: a ring with four ticks pointing in at it.
        cx, cy, r = x + s * 0.5, y + s * 0.5, s * 0.20
        draw.add_circle(imgui.ImVec2(cx, cy), r, fg, 0, 1.5)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            draw.add_line(
                imgui.ImVec2(cx + dx * (r + 2.0), cy + dy * (r + 2.0)),
                imgui.ImVec2(cx + dx * s * 0.36, cy + dy * s * 0.36), fg, 1.5)
        return hovered

    def _draw_map(self, ast, arc, proj, pts, size):
        """Draw the canvas. -> (row under the cursor or -1, was it clicked).

        A ROW, not an entry: one archive holds every brain and an id is only
        unique within one of them, so every one-shot the browser sets has to
        name a position.
        """
        origin = imgui.get_cursor_screen_pos()
        # Lets the Home button, drawn last, sit on top and take the click.
        imgui.set_next_item_allow_overlap()
        imgui.invisible_button("map_hit", size)
        # The canvas OWNS the wheel while it is hovered. ImGui routes the wheel
        # in NewFrame, before any of this runs, so nothing read here can stop
        # it - only claiming the key can.
        imgui.set_item_key_owner(imgui.Key.mouse_wheel_y)
        hovering = imgui.is_item_hovered()
        clicked = self._map_interact(ast, origin, size, hovering)

        draw = imgui.get_window_draw_list()
        far = imgui.ImVec2(origin.x + size.x, origin.y + size.y)
        draw.add_rect_filled(origin, far, imgui.IM_COL32(20, 20, 24, 255))
        draw.push_clip_rect(origin, far, True)

        xs, ys = self._map_to_screen(ast, pts.unit, origin, size)
        # Only what is actually on the canvas: at high zoom almost nothing is,
        # and a draw call per archive entry per frame is the cost otherwise.
        on = ((xs >= origin.x) & (xs <= far.x) & (ys >= origin.y) & (ys <= far.y))
        sel = np.flatnonzero(on)

        if not ast.map_thumbs and ast.map_render in ("density", "points+density"):
            self._draw_density(
                draw, xs[sel], ys[sel], origin, size, pts.colors[sel],
                None if pts.tvals is None else pts.tvals[sel])

        # The atlas REPLACES the scatter. Points drawn as well would compete
        # with the pictures for the same entries on every pan, and a dot under
        # a picture is invisible while still costing a draw call.
        if ast.map_thumbs:
            self._draw_atlas(ast, arc, draw, pts, origin, size)
        elif ast.map_render != "density":
            # .tolist() first: indexing a numpy array with a Python int inside
            # the loop costs more than the draw call it feeds.
            px, py = xs[sel].tolist(), ys[sel].tolist()
            pc = pts.colors[sel].tolist()
            for x, y, c in zip(px, py, pc):
                draw.add_circle_filled(imgui.ImVec2(x, y), 3.0, c)

        best_i, best_d, picked = -1, 1e9, -1
        if hovering and ast.map_thumbs:
            # The PICTURE under the pointer, not the nearest hidden dot: the
            # cell's winner is what is drawn there, and a hover card showing
            # some other entry is a card describing something you cannot see.
            picked = self._atlas_pick(ast, imgui.get_mouse_pos(), origin, size)
        elif hovering and len(sel):
            mouse = imgui.get_mouse_pos()
            d = np.abs(xs[sel] - mouse.x) + np.abs(ys[sel] - mouse.y)
            j = int(np.argmin(d))
            best_i, best_d = int(sel[j]), float(d[j])

        goal_pt = getattr(self, "archive_goal_point", None)
        if goal_pt is not None:
            gu = (proj.transform(np.asarray(goal_pt)[None]) - pts.lo) / pts.span
            gx, gy = self._map_to_screen(ast, gu, origin, size)
            draw.add_circle(imgui.ImVec2(float(gx[0]), float(gy[0])), 7.0,
                            imgui.IM_COL32(255, 90, 90, 255), 0, 2.0)
        draw.pop_clip_rect()

        # Over the Home button is not over the dots beneath it, or the hover
        # card covers the button you are reaching for.
        if self._render_map_home(ast, draw, origin, size):
            hovering = False

        # The atlas resolves its own entry: the plan on screen may belong to
        # an EARLIER points object, so this one's idx would name someone else.
        if ast.map_thumbs:
            return (picked, clicked) if hovering and picked >= 0 else (-1, False)
        # Through pts.idx: with a filter on, row i is not entry i, and hovering
        # the wrong entry is worse than not hovering at all.
        if hovering and best_i >= 0 and best_d < 12.0:
            return int(pts.idx[best_i]), clicked
        return -1, False

    def _atlas_plan(self, ast, pts, size):
        """Which entry stands for which cell, in UNIT space. Cached.

        Binned in unit space at a zoom-QUANTISED cell size, so panning cannot
        change it and zooming only changes it when a level boundary is crossed.
        Binning in screen space instead reshuffles every picture on a one-pixel
        pan, which is what made the atlas flicker.
        """
        cell_u = map_view.atlas_cell(size.x * ast.map_zoom,
                                     size.y * ast.map_zoom,
                                     float(ast.map_thumb_px))
        key = (cell_u,)
        hit = getattr(self, "_atlas_plan_memo", None)
        # `pts` by identity, not by id(): CPython hands a freed object's
        # address to the next one of its type, so an id alone can collide.
        if hit is not None and hit[0] is pts and hit[1] == key:
            return hit[2]
        # UNCAPPED: the plan is a few arrays of ints and costs nothing to
        # hold, and capping it here is what left gaps where the map was
        # zoomed in. The cap that matters is on the cells actually DRAWN.
        plan = map_view.atlas_winners(pts.unit, pts.novelty, cell_u)
        self._atlas_plan_memo = (pts, key, (plan, cell_u))
        return plan, cell_u

    def _atlas_rects(self, ast, shown, origin, size):
        """-> screen rects for a plan's cells, and which of them are visible.

        The visible set is CAPPED, keeping the most novel: it is the working
        set the cache has to hold, and one larger than the cache evicts its
        own cells and re-decodes them for as long as the map is open. Capping
        the whole PLAN instead leaves gaps exactly where you zoom in, since
        the cells in view need not be among the map's most novel.
        """
        _arc, pts, (ux, uy, win), cell = shown
        corner = np.stack([ux, uy], axis=1)
        sx, sy = self._map_to_screen(ast, corner, origin, size)
        # y is flipped by _map_to_screen, so the cell's top edge comes from
        # its FAR corner in unit space.
        far = np.stack([ux + cell[0], uy + cell[1]], axis=1)
        fx, fy = self._map_to_screen(ast, far, origin, size)
        on = ((fx >= origin.x) & (sx <= origin.x + size.x)
              & (sy >= origin.y) & (fy <= origin.y + size.y))
        live = np.flatnonzero(on)
        if len(live) > ATLAS_BUDGET:
            keep = np.argpartition(pts.novelty[win[live]],
                                   -ATLAS_BUDGET)[-ATLAS_BUDGET:]
            live = live[np.sort(keep)]
        return sx, fy, fx, sy, live

    def _atlas_pick(self, ast, mouse, origin, size) -> int:
        """-> the archive row whose picture is under `mouse`, or -1.

        Read off the plan being DRAWN, never the one being decoded, so what
        the card shows is what is on screen.
        """
        shown = getattr(self, "_atlas_shown", None)
        if shown is None:
            return -1
        _arc, pts, (_ux, _uy, win), _cell = shown
        if not len(win):
            return -1
        sx, fy, fx, sy, _live = self._atlas_rects(ast, shown, origin, size)
        hit = ((mouse.x >= sx) & (mouse.x < fx)
               & (mouse.y >= fy) & (mouse.y < sy))
        j = np.flatnonzero(hit)
        return int(pts.idx[win[j[0]]]) if len(j) else -1

    def _atlas_ready(self, arc, pts, win, rows, cache):
        """Decode what the cells in `rows` still need, up to this frame's
        budget. -> is every one of their pictures resident now?

        A plan is only put ON SCREEN once it is, which is what stops a
        wholesale change - a zoom level, or switching PCA to UMAP - wiping
        across the map one row of thumbnails at a time. Only the VISIBLE
        cells are waited for; a level holds thousands, and none of the ones
        off screen are what the eye is on.
        """
        budget = ATLAS_NEW_PER_FRAME
        missing = 0
        for row in win[rows].tolist():
            key = arc.thumb_key(int(pts.idx[row]))
            if self._peek_thumb(cache, key) is not None:
                continue
            if budget <= 0:
                missing += 1
                continue
            budget -= 1
            if cache.get(key) is None:
                # An unreadable thumbnail never arrives; counting it as
                # missing would hold the plan back for good.
                continue
        return missing == 0

    def _draw_atlas(self, ast, arc, draw, pts, origin, size):
        """One thumbnail per occupied cell, and nothing else.

        Double-buffered: the plan being decoded is not the plan being drawn.
        The previous one keeps its cells until the new one is complete, so a
        level change rescales the pictures already on screen instead of
        sweeping new ones across it. Nothing is ever drawn half-filled.
        """
        cache = getattr(self, "atlas_cache", None) or getattr(
            self, "thumb_cache", None)
        if cache is None or not len(pts.idx):
            return
        plan, cell_u = self._atlas_plan(ast, pts, size)
        shown = getattr(self, "_atlas_shown", None)
        # The snapshot names rows of the archive it was built from. Switching
        # archive replaces the entry list - a shorter one indexes past the end
        # mid-frame - so the previous atlas is dropped rather than held.
        if shown is not None and shown[0] is not arc:
            self._atlas_shown = shown = None
        if len(plan[2]):
            cand = (arc, pts, plan, cell_u)
            _sx, _fy, _fx, _sy, live = self._atlas_rects(ast, cand, origin,
                                                         size)
            # A constant reserve, never the visible count: sizing the cache to
            # the viewport shrinks it as you zoom in and evicts the level you
            # came from, so zooming back out decodes it all again.
            self._reserve_thumbs(cache, ATLAS_BUDGET + ATLAS_HEADROOM)
            ready = self._atlas_ready(arc, pts, plan[2], live, cache)
            # Nothing to hold on to yet - a first open - so fill in view
            # rather than showing an empty canvas for a second.
            if ready or shown is None:
                self._atlas_shown = shown = cand
        if shown is None:
            return
        # Everything from the SNAPSHOT, positions and rows alike: switching
        # PCA to UMAP replaces `pts`, and the plan on screen still belongs to
        # the old one until the new one is complete.
        _arc, spts, (_ux, _uy, win), _cell = shown
        if not len(win):
            return
        sx, fy, fx, sy, live = self._atlas_rects(ast, shown, origin, size)
        rows, x0, y0, x1, y1 = (win[live], sx[live], fy[live],
                                fx[live], sy[live])
        for row, ax, ay, bx, by in zip(rows.tolist(), x0.tolist(), y0.tolist(),
                                       x1.tolist(), y1.tolist()):
            tex = self._peek_thumb(cache, arc.thumb_key(int(spts.idx[row])))
            if tex is not None:
                draw.add_image(imgui.ImTextureRef(tex.glo),
                               imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by))

    def _map_points(self, arc, proj, ast):
        """-> a map_view.MapPoints, or None if the filter matched nothing.

        Cached against (archive revision, projection version, view options):
        none of them change between generations, and the map redraws every
        frame.

        The unit square is normalised over the FILTERED entries, so narrowing
        the filter also expands what is left to fill the canvas.
        """
        key = (getattr(arc, "revision", None), getattr(proj, "version", None),
               len(arc), ast.map_color_by, ast.map_filter, ast.map_recent_gens,
               ast.map_novel_pct, ast.map_filter_goal, ast.map_filter_source)
        hit = getattr(self, "_map_cache", None)
        if hit is not None and hit[0] is arc and hit[1] is proj and hit[2] == key:
            return hit[3]

        idx = map_view.filter_indices(
            arc.entries, ast.map_filter,
            recent_gens=ast.map_recent_gens, novel_pct=ast.map_novel_pct,
            goal=ast.map_filter_goal, source=ast.map_filter_source)
        if not len(idx):
            self._map_cache = (arc, proj, key, None)
            return None

        pts = proj.transform_rows(arc, idx)
        if not len(pts):
            return None
        lo = pts.min(axis=0)
        span = np.maximum(pts.max(axis=0) - lo, 1e-6)
        unit = (pts - lo) / span

        shown = [arc.entries[i] for i in idx]
        values = map_view.scalar_values(shown, ast.map_color_by)
        if values is None:
            fallback = self._MAP_COLORS["expansion"]
            tvals = None
            colors = np.array(
                [self._MAP_COLORS.get("pin" if e.pinned else e.source, fallback)
                 for e in shown], dtype=np.int64)
        else:
            tvals = map_view.normalise(values)
            colors = map_view.ramp_colors(tvals)

        novelty = np.array([e.novelty for e in shown], dtype=np.float32)
        out = map_view.MapPoints(unit, lo, span, colors, idx, tvals,
                                 novelty)
        self._map_cache = (arc, proj, key, out)
        return out

    _MAP_COMBO_W = 130.0

    def _render_map_controls(self, ast, arc) -> None:
        """Colour / Show / Draw. Every default is the historical map, so the
        plain scatter coloured by regime is always one combo away."""
        w = self._MAP_COMBO_W
        right = layout.row_right_edge()

        # Pictures carry their own colour and their own marks, so neither
        # combo has anything to say while the atlas is on.
        pictures = bool(ast.map_thumbs)
        imgui.begin_disabled(pictures)
        imgui.set_next_item_width(w)
        changed, i = imgui.combo("Colour##map",
                                 map_view.COLOR_MODES.index(ast.map_color_by)
                                 if ast.map_color_by in map_view.COLOR_MODES else 0,
                                 list(map_view.COLOR_MODES))
        if changed:
            ast.map_color_by = map_view.COLOR_MODES[i]
        imgui.end_disabled()
        hints.tip("Thumbnails carry their own colour." if pictures
                  else "What the colours mean.")

        layout.wrap_row(right, layout.labelled_width(w, "Show"))
        imgui.set_next_item_width(w)
        labels = [map_view.FILTER_LABELS[m] for m in map_view.FILTER_MODES]
        changed, i = imgui.combo("Show##map",
                                 map_view.FILTER_MODES.index(ast.map_filter)
                                 if ast.map_filter in map_view.FILTER_MODES else 0,
                                 labels)
        if changed:
            ast.map_filter = map_view.FILTER_MODES[i]
        hints.tip("Which entries to draw.")

        layout.wrap_row(right, layout.labelled_width(w, "Draw"))
        imgui.begin_disabled(pictures)
        imgui.set_next_item_width(w)
        changed, i = imgui.combo("Draw##map",
                                 map_view.RENDER_MODES.index(ast.map_render)
                                 if ast.map_render in map_view.RENDER_MODES else 0,
                                 list(map_view.RENDER_MODES))
        if changed:
            ast.map_render = map_view.RENDER_MODES[i]
        imgui.end_disabled()
        hints.tip("Dots, or a heatmap of the same colours.")

        self._render_map_filter_arg(ast, arc)

    def _render_atlas_controls(self, ast) -> None:
        """The atlas toggle and its cell size."""
        right = layout.row_right_edge()
        _, ast.map_thumbs = imgui.checkbox("Thumbnails", ast.map_thumbs)
        hints.tip("A picture per cell instead of dots. Zoom in to split them.")
        if not ast.map_thumbs:
            return
        layout.wrap_row(right, layout.labelled_width(self._MAP_COMBO_W, "Size"))
        imgui.set_next_item_width(self._MAP_COMBO_W)
        ch, px = imgui.slider_int("Size##atlas", int(ast.map_thumb_px),
                                  ATLAS_PX_MIN, ATLAS_PX_MAX, "%d px")
        if ch:
            ast.map_thumb_px = int(px)

    def _render_map_filter_arg(self, ast, arc) -> None:
        """The one extra control the chosen filter needs, and nothing else."""
        if ast.map_filter == "recent":
            layout.push_settings_width()
            _, ast.map_recent_gens = imgui.slider_int(
                "Generations##map", ast.map_recent_gens, 1, 2000)
            imgui.pop_item_width()
        elif ast.map_filter == "novel":
            layout.push_settings_width()
            _, ast.map_novel_pct = imgui.slider_int(
                "Top %##map", ast.map_novel_pct, 1, 100)
            imgui.pop_item_width()
        elif ast.map_filter in ("goal", "source"):
            field = "goal" if ast.map_filter == "goal" else "source"
            attr = "map_filter_goal" if field == "goal" else "map_filter_source"
            values = map_view.present_values(arc.entries, field)
            if not values:
                imgui.text_colored(imgui.ImVec4(*_DIM),
                                   f"No entry carries a {field} yet.")
                return
            current = getattr(ast, attr)
            if current not in values:
                setattr(ast, attr, values[0])
                current = values[0]
            layout.push_settings_width()
            changed, i = imgui.combo(f"{field.title()}##mapfilter",
                                     values.index(current), values)
            imgui.pop_item_width()
            if changed:
                setattr(ast, attr, values[i])

    def _draw_density(self, draw, xs, ys, origin, size, colors, tvals) -> None:
        """A heatmap carrying the SAME colour the dots would have, with the
        entry count in the opacity - so the Colour combo means one thing in
        both draw modes.

        Binned in screen space so resolution follows the zoom, and drawn under
        the points: overlapping dots hide their own density, which is why a
        full archive reads as one blob.
        """
        flat, on, nx, ny, cell = map_view.bin_points(
            xs, ys, (origin.x, origin.y), (size.x, size.y))
        if not len(flat):
            return

        ncells = nx * ny
        if tvals is None:
            # A category cannot be averaged: the cell takes its winner's colour.
            cell_col, counts = map_view.cell_majority(flat, colors[on], ncells)
            filled = np.flatnonzero(counts)
            cols = map_view.with_alpha(
                cell_col[filled], map_view.density_alpha(counts[filled]))
        else:
            means, counts = map_view.cell_means(flat, tvals[on], ncells)
            filled = np.flatnonzero(counts)
            cols = map_view.ramp_colors(
                means[filled], alpha=map_view.density_alpha(counts[filled]))

        x0 = origin.x + (filled % nx) * cell
        y0 = origin.y + (filled // nx) * cell
        for x, y, c in zip(x0.tolist(), y0.tolist(), cols.tolist()):
            draw.add_rect_filled(imgui.ImVec2(x, y),
                                 imgui.ImVec2(x + cell, y + cell), c)

    def _render_map_legend(self, ast, shown: int, total: int) -> None:
        """How much of the archive is on screen, and what the colours mean."""
        imgui.text_colored(imgui.ImVec4(*_DIM), f"showing {shown} of {total}")
        if ast.map_render != "points":
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*_DIM), "   opacity: entries per cell")
        if ast.map_color_by not in ("novelty", "liveness"):
            return

        imgui.text_colored(imgui.ImVec4(*_DIM), f"{ast.map_color_by}  low")
        for k, t in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            imgui.same_line(0.0, 2.0)
            c = int(map_view.ramp_colors(np.array([t], dtype=np.float32))[0])
            # A real swatch widget: imgui.text renders "##" literally, and a
            # coloured glyph depends on the font having a block character.
            imgui.color_button(f"##ramp{k}",
                               imgui.ImVec4((c & 255) / 255.0,
                                            ((c >> 8) & 255) / 255.0,
                                            ((c >> 16) & 255) / 255.0, 1.0),
                               0, imgui.ImVec2(14, 14))
        imgui.same_line(0.0, 4.0)
        imgui.text_colored(imgui.ImVec4(*_DIM), "high")

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
            # On the frame the canvas BECOMES active, mouse_delta is the
            # movement that ARRIVED at the dot, not a drag of it. Panning by
            # that slides every dot out from under the cursor before the hit
            # test below runs, so the click lands on whatever slid into its
            # place - and a second, stationary click on the same spot works.
            # It also counts toward the drag slop, which can suppress the click
            # outright instead.
            d = imgui.ImVec2(0.0, 0.0) if imgui.is_item_activated() else io.mouse_delta
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

    def _map_hover_card(self, arc, row) -> None:
        """The picture, on hover. A dot's position is not what it IS."""
        entry = arc.entries[row]
        cache = getattr(self, "thumb_cache", None)
        tex = cache.get(arc.thumb_key(row)) if cache is not None else None
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
        """The picked dot's actual image - a position is not what it IS."""
        row = ast.selected_entry_id
        entry = arc.entries[row] if 0 <= row < len(arc.entries) else None
        if entry is None:
            imgui.text_colored(imgui.ImVec4(*_DIM),
                               "Click a point to see what it is.")
            return

        imgui.separator()
        cache = getattr(self, "thumb_cache", None)
        tex = cache.get(arc.thumb_key(row)) if cache is not None else None
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
            self.open_save_popup(save_targets.ARCHIVE_ENTRY, arg=row)
        imgui.same_line()
        if imgui.button("Delete##map"):
            ast.delete_entry_id = row
        imgui.end_group()
