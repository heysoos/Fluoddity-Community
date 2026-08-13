"""Auto (Prompt) tab of the tournament window.

Passive: renders widgets and sets state, runs no logic.
"""
from __future__ import annotations

from imgui_bundle import imgui

from services import save_targets
from services.capture_health import sweeping_parameters
from services.cohort_tiling import cohorts_for, max_variants
from ui import layout
from ui.notices import OK, render_banner

ALGORITHM_NAMES = ["CMA-ES", "Sep-CMA-ES", "GA", "Random Search"]

COHORT_TOOLTIP = "Gives each tile several variants of its genome."

PHYSICS_TOOLTIP = "Searches the physics sliders as well as the brain."

_WARN = (1.0, 0.6, 0.2, 1.0)
_BAD = (1.0, 0.4, 0.3, 1.0)
_OK = (0.4, 0.9, 0.5, 1.0)

_FIT_COLOR = (0.35, 0.85, 0.45, 1.0)      # green, climbing
_SIGMA_COLOR = (1.0, 0.65, 0.25, 1.0)     # orange, decaying


def normalize_series(values) -> list[float]:
    """Map a series onto 0..1 against its OWN min/max, so series on different
    scales can share a plot. A flat series sits in the middle."""
    vals = [float(v) for v in values]
    finite = [v for v in vals if v == v and abs(v) != float("inf")]
    if not vals:
        return []
    if not finite:
        return [0.5] * len(vals)
    lo, hi = min(finite), max(finite)
    if hi - lo <= 0.0:
        return [0.5] * len(vals)
    span = hi - lo
    return [
        (min(max(v, lo), hi) - lo) / span if v == v else 0.5
        for v in vals
    ]


class AutoTournamentWindowMixin:
    auto_service = None
    auto_unavailable = ""
    _auto_load_path = ""

    def render_auto_tournament_tab(self):
        ats = self.state.auto_tournament
        ats.enabled = True

        self._render_auto_banners(ats)

        if self.auto_unavailable:
            self._render_auto_unavailable(ats)
            return

        svc = self.auto_service
        gen = svc.generation if svc is not None else 0
        sigma = svc.sigma if svc is not None else ats.sigma0

        imgui.text(f"Generation {gen}")
        imgui.separator()

        # One item width for the whole tab: every label then stays on screen
        # however narrow the window is.
        layout.push_settings_width()
        self._render_goal(ats, svc)

        idx = (ALGORITHM_NAMES.index(ats.algorithm)
               if ats.algorithm in ALGORITHM_NAMES else 0)
        ch, idx = imgui.combo("Algorithm", idx, ALGORITHM_NAMES)
        if ch:
            ats.algorithm = ALGORITHM_NAMES[idx]

        # Beside the goal it scores. Free to move, unlike Explore's: a prompt
        # run stores nothing, so there are no vectors a change invalidates.
        self._render_encoder_picker(ats)

        self._render_rollout_controls(ats)

        imgui.separator()
        self._render_transport(ats)

        imgui.separator()
        _, ats.sigma0 = imgui.slider_float("Initial Sigma", ats.sigma0, 0.05, 1.5)

        imgui.separator()
        self._render_physics_search(ats, svc)

        imgui.separator()
        self._render_tile_mutation(ats, sigma)

        imgui.separator()
        self._render_metrics(svc)

        imgui.separator()
        self._render_save_load(ats, svc)
        imgui.pop_item_width()

    # -- pieces ---------------------------------------------------------

    def _render_auto_banners(self, ats):
        render_banner(ats, "warning", _BAD, scope="auto")
        render_banner(ats, "notice", OK, scope="auto")

        # A right-click on a tile asks for a name rather than saving silently.
        # The orchestrator sets this because only it knows whether Explore mode
        # has claimed the right-click for 'chase this tile' instead.
        if ats.pending_save_tile >= 0:
            svc = self.auto_service
            self.open_save_popup(
                save_targets.AUTO_TILE, arg=ats.pending_save_tile,
                generation=int(getattr(svc, "generation", 0) or 0))
            ats.pending_save_tile = -1

        sweeps = sweeping_parameters(self.state.sim)
        if sweeps:
            layout.text_colored_wrapped(
                _WARN, "Tiles are not comparable: " + ", ".join(sweeps))
            layout.text_disabled_wrapped(
                "These sweep across tiles, so fitness is confounded by "
                "position until they are cleared.")

    def _render_encoder_picker(self, ats):
        """Which encoder scores this prompt. Each option explains itself on
        hover, which is why this is not a plain combo."""
        from ui.encoder_widgets import encoder_combo

        _, ats.model_key = encoder_combo("Encoder", ats.model_key)

    def _render_auto_unavailable(self, ats):
        if self.auto_unavailable == "model_missing":
            self._render_encoder_picker(ats)
            imgui.text_wrapped("Encoder weights are not downloaded.")
            if imgui.button(f"Download {ats.model_key}"):
                ats.download_model_requested = True
        else:
            imgui.text_wrapped(f"Auto mode unavailable: {self.auto_unavailable}")
            imgui.text_disabled("pip install onnxruntime-directml tokenizers cmaes")

    def _render_goal(self, ats, svc):
        """Typed text is not the goal until submitted, so the box is tinted
        while the two differ."""
        active = (svc.prompt if svc is not None else "").strip()
        pending = ats.prompt.strip() != active

        if pending:
            imgui.push_style_color(imgui.Col_.frame_bg,
                                   imgui.ImVec4(0.42, 0.28, 0.05, 1.0))
            imgui.push_style_color(imgui.Col_.frame_bg_hovered,
                                   imgui.ImVec4(0.52, 0.35, 0.07, 1.0))
            imgui.push_style_color(imgui.Col_.frame_bg_active,
                                   imgui.ImVec4(0.58, 0.40, 0.09, 1.0))
        # Room for the label AND the Set button that follows it.
        style = imgui.get_style()
        imgui.set_next_item_width(
            -(imgui.calc_text_size("Goal").x + style.item_inner_spacing.x
              + style.item_spacing.x + layout.button_width("Set")))
        changed, ats.prompt = imgui.input_text(
            "Goal", ats.prompt, imgui.InputTextFlags_.enter_returns_true
        )
        if pending:
            imgui.pop_style_color(3)

        if changed:
            ats.prompt_changed = True
        imgui.same_line()
        imgui.begin_disabled(not pending)
        if imgui.button("Set"):
            ats.prompt_changed = True
        imgui.end_disabled()

        if pending and ats.prompt.strip():
            layout.text_colored_wrapped(_WARN,
                                        "not set - press Enter or click Set")
        elif active:
            layout.text_colored_wrapped(_OK, f'steering toward: "{active}"')
        else:
            imgui.text_disabled("no goal set")

    def _render_rollout_controls(self, ats, grid_note=None):
        """Grid and rollout timing, shared by Auto and Explore so the two tabs
        cannot disagree about what a generation is.

        grid_note overrides the last hint line: a grid change costs Auto its
        covariance but only ends Explore's expedition."""
        ch, g = imgui.slider_int("Grid", ats.grid, 2, 8)
        if ch and g != ats.grid:
            ats.grid = g
            ats.grid_changed = True
        self._render_grid_hints(ats, grid_note)

        _, ats.steps_per_gen = imgui.slider_int(
            "Steps per Gen", ats.steps_per_gen, 50, 2000)
        _, ats.sim_steps_per_frame = imgui.slider_int(
            "Sim Steps per Frame", ats.sim_steps_per_frame, 1, 50)
        _, ats.snapshots_per_gen = imgui.slider_int(
            "Snapshots per Gen", ats.snapshots_per_gen, 1, 8)

    def _render_grid_hints(self, ats, note=None):
        tiles = ats.grid * ats.grid
        src_px = 1024 // ats.grid
        layout.text_disabled_wrapped(
            f"population {tiles}   source {src_px}px/tile")
        if src_px < 224:
            layout.text_disabled_wrapped(
                "  upscaled to 224 for the encoder - consider a larger canvas")
        if ats.grid == 2:
            layout.text_disabled_wrapped("  popsize 4 is small for CMA-ES")
        layout.text_disabled_wrapped(
            note or "changing the grid resets the optimizer")

    def _render_transport(self, ats):
        right = layout.row_right_edge()
        if ats.running:
            if imgui.button("Pause"):
                ats.pause_requested = True
        else:
            imgui.begin_disabled(not ats.prompt.strip())
            if imgui.button("Start"):
                ats.start_requested = True
            imgui.end_disabled()
        layout.wrap_row(right, layout.button_width("Reset"))
        if imgui.button("Reset"):
            ats.reset_requested = True

    def _render_physics_search(self, ats, svc):
        from services.physics_genome import PHYSICS_PARAMS

        was = ats.physics_enabled
        _, ats.physics_enabled = imgui.checkbox(
            "Search Physics Too", ats.physics_enabled)
        if imgui.is_item_hovered():
            imgui.set_tooltip(PHYSICS_TOOLTIP)
        if ats.physics_enabled != was:
            ats.reset_requested = True     # the search space changed dimension

        if not ats.physics_enabled:
            layout.text_disabled_wrapped(
                "brain only - the loaded preset fixes the overall look")
            return

        layout.text_disabled_wrapped(
            "searching " + ", ".join(n.replace('_', ' ').title()
                                     for n, _g, _lo, _hi in PHYSICS_PARAMS))
        layout.text_colored_wrapped(
            _WARN,
            "the preset's physics sliders no longer apply while this is on")

    def _render_tile_mutation(self, ats, sigma):
        _, ats.tile_mutation_enabled = imgui.checkbox(
            "Per-tile Mutation", ats.tile_mutation_enabled)
        if not ats.tile_mutation_enabled:
            return

        kmax = max_variants(ats.grid)
        ats.variants_per_tile = max(1, min(ats.variants_per_tile, kmax))
        _, ats.variants_per_tile = imgui.slider_int(
            "Variants per Tile", ats.variants_per_tile, 1, kmax)
        if imgui.is_item_hovered():
            imgui.set_tooltip(COHORT_TOOLTIP)
        layout.text_disabled_wrapped(
            f"cohorts driven to {cohorts_for(ats.grid, ats.variants_per_tile)} "
            f"(max {kmax} variants at this grid)")

        _, ats.tile_mutation_strength = imgui.slider_float(
            "Mutation Strength", ats.tile_mutation_strength, 0.0, 0.5)
        # Once mutation approaches sigma, the spread between tiles no longer
        # exceeds the wobble inside each one and the search stalls silently.
        if ats.tile_mutation_strength < sigma / 3.0:
            imgui.text_disabled(f"sigma {sigma:.3f} - ok")
        else:
            layout.text_colored_wrapped(
                _WARN,
                f"sigma {sigma:.3f} - mutation too high, search may stall")

    def _render_metrics(self, svc):
        if svc is None or svc.logger is None:
            imgui.text_disabled("no metrics yet")
            return
        h = svc.logger.history()
        best = h.get("fit_best") or []
        mean = h.get("fit_mean") or []
        if not best:
            imgui.text_disabled("no generations completed yet")
            return
        imgui.text(f"best {max(best):.3f}   last {best[-1]:.3f}")
        self._render_trace(best, h.get("sigma") or [])
        if imgui.begin_table("autolog", 3, imgui.TableFlags_.borders):
            imgui.table_setup_column("gen")
            imgui.table_setup_column("best")
            imgui.table_setup_column("mean")
            imgui.table_headers_row()
            for i in range(max(0, len(best) - 50), len(best)):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(str(i + 1))
                imgui.table_next_column()
                imgui.text(f"{best[i]:.3f}")
                imgui.table_next_column()
                imgui.text(f"{mean[i]:.3f}" if i < len(mean) else "-")
            imgui.end_table()

    _TRACE_H = 92.0
    _TRACE_PAD = 6.0
    _TRACE_MAX_PTS = 400      # one run can reach thousands of generations

    def _render_trace(self, best, sigma):
        """Fitness and sigma overlaid, each on its own axis."""
        self._render_series([(sigma, _SIGMA_COLOR, "sigma"),   # behind
                             (best, _FIT_COLOR, "fitness")])   # in front

    def _render_series(self, series, height=None):
        """Overlay several series, each normalised against its OWN range.

        `series` is [(values, rgba, label), ...], drawn back to front, with the
        true range of each printed under the plot next to its colour.
        """
        h = float(height if height is not None else self._TRACE_H)
        w = max(120.0, imgui.get_content_region_avail().x)
        p0 = imgui.get_cursor_screen_pos()
        imgui.dummy(imgui.ImVec2(w, h))
        dl = imgui.get_window_draw_list()

        x0, y0 = p0.x, p0.y
        x1, y1 = x0 + w, y0 + h
        dl.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1),
                           imgui.get_color_u32(imgui.ImVec4(0.09, 0.09, 0.11, 1.0)))
        dl.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1),
                    imgui.get_color_u32(imgui.ImVec4(0.30, 0.30, 0.34, 1.0)))

        pad = self._TRACE_PAD
        iw, ih = w - 2 * pad, h - 2 * pad

        def draw(values, color):
            vals = [float(v) for v in values]
            if not vals:
                return
            # Subsample the RAW values, then normalise, so a NaN stays lined up
            # with the point it belongs to.
            if len(vals) > self._TRACE_MAX_PTS:
                step = len(vals) / self._TRACE_MAX_PTS
                vals = [vals[min(int(i * step), len(vals) - 1)]
                        for i in range(self._TRACE_MAX_PTS)]
            norm = normalize_series(vals)
            col = imgui.get_color_u32(imgui.ImVec4(*color))
            n = len(norm)
            if n == 1:
                if vals[0] == vals[0]:
                    dl.add_circle_filled(
                        imgui.ImVec2(x0 + pad + iw * 0.5, y0 + pad + ih * 0.5),
                        2.5, col)
                return
            prev = None
            for i, (raw, v) in enumerate(zip(vals, norm)):
                # NaN marks a generation this series does not cover - fitness
                # outside an expedition. Break the line rather than
                # interpolating across it; the gap is the information.
                if raw != raw:
                    prev = None
                    continue
                p = imgui.ImVec2(x0 + pad + iw * i / (n - 1),
                                 y0 + pad + ih * (1.0 - v))
                if prev is not None:
                    dl.add_line(prev, p, col, 1.6)
                else:
                    dl.add_circle_filled(p, 1.6, col)
                prev = p

        for values, color, _label in series:
            draw(values, color)

        first = True
        for values, color, label in series:
            finite = [v for v in values if v == v]
            if not finite:
                continue
            if not first:
                imgui.same_line()
            first = False
            imgui.text_colored(imgui.ImVec4(*color),
                               f"{label} {min(finite):.4g} - {max(finite):.4g}  ")

    def _render_save_load(self, ats, svc):
        gen = int(getattr(svc, "generation", 0) or 0)
        right = layout.row_right_edge()
        if imgui.button("Save best genome..."):
            self.open_save_popup(save_targets.AUTO_BEST, generation=gen)
        layout.wrap_row(right, layout.button_width("Save checkpoint"))
        if imgui.button("Save checkpoint"):
            ats.save_checkpoint_requested = True
        imgui.set_item_tooltip("Resumes the optimizer; it is not a config.")

        _, self._auto_load_path = imgui.input_text(
            "Load path", getattr(self, "_auto_load_path", ""))
        right = layout.row_right_edge()
        if imgui.button("Load genome"):
            ats.load_genome_path = self._auto_load_path
        imgui.set_item_tooltip("Takes the starting point only.")
        layout.wrap_row(right, layout.button_width("Load checkpoint"))
        if imgui.button("Load checkpoint"):
            ats.load_checkpoint_path = self._auto_load_path
        imgui.set_item_tooltip("Restores the whole search, and the saved grid.")

        if svc is not None and svc.logger is not None and svc.logger.enabled:
            right = layout.row_right_edge()
            if imgui.button("Open run folder"):
                import os
                import subprocess

                subprocess.Popen(["explorer", os.path.abspath(svc.logger.dir)])
            layout.wrap_row(right, imgui.calc_text_size(
                str(svc.logger.run_id)).x)
            imgui.text_disabled(str(svc.logger.run_id))

        _, ats.autosave_every = imgui.slider_int(
            "Autosave every N gens", ats.autosave_every, 0, 100)
