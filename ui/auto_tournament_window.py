"""Auto (CLIP) tab of the tournament window.

Passive: renders widgets and sets state, runs no logic.
"""
from __future__ import annotations

from imgui_bundle import imgui

from services.capture_health import sweeping_parameters
from services.cohort_tiling import cohorts_for, max_variants

ALGORITHM_NAMES = ["CMA-ES", "Sep-CMA-ES", "GA", "Random Search"]

# One sentence each. The reasoning lives in services/cohort_tiling.py and
# services/physics_genome.py, where it can be as long as it needs to be.
COHORT_TOOLTIP = (
    "Gives each tile several variants of its genome; the cohort count is set "
    "for you."
)

PHYSICS_TOOLTIP = (
    "Searches the physics sliders as well as the brain; toggling it resets the "
    "search."
)

_WARN = (1.0, 0.6, 0.2, 1.0)
_BAD = (1.0, 0.4, 0.3, 1.0)
_OK = (0.4, 0.9, 0.5, 1.0)

_FIT_COLOR = (0.35, 0.85, 0.45, 1.0)      # green, climbing
_SIGMA_COLOR = (1.0, 0.65, 0.25, 1.0)     # orange, decaying


def normalize_series(values) -> list[float]:
    """Map a series onto 0..1 against its own min/max.

    Fitness and sigma live on different scales, so a shared axis would flatten
    one of them. Each gets its own normalisation and its real range is printed
    in the legend. A flat series sits in the middle rather than dividing by zero.
    """
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

        self._render_goal(ats, svc)

        idx = (ALGORITHM_NAMES.index(ats.algorithm)
               if ats.algorithm in ALGORITHM_NAMES else 0)
        ch, idx = imgui.combo("Algorithm", idx, ALGORITHM_NAMES)
        if ch:
            ats.algorithm = ALGORITHM_NAMES[idx]

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

    # -- pieces ---------------------------------------------------------

    def _render_auto_banners(self, ats):
        if ats.warning:
            imgui.text_colored(imgui.ImVec4(*_BAD), ats.warning)
            imgui.same_line()
            if imgui.button("Dismiss"):
                ats.warning = ""

        sweeps = sweeping_parameters(self.state.sim)
        if sweeps:
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
                "Tiles are not comparable: " + ", ".join(sweeps),
            )
            imgui.text_disabled(
                "These have a spatial or cohort sweep, so they take different "
                "values in different tiles. Fitness is confounded by position "
                "until they are cleared."
            )

    def _render_auto_unavailable(self, ats):
        if self.auto_unavailable == "model_missing":
            imgui.text_wrapped("CLIP model weights are not downloaded.")
            if imgui.button("Download CLIP model (~330 MB)"):
                ats.download_model_requested = True
        else:
            imgui.text_wrapped(f"Auto mode unavailable: {self.auto_unavailable}")
            imgui.text_disabled("pip install onnxruntime-directml tokenizers cmaes")

    def _render_goal(self, ats, svc):
        """Typed text is not the goal until it is submitted. The box is tinted
        while the two differ, so there is never a moment where the UI shows one
        prompt and CLIP is scoring another."""
        active = (svc.prompt if svc is not None else "").strip()
        pending = ats.prompt.strip() != active

        if pending:
            imgui.push_style_color(imgui.Col_.frame_bg,
                                   imgui.ImVec4(0.42, 0.28, 0.05, 1.0))
            imgui.push_style_color(imgui.Col_.frame_bg_hovered,
                                   imgui.ImVec4(0.52, 0.35, 0.07, 1.0))
            imgui.push_style_color(imgui.Col_.frame_bg_active,
                                   imgui.ImVec4(0.58, 0.40, 0.09, 1.0))
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
            imgui.text_colored(imgui.ImVec4(*_WARN),
                               "not set - press Enter or click Set")
        elif active:
            imgui.text_colored(imgui.ImVec4(*_OK), f'steering toward: "{active}"')
        else:
            imgui.text_disabled("no goal set")

    def _render_rollout_controls(self, ats, grid_note=None):
        """Grid and rollout timing. Shared verbatim by Auto and Explore - both
        drive the same AutoTournamentService rollout machine, so duplicating
        these widgets would let the two tabs disagree about what a generation
        is.

        grid_note overrides the last hint line because the consequence of a
        grid change differs: Auto mode loses its accumulated covariance, while
        Explore mode only ends any expedition in flight."""
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
        imgui.text_disabled(f"population {tiles}   source {src_px}px/tile")
        if src_px < 224:
            imgui.text_disabled(
                "  upscaled to 224 for CLIP - consider a larger canvas")
        if ats.grid == 2:
            imgui.text_disabled("  popsize 4 is small for 80-D CMA-ES")
        imgui.text_disabled(note or "changing the grid resets the optimizer")

    def _render_transport(self, ats):
        if ats.running:
            if imgui.button("Pause"):
                ats.pause_requested = True
        else:
            imgui.begin_disabled(not ats.prompt.strip())
            if imgui.button("Start"):
                ats.start_requested = True
            imgui.end_disabled()
        imgui.same_line()
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
            imgui.text_disabled(
                "brain only - the loaded preset fixes the overall look")
            return

        imgui.text_disabled(
            "searching " + ", ".join(n.replace('_', ' ').title()
                                     for n, _g, _lo, _hi in PHYSICS_PARAMS))
        imgui.text_colored(
            imgui.ImVec4(*_WARN),
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
        imgui.text_disabled(
            f"cohorts driven to {cohorts_for(ats.grid, ats.variants_per_tile)} "
            f"(max {kmax} variants at this grid)")

        _, ats.tile_mutation_strength = imgui.slider_float(
            "Mutation Strength", ats.tile_mutation_strength, 0.0, 0.5)
        # Sigma shrinks as CMA-ES converges. Once it approaches the mutation
        # strength, the spread between tiles no longer exceeds the wobble inside
        # each tile and the search stalls with no error.
        if ats.tile_mutation_strength < sigma / 3.0:
            imgui.text_disabled(f"sigma {sigma:.3f} - ok")
        else:
            imgui.text_colored(
                imgui.ImVec4(*_WARN),
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
        """Fitness and sigma overlaid, each on its own axis.

        Sharing an axis would be misleading: fitness climbs through ~0.05-0.5
        while sigma decays from ~0.5 toward 0, so whichever has the wider range
        flattens the other. Each is normalised separately and the true range is
        printed next to its colour.
        """
        w = max(120.0, imgui.get_content_region_avail().x)
        p0 = imgui.get_cursor_screen_pos()
        imgui.dummy(imgui.ImVec2(w, self._TRACE_H))
        dl = imgui.get_window_draw_list()

        x0, y0 = p0.x, p0.y
        x1, y1 = x0 + w, y0 + self._TRACE_H
        dl.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1),
                           imgui.get_color_u32(imgui.ImVec4(0.09, 0.09, 0.11, 1.0)))
        dl.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1),
                    imgui.get_color_u32(imgui.ImVec4(0.30, 0.30, 0.34, 1.0)))

        pad = self._TRACE_PAD
        iw, ih = w - 2 * pad, self._TRACE_H - 2 * pad

        def draw(values, color):
            if not values:
                return
            norm = normalize_series(values)
            # Never more points than the plot is wide.
            if len(norm) > self._TRACE_MAX_PTS:
                step = len(norm) / self._TRACE_MAX_PTS
                norm = [norm[min(int(i * step), len(norm) - 1)]
                        for i in range(self._TRACE_MAX_PTS)]
            col = imgui.get_color_u32(imgui.ImVec4(*color))
            n = len(norm)
            if n == 1:
                dl.add_circle_filled(
                    imgui.ImVec2(x0 + pad + iw * 0.5, y0 + pad + ih * 0.5), 2.5, col)
                return
            pts = [imgui.ImVec2(x0 + pad + iw * i / (n - 1),
                                y0 + pad + ih * (1.0 - v))
                   for i, v in enumerate(norm)]
            for a, b in zip(pts, pts[1:]):
                dl.add_line(a, b, col, 1.6)

        draw(sigma, _SIGMA_COLOR)     # behind
        draw(best, _FIT_COLOR)        # in front - it is the thing being optimised

        imgui.text_colored(imgui.ImVec4(*_FIT_COLOR),
                           f"fitness {min(best):.3f} - {max(best):.3f}")
        if sigma:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*_SIGMA_COLOR),
                               f"   sigma {min(sigma):.3f} - {max(sigma):.3f}")

    def _render_save_load(self, ats, svc):
        if imgui.button("Save best genome"):
            ats.save_best_requested = True
        imgui.same_line()
        if imgui.button("Save checkpoint"):
            ats.save_checkpoint_requested = True

        _, self._auto_load_path = imgui.input_text(
            "Load path", getattr(self, "_auto_load_path", ""))
        if imgui.button("Load genome"):
            ats.load_genome_path = self._auto_load_path
        imgui.same_line()
        if imgui.button("Load checkpoint"):
            ats.load_checkpoint_path = self._auto_load_path
        imgui.text_disabled(
            "Load genome takes only the starting point - sigma, algorithm and "
            "grid stay as set here. Load checkpoint restores the whole search "
            "and forces the grid to the saved value.")

        if svc is not None and svc.logger is not None and svc.logger.enabled:
            if imgui.button("Open run folder"):
                import os
                import subprocess

                subprocess.Popen(["explorer", os.path.abspath(svc.logger.dir)])
            imgui.same_line()
            imgui.text_disabled(str(svc.logger.run_id))

        _, ats.autosave_every = imgui.slider_int(
            "Autosave every N gens", ats.autosave_every, 0, 100)
