"""Auto (CLIP) tab of the tournament window.

Passive: renders widgets and sets state, runs no logic.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services.capture_health import sweeping_parameters
from services.cohort_tiling import cohorts_for, max_variants

ALGORITHM_NAMES = ["CMA-ES", "Sep-CMA-ES", "GA", "Random Search"]

COHORT_TOOLTIP = (
    "Particles are numbered in one long list. Both 'which tile' and 'which "
    "cohort' are just slices of that list, so they nest: with 64 cohorts across "
    "16 tiles, each tile contains 4 cohorts, and each cohort gets its own small "
    "tweak of that tile's genome.\n\n"
    "This is why cohorts must be a multiple of the tile count. If they were "
    "equal, each tile would contain exactly one cohort - one tweak applied to "
    "every particle - and the tile would still be uniform, just shifted. "
    "Cohorts are set for you while this is enabled."
)

_WARN = (1.0, 0.6, 0.2, 1.0)
_BAD = (1.0, 0.4, 0.3, 1.0)


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

        changed, ats.prompt = imgui.input_text(
            "Goal", ats.prompt, imgui.InputTextFlags_.enter_returns_true
        )
        if changed:
            ats.prompt_changed = True
        imgui.same_line()
        if imgui.button("Set"):
            ats.prompt_changed = True

        idx = (ALGORITHM_NAMES.index(ats.algorithm)
               if ats.algorithm in ALGORITHM_NAMES else 0)
        ch, idx = imgui.combo("Algorithm", idx, ALGORITHM_NAMES)
        if ch:
            ats.algorithm = ALGORITHM_NAMES[idx]

        ch, g = imgui.slider_int("Grid", ats.grid, 2, 8)
        if ch and g != ats.grid:
            ats.grid = g
            ats.grid_changed = True
        self._render_grid_hints(ats)

        imgui.separator()
        self._render_transport(ats)

        imgui.separator()
        _, ats.steps_per_gen = imgui.slider_int(
            "Steps per Gen", ats.steps_per_gen, 50, 2000)
        _, ats.sim_steps_per_frame = imgui.slider_int(
            "Sim Steps per Frame", ats.sim_steps_per_frame, 1, 50)
        _, ats.snapshots_per_gen = imgui.slider_int(
            "Snapshots per Gen", ats.snapshots_per_gen, 1, 8)
        _, ats.sigma0 = imgui.slider_float("Initial Sigma", ats.sigma0, 0.05, 1.5)

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

    def _render_grid_hints(self, ats):
        tiles = ats.grid * ats.grid
        src_px = 1024 // ats.grid
        imgui.text_disabled(f"population {tiles}   source {src_px}px/tile")
        if src_px < 224:
            imgui.text_disabled(
                "  upscaled to 224 for CLIP - consider a larger canvas")
        if ats.grid == 2:
            imgui.text_disabled("  popsize 4 is small for 80-D CMA-ES")
        imgui.text_disabled("changing the grid resets the optimizer")

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
        imgui.plot_lines("fitness", np.asarray(best, dtype=np.float32),
                         graph_size=imgui.ImVec2(0, 60))
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
