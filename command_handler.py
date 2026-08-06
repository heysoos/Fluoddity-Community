"""Command handler: processes one-shot UI commands each frame."""
import random
import numpy as np
from utilities.gl_helpers import readback_rule


class CommandHandler:
    """Processes one-shot commands from UI state.

    Receives references to app components in __init__ and coordinates
    them in response to UI flags (the "one-shot flag" pattern).
    """

    def __init__(self, sim, camera, ui, rule_manager, entity_picker,
                 video_service, config_saver, multi_load_service, user_configs_dir,
                 field_handler=None, param_lock_service=None, tournament_service=None,
                 auto_service=None):
        self.sim = sim
        self.camera = camera
        self.ui = ui
        self.rule_manager = rule_manager
        self.entity_picker = entity_picker
        self.video_service = video_service
        self.config_saver = config_saver
        self.multi_load_service = multi_load_service
        self.user_configs_dir = user_configs_dir
        self.field_handler = field_handler
        self.param_lock_service = param_lock_service
        self.tournament_service = tournament_service
        self.auto_service = auto_service
        self._auto_capture_warned = False

        # Preview state
        self.preview_rule_active = False  # File->load preview
        self._preview_rule_was_pushed = False  # Whether we actually pushed a rule (vs blocked by lock)
        self.clipboard_preview_active = False  # Config clipboard preview
        self._clipboard_rule_was_pushed = False  # Whether clipboard preview actually pushed a rule
        self._clipboard_cached_config = None  # Full config saved before clipboard preview

        # Video pending state (waiting for scheduled start frame)
        self.video_pending = False
        self.video_scheduled_start_frame = 0

        # Deferred entity selection state (waits one frame for rule buffer to be written)
        self._pending_entity_selection = None  # Tuple of (entity_id, entity_pos, entity_cohort) or None

    def _apply_config_with_locks(self, config, ui_state, watercolor_override=None):
        """Apply config with parameter lock snapshot/restore. Returns rule."""
        pls = self.param_lock_service
        snapshot = pls.snapshot_locked(ui_state.sim, ui_state.preferences) if pls else {}
        rule = self.config_saver.apply_config(config, ui_state.sim, watercolor_override)
        if pls:
            pls.restore_locked(ui_state.sim, ui_state.preferences, snapshot)
        return rule

    def _push_and_apply_rule(self, rule, ui_state):
        """Push rule to manager and apply to GPU, unless rule lock is active."""
        pls = self.param_lock_service
        if pls and pls.should_block_rule_push():
            return
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        self.sim.apply_rule(rule)

    @property
    def has_pending_entity_selection(self):
        """Whether there's a pending entity selection waiting for rule readback."""
        return self._pending_entity_selection is not None

    def try_complete_entity_selection(self, ui_state):
        """Try to complete a pending entity selection after rule buffer update.

        Called from simulation_runner after sim.update() on the first physics step.
        Returns True if selection was completed.
        """
        if self._pending_entity_selection is None:
            return False

        pending_entity_id = self.sim.consume_pending_rule_readback()
        if pending_entity_id is None:
            return False

        entity_id, entity_pos, entity_cohort = self._pending_entity_selection
        self._pending_entity_selection = None

        # Read back the rule (buffer was just written by entity_update)
        rule = readback_rule(self.sim.get_rule_buffer(), entity_id)
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        self.sim.apply_rule(rule)
        self.sim.update_sliders_from_particle(entity_pos, entity_cohort)
        print(f"Deferred rule readback complete for entity {entity_id}")
        return True

    def process_commands(self, ui_state, tiling_mode):
        """Handle one-shot commands from UI state."""

        # Handle world size change
        if ui_state.request_world_size_change:
            self._handle_world_size_change(ui_state)

        # Toggle recording (with delayed start support)
        if ui_state.toggle_recording:
            self._handle_toggle_recording(ui_state)

        # Screenshot request (Shift+P) - set pending flag
        if ui_state.request_screenshot:
            return 'screenshot_pending'

        # Shader reload
        if ui_state.request_reload:
            self.sim.reload()
            if self.rule_manager.has_rules():
                self.sim.apply_rule(self.rule_manager.get_current_rule())
            self.camera.reload()
            if self.field_handler and self.field_handler.adv_draw:
                self.field_handler.adv_draw.reload()
            # A partial rollout across a shader swap is not a valid sample.
            if self.auto_service is not None:
                self.auto_service.abort_generation()

        # Simple reset (R key)
        if ui_state.request_reset:
            self.sim.reset()

        # Full reset (Z key)
        if ui_state.request_full_reset:
            self._handle_full_reset(ui_state)

        # Randomize mutations (M key)
        if ui_state.request_randomize_mutations:
            self._handle_randomize_mutations(ui_state)

        # Handle sweep preview restore
        restored_sweep_preview = self._handle_sweep_preview_restore(ui_state)

        # Handle mouse clicks
        if not restored_sweep_preview:
            self._handle_mouse_clicks(ui_state, tiling_mode)

        # Handle config save/load/delete
        self._handle_config_commands(ui_state)

        # Handle field image load requests
        if ui_state.request_load_force_field_image and ui_state.field_load_image_path:
            if self.field_handler:
                self.field_handler.load_field_from_image(ui_state.field_load_image_path, "force")
        if ui_state.request_load_strafe_field_image and ui_state.field_load_image_path:
            if self.field_handler:
                self.field_handler.load_field_from_image(ui_state.field_load_image_path, "strafe")

        # Handle preview commands (file browser)
        self._handle_preview_commands(ui_state)

        # Handle config clipboard commands
        self._handle_clipboard_commands(ui_state)

        # Tournament mode
        self._handle_tournament(ui_state)

        # Automatic (CLIP-guided) tournament mode
        self._handle_auto_tournament(ui_state)

        return None

    def _handle_world_size_change(self, ui_state):
        """Handle world size change request (also handles aspect ratio changes)."""
        self.sim.world_size = ui_state.preferences.world_size
        self.sim.canvas_aspect_ratio = ui_state.preferences.canvas_aspect_ratio
        self.sim.particle_density = ui_state.preferences.particle_density
        self.sim.setup_simulation_state()
        self.sim.setup_shaders()
        self.entity_picker.update_buffer(self.sim.get_entity_buffer())
        if self.rule_manager.has_rules():
            self.sim.apply_rule(self.rule_manager.get_current_rule())
        self.sim.reset()
        # setup_simulation_state() reallocated every GPU buffer, which zeroes the
        # buffer holding the 16 tournament genomes. Without a re-upload every
        # particle gets a null brain and the whole grid freezes.
        if self.tournament_service is not None:
            self.tournament_service.mark_dirty()
        # A partial rollout is not a valid fitness sample.
        if self.auto_service is not None:
            self.auto_service.abort_generation()
        # Reinitialize field texture at new canvas dimensions (if it exists)
        if self.field_handler and self.field_handler._has_field_tex:
            canvas_dim_x, canvas_dim_y = self.sim.get_canvas_dimensions()
            self.field_handler.adv_draw.ensure_initialized(canvas_dim_x, canvas_dim_y)
        self.ui._last_applied_world_size = ui_state.preferences.world_size
        self.ui._last_applied_particle_density = ui_state.preferences.particle_density
        w, h = self.sim.get_canvas_dimensions()
        print(f"World size changed to {self.sim.world_size} "
              f"(density: {self.sim.particle_density}, "
              f"entity_count: {self.sim.entity_count}, "
              f"canvas: {w}x{h}, "
              f"{self.sim.entity_count / (w * h):.3f} particles/texel)")

    def _handle_toggle_recording(self, ui_state):
        """Handle video recording toggle with delayed start support."""
        if self.video_pending:
            self.video_pending = False
            self.video_scheduled_start_frame = 0
        elif self.video_service.is_active():
            self.video_service.stop()
        else:
            video_end_frame = ui_state.preferences.video_end_frame
            video_simulation_frames = ui_state.preferences.max_frames * ui_state.preferences.motion_blur_samples
            scheduled_start_frame = video_end_frame - video_simulation_frames

            if video_end_frame == 0 or scheduled_start_frame <= self.sim.frame_count:
                self.video_service.start()
            else:
                self.video_pending = True
                self.video_scheduled_start_frame = scheduled_start_frame

    def _handle_full_reset(self, ui_state):
        """Handle full reset (Z key): reset entities, apply zero rule, randomize, push new state."""
        self.sim.reset()
        zero_rule = np.zeros((10, 8), dtype=np.float32)
        self.sim.apply_rule(zero_rule)
        ui_state.sim.rule_seed = random.random()
        self.rule_manager.push_rule(zero_rule, ui_state.sim.rule_seed)

    def _handle_randomize_mutations(self, ui_state):
        """Handle randomize mutations (M key)."""
        current_rule = self.rule_manager.get_current_rule()
        if current_rule is not None:
            ui_state.sim.rule_seed = random.random()
            self.rule_manager.push_rule(current_rule.copy(), ui_state.sim.rule_seed)
            self.sim.apply_rule(current_rule)

    def _handle_sweep_preview_restore(self, ui_state):
        """Handle sweep preview restore: ANY click re-enables sweeps. Returns True if restored."""
        if ui_state.sim.sweep_preview_pending_restore:
            if ui_state.any_left_click_this_frame or ui_state.any_right_click_this_frame:
                ui_state.sim.parameter_sweeps_enabled = True
                ui_state.sim.sweep_preview_pending_restore = False
                return True
        return False

    def _handle_mouse_clicks(self, ui_state, tiling_mode):
        """Handle left/right mouse click behavior based on mode."""
        # Tournament mode: left click selects the tile under the cursor
        if ui_state.tournament.enabled and ui_state.left_click_this_frame:
            tex = self.camera.screen_to_tex(ui_state.mouse_pos, self.sim.view_tex.size)
            grid = self.tournament_service.grid if self.tournament_service else 4
            tx = min(grid - 1, max(0, int(tex[0] * grid)))
            ty = min(grid - 1, max(0, int(tex[1] * grid)))
            ui_state.tournament.clicked_tile = ty * grid + tx
            return

        if ui_state.left_click_this_frame:
            if ui_state.sim.parameter_sweeps_enabled:
                self._handle_sweep_click(ui_state, tiling_mode)
            elif ui_state.preferences.mouse_mode == "Select Particle":
                self._handle_entity_pick(ui_state, tiling_mode)

        elif ui_state.right_click_this_frame:
            if ui_state.sim.parameter_sweeps_enabled:
                if self.sim.has_active_xy_sweep() or self.sim.has_active_cohort_sweep():
                    ui_state.sim.parameter_sweeps_enabled = False
                    ui_state.sim.sweep_preview_pending_restore = True
            elif ui_state.preferences.mouse_mode == "Select Particle":
                if self.rule_manager.length() > 1:
                    prev_rule, prev_seed = self.rule_manager.pop_rule()
                    if prev_seed is not None:
                        ui_state.sim.rule_seed = prev_seed
                    self.sim.apply_rule(prev_rule)

    def _handle_tournament(self, ui_state):
        """Drive the TournamentService from tournament one-shot flags."""
        svc = self.tournament_service
        if svc is None:
            return
        ts = ui_state.tournament

        # Sync persistent controls
        svc.mutation_strength = ts.mutation_strength
        svc.inject_randoms = ts.inject_randoms
        svc.crossover_enabled = ts.crossover_enabled

        if not ts.enabled:
            self._clear_tournament_flags(ts)
            return

        # Lazy-init the population the first time tournament turns on
        if not svc.initialized:
            svc.init_population()

        if ts.clicked_tile >= 0:
            svc.toggle_select(ts.clicked_tile)
        if ts.next_gen_requested:
            svc.next_generation()
        if ts.undo_requested:
            svc.undo()
        if ts.reset_requested:
            svc.reset()
        if ts.save_requested:
            self._save_tournament_selection(ui_state)

        # These are one-shot: clear them now that they've been consumed.
        # (UI.get_state() returns the live state object and only clears its own
        #  private mirrors, so the consumer must clear these.)
        self._clear_tournament_flags(ts)

        # Upload + clear/reseed whenever the population changed
        if svc.is_dirty():
            self.sim.write_tournament_rules(svc.pack_rule_bytes())
            self.sim.reset()   # clears canvas + frame_count=0 => reseed into tiles
            svc.clear_dirty()

    @staticmethod
    def _clear_tournament_flags(ts):
        """Reset tournament one-shot flags after consumption."""
        ts.clicked_tile = -1
        ts.next_gen_requested = False
        ts.undo_requested = False
        ts.reset_requested = False
        ts.save_requested = False

    def _save_tournament_selection(self, ui_state):
        """Save each selected genome to a config JSON in the user configs dir."""
        svc = self.tournament_service
        if not svc.selected:
            print("Tournament save: no tiles selected")
            return
        for tile in sorted(svc.selected):
            config = self.config_saver.create_config(ui_state.sim, svc.population[tile])
            filepath = self.user_configs_dir / f"tournament_tile{tile}.json"
            self.config_saver.save_to_file(config, filepath)
            print(f"Saved tournament tile {tile} -> {filepath}")

    # ------------------------------------------------------------------
    # Automatic (CLIP-guided) tournament
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_auto_flags(ats):
        """Reset auto one-shot flags after consumption.

        Cleared here rather than in UI.get_state(): get_state returns the live
        state object, so clearing there would wipe flags before this handler
        ever read them. `warning` is persistent and is NOT cleared.
        """
        ats.start_requested = False
        ats.pause_requested = False
        ats.reset_requested = False
        ats.prompt_changed = False
        ats.grid_changed = False
        ats.save_checkpoint_requested = False
        ats.save_best_requested = False
        ats.save_tile_requested = -1
        ats.load_checkpoint_path = ""
        ats.load_genome_path = ""
        ats.download_model_requested = False

    def _handle_auto_tournament(self, ui_state):
        """Drive the AutoTournamentService from auto one-shot flags."""
        ats = ui_state.auto_tournament

        if ats.download_model_requested:
            self._start_model_download()

        svc = self.auto_service
        if svc is None:
            self._clear_auto_flags(ats)
            return

        if ats.grid_changed and self.tournament_service is not None:
            # Applied at a generation boundary; resets the optimizer because
            # cmaes.CMA fixes popsize at construction.
            self.tournament_service.set_grid(ats.grid)
            svc.reset()

        svc.configure(
            steps_per_gen=ats.steps_per_gen,
            snapshots_per_gen=ats.snapshots_per_gen,
            sim_steps_per_frame=ats.sim_steps_per_frame,
            sigma0=ats.sigma0,
            algorithm=ats.algorithm,
            autosave_every=ats.autosave_every,
            physics_enabled=ats.physics_enabled,
            tile_mutation_enabled=ats.tile_mutation_enabled,
            variants_per_tile=ats.variants_per_tile,
            tile_mutation_strength=ats.tile_mutation_strength,
        )

        if ats.prompt_changed:
            svc.set_prompt(ats.prompt)
        if ats.reset_requested:
            svc.reset()
        if ats.pause_requested:
            svc.pause()
        if ats.load_genome_path:
            self._load_auto_genome(svc, ats)
        if ats.load_checkpoint_path:
            self._load_auto_checkpoint(svc, ats)
        if ats.start_requested and ats.prompt.strip():
            self._auto_capture_warned = False
            svc.start(ats.prompt)
        if ats.save_best_requested:
            self._save_auto_genome(svc, ui_state, tile=None)
        if ats.save_tile_requested >= 0:
            self._save_auto_genome(svc, ui_state, tile=ats.save_tile_requested)
        if ats.save_checkpoint_requested:
            self._save_auto_checkpoint(svc)

        ats.running = svc.phase.value == "rollout"
        self._clear_auto_flags(ats)

    def _start_model_download(self):
        import threading

        from tools.fetch_clip_onnx import fetch

        threading.Thread(target=fetch, daemon=True).start()
        print("[auto] downloading CLIP model in the background")

    def report_capture_health(self, crops, ui_state):
        """Reported once per run rather than once per frame."""
        if self._auto_capture_warned:
            return
        from services.capture_health import check_capture

        msg = check_capture(crops)
        if msg:
            self._auto_capture_warned = True
            ui_state.auto_tournament.warning = msg
            print(f"[auto] {msg}")

    def _load_auto_genome(self, svc, ats):
        from services.genome_io import import_genome

        try:
            z, clamped, _meta = import_genome(ats.load_genome_path)
        except (OSError, ValueError, KeyError) as exc:
            ats.warning = f"could not load genome: {exc}"
            print(f"[auto] {ats.warning}")
            return
        svc.set_x0(z)
        if clamped:
            ats.warning = (f"loaded genome with {clamped}/80 coefficients "
                           "clamped to the searchable range")
            print(f"[auto] {ats.warning}")

    def _load_auto_checkpoint(self, svc, ats):
        from services.run_checkpoint import CheckpointError, load_checkpoint

        try:
            state = load_checkpoint(ats.load_checkpoint_path,
                                    expect_signature=svc.spec.signature())
            svc.restore(state)
        except CheckpointError as exc:
            ats.warning = f"checkpoint not loaded: {exc}"
            print(f"[auto] {ats.warning}")
            return
        # Loading forces the grid to the checkpoint's value; keep the UI in sync.
        ats.grid = svc.tournament.grid
        ats.algorithm = svc.algorithm
        ats.prompt = svc.prompt
        ats.steps_per_gen = svc.steps_per_gen
        ats.snapshots_per_gen = svc.snapshots_per_gen
        ats.sim_steps_per_frame = svc.sim_steps_per_frame
        ats.sigma0 = svc.sigma0
        print(f"[auto] resumed at generation {svc.generation}")

    def _save_auto_checkpoint(self, svc):
        from services.run_checkpoint import save_checkpoint

        if svc.logger is None or not svc.logger.enabled:
            print("[auto] no run directory; checkpoint not saved")
            return
        path = svc.logger.dir / f"checkpoint_gen{svc.generation:06d}.npz"
        save_checkpoint(path, svc.checkpoint_state())
        print(f"[auto] saved {path}")

    def _save_auto_genome(self, svc, ui_state, tile):
        from services.genome_io import export_genome

        if tile is None:
            z = svc.optimizer.best()[0] if svc.optimizer is not None else None
            name = f"evolved_best_gen{svc.generation:04d}.json"
        else:
            cz = svc.current_z
            z = cz[tile] if cz is not None and tile < len(cz) else None
            name = f"evolved_tile{tile}_gen{svc.generation:04d}.json"
        if z is None:
            print("[auto] nothing to save yet")
            return
        canvas_px = self.sim.get_canvas_dimensions()[0]
        meta = {
            "generation": svc.generation,
            "prompt": svc.prompt,
            "algorithm": svc.algorithm,
            "evolved_at_canvas_px": int(canvas_px),
            "evolved_at_tile_px": 224,
            "evolved_with_tile_mutation": bool(svc.tile_mutation_enabled),
            "mutation_strength": float(svc.tile_mutation_strength),
            "variants_per_tile": int(svc.variants_per_tile),
        }
        path = self.user_configs_dir / name
        export_genome(path, z, ui_state.sim, meta)
        print(f"[auto] saved {path}")

    def _handle_sweep_click(self, ui_state, tiling_mode):
        """Handle left click when parameter sweeps are enabled."""
        if not (self.sim.has_active_xy_sweep() or self.sim.has_active_cohort_sweep()):
            return

        tex_coords = self.camera.screen_to_tex(
            ui_state.mouse_pos, self.sim.view_tex.size
        )
        if tiling_mode:
            tex_coords = (
                np.fmod(tex_coords[0] + 10.0, 1.0),
                np.fmod(tex_coords[1] + 10.0, 1.0)
            )
        world_pos = (tex_coords[0] * 2 - 1, tex_coords[1] * 2 - 1)

        if self.sim.has_active_cohort_sweep():
            entity_id, entity_pos, entity_cohort = self.entity_picker.find_nearest_entity(tex_coords)
            self.sim.update_sliders_from_particle(world_pos, entity_cohort)
        else:
            self.sim.update_sliders_from_position(world_pos)

    def _handle_entity_pick(self, ui_state, tiling_mode):
        """Handle entity selection via left click in Select Particle mode."""
        tex_coords = self.camera.screen_to_tex(
            ui_state.mouse_pos, self.sim.view_tex.size
        )
        if tiling_mode:
            tex_coords = (
                np.fmod(tex_coords[0] + 10.0, 1.0),
                np.fmod(tex_coords[1] + 10.0, 1.0)
            )
        entity_id, entity_pos, entity_cohort = self.entity_picker.find_nearest_entity(tex_coords)

        if entity_id >= 0 and entity_id < self.sim.entity_count:
            print(f"Entity {entity_id} at pos {entity_pos}, cohort {entity_cohort} - requesting rule buffer update")
            self.sim.request_rule_buffer_update(entity_id)
            self._pending_entity_selection = (entity_id, entity_pos, entity_cohort)
        else:
            print(f"Warning: entity_id {entity_id} out of bounds (max: {self.sim.entity_count - 1})")

    def _handle_config_commands(self, ui_state):
        """Handle config save/load/delete commands."""
        fh = self.field_handler

        # Config save (Ctrl+C)
        if ui_state.request_save_config:
            current_rule = self.rule_manager.get_current_rule()

            field_snapshot, field_strengths = (
                fh.snapshot_with_strengths(ui_state) if fh else (None, None))

            config = self.config_saver.create_config(
                ui_state.sim, current_rule, field_strengths=field_strengths)
            config_string = self.config_saver.encode_clipboard(config)
            self.ui.set_clipboard(config_string)
            self.ui.add_to_config_clipboard(
                config, self.ui.currently_open_project, field_snapshot=field_snapshot)
            if fh:
                fh.enforce_snapshot_cap(self.ui.config_clipboard)
            print(f"Config copied to clipboard ({len(config_string)} chars)")

        # Config load (Ctrl+V)
        if ui_state.request_load_config:
            config_string = ui_state.clipboard_text
            if config_string:
                config = self.config_saver.decode_clipboard(config_string)
                if config is not None:
                    rule = self._apply_config_with_locks(config, ui_state)
                    self._push_and_apply_rule(rule, ui_state)
                    if fh:
                        fh.apply_last_copied(ui_state)
                    print("Config loaded from clipboard")
                else:
                    print("Failed to load config from clipboard")

        # File save (menu)
        if ui_state.request_save_file:
            self._handle_file_save(ui_state)

        # File load (menu)
        if ui_state.request_load_file:
            self._handle_file_load(ui_state)

        # File delete (menu)
        if ui_state.request_delete_file:
            filename = ui_state.delete_filename
            category = ui_state.delete_category
            if filename:
                filepath = self.ui._get_config_path(filename, category)
                if filepath.exists():
                    filepath.unlink()
                    if fh:
                        fh.delete_field_png(filepath)
                    print(f"Config deleted: {filepath}")

    def _handle_file_save(self, ui_state):
        """Handle file save from menu, including field texture PNG."""
        filename = ui_state.save_filename
        if not filename:
            return

        fh = self.field_handler
        current_rule = self.rule_manager.get_current_rule()

        field_data, field_strengths = (
            fh.snapshot_with_strengths(ui_state) if fh else (None, None))

        config = self.config_saver.create_config(
            ui_state.sim, current_rule, field_strengths=field_strengths)
        filepath = self.user_configs_dir / f"{filename}.json"
        self.config_saver.save_to_file(config, filepath)

        if fh:
            fh.save_field_png(field_data, filename, self.user_configs_dir)
            fh.invalidate_cache(filepath)

        print(f"Config saved to {filepath}")
        self.ui.update_physics_defaults(filename)

    def _handle_file_load(self, ui_state):
        """Handle file load from menu, including multi-load and preview modes."""
        filename = ui_state.load_filename
        category = ui_state.load_category
        if not filename:
            return

        fh = self.field_handler

        # Multi-load mode (ignores fields entirely)
        if ui_state.multi_load.multi_load_enabled:
            filepath = self.ui._get_config_path(filename, category)
            config = self.config_saver.load_from_file(filepath)
            if config is not None:
                success = self.multi_load_service.add_config(config, filename)
                if success:
                    print(f"Config added to multi-load: {filename}")
                else:
                    print(f"Failed to add config: multi-load list is full "
                          f"({self.multi_load_service.get_config_count()}/64)")
            else:
                print(f"Failed to load config from {filepath}")
            return

        # Normal mode
        if self.preview_rule_active:
            # Preview already applied config, rule, and field texture - just finalize
            self.preview_rule_active = False
            self._preview_rule_was_pushed = False
            if fh:
                fh.discard_preview_cache()
            if ui_state.load_watercolor_override is not None:
                ui_state.sim.watercolor_mode = ui_state.load_watercolor_override
            print(f"Config loaded (from preview): {filename}")
            self.ui.update_physics_defaults(filename)
        else:
            # No preview active - load fresh from file
            filepath = self.ui._get_config_path(filename, ui_state.load_category)
            config = self.config_saver.load_from_file(filepath)
            if config is not None:
                rule = self._apply_config_with_locks(
                    config, ui_state,
                    watercolor_override=ui_state.load_watercolor_override
                )
                self._push_and_apply_rule(rule, ui_state)
                if fh:
                    fh.apply_for_config(config, filepath, ui_state)
                print(f"Config loaded from {filepath}")
                self.ui.update_physics_defaults(filename)
            else:
                print(f"Failed to load config from {filepath}")

    def _handle_preview_commands(self, ui_state):
        """Handle config preview (hover in Load submenu) and clear preview."""
        fh = self.field_handler

        # Clear preview must happen before new preview
        if ui_state.request_clear_preview:
            if self.preview_rule_active:
                if self._preview_rule_was_pushed:
                    prev_rule, prev_seed = self.rule_manager.pop_rule()
                    if prev_seed is not None:
                        ui_state.sim.rule_seed = prev_seed
                    self.sim.apply_rule(prev_rule)
                self.preview_rule_active = False
                self._preview_rule_was_pushed = False

                if fh:
                    fh.restore_from_preview(ui_state)

        # New preview
        if ui_state.request_preview_config:
            filename = ui_state.preview_filename
            category = ui_state.preview_category
            if filename:
                filepath = self.ui._get_config_path(filename, category)
                config = self.config_saver.load_from_file(filepath)
                if config and config.rule is not None:
                    # Cache field texture on first preview entry
                    if not self.preview_rule_active and fh:
                        fh.cache_for_preview(ui_state)

                    pls = self.param_lock_service
                    if not (pls and pls.should_block_rule_push()):
                        if not (pls and pls.is_locked('rule_seed')):
                            ui_state.sim.rule_seed = config.rule_seed
                        self.rule_manager.push_rule(config.rule, ui_state.sim.rule_seed)
                        self.sim.apply_rule(config.rule)
                        self._preview_rule_was_pushed = True
                    self.preview_rule_active = True

                    if fh:
                        fh.apply_for_config(config, filepath, ui_state)

    def _handle_clipboard_commands(self, ui_state):
        """Handle config clipboard preview, load, delete, and import-to-multiload."""
        fh = self.field_handler

        # Clear clipboard preview (must happen before new preview)
        if ui_state.request_clear_clipboard_preview:
            if self.clipboard_preview_active:
                if self._clipboard_rule_was_pushed:
                    self.rule_manager.pop_rule()
                # Restore the full cached config (not just the rule)
                if self._clipboard_cached_config is not None:
                    rule = self._apply_config_with_locks(
                        self._clipboard_cached_config, ui_state)
                    self.sim.apply_rule(rule)
                    self._clipboard_cached_config = None

                if fh:
                    fh.restore_from_clipboard_preview(ui_state)

                self.clipboard_preview_active = False
                self._clipboard_rule_was_pushed = False

        # New clipboard preview
        if ui_state.request_preview_clipboard_config:
            idx = ui_state.clipboard_config_index
            if 0 <= idx < len(self.ui.config_clipboard):
                # Cache current full config before applying preview
                if not self.clipboard_preview_active:
                    current_rule = self.rule_manager.get_current_rule()
                    self._clipboard_cached_config = self.config_saver.create_config(
                        ui_state.sim, current_rule)
                    if fh:
                        fh.cache_for_clipboard_preview(ui_state)

                config, _label, field_snapshot = self.ui.config_clipboard[idx]
                rule = self._apply_config_with_locks(config, ui_state)
                pls = self.param_lock_service
                if not (pls and pls.should_block_rule_push()):
                    self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
                    self.sim.apply_rule(rule)
                    self._clipboard_rule_was_pushed = True
                self.clipboard_preview_active = True

                if fh:
                    fh.apply_snapshot(field_snapshot, config, ui_state)

        # Load clipboard config (click)
        if ui_state.request_load_clipboard_config:
            self._load_clipboard_config(ui_state)

        # Delete clipboard entry
        if ui_state.request_delete_clipboard_config:
            self._delete_clipboard_config(ui_state)

        # Import clipboard to multi-load
        if ui_state.request_import_clipboard_to_multiload:
            self._import_clipboard_to_multiload()

    def _load_clipboard_config(self, ui_state):
        """Load a config from the clipboard (apply it permanently)."""
        fh = self.field_handler

        # Clear preview first (discard cached config since we're committing)
        if self.clipboard_preview_active:
            if self._clipboard_rule_was_pushed:
                self.rule_manager.pop_rule()
            self.clipboard_preview_active = False
            self._clipboard_rule_was_pushed = False
            self._clipboard_cached_config = None
            if fh:
                fh.discard_clipboard_preview_cache()

        idx = ui_state.clipboard_config_index
        if 0 <= idx < len(self.ui.config_clipboard):
            config, label, field_snapshot = self.ui.config_clipboard[idx]
            rule = self._apply_config_with_locks(config, ui_state)
            self._push_and_apply_rule(rule, ui_state)

            if fh:
                fh.apply_snapshot(field_snapshot, config, ui_state)

            # Extract original filename from label (everything before the *)
            original_filename = label.rsplit("*", 1)[0]
            self.ui.update_physics_defaults(original_filename)
            print(f"Config loaded from clipboard: {label}")

    def _delete_clipboard_config(self, ui_state):
        """Delete an entry from the config clipboard."""
        fh = self.field_handler

        # Clear preview first, restore cached config
        if self.clipboard_preview_active:
            if self._clipboard_rule_was_pushed:
                self.rule_manager.pop_rule()
            if self._clipboard_cached_config is not None:
                rule = self._apply_config_with_locks(
                    self._clipboard_cached_config, ui_state)
                self.sim.apply_rule(rule)
                self._clipboard_cached_config = None

            if fh:
                fh.restore_from_clipboard_preview(ui_state)

            self.clipboard_preview_active = False
            self._clipboard_rule_was_pushed = False

        idx = ui_state.clipboard_config_index
        if 0 <= idx < len(self.ui.config_clipboard):
            self.ui.config_clipboard.pop(idx)

    def _import_clipboard_to_multiload(self):
        """Replace multi-load configs with contents of config clipboard."""
        # Clear existing multi-load configs
        while self.multi_load_service.get_config_count() > 0:
            self.multi_load_service.remove_config(0)

        # Add each clipboard entry
        for config, label, _field in self.ui.config_clipboard:
            self.multi_load_service.add_config(config, label)

        print(f"Imported {len(self.ui.config_clipboard)} configs from clipboard to multi-load")
