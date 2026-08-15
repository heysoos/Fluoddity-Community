"""Command handler: processes one-shot UI commands each frame."""
import copy
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
        # Explore (IMGEP) mode; set by App._ensure_archive_service().
        self.imgep_driver = None
        self.archive = None
        self.archive_store = None
        self.goal_list = None
        self.archive_projection = None
        # Which run's physics is already on disk, and the configs read back for
        # the preview. Both are per-archive and cleared on a switch.
        self._run_physics_written = ""
        self._run_config_cache = {}
        # App._switch_archive; None until Explore mode has been opened once.
        self.switch_archive = None
        # App._release_archive - see _handle_archive_management.
        self.release_archive = None
        # App._apply_brain_layout, wired by the orchestrator at startup.
        self.apply_brain_layout = None
        # A rule that arrived on a config naming a brain other than the one
        # running. Lives exactly one frame; see take_pending_brain_rule.
        self._pending_brain_rule = None

        # Preview state
        self.preview_rule_active = False  # File->load preview
        self._preview_rule_was_pushed = False  # Whether we actually pushed a rule (vs blocked by lock)
        # Archive browser live preview: which entry is on the GPU right now.
        self._archive_preview_id = -1
        self._archive_preview_pushed = False
        self._archive_preview_physics = None
        self._archive_preview_arc = None
        # (owner, layout) while a preview has BORROWED another brain, where
        # layout is the one the sim had before it. See _borrow_layout.
        self._borrow = None
        self.clipboard_preview_active = False  # Config clipboard preview
        self._clipboard_rule_was_pushed = False  # Whether clipboard preview actually pushed a rule
        self._clipboard_cached_config = None  # Full config saved before clipboard preview

        # Video pending state (waiting for scheduled start frame)
        self.video_pending = False
        self.video_scheduled_start_frame = 0

        # Deferred entity selection state (waits one frame for rule buffer to be written)
        self._pending_entity_selection = None  # Tuple of (entity_id, entity_pos, entity_cohort) or None

    @property
    def preview_active(self) -> bool:
        """Is something BORROWED on screen right now - a hover, not a choice?

        Every preview writes the rule and the physics the undo journal's
        per-frame diff watches, so each one recorded a step, and un-hovering
        restored and recorded another. Sliding down File > Load filled the
        history in seconds and pushed the real steps off the end.

        One predicate rather than a check per site: a fourth preview added
        later is covered by naming its flag HERE, and forgetting to is a bug
        with one home instead of three.
        """
        return (self.preview_rule_active
                or self._archive_preview_id >= 0
                or self.clipboard_preview_active)

    def _apply_config_with_locks(self, config, ui_state, watercolor_override=None):
        """Apply config with parameter lock snapshot/restore. Returns rule."""
        pls = self.param_lock_service
        snapshot = pls.snapshot_locked(ui_state.sim, ui_state.preferences) if pls else {}
        rule = self.config_saver.apply_config(config, ui_state.sim, watercolor_override)
        if pls:
            pls.restore_locked(ui_state.sim, ui_state.preferences, snapshot)
        self._restore_brain_settings(config, ui_state)
        return rule

    def _push_and_apply_rule(self, rule, ui_state):
        """Push rule to manager and apply to GPU, unless rule lock is active."""
        pls = self.param_lock_service
        if pls and pls.should_block_rule_push():
            return
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        if self._switch_will_apply(rule):
            return
        self.sim.apply_rule(rule)

    def apply_undo_snapshot(self, snap, ui_state) -> bool:
        """Put a snapshot back. Returns True if the brain half was skipped.

        The layout goes back BEFORE the rule: apply_rule measures a rule
        against the live layout and silently refuses a mismatch. Under a
        tournament the brain belongs to the grid's owner, so only the settings
        are restored - the same split G already makes.
        """
        from services.undo_history import CONTAINERS

        pls = self.param_lock_service
        locked = (pls.snapshot_locked(ui_state.sim, ui_state.preferences)
                  if pls else {})
        for name, module, _panel in CONTAINERS:
            target = getattr(ui_state, name)
            held = snap.fields.get(name, {})
            for f in module.UNDOABLE_FIELDS:
                if f in held:            # absent means "keep what is on screen"
                    setattr(target, f, copy.deepcopy(held[f]))
        if pls:
            pls.restore_locked(ui_state.sim, ui_state.preferences, locked)

        if self._grid_owner(ui_state) is not None:
            return True
        if snap.rule is not None:
            self._restore_snapshot_brain(snap, ui_state)
        return False

    def _restore_snapshot_brain(self, snap, ui_state) -> None:
        """Layout first, then the rule."""
        if snap.brain_signature and self.apply_brain_layout is not None:
            live = getattr(getattr(self, "sim", None), "brain_layout", None)
            if live is None or live.signature() != snap.brain_signature:
                from services.brains import layout_from_signature

                layout = layout_from_signature(snap.brain_signature,
                                               snap.brain_settings)
                if layout is not None:
                    # The WINDOW as well as the sim. _handle_brain_layout
                    # applies whatever it finds in ui_state.brain every frame,
                    # so a restore that moves only the sim is put back by the
                    # next one - an undo that appears to work and then keeps
                    # the new brain on top. Applied here as well rather than
                    # left to that frame, because apply_rule below measures the
                    # rule against the LIVE layout and refuses a mismatch.
                    self._put_brain_window(layout, ui_state)
                    self.apply_brain_layout(layout, ui_state)
        rule = np.array(snap.rule, copy=True)
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        self.sim.apply_rule(rule)

    @staticmethod
    def _put_brain_window(layout, ui_state) -> None:
        """Point the Brain window at `layout`, so it stops asking for another.

        Through settings_of, which is what layout_for reads back - the same
        round trip a config load makes, rather than a second opinion about how
        a layout is spelled.
        """
        from services.brains import settings_of

        bst = getattr(ui_state, "brain", None)
        if bst is None:
            return
        bst.modality = layout.modality
        bst.settings = dict(settings_of(layout))

    def _switch_will_apply(self, rule) -> bool:
        """Is a layout switch already going to apply this rule, this frame?

        Its config named another brain, so _handle_brain_layout switches and
        hands it over. Pushing it here first only reaches apply_rule's width
        guard, which prints an 'ignoring' line for a load that then succeeds -
        a load that worked, reported as one that did not.
        """
        pending = self._pending_brain_rule
        if pending is None or rule is None or self.apply_brain_layout is None:
            return False
        lay = getattr(getattr(self, "sim", None), "brain_layout", None)
        return lay is not None and pending[1] != lay.signature()

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

        # Read back the rule (buffer was just written by entity_update).
        # The buffer holds the active brain's floats, so the layout must come
        # from the sim rather than defaulting to Fourier.
        rule = readback_rule(self.sim.get_rule_buffer(), self.sim.brain_layout)
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

        # In Explore mode a right-click on a tile means 'chase this' rather
        # than 'save this tile as a config'. Redirected here, before
        # _handle_auto_tournament clears the flag.
        if ui_state.archive.enabled and self.imgep_driver is not None:
            if ui_state.auto_tournament.save_tile_requested >= 0:
                ui_state.archive.chase_tile = ui_state.auto_tournament.save_tile_requested
                ui_state.auto_tournament.save_tile_requested = -1

        # Brain layout, before anything that runs a generation: it reallocates
        # the GPU buffers and resets the optimizer.
        self._handle_brain_layout(ui_state)

        # Automatic (CLIP-guided) tournament mode
        self._handle_auto_tournament(ui_state)

        # Archive management, before Explore so a switch lands this frame
        self._handle_archive_management(ui_state)

        # Explore (IMGEP) mode
        self._handle_explore(ui_state)

        # Live preview from the archive browser. AFTER _handle_explore, which
        # is what clears the browser's other one-shots, and after both mode
        # handlers so it sees this frame's enabled flags.
        self._handle_archive_preview(ui_state)

        return None

    def _handle_brain_layout(self, ui_state):
        """Apply a requested brain layout, and keep the window's readout live.

        Reads and clears the one-shot flag; App._apply_brain_layout does the
        work, because it owns the archive teardown and rebuild.
        """
        from ui.brain_window import layout_for

        bst = ui_state.brain
        bst.archive_entries = len(self.archive) if self.archive is not None else 0
        bst.best_z = self._active_best_z()

        bst.borrow_active = self._borrow is not None
        self._handle_brain_source(ui_state)

        bst.request_layout_change = False
        if self.apply_brain_layout is None:
            return
        if self._borrow is not None:
            # A preview is borrowing another brain's layout. The window still
            # reads the user's own, so applying it here would switch archive
            # and reset the optimizer to undo the hover - every frame.
            return
        # Called every frame, not only on the flag: a count slider commits on
        # release and a scale slider immediately, and both have to reach the
        # decode. _apply_brain_layout early-returns when nothing differs and
        # takes a light path when only the scales do.
        self.apply_brain_layout(layout_for(bst.modality, bst.settings), ui_state)
        # One frame only. A same-brain load needs no switch, so nothing consumed
        # it, and a rule left here would be applied by the next unrelated one.
        self._pending_brain_rule = None

    def _handle_brain_source(self, ui_state) -> None:
        """The Adopt button and the per-layer operations menu.

        Both are GENOME edits: they go through apply_rule and push onto the rule
        stack, so Z undoes them like anything else, and neither touches the
        layout, the archive or the optimizer.
        """
        from ui.brain_window import BrainWindowMixin

        bst = ui_state.brain
        op, bst.layer_op = bst.layer_op, None
        adopt, bst.adopt_requested = bst.adopt_requested, False
        sim = getattr(self, "sim", None)
        if sim is None:
            return

        kind = BrainWindowMixin.source_kind(bst)
        bst.source_count = self._brain_source_count(ui_state, sim, kind)
        if kind == "tile" or self._borrow is not None:
            # Slot 0 is the grid's, or someone else's for the length of a hover.
            self._layer_base = None
            return

        i = int(bst.source_index)
        if adopt and kind == "cohort":
            params = sim.cohort_brain(i)
            if params is not None:
                self.rule_manager.push_rule(np.asarray(params),
                                            ui_state.sim.rule_seed)
                sim.apply_rule(params)
            return
        if op is None:
            return
        self._apply_layer_op(ui_state, sim, kind, i, op)

    def _brain_source_count(self, ui_state, sim, kind: str) -> int:
        if kind == "cohort":
            from services.brains import MAX_COHORT_BRAINS

            return max(1, min(int(ui_state.sim.num_cohorts or 1),
                              MAX_COHORT_BRAINS))
        if kind == "tile":
            return sim.tournament_grid ** 2
        return 1

    def _current_brain(self, sim, kind: str, i: int):
        """The genome the layer menu is editing, as a flat float32 vector."""
        if kind == "cohort":
            params = sim.cohort_brain(i)
        else:
            params = self.rule_manager.get_current_rule()
        if params is None:
            return None
        flat = np.asarray(params, dtype=np.float32).reshape(-1)
        return flat if flat.size == sim.brain_layout.length else None

    def _put_brain(self, ui_state, sim, kind: str, i: int, params,
                   push: bool) -> None:
        if kind == "cohort":
            sim.write_cohort_brain(i, params)
            return
        if push:
            self.rule_manager.push_rule(np.asarray(params),
                                        ui_state.sim.rule_seed)
        sim.apply_rule(params)

    def _apply_layer_op(self, ui_state, sim, kind: str, i: int, op) -> None:
        """One operation on one layer.

        Three snapshots, because they answer three different questions.
        `_layer_open` is the genome as the menu opened and is what Reset puts
        back. `_layer_base` is what Scale multiplies - rebased after a reroll,
        so the next drag scales what is on screen rather than jumping back past
        it - and multiplying a snapshot rather than the live value is what stops
        a drag compounding. `_layer_live` is what the GPU currently holds, since
        a scale in progress has not been pushed onto the rule stack and so
        cannot be read back from it.
        """
        from services.brains import get

        index, name, arg = op
        layout = sim.brain_layout
        modality = get(layout.modality)
        parts = getattr(modality, "layer_parts", None)
        if parts is None:
            return                      # this modality has no editable layers
        try:
            spans = parts(layout, int(index))
        except (IndexError, ValueError, TypeError):
            return                      # the stack changed under the menu

        if name == "scale_begin":
            self._layer_open = self._layer_base = self._current_brain(
                sim, kind, i)
            self._layer_live = None
            return
        if name == "scale_end":
            base = getattr(self, "_layer_base", None)
            live = getattr(self, "_layer_live", None)
            self._layer_open = self._layer_base = self._layer_live = None
            # ONE history entry for the whole drag, and none at all if the
            # slider never moved.
            if base is not None and live is not None \
                    and not np.array_equal(base, live):
                self._put_brain(ui_state, sim, kind, i, live, push=True)
            return

        base = getattr(self, "_layer_base", None)
        if base is None or base.size != layout.length:
            return
        out = (getattr(self, "_layer_live", None) if name == "scale"
               else self._current_brain(sim, kind, i))
        out = base.copy() if out is None else out.copy()

        if name == "scale":
            for lo, hi in spans.values():
                out[lo:hi] = base[lo:hi] * float(arg)
            self._layer_live = out
            self._put_brain(ui_state, sim, kind, i, out, push=False)
            return
        if name in ("reroll_weights", "reroll_biases"):
            part = "weights" if name.endswith("weights") else "biases"
            lo, hi = spans[part]
            out[lo:hi] = modality.layer_reroll(
                np.random.default_rng(), layout, int(index), part, int(arg or 0))
        elif name == "reset":
            opened = getattr(self, "_layer_open", None)
            if opened is None or opened.size != out.size:
                return
            for lo, hi in spans.values():
                out[lo:hi] = opened[lo:hi]
        else:
            return
        # The next Scale is relative to what is on screen now, not to what was
        # there before the reroll. Reset keeps its own snapshot, or it would be
        # rebased by the very edits it exists to undo.
        self._layer_base = out.copy()
        self._layer_live = None
        self._put_brain(ui_state, sim, kind, i, out, push=True)

    def _active_best_z(self):
        """The best search vector the running driver has found, or None.

        Whichever driver owns the search: Auto's is auto_service.driver, Explore
        swaps its own in. None when no optimizer exists yet, and also when one
        exists but has never been told a fitness - best() answers zeros and -inf
        there, and reporting zeros as an unsaturated genome would be a lie.
        """
        import numpy as np

        drv = getattr(self.auto_service, "driver", None) or self.imgep_driver
        opt = getattr(drv, "optimizer", None)
        if opt is None:
            return None
        z, f = opt.best()
        return z if np.isfinite(f) else None

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

    def _grid_owner(self, ui_state):
        """Who owns the population right now, or None outside a tournament.

        sim.apply_rule writes SLOT 0, which under a tournament is tile 0 and is
        overwritten by the next generation - so the single-brain keys changed
        one square in the corner and nothing else. Under Auto or Explore the
        optimizer owns every tile, so the equivalent of a blank slate there is
        its own reset.
        """
        if not getattr(ui_state.tournament, "enabled", False):
            return None
        auto_on = (getattr(ui_state.auto_tournament, "enabled", False)
                   or getattr(ui_state.archive, "enabled", False))
        if auto_on and self.auto_service is not None:
            return self.auto_service
        return self.tournament_service

    def _handle_full_reset(self, ui_state):
        """Z: a blank slate. Under a tournament that is the whole grid."""
        owner = self._grid_owner(ui_state)
        if owner is not None:
            owner.reset()
            return
        self.sim.reset()
        zero_rule = np.zeros((10, 8), dtype=np.float32)
        self.sim.apply_rule(zero_rule)
        ui_state.sim.rule_seed = random.random()
        self.rule_manager.push_rule(zero_rule, ui_state.sim.rule_seed)

    def _handle_randomize_mutations(self, ui_state):
        """G: a fresh crop of mutations.

        Under a tournament the seed is the whole of it - the rule the slot-0
        write would push belongs to the grid's owner, not to the user.
        """
        ui_state.sim.rule_seed = random.random()
        if self._grid_owner(ui_state) is not None:
            return
        current_rule = self.rule_manager.get_current_rule()
        if current_rule is not None:
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

        # Sync persistent controls. Above the `enabled` early-return, and ahead
        # of _handle_brain_layout in process_commands, so a layout switch this
        # frame reseeds from the CURRENT seed - that switch is what rerolls the
        # tiles, and it happens whether or not the tournament is on screen.
        svc.seed = float(getattr(ui_state.sim, "rule_seed", 0.0) or 0.0)
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
        # Saving is not here: Save Selected opens the shared name dialog, and
        # comes back through _handle_file_save like every other save.

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

    def _rule_fits(self, config) -> bool:
        """Does this config's brain belong to the layout that is running?

        A rule saved under another brain cannot be converted - its floats mean
        different things - so it is dropped rather than reinterpreted.
        sim.apply_rule refuses it too; this exists so that HOVERING the Load
        menu under a different brain does not print one line per config it
        passes over, which is what buried the real errors.

        With no signature the width is all there is to go on. That is not a
        weaker check by choice: files written before brains were swappable are
        Fourier, and a width match is what lets a tile saved by an earlier build
        of this branch still load.
        """
        lay = getattr(getattr(self, "sim", None), "brain_layout", None)
        if lay is None or config is None or getattr(config, "rule", None) is None:
            return True
        sig = getattr(config, "brain_layout", "")
        if sig:
            return sig == lay.signature()
        return int(np.asarray(config.rule).size) == int(lay.length)

    def _brain_layout(self):
        """The layout to stamp on a saved config, or None.

        From the sim, which is the one holder of the layout the rule was
        actually decoded under.
        """
        return getattr(getattr(self, "sim", None), "brain_layout", None)

    def _restore_brain_settings(self, config, ui_state) -> None:
        """Put the Brain window back where the creature was authored.

        Loading a config already moves every physics slider; the brain's
        settings are part of the same preset, and leaving them behind is what
        makes a reloaded creature un-editable. Its rule plays back correctly
        either way - the stored rule is decoded - but re-encoding it under
        different scales pins coordinates at the rails, and a pinned coordinate
        never comes back.

        _handle_brain_layout runs every frame and applies whatever it finds
        here, so writing the state IS applying it. Nothing to request.
        """
        sig = getattr(config, "brain_layout", "")
        bst = getattr(ui_state, "brain", None)
        if not sig or bst is None:
            return                  # pre-modality file: Fourier, nothing to say
        from services.brains import layout_from_signature, settings_of
        from ui.brain_window import layout_for

        settings = dict(getattr(config, "brain_settings", None) or {})
        bst.modality = sig.split("-")[0]
        bst.settings = settings
        got = layout_for(bst.modality, settings).signature()
        if got != sig:
            # The file disagrees with itself - reachable when a modality's
            # defaults move between builds, or when the settings are missing
            # altogether. The SIGNATURE wins, because it is what the rule was
            # decoded under and what apply_rule measures its width against;
            # the settings only supply the decode scales it leaves out.
            named = layout_from_signature(sig, settings)
            if named is None:
                print(f"[brain] {sig} was saved, but this build cannot rebuild "
                      f"it; loading the settings, which give {got}")
            else:
                bst.settings = settings_of(named)
                print(f"[brain] {sig} was saved, but its settings rebuild "
                      f"{got}; loading the signature")

        # Hand the creature to the switch this config just asked for. The rule
        # was already pushed, under the OLD layout, where apply_rule refused it
        # on width - and the switch would then seed a generated brain over it.
        rule = getattr(config, "rule", None)
        if rule is not None:
            self._pending_brain_rule = (
                np.asarray(rule, dtype=np.float32).reshape(-1).copy(), sig)

    def take_pending_brain_rule(self, signature):
        """-> the rule a just-loaded config carried, if it is `signature`'s.

        Consumed, so it can never be applied twice. Matched on SIGNATURE and
        never on width: Fourier at 10 centres and a Lenia layout can both be 80
        floats, over completely different meanings.
        """
        pending, self._pending_brain_rule = self._pending_brain_rule, None
        if pending is None:
            return None
        rule, sig = pending
        return rule if sig == signature else None

    def _save_tournament_selection(self, ui_state, filename):
        """Save each selected genome under the chosen name (tiles get suffixed
        names, see save_targets.target_stems). Uses the tiles from the
        request, not the live selection, so a click while the dialog is open
        can't change what gets written."""
        from services import save_targets

        svc = self.tournament_service
        tiles = [t for t in (ui_state.save_tiles or sorted(svc.selected))
                 if 0 <= t < len(svc.population)]
        if not tiles:
            ui_state.tournament.notice = "Nothing saved: no tiles were selected."
            return
        stems = save_targets.target_stems(
            save_targets.TOURNAMENT_TILE, filename, tiles)
        written = []
        for tile, stem in zip(tiles, stems):
            # The TOURNAMENT's layout, not the sim's: these genomes are its
            # population, and it is the object that bred them.
            config = self.config_saver.create_config(
                ui_state.sim, svc.population[tile], layout=svc.layout)
            filepath = self.user_configs_dir / f"{stem}.json"
            self.config_saver.save_to_file(config, filepath)
            written.append(filepath.name)
        ui_state.tournament.notice = (
            f"Saved {', '.join(written)} to your configs folder "
            f"(File > Load > Custom).")
        print(f"[tournament] saved {written}")

    # ------------------------------------------------------------------
    # Automatic (CLIP-guided) tournament
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_auto_flags(ats):
        """Reset auto one-shot flags after consumption. `warning` is
        persistent and is NOT cleared."""
        ats.start_requested = False
        ats.pause_requested = False
        ats.reset_requested = False
        ats.prompt_changed = False
        ats.grid_changed = False
        ats.save_checkpoint_requested = False
        ats.save_tile_requested = -1
        # pending_save_tile is deliberately NOT cleared here: it travels the
        # other way, and the UI clears it when it opens the dialog.
        ats.load_checkpoint_path = ""
        ats.load_genome_path = ""
        ats.download_model_requested = False

    def _handle_auto_tournament(self, ui_state):
        """Drive the AutoTournamentService from auto one-shot flags."""
        ats = ui_state.auto_tournament

        if ats.download_model_requested:
            self._start_model_download(getattr(ats, "model_key", None))

        svc = self.auto_service
        if svc is None:
            self._clear_auto_flags(ats)
            return

        # Explore drives this same service through a different driver; skip so
        # Auto's settings don't fight _handle_explore over the same fields.
        if self.imgep_driver is not None and svc.driver is self.imgep_driver:
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
        if ats.save_tile_requested >= 0:
            # Not saved here: handed back to the UI so it can ask for a name.
            # Only this side knows Explore has not already claimed the click.
            ats.pending_save_tile = ats.save_tile_requested
        if ats.save_checkpoint_requested:
            self._save_auto_checkpoint(svc, ats)

        ats.running = svc.phase.value == "rollout"
        self._clear_auto_flags(ats)

    # ---- Explore (IMGEP) mode ------------------------------------------

    @staticmethod
    def _clear_explore_flags(ast):
        ast.start_requested = False
        ast.pause_requested = False
        ast.reset_requested = False
        ast.add_goal_requested = False
        ast.remove_goal_index = -1
        ast.move_goal_index = -1
        ast.move_goal_delta = 0
        ast.grid_changed = False
        ast.cancel_expedition_requested = False
        ast.chase_tile = -1
        ast.pin_tile = -1
        ast.seed_entry_id = -1
        ast.delete_entry_id = -1
        ast.refit_projection_requested = False

    @staticmethod
    def _clear_archive_flags(ast):
        ast.switch_archive_name = ""
        ast.new_archive_requested = False
        ast.clear_archive_requested = False
        ast.delete_archive_requested = False
        ast.refresh_archive_list_requested = False

    def _handle_archive_management(self, ui_state):
        """Create, empty, delete and switch archives. Runs BEFORE
        _handle_explore so a switch lands the same frame the button was
        pressed, before the settings push that follows."""
        from services.archive_library import (clear, create, delete,
                                              list_archives)
        from utilities.paths import get_archives_root

        ast = ui_state.archive
        if self.switch_archive is None:
            self._clear_archive_flags(ast)
            return

        root = get_archives_root()
        target = ""
        new_encoder = ""

        if ast.new_archive_requested:
            res = create(root, ast.new_archive_name, ast.new_archive_encoder)
            if res.ok:
                ast.new_archive_name = ""
                target = res.name
                # A bar calibrated in one encoder's space means nothing in
                # another's, and load_settings returns {} for a new archive -
                # so without this it inherits the outgoing one. See CLAUDE.md.
                new_encoder = ast.new_archive_encoder
            else:
                ast.warning = res.message
        elif ast.clear_archive_requested:
            # Release first: Windows refuses to rename a directory with an
            # open handle, and ArchiveStore keeps index.jsonl open for append.
            self._release_current_archive(ui_state)
            res = clear(root, ast.archive_name)
            target = res.name if res.ok else ast.archive_name
            if not res.ok:
                ast.warning = res.message
        elif ast.delete_archive_requested:
            self._release_current_archive(ui_state)
            res = delete(root, ast.archive_name)
            if res.ok:
                remaining = list_archives(root)
                target = remaining[0]["name"] if remaining else ""
            else:
                ast.warning = res.message
                target = ast.archive_name      # reopen what we just closed
        elif ast.switch_archive_name:
            target = ast.switch_archive_name

        if target:
            switched = self.switch_archive(target, ui_state)
            if switched and new_encoder:
                from services.vision_models import REGISTRY

                model = REGISTRY.get(new_encoder)
                if model is not None:
                    ast.min_separation = model.default_min_separation
        elif ast.refresh_archive_list_requested:
            # A switch refreshes the listing itself, so this is only for the
            # Refresh button on its own.
            ast.archive_list = list_archives(root)

        self._clear_archive_flags(ast)

    def _release_current_archive(self, ui_state):
        """Close everything holding the active archive directory, if we can.
        Optional so this still works when no release callback is wired up."""
        if self.release_archive is not None:
            self.release_archive(ui_state)

    def _handle_explore(self, ui_state):
        ast = ui_state.archive
        svc, drv = self.auto_service, self.imgep_driver
        if drv is None or svc is None or svc.driver is not drv:
            self._clear_explore_flags(ast)
            return

        if ast.grid_changed and self.tournament_service is not None:
            self.tournament_service.set_grid(int(ast.grid))
            svc.abort_generation()

        svc.configure(
            steps_per_gen=ast.steps_per_gen,
            snapshots_per_gen=ast.snapshots_per_gen,
            sim_steps_per_frame=ast.sim_steps_per_frame,
            sigma0=ast.sigma0,
            physics_enabled=ast.physics_enabled,
            tile_mutation_enabled=ast.tile_mutation_enabled,
            variants_per_tile=ast.variants_per_tile,
            tile_mutation_strength=ast.tile_mutation_strength,
        )
        for name in ("sigma_expand", "alpha", "k", "seed_n", "liveness_min",
                     "n_views",
                     "refresh_sweep_gens", "expansion_between", "expedition_gens",
                     "expedition_sigma", "latent_share", "novelty_share",
                     "goal_order",
                     "seed_ess_min", "seed_ess_max"):
            setattr(drv, name, getattr(ast, name))
        if self.archive is not None:
            self.archive.capacity = int(ast.capacity)
            self.archive.min_separation = float(ast.min_separation)

        if self.goal_list is not None:
            if ast.add_goal_requested and ast.new_goal_text.strip():
                if self.goal_list.add(ast.new_goal_text):
                    ast.new_goal_text = ""
                self.goal_list.save()
            if ast.remove_goal_index >= 0:
                self.goal_list.remove(ast.remove_goal_index)
                self.goal_list.save()
            if ast.move_goal_index >= 0 and ast.move_goal_delta:
                self.goal_list.move(ast.move_goal_index, ast.move_goal_delta)
                self.goal_list.save()

        if ast.pause_requested:
            svc.pause()
        if ast.reset_requested:
            # Resets the SEARCH. The archive is the product and survives.
            svc.reset()
        if ast.start_requested:
            svc.start()
        # Before chase: a chase in the same frame is a request for a NEW
        # expedition and must not be undone by the cancel.
        if ast.cancel_expedition_requested:
            drv.end_expedition()
        if ast.chase_tile >= 0 and not drv.chase(int(ast.chase_tile)):
            ast.warning = "nothing captured yet - chase needs one generation first"
        if (ast.refit_projection_requested and self.archive is not None
                and self.archive_projection is not None):
            self.archive_projection.fit(self.archive.embeddings)

        if ast.seed_entry_id >= 0:
            self._seed_from_archive(ast)
        if ast.delete_entry_id >= 0:
            self._delete_archive_entry(ast)

        ast.running = svc.phase.value == "rollout"
        if ast.running:
            self._record_run_physics(ui_state, svc.run_id)
        self._clear_explore_flags(ast)

    def _record_run_physics(self, ui_state, run_id):
        """File the physics this run is producing entries under.

        Driven off `running` rather than off start_requested, so a resume, a
        reset and a run already going when the app opened all reach it; the
        store refuses to overwrite, so calling it every frame files exactly one
        config per run. Without it an entry replays under whatever the sliders
        say at browse time, which - with physics search off - is the entry's
        whole physics.
        """
        if self.archive_store is None or not run_id:
            return
        if run_id == self._run_physics_written:
            return
        from services.config_saver import ConfigSaver

        # Stamped with the layout so the run's decode scales survive the round
        # trip: _restore_brain_settings declines a config that names no brain,
        # and the store's own signature directory cannot supply the scales.
        cfg = ConfigSaver().create_config(ui_state.sim, None,
                                          layout=self._brain_layout())
        if self.archive_store.save_run_config(run_id, cfg.to_json()):
            self._run_physics_written = run_id

    def _archive_index(self, row):
        """The browser's one-shots address entries by ROW, not by id.

        An id is unique inside one layout's directory and nowhere else - one
        archive now holds every brain, and two of them each hold an entry 0 -
        so a bare id cannot name an entry here.
        """
        r = int(row)
        return r if 0 <= r < len(self.archive.entries) else None

    def _adopt_foreign_entry(self, ui_state, row) -> bool:
        """Switch to the brain this entry belongs to and run it. -> did it.

        The same handoff a cross-brain config load uses: stash the genome
        against its signature and point the Brain window at that modality. The
        switch itself is performed HERE rather than left to next frame's
        _handle_brain_layout, because a hover has the borrowed layout live and
        the frame between the two would render the previewed brain under the
        user's - a flash of a creature that never existed.

        The layout is rebuilt FROM THE SIGNATURE, not from the modality's
        defaults: an entry saved under gabor-n7 is not reachable by asking
        gabor, which answers gabor-n12. The archive does not store an entry's
        decode scales, so those come from the defaults - the genome is stored
        decoded, so it plays back as authored either way.
        """
        bst = getattr(ui_state, "brain", None)
        if bst is None or self.apply_brain_layout is None:
            return False
        from services.brains import layout_from_signature, settings_of

        sig = self.archive.layout_at(row)
        layout = layout_from_signature(sig)
        if layout is None:
            ui_state.archive.warning = (
                f"#{self.archive.entries[row].id} is a {sig} brain, which this "
                f"build cannot rebuild; switch by hand in the Brain window.")
            return True
        self._pending_brain_rule = (
            np.asarray(self.archive.brain_at(row), dtype=np.float32).copy(), sig)
        bst.modality = layout.modality
        bst.settings = settings_of(layout)
        entry_id = self.archive.entries[row].id
        self.apply_brain_layout(layout, ui_state)
        ui_state.archive.notice = f"Switched to {sig} and loaded #{entry_id}."
        return True

    # ---- borrowing another brain's layout for a preview ----------------

    def _borrow_layout(self, layout, owner) -> bool:
        """Point the sim at `layout` for the duration of a preview. -> did it.

        Only the GPU side moves: the buffer is resized and the modality uniform
        follows the sim's own layout. The archive, the optimizer and the Brain
        window keep the user's brain, so nothing here pays for the teardown a
        real switch does - which is what makes this cheap enough to hover.

        `owner` names the preview holding it, and only that owner may give it
        back. Two of them borrow - the Load menu and the archive gallery - and
        both run every frame, the gallery second.

        Re-borrowing keeps the FIRST base. Sliding down the Load menu across
        two brains borrows twice with no return in between, and remembering the
        second base would restore the user to a layout they never chose.
        """
        current = getattr(self.sim, "brain_layout", None)
        if current is None or not hasattr(self.sim, "realloc_brain_buffers"):
            return False
        if self._borrow is None:
            self._borrow = (owner, current)
        if layout != current:
            self.sim.realloc_brain_buffers(layout)
        return True

    def _holds_borrow(self, owner) -> bool:
        return self._borrow is not None and self._borrow[0] == owner

    def _return_layout(self, owner) -> None:
        """Give the user's own brain back. Must run before the rule under it.

        A no-op unless `owner` is the preview that took it: the gallery's
        teardown runs every frame, and returning a layout it does not own left
        the Load menu's hover undone before the click could keep it.
        """
        if not self._holds_borrow(owner):
            return
        _who, base = self._borrow
        self._borrow = None
        if base != self.sim.brain_layout:
            self.sim.realloc_brain_buffers(base)

    def _config_borrow_layout(self, config):
        """The layout a config must be previewed under, or None to stay put.

        None covers three cases that all mean 'no borrow': a file naming no
        brain (those are Fourier by history), one naming the brain already
        running, and one this build cannot rebuild.

        The config's own `brain_settings` supply the decode SCALES, which a
        signature leaves out - a file has them and an archive entry does not.
        """
        from services.brains import layout_from_signature

        sig = getattr(config, "brain_layout", "")
        live = getattr(getattr(self, "sim", None), "brain_layout", None)
        if not sig or live is None or sig == live.signature():
            return None
        return layout_from_signature(sig, getattr(config, "brain_settings", None))

    def _preview_layout(self, i):
        """The layout entry `i` must be run under, or None if it cannot be.

        Native entries answer None: there is nothing to borrow, and the caller
        distinguishes that from a refusal by asking is_native itself.
        """
        from services.brains import layout_from_signature

        return layout_from_signature(self.archive.layout_at(i))

    def _foreign_notice(self, i) -> str:
        """Why an entry of another brain cannot be decoded here, or ""."""
        if self.archive.is_native(i):
            return ""
        return (f"#{self.archive.entries[i].id} is a "
                f"{self.archive.layout_at(i)} brain; switch to it in the Brain "
                f"window to run or export this one.")

    # ---- live preview of an archive entry -----------------------------

    def _handle_archive_preview(self, ui_state):
        """Run the hovered archive entry in the live sim, and commit a click.

        Outside tournament mode only - there the canvas is a grid of
        simulations and swapping one rule into it would mean nothing.

        Runs on its own rather than inside _handle_explore, which bails as soon
        as the Explore driver is detached; the browser stays usable after that
        and this is the whole point of it.
        """
        ast = ui_state.archive
        tournament = ast.enabled or ui_state.auto_tournament.enabled
        if self.archive is None or tournament:
            self._end_archive_preview(ui_state)
            ast.load_entry_id = -1
            return

        # Ids restart at 0 in every archive, so after a switch the id being
        # previewed names a different creature - or none. Run ids do not
        # restart, but they are looked up in the store that just changed.
        if self._archive_preview_arc is not self.archive:
            self._end_archive_preview(ui_state)
            self._run_config_cache = {}
            self._run_physics_written = ""

        if ast.load_entry_id >= 0:
            # The previewed rule is already on the GPU and on the rule stack;
            # committing is dropping the restore point, not applying anything.
            entry_id = ast.load_entry_id
            ast.load_entry_id = -1
            # The one action that crosses to another brain, because it is
            # explicit. Hand the genome to the switch the way a cross-brain
            # config load does, and let _handle_brain_layout apply both.
            row = self._archive_index(entry_id)
            if row is not None and not self.archive.is_native(row):
                # The borrow ends here and the real switch takes over, so the
                # sim must be holding the user's own layout for it to switch
                # FROM. The physics stay: this is a commit, not a restore.
                self._return_layout("gallery")
                if self._adopt_foreign_entry(ui_state, row):
                    self._archive_preview_id = -1
                    self._archive_preview_physics = None
                    self._archive_preview_pushed = False
                    self._archive_preview_arc = None
                    return
            if self._archive_preview_id != entry_id:
                self._show_archive_preview(ui_state, entry_id)
            self._archive_preview_id = -1
            self._archive_preview_physics = None
            self._archive_preview_pushed = False
            ast.notice = f"Loaded #{entry_id}."
            return

        want = int(ast.preview_entry_id) if ast.live_preview else -1
        if want == self._archive_preview_id:
            return
        self._end_archive_preview(ui_state)
        if want >= 0:
            self._show_archive_preview(ui_state, want)

    def _run_config_for(self, entry):
        """-> the PhysicsConfig this entry's RUN was carried out under, or None
        for a run that predates run configs. Cached: the browser asks once per
        hovered entry, and a run has one config."""
        if self.archive_store is None:
            return None
        run_id = getattr(entry, "run_id", "")
        if run_id not in self._run_config_cache:
            from services.config_saver import PhysicsConfig

            raw = self.archive_store.load_run_config(run_id)
            try:
                cfg = PhysicsConfig.from_json(raw) if raw is not None else None
            except (ValueError, KeyError, TypeError) as exc:
                print(f"[archive] run {run_id}'s physics is unreadable ({exc})")
                cfg = None
            self._run_config_cache[run_id] = cfg
        return self._run_config_cache[run_id]

    def _show_archive_preview(self, ui_state, entry_id):
        """Push entry `entry_id`'s brain, and the physics it actually ran under.

        Two layers, because the search only ever moves part of the physics: the
        RUN's config is the base, and the entry's own vector overrides it for
        the parameters the optimizer searched. With physics search off there is
        no vector and the run config is the entry's physics entirely.
        """
        from services.config_saver import ConfigSaver
        from services.physics_genome import PHYSICS_PARAMS

        i = self._archive_index(entry_id)
        if i is None:
            return
        # Another brain's entry runs under a BORROWED layout - the archive and
        # the search keep the user's own, so a hover costs a buffer resize and
        # nothing else. Silent when the layout cannot be rebuilt: the click says
        # so out loud, and a notice per hovered pixel is not feedback.
        borrowed = None
        if not self.archive.is_native(i):
            borrowed = self._preview_layout(i)
            if borrowed is None:
                return
        sim_state = ui_state.sim
        entry = self.archive.entries[i]
        run_config = self._run_config_for(entry)
        searched = "physics" in entry.spec
        # Snapshotted WHOLE before the first push and restored whole, because a
        # genome is not the same creature under the sliders that happen to be
        # set, and the run config reaches trails and boundaries as well.
        if run_config is not None or searched:
            saver = ConfigSaver()
            if self._archive_preview_physics is None:
                self._archive_preview_physics = saver.create_config(sim_state, None)
            if run_config is not None:
                saver.apply_config(run_config, sim_state)
            if searched:
                for j, (name, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS):
                    setattr(sim_state, name, float(self.archive.physics[i][j]))

        # brain_at, not brains: the pooled column is padded to the WIDEST layout
        # the archive holds, and apply_rule refuses a rule of the wrong width.
        rule = np.asarray(self.archive.brain_at(i), dtype=np.float32)
        pls = self.param_lock_service
        blocked = bool(pls and pls.should_block_rule_push())
        if borrowed is not None and not blocked:
            blocked = not self._borrow_layout(borrowed, "gallery")
        if not blocked:
            self.rule_manager.push_rule(rule, sim_state.rule_seed)
            self.sim.apply_rule(rule)
            self._archive_preview_pushed = True
        self._archive_preview_id = int(entry_id)
        self._archive_preview_arc = self.archive

    def _end_archive_preview(self, ui_state):
        """Put back whatever was running before the preview."""
        if self._archive_preview_id < 0:
            self._return_layout("gallery")
            return
        # Before the rule: what comes off the stack is the user's own brain,
        # and apply_rule measures it against whatever layout is live.
        self._return_layout("gallery")
        if self._archive_preview_pushed:
            prev_rule, prev_seed = self.rule_manager.pop_rule()
            if prev_seed is not None:
                ui_state.sim.rule_seed = prev_seed
            # (None, None) means the preview was the only rule on the stack:
            # there is nothing to go back to, and handing that None to the GPU
            # is not a restore.
            if prev_rule is not None:
                self.sim.apply_rule(prev_rule)
        if self._archive_preview_physics is not None:
            from services.config_saver import ConfigSaver

            ConfigSaver().apply_config(self._archive_preview_physics, ui_state.sim)
        self._archive_preview_id = -1
        self._archive_preview_physics = None
        self._archive_preview_pushed = False
        self._archive_preview_arc = None

    def _export_archive_entry(self, ui_state, entry_id, filename):
        """Write an archive entry as an ordinary Fluoddity config, so it opens
        in the normal single-simulation view at any resolution."""
        from services.genome_io import export_genome
        from services.genome_spec import encode
        from services.physics_genome import PHYSICS_PARAMS

        i = self._archive_index(entry_id)
        if i is None:
            ui_state.archive.warning = (
                f"Entry #{entry_id} is no longer in the archive.")
            return
        e = self.archive.entries[i]
        warn = self._foreign_notice(i)
        if warn:
            # Exporting re-encodes, and re-encoding under the wrong squash
            # writes out a different creature than the one on screen.
            ui_state.archive.warning = warn
            return
        layout = self.archive.layout
        z, _clamped = encode(self.archive.brain_at(i), layout)
        sim_state = ui_state.sim
        if "physics" in e.spec:
            # Archive physics is stored ABSOLUTE, so no origin is needed here.
            for j, (name, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS):
                setattr(sim_state, name, float(self.archive.physics[i][j]))
        meta = {"archive_id": int(e.id), "novelty": float(e.novelty),
                "liveness": float(e.liveness), "source": e.source,
                "goal": e.goal, "run_id": e.run_id, "spec": e.spec}
        path = self.user_configs_dir / f"{filename}.json"
        export_genome(path, z, sim_state, meta, layout=layout)
        print(f"[archive] saved {path}")
        ui_state.archive.notice = (
            f"Saved {path.name} to your configs folder (File > Load > Custom).")

    def _seed_from_archive(self, ast):
        """Load an archive entry as a search starting point. Only Auto mode
        has an x0; Explore draws parents from the archive by novelty."""
        from services.genome_spec import encode

        i = self._archive_index(ast.seed_entry_id)
        if i is None or self.auto_service is None:
            return
        if not hasattr(self.auto_service.driver, "set_x0"):
            ast.warning = ("'Seed a run from here' applies to the Auto (Prompt) "
                           "tab; Explore picks its own parents from the archive.")
            return
        warn = self._foreign_notice(i)
        if warn:
            # An x0 is the optimizer's mean, so it has to be a genome this
            # search can decode and mutate.
            ast.warning = warn
            return
        z, _ = encode(self.archive.brain_at(i), self.archive.layout)
        self.auto_service.set_x0(z)
        ast.notice = (f"Auto mode's search will start from #{ast.seed_entry_id} "
                      "on its next generation.")

    def _delete_archive_entry(self, ast):
        i = self._archive_index(ast.delete_entry_id)
        if i is None:
            return
        self.archive._remove(i)
        self.archive.maybe_flush(force=True)
        if ast.selected_entry_id == ast.delete_entry_id:
            ast.selected_entry_id = -1

    def _start_model_download(self, model_key: str | None = None):
        import threading

        from services.vision_models import DEFAULT_KEY
        from tools.fetch_models import fetch

        key = model_key or DEFAULT_KEY
        threading.Thread(target=fetch, args=(key,), daemon=True).start()
        print(f"[auto] downloading {key} in the background")

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
            z, clamped, _meta = import_genome(ats.load_genome_path,
                                             layout=svc.spec.layout)
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
                                    expect_signature=svc.spec.signature(),
                                    expect_layout=svc.spec.layout.signature())
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

    def _save_auto_checkpoint(self, svc, ats):
        """A checkpoint resumes the OPTIMIZER, so it skips the name dialog
        and configs folder, saving into the run folder instead."""
        from services.run_checkpoint import save_checkpoint

        if svc.logger is None or not svc.logger.enabled:
            ats.warning = ("No run folder yet, so the checkpoint was not "
                           "saved. Start a run first.")
            return
        path = svc.logger.dir / f"checkpoint_gen{svc.generation:06d}.npz"
        save_checkpoint(path, svc.checkpoint_state())
        ats.notice = f"Checkpoint saved to {path} (not a config)."
        print(f"[auto] saved {path}")

    def _save_auto_genome(self, ui_state, filename, tile):
        import copy

        from services.genome_io import export_genome
        from services.physics_genome import decode_physics

        svc = self.auto_service
        ats = ui_state.auto_tournament
        if svc is None:
            ats.warning = "Auto mode is not running; there is nothing to save."
            return
        if tile is None:
            z = svc.optimizer.best()[0] if svc.optimizer is not None else None
        else:
            cz = svc.current_z
            z = cz[tile] if cz is not None and tile < len(cz) else None
        if z is None:
            ats.warning = ("Nothing to save yet - run at least one generation "
                           "first.")
            return

        # export_genome decodes a BRAIN only; with physics search on, z also
        # carries physics genes and must be split first.
        blocks = svc.spec.split(z)
        brain_z = blocks["brain"]

        # Physics travels with the brain, applied to a COPY so Save doesn't
        # move the user's sliders.
        sim_state = ui_state.sim
        if "physics" in blocks:
            sim_state = copy.copy(sim_state)
            for field, value in decode_physics(blocks["physics"],
                                               svc.physics_origin).items():
                setattr(sim_state, field, float(value))

        canvas_px = self.sim.get_canvas_dimensions()[0]
        meta = {
            "generation": svc.generation,
            "prompt": svc.prompt,
            "algorithm": svc.algorithm,
            "physics_search": "physics" in blocks,
            "evolved_at_canvas_px": int(canvas_px),
            "evolved_at_tile_px": 224,
            "evolved_with_tile_mutation": bool(svc.tile_mutation_enabled),
            "mutation_strength": float(svc.tile_mutation_strength),
            "variants_per_tile": int(svc.variants_per_tile),
        }
        path = self.user_configs_dir / f"{filename}.json"
        export_genome(path, brain_z, sim_state, meta, layout=svc.spec.layout)
        ats.notice = (f"Saved {path.name} to your configs folder "
                      f"(File > Load > Custom).")
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
                ui_state.sim, current_rule, field_strengths=field_strengths,
                layout=self._brain_layout())
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

        # Audio rig preset (Audio Reactive panel)
        self._handle_audio_preset(ui_state)

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
        """Every save in the app arrives here, named by the user. Dispatches
        on `kind`; "" is the live configuration. See services/save_targets."""
        from services import save_targets

        filename = ui_state.save_filename
        if not filename:
            return
        kind = getattr(ui_state, "save_kind", save_targets.CONFIG)
        if kind == save_targets.TOURNAMENT_TILE:
            return self._save_tournament_selection(ui_state, filename)
        if kind == save_targets.AUTO_BEST:
            return self._save_auto_genome(ui_state, filename, tile=None)
        if kind == save_targets.AUTO_TILE:
            return self._save_auto_genome(ui_state, filename,
                                          tile=int(ui_state.save_arg))
        if kind == save_targets.ARCHIVE_ENTRY:
            return self._export_archive_entry(ui_state, int(ui_state.save_arg),
                                              filename)
        if kind == save_targets.AUDIO_RIG:
            return self._save_audio_rig(ui_state, filename)
        return self._save_live_config(ui_state, filename)

    def _save_audio_rig(self, ui_state, filename):
        """Write the live rig under a name, and say so on the panel."""
        from services.audio_rig_io import preset_path, save_rig

        path = preset_path(filename)
        ast = ui_state.audio
        if save_rig(ast, path):
            ast.preset_name = path.stem
            ast.notice = f"Rig saved as {path.stem}"
        else:
            ast.warning = f"Could not write {path}"

    def _handle_audio_preset(self, ui_state):
        """Load a named rig over the live one. Cleared before the attempt, so
        a preset that will not read cannot retry every frame."""
        ast = ui_state.audio
        if not ast.request_load_preset:
            return
        ast.request_load_preset = False
        from services.audio_rig_io import load_rig, preset_path

        name = ast.preset_name
        if not name:
            return
        if load_rig(ast, preset_path(name)):
            ast.notice = f"Rig loaded from {name}"
        else:
            ast.warning = f"Could not read the rig '{name}'"

    def _save_live_config(self, ui_state, filename):
        """Handle file save from menu, including field texture PNG."""
        fh = self.field_handler
        current_rule = self.rule_manager.get_current_rule()

        field_data, field_strengths = (
            fh.snapshot_with_strengths(ui_state) if fh else (None, None))

        config = self.config_saver.create_config(
            ui_state.sim, current_rule, field_strengths=field_strengths,
            layout=self._brain_layout())
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
            # Preview already applied config, rule, and field texture - just
            # finalize. EXCEPT the brain: the hover only borrows a layout, so a
            # config naming another one still needs the real switch, and
            # finalizing alone left the preset unloaded while reporting success.
            borrowed = self._holds_borrow("menu")
            self.preview_rule_active = False
            self._preview_rule_was_pushed = False
            if borrowed:
                # Give the layout back first, or the switch has nothing to
                # switch FROM and early-returns.
                self._return_layout("menu")
                config = self.config_saver.load_from_file(
                    self.ui._get_config_path(filename, category))
                if config is not None:
                    # Sets the modality and stashes the creature;
                    # _handle_brain_layout applies both later this frame.
                    self._restore_brain_settings(config, ui_state)
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
                # Before the rule: what comes off the stack is the user's own
                # brain, and apply_rule measures a rule against the live layout.
                self._return_layout("menu")
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

                    # A config naming another brain is previewed under a
                    # BORROWED layout, so hovering the Load menu shows its
                    # creature - the switch itself waits for the click, because
                    # it rebuilds the archive and would fire per menu item.
                    borrow = self._config_borrow_layout(config)
                    if borrow is not None:
                        self._borrow_layout(borrow, "menu")

                    pls = self.param_lock_service
                    if not (pls and pls.should_block_rule_push()) and self._rule_fits(config):
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
                        ui_state.sim, current_rule, layout=self._brain_layout())
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
