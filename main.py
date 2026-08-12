import glfw
import moderngl
import time
import numpy as np
from camera import Camera
from sim import Sim, SIZE_OF_ENTITY_STRUCT
from ui import UI
from services import RuleManager, EntityPicker, VideoRecorderService, ConfigSaver, ArrowDebugService, MultiLoadService, TournamentService
from services.field_handler import FieldHandler
from services.parameter_lock_service import ParameterLockService
from utilities.paths import initialize_user_data, get_user_physics_configs_dir, get_app_physics_configs_dir, get_screenshots_dir
from state import load_preferences, save_preferences, SimState
from command_handler import CommandHandler
from simulation_runner import SimulationRunner
from camera_input import process_camera_input
from controller_input import ControllerCam, process_controller_input, find_joystick
from utilities.advanced_drawing import AdvancedDrawingProcessor


def put_back_auto_overrides(ui_state, prev_aspect, prev_speedmult,
                            prev_motion_blur):
    """Restore the three preferences an automatic mode commandeers.

    Module-level so both the mode-off edge and the quit path can share it.
    """
    prefs = ui_state.preferences
    if prev_aspect and prev_aspect != "1:1":
        prefs.canvas_aspect_ratio = prev_aspect
        ui_state.request_world_size_change = True
    if prev_speedmult is not None:
        prefs.speedmult = prev_speedmult
    if prev_motion_blur is not None:
        prefs.motion_blur = prev_motion_blur


class App:
    """Main application orchestrator.

    Coordinates all components each frame: reads UI state, delegates commands
    to CommandHandler, physics to SimulationRunner, and manages recording/
    screenshot state machines.
    """

    def __init__(self):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        self.window = glfw.create_window(800, 600, "Fluoddity", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync

        # Initialize ModernGL
        self.ctx = moderngl.create_context()
        self.ctx.gc_mode = 'auto'

        # Always on top ONLY FOR WHEN LIVE EDITING THE SHADERS, NOT IN DISTRIBUTION
        #glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)

        # Initialize user data directory (creates Documents/Fluoddity on first run)
        initialize_user_data()

        # Load preferences first to get world_size
        loaded_prefs = load_preferences()

        # Create components (no cross-references between UI and sim/camera)
        self.sim = Sim(self.ctx, world_size=loaded_prefs.world_size, canvas_aspect_ratio=loaded_prefs.canvas_aspect_ratio,
                       particle_density=loaded_prefs.particle_density)
        self.camera = Camera(self.ctx, self.sim, self.window)
        self.ui = UI(self.window, self.ctx, self.sim.view_option_labels)

        # Apply loaded preferences to UI
        self.ui.state.preferences = loaded_prefs
        self.ui._last_applied_world_size = loaded_prefs.world_size
        self.ui._last_applied_particle_density = loaded_prefs.particle_density

        # Create services (Orchestrator owns these)
        self.rule_manager = RuleManager()
        entity_stride = SIZE_OF_ENTITY_STRUCT // 4
        self.entity_picker = EntityPicker(self.sim.get_entity_buffer(), entity_stride)
        self.video_service = VideoRecorderService()
        self.config_saver = ConfigSaver()
        self.arrow_debug_service = ArrowDebugService(self.ctx)
        self.multi_load_service = MultiLoadService()
        self.tournament_service = TournamentService(grid=4)
        # Built lazily on first use of Auto mode - onnxruntime and cmaes must
        # never be imported at startup.
        self.auto_service = None
        self.tile_capture = None
        self.capture_blit = None
        self.capture_view = None
        self.clip_scorer = None
        self._auto_prev_aspect = None
        self._auto_prev_speedmult = None
        self._auto_prev_motion_blur = None
        self._auto_was_enabled = False
        self._last_crops = None
        # Explore (IMGEP) mode, also lazy - it needs the same CLIP scorer.
        self.imgep_driver = None
        self.prompt_driver = None
        self.archive = None
        self.archive_store = None
        self.goal_list = None
        self.archive_projection = None
        self.thumb_cache = None
        # Brain Inspector: None until the window is first opened, False if it
        # could not be built (a diagnostic panel must not take the app down).
        self.brain_preview = None
        self._last_projection_size = 0
        self._explore_was_enabled = False
        self.advanced_drawing_processor = AdvancedDrawingProcessor(self.ctx)
        self.ui.multi_load_service = self.multi_load_service
        self.ui.tournament_service = self.tournament_service
        self.ui.advanced_drawing_processor = self.advanced_drawing_processor

        # Physics configs directories
        self.app_configs_dir = get_app_physics_configs_dir()
        self.user_configs_dir = get_user_physics_configs_dir()
        self.user_configs_dir.mkdir(exist_ok=True)

        # Create delegated handlers
        self.param_lock_service = ParameterLockService()
        self.field_handler = FieldHandler(
            self.advanced_drawing_processor, self.sim,
            param_lock_service=self.param_lock_service)
        self.ui.param_lock_service = self.param_lock_service
        self.command_handler = CommandHandler(
            self.sim, self.camera, self.ui, self.rule_manager,
            self.entity_picker, self.video_service, self.config_saver,
            self.multi_load_service, self.user_configs_dir,
            field_handler=self.field_handler,
            param_lock_service=self.param_lock_service,
            tournament_service=self.tournament_service,
            auto_service=None,  # set by _ensure_auto_service()
        )
        # Archive switching is orchestration, so CommandHandler asks for it
        # rather than reaching into App.
        self.command_handler.switch_archive = self._switch_archive
        self.command_handler.release_archive = self._release_archive
        self.command_handler.apply_brain_layout = self._apply_brain_layout
        # Xbox controller (FPS camera for shader-driven field)
        self.controller_cam = ControllerCam()
        self.joystick_state = {'joystick_id': find_joystick(), 'prev_buttons': []}

        self.sim_runner = SimulationRunner(
            self.sim, self.camera, self.video_service,
            self.command_handler, self.window,
            advanced_drawing_processor=self.advanced_drawing_processor,
            controller_cam=self.controller_cam
        )

        # Frame timing
        self.last_update_time = time.time()

        # Track user's desired settings (for restoration after recording)
        self.user_speedmult = 1
        self.was_recording = False
        self.user_motion_blur = True
        self.user_blur_quality = 1

        # Screenshot state machine
        self.screenshot_pending = False
        self.screenshot_in_progress = False
        self.screenshot_saved_settings = {}

        # Track previous view option for camera repositioning when leaving tiling mode
        self.prev_view_option = 0

        # Track tournament enable edge (to frame the grid when it turns on)
        self._tournament_was_enabled = False

        # Ensure _Default.json exists and load it
        self._ensure_default_config()
        self._load_default_config()
        self.sim.reload()
        self.sim.reset()

    # ------------------------------------------------------------------
    # Automatic (CLIP-guided) tournament
    # ------------------------------------------------------------------

    def _ensure_auto_service(self):
        """Build the CLIP scorer, capture buffer and service on first use.

        Imports stay lazy (onnxruntime/cmaes must not load at startup); Auto
        mode degrades to a message instead of crashing when they're absent.
        """
        if self.auto_service is not None:
            return True
        try:
            from services.auto_tournament_service import AutoTournamentService
            from services.capture_blit import CaptureBlit
            from services.capture_view import CaptureView
            from services.clip_scorer import CLIPScorer
            from services.run_logger import RunLogger
            from services.tile_capture import TileCapture
            from tools.fetch_clip_onnx import MODEL_DIR, is_present
        except ImportError as exc:
            self.ui.auto_unavailable = f"missing package: {exc.name}"
            return False

        if not is_present(MODEL_DIR):
            self.ui.auto_unavailable = "model_missing"
            return False

        try:
            self.clip_scorer = CLIPScorer(MODEL_DIR)
        except Exception as exc:
            self.ui.auto_unavailable = f"could not load CLIP: {exc}"
            return False

        self.tile_capture = TileCapture(self.ctx, self.tournament_service.grid)
        self.capture_blit = CaptureBlit(self.ctx)
        self.capture_view = CaptureView(self.ctx, self.sim, self.camera)
        self.auto_service = AutoTournamentService(
            self.tournament_service,
            scorer=self.clip_scorer,
            logger=RunLogger(config={"grid": self.tournament_service.grid}),
        )
        self.command_handler.auto_service = self.auto_service
        self.ui.auto_service = self.auto_service
        self.ui.auto_unavailable = ""
        return True

    def _build_archive_set(self, path):
        """(Re)build everything that hangs off ONE archive directory, and point
        every holder at it. Keeps the existing ImgepDriver - its sigma, alpha
        and expedition cadence are the user's settings, not the archive's.
        """
        from services.archive import Archive
        from services.archive_io import ArchiveStore
        from services.archive_projection import Projection
        from services.goal_source import GoalList
        from services.thumb_cache import ThumbCache, gl_loader

        from services.archive_io import migrate_archive

        # An archive belongs to ONE brain layout: the floats it stores mean
        # nothing without it. Entries live in <archive>/<signature>/, so
        # switching modality moves to a sibling directory and both survive.
        from services.brains import default_layout

        migrate_archive(path)
        # From the sim when there is one. Falling back rather than requiring it
        # keeps this callable before the sim exists, and from the switch path.
        layout = (getattr(getattr(self, "sim", None), "brain_layout", None)
                  or default_layout())
        store = ArchiveStore(path, layout)
        archive = Archive(store=store, layout=layout)
        loaded, dropped = archive.load_from_store()
        print(f"[archive] {path.name}/{layout.signature()}: "
              f"loaded {loaded} entries ({dropped} dropped)")

        goals = GoalList(store=store)
        goals.load()

        self.archive_store = store
        self.archive = archive
        self.goal_list = goals
        self.archive_projection = Projection()
        self.archive_projection.fit(archive.embeddings)
        self._last_projection_size = len(archive)
        # Keyed by signature, because one archive holds every brain and each
        # of them has a 000000.jpg.
        self.thumb_cache = ThumbCache(gl_loader(self.ctx, archive.stores),
                                      capacity=256)

        if self.imgep_driver is not None:
            self.imgep_driver.archive = archive
            self.imgep_driver.goals = goals
        self.ui.archive_obj = archive
        self.ui.archive_goals = goals
        self.ui.archive_projection = self.archive_projection
        self.ui.thumb_cache = self.thumb_cache
        self.command_handler.archive = archive
        self.command_handler.archive_store = store
        self.command_handler.goal_list = goals
        self.command_handler.archive_projection = self.archive_projection

    def _archive_path_for(self, name, ast):
        """The directory for `name`, falling back to 'default' when it is gone."""
        from services.archive_library import create, resolve, safe_name
        from utilities.paths import DEFAULT_ARCHIVE, get_archives_root

        root = get_archives_root()
        safe = safe_name(name) or DEFAULT_ARCHIVE
        path = resolve(root, safe)
        if path.is_dir():
            return path
        if safe != DEFAULT_ARCHIVE:
            ast.warning = (f"Archive '{safe}' is gone; "
                           f"loaded '{DEFAULT_ARCHIVE}' instead.")
        create(root, DEFAULT_ARCHIVE)
        return resolve(root, DEFAULT_ARCHIVE)

    def _release_archive(self, ui_state):
        """Let go of the archive DIRECTORY: flush, save, close, drop textures.

        Must run before an archive is emptied or deleted - Windows refuses to
        remove a directory with an open handle, and ArchiveStore keeps
        index.jsonl open for append. Flush before closing the store or
        unflushed entries are lost; release the thumbnail cache too, since
        entry ids restart at 0 in every archive.
        """
        ui_state.archive.running = False
        if self.auto_service is not None:
            self.auto_service.pause()
        if self.imgep_driver is not None:
            # The CMA-ES mean was seeded from a parent in the OUTGOING archive.
            self.imgep_driver.end_expedition()
        if self.archive is not None:
            self.archive.maybe_flush(force=True)
        if self.goal_list is not None:
            self.goal_list.save()
        if self.archive_store is not None:
            self.archive_store.close()
        if self.thumb_cache is not None:
            self.thumb_cache.release()

    def _save_archive_settings(self, ui_state):
        """Persist the Explore settings into the archive's own folder.

        Called before anything lets go of an archive - a switch, and quitting.
        """
        if self.archive_store is None:
            return
        self.archive_store.save_settings(ui_state.archive.to_settings())

    def _load_archive_settings(self, ui_state):
        """Restore an archive's settings, and make a restored grid take effect.

        `grid` needs `grid_changed` set too - the tournament grid only
        rebuilds on that flag, not on the per-frame configure() push.
        """
        if self.archive_store is None:
            return
        ast = ui_state.archive
        before = ast.grid
        applied = ast.apply_settings(self.archive_store.load_settings())
        if "grid" in applied and ast.grid != before:
            ast.grid_changed = True

    def _switch_archive(self, name, ui_state):
        """Point the search at a different archive directory. -> success."""
        from services.archive_library import list_archives, resolve, safe_name
        from utilities.paths import get_archives_root

        ast = ui_state.archive
        root = get_archives_root()
        safe = safe_name(name)
        if not safe:
            ast.warning = "That name has no usable characters."
            return False
        path = resolve(root, safe)
        if not path.is_dir():
            ast.warning = f"'{safe}' is no longer on disk."
            ast.archive_list = list_archives(root)
            return False

        # Before the release, while the outgoing store is still open.
        self._save_archive_settings(ui_state)
        self._release_archive(ui_state)
        self._build_archive_set(path)
        # show_browser is one of the restored settings, and the switch can be
        # driven FROM the browser - so an archive last closed with it shut must
        # not make the window vanish under the user. A switch may open the
        # browser; it may never close it.
        was_open = ast.show_browser
        self._load_archive_settings(ui_state)
        ast.show_browser = ast.show_browser or was_open

        ui_state.preferences.archive_name = safe
        ast.archive_name = safe
        ast.archive_list = list_archives(root)
        ast.selected_entry_id = -1
        # Deliberately NOT resumed: the user pressed a management button, not
        # Start.
        return True

    def _render_brain_preview(self, ui_state) -> None:
        """Draw the Inspector atlas, if the Brain window is open.

        Built lazily and never rebuilt: the shader is fixed, only its uniforms
        change with the layout. A failure here must never stop the app - it is a
        diagnostic panel - so it degrades to no texture and the window says so.
        """
        bst = ui_state.brain
        if not bst.enabled:
            return
        if self.brain_preview is None:
            try:
                from services.brain_preview import BrainPreview

                self.brain_preview = BrainPreview(self.ctx)
                self.ui.brain_preview = self.brain_preview
            except Exception as exc:
                print(f"[brain] inspector unavailable ({exc})")
                self.brain_preview = False      # do not retry every frame
                return
        if self.brain_preview is False:
            return

        from services.brain_preview import AXES
        from ui.brain_window import layout_for

        axes = AXES[min(bst.preview_axes, len(AXES) - 1)][1]
        layout = layout_for(bst.modality, bst.settings)
        # Slot 0 is always a real brain now - the loaded rule, or cohort 0's
        # generated one - so the Inspector just draws it. It used to have to
        # re-derive the shader's "is this buffer blank" verdict, and when it did
        # not, it drew the blank buffer: black tiles while the particles ran.
        bst.preview_per_cohort = self.sim.brain_per_cohort
        bst.preview_tile0 = self.sim.tournament_enabled
        try:
            self.ui.brain_preview_tex = self.brain_preview.render(
                layout,
                self.sim.multi_load_rule_buffer,
                axes=axes,
                channel=bst.preview_channel,
                value_range=bst.preview_range,
                gain=bst.preview_gain,
                seed=bst.preview_seed,
            )
        except Exception as exc:
            print(f"[brain] inspector render failed ({exc})")
            self.ui.brain_preview_tex = None
            self.brain_preview = False

    def _refresh_driver_specs(self, layout, reset: bool = False) -> None:
        """Point every driver's genome spec at `layout`.

        reset=True only when the WIDTH changed: a decode-scale change leaves the
        search dimension and the archive intact, so throwing away the optimizer's
        covariance would cost the run for nothing.
        """
        from services.genome_spec import physics_spec_for, spec_for

        for drv in (getattr(self.auto_service, "driver", None),
                    self.imgep_driver):
            if drv is None:
                continue
            physics = bool(getattr(drv, "physics_enabled", False))
            spec = physics_spec_for(layout) if physics else spec_for(layout)
            if hasattr(drv, "set_spec"):
                drv.set_spec(spec)
            if reset and hasattr(drv, "reset"):
                drv.reset()

    def _apply_brain_layout(self, layout, ui_state) -> bool:
        """Switch the brain layout. A hard reset of the search, never partial.

        The archive changes because its directory is keyed by the layout
        signature, so a layout change IS an archive switch - to a sibling
        directory under the same archive name. It therefore runs the same
        sequence a name switch does, rather than an inline copy of it: the
        settings are saved while the outgoing store is still open, and
        _release_archive is what flushes, closes and drops the thumbnails.
        """
        from services.archive_library import resolve
        from services.genome_spec import physics_spec_for, spec_for
        from utilities.paths import get_archives_root

        current = self.sim.brain_layout
        if layout == current:
            if tuple(layout.scales) == tuple(current.scales):
                return False
            # SCALES ONLY. What a z means changed, but not how wide it is, so
            # the archive stays valid (it stores decoded brains) and the
            # optimizer keeps its covariance. Refresh the decode and stop -
            # no teardown, no reallocation.
            self.sim.set_brain_scales(layout)
            self.tournament_service.set_layout(layout)
            if self.auto_service is not None:
                self.auto_service.set_layout(layout)
            self._refresh_driver_specs(layout)
            return True

        # Before the release, while the outgoing store is still open.
        self._save_archive_settings(ui_state)
        self._release_archive(ui_state)

        # The GPU side first: the per-particle readback buffer is sized by the
        # active length, and slot 0 is re-uploaded from whatever rule is live.
        self.sim.realloc_brain_buffers(layout)
        # The old genome's floats mean something else under a new layout, so it
        # is dropped - UNLESS a config naming this very layout was loaded in the
        # same frame, because the switch is that config's own doing and its
        # creature is what the user asked for. The load runs first and pushes
        # the rule while the OLD brain is still live, so apply_rule refuses it
        # on width; without this handoff the switch then buries it under a
        # generated brain. Otherwise apply_rule(None) seeds what "no rule"
        # means: one generated brain per cohort, for every modality alike.
        pending = None
        if self.command_handler is not None:
            pending = self.command_handler.take_pending_brain_rule(
                layout.signature())
        self.sim.apply_rule(pending)

        # The interactive tournament breeds genomes of the layout it is told
        # about; without this it keeps producing the old width and the tiles are
        # uploaded into slots that expect the new one.
        self.tournament_service.set_layout(layout)
        if self.auto_service is not None:
            self.auto_service.set_layout(layout)

        # The optimizer searches a different number of dimensions now, so its
        # covariance and population are meaningless. Reset rather than resize.
        self._refresh_driver_specs(layout, reset=True)

        if self.archive is not None or self.archive_store is not None:
            path = resolve(get_archives_root(),
                           ui_state.preferences.archive_name)
            self._build_archive_set(path)
            # settings.json lives inside the signature directory, so the
            # incoming layout has its own. As in _switch_archive, the switch
            # may open the browser but must never close it under the user.
            ast = ui_state.archive
            was_open = ast.show_browser
            self._load_archive_settings(ui_state)
            ast.show_browser = ast.show_browser or was_open
        return True

    def _open_archive(self, ui_state):
        """Load the archive directory into memory, if it is not already.

        Deliberately CLIP-FREE: browsing, the map and live preview need the
        entries, their thumbnails and the projection, none of which involve the
        scorer. Building the ONNX sessions to open a window would cost a second
        for nothing, and would fail outright wherever the optional packages are
        missing - where browsing still ought to work.
        """
        if self.archive is not None:
            return
        from services.archive_library import list_archives
        from utilities.paths import get_archives_root

        ast = ui_state.archive
        path = self._archive_path_for(ui_state.preferences.archive_name, ast)
        self._build_archive_set(path)
        # First open of the session: this is what makes relaunching the app and
        # reopening an archive show that archive's settings rather than the
        # class defaults.
        self._load_archive_settings(ui_state)
        ast.archive_name = path.name
        ast.archive_list = list_archives(get_archives_root())
        ui_state.preferences.archive_name = path.name

    def _open_archive_browser(self, ui_state):
        """Extras > Archive Browser: show the gallery with no tournament mode.

        show_browser is set AFTER the archive's settings load, because it is
        one of the settings - restoring an archive last closed would otherwise
        shut the window the user just asked for.
        """
        self._open_archive(ui_state)
        ui_state.archive.show_browser = True

    def _ensure_archive_service(self, ui_state):
        """Build the archive, goal list and IMGEP driver on first use.

        Reuses the same AutoTournamentService as Auto mode, swapping only
        its driver. Imports stay lazy.
        """
        if not self._ensure_auto_service():
            self.ui.archive_unavailable = self.ui.auto_unavailable
            return False
        # Idempotent, and it may already have happened: the browser opens the
        # archive on its own, without ever building a driver.
        self._open_archive(ui_state)
        if self.imgep_driver is not None:
            return True

        from services.imgep_driver import ImgepDriver

        # The archive, its name and the settings are _open_archive's job.
        self.imgep_driver = ImgepDriver(
            self.tournament_service, self.clip_scorer,
            self.archive, self.goal_list)
        self.prompt_driver = self.auto_service.driver

        self.ui.archive_driver = self.imgep_driver
        self.ui.archive_service = self.auto_service
        self.ui.archive_unavailable = ""
        self.command_handler.imgep_driver = self.imgep_driver
        return True

    def _capture_tiles(self, ui_state):
        """Render the tournament grid into the square capture FBO.

        Re-renders the canvas at grid*224 with an identity camera, so the
        displayed pan/zoom/window size cannot affect what the optimizer scores.
        """
        from services.tile_capture import TILE_PX

        kwargs = getattr(self.sim_runner, "last_assemble_kwargs", None)
        if kwargs is None:
            return None            # nothing rendered yet this session
        grid = self.tournament_service.grid
        tex = self.capture_view.render(ui_state, kwargs, grid * TILE_PX)
        if tex is None:
            return None

        self.tile_capture.resize(grid)
        crops = self.tile_capture.capture(
            lambda fbo: self.capture_view.draw_grid(
                fbo, tex, grid, self.capture_blit, ui_state, TILE_PX)
        )
        self.ctx.screen.use()
        width, height = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, width, height)
        return crops

    def _drive_auto_tournament(self, ui_state):
        """Advance the auto loop one frame. Returns the number of physics steps
        the simulation should run this frame."""
        from services.auto_tournament_service import Action

        svc = self.auto_service
        action = svc.update()

        if action is Action.WRITE_RULES:
            self.sim.write_tournament_rules(self.tournament_service.pack_rule_bytes())
            self.tournament_service.clear_dirty()
            # Every tile in a generation shares this seed for a fair comparison;
            # it changes between generations.
            if svc.tile_physics:
                self.sim.write_tournament_physics(svc.tile_physics)
            self.sim.reset_seed = float(svc.gen_seed)
            self.sim.reset()
            return 0
        if action is Action.CAPTURE:
            crops = self._capture_tiles(ui_state)
            if crops is not None:
                self._last_crops = crops
                self.command_handler.report_capture_health(crops, ui_state)
                svc.submit_frames(crops)
            return 0
        if action is Action.SCORE:
            fit = svc.score_and_tell()
            # None while the CLIP pass runs off-thread; retry next frame.
            if fit is not None:
                self._after_generation(fit)
            return 0
        if action is Action.STEP:
            return max(1, int(svc.sim_steps_per_frame))
        return 1

    def _after_generation(self, fit):
        """Periodic best-tile frame dump and checkpoint autosave."""
        from services.run_checkpoint import save_checkpoint

        svc = self.auto_service
        gen = svc.generation

        # Refit every 500 admissions, not per frame (Projection.fit sign-aligns
        # to the previous components, so the map doesn't mirror itself). Also
        # refit once while unfitted so a cold archive doesn't wait 500 entries.
        proj = self.archive_projection
        if self.archive is not None and proj is not None:
            grown = len(self.archive) - self._last_projection_size
            if grown >= 500 or (not proj.fitted and len(self.archive) > 2):
                proj.fit(self.archive.embeddings)
                self._last_projection_size = len(self.archive)
        if self.imgep_driver is not None:
            g = getattr(self.imgep_driver, "_goal", None)
            self.ui.archive_goal_point = g.embedding if g is not None else None

        log = svc.logger
        if log is None or not log.enabled:
            return
        if self._last_crops is not None and gen % 10 == 0:
            log.save_frame(self._last_crops[int(np.argmax(fit))], gen)
        if svc.autosave_every and gen % svc.autosave_every == 0:
            save_checkpoint(log.dir / f"checkpoint_gen{gen:06d}.npz",
                            svc.checkpoint_state())

    def _ensure_default_config(self):
        """Ensure _Default.json exists in physics_configs directory. Create it if missing."""
        default_path = self.app_configs_dir / "Core/_Default.json"
        if not default_path.exists():
            default_state = SimState()
            zero_rule = np.zeros((10, 8), dtype=np.float32)
            config = self.config_saver.create_config(default_state, zero_rule)
            self.config_saver.save_to_file(config, default_path)
            print(f"Created default config: {default_path}")

    def _load_default_config(self):
        """Load _Default.json on startup."""
        default_path = self.app_configs_dir / "Core/_Default.json"
        config = self.config_saver.load_from_file(default_path)
        if config is not None:
            rule = self.config_saver.apply_config(config, self.ui.state.sim)
            self.rule_manager.push_rule(rule, self.ui.state.sim.rule_seed)
            self.sim.apply_rule(rule)
            print(f"Loaded default config from {default_path}")
            self.ui.update_physics_defaults("_Default")
        else:
            print(f"Failed to load default config from {default_path}")

    def run(self):
        """The frame loop. Wrapped in try/finally so a crash still logs its
        traceback and runs cleanup() to flush the archive and settings."""
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.orchestrate_frame()
                glfw.swap_buffers(self.window)
        except BaseException:
            # Before cleanup, in case cleanup dies too on a lost GL context.
            self._write_crash_log()
            raise
        finally:
            self._cleanup_safely()

    def _write_crash_log(self) -> None:
        """Record the traceback where an overnight run can still find it."""
        import traceback

        text = traceback.format_exc()
        print(text)
        try:
            from utilities.paths import get_user_data_dir

            path = get_user_data_dir() / "crash.log"
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {stamp} =====\n{text}")
            print(f"[crash] traceback appended to {path}")
        except Exception as exc:      # a logging failure must not mask the bug
            print(f"[crash] could not write the crash log ({exc})")

    def _cleanup_safely(self) -> None:
        """Run cleanup(); a failure in it must not mask the real exception
        (a lost GL context makes every GL call raise)."""
        try:
            self.cleanup()
        except BaseException as exc:
            print(f"[cleanup] failed during shutdown ({exc!r})")

    def orchestrate_frame(self):
        """Main orchestration logic - reads UI state, coordinates components."""

        # 1. Get current UI state
        ui_state = self.ui.get_state()
        tiling_mode = (ui_state.sim.current_view_option == 3)

        # 1.5. Auto-mode enable edge. MUST run before process_commands, which
        # clears the one-shot start_requested flag. Also forces square tiles -
        # CLIP needs square input.
        auto = ui_state.auto_tournament
        if auto.enabled and not self._auto_was_enabled:
            self._save_auto_overrides(ui_state)
            self._ensure_auto_service()
        elif not auto.enabled and self._auto_was_enabled:
            self._undo_auto_overrides(ui_state)
            if self.auto_service is not None:
                self.auto_service.pause()
        self._auto_was_enabled = auto.enabled

        # Extras > Archive Browser. Before process_commands, which is where the
        # browser's own flags are read, and cleared first so a failure to open
        # cannot retry every frame.
        if ui_state.archive.open_browser_requested:
            ui_state.archive.open_browser_requested = False
            self._open_archive_browser(ui_state)

        # Explore mode reuses Auto mode's rollout machine and capture path;
        # only the driver is swapped.
        expl = ui_state.archive
        if expl.enabled and not self._explore_was_enabled:
            self._save_auto_overrides(ui_state)
            if self._ensure_archive_service(ui_state):
                self.auto_service.driver = self.imgep_driver
                self.auto_service.abort_generation()
        elif not expl.enabled and self._explore_was_enabled:
            if self.auto_service is not None and self.prompt_driver is not None:
                self.auto_service.pause()
                self.auto_service.driver = self.prompt_driver
            if self.archive is not None:
                self.archive.maybe_flush(force=True)
            self._undo_auto_overrides(ui_state)
        self._explore_was_enabled = expl.enabled

        # 2. Process one-shot commands
        result = self.command_handler.process_commands(ui_state, tiling_mode)
        if result == 'screenshot_pending' and not self.screenshot_pending and not self.screenshot_in_progress:
            self.screenshot_pending = True

        # 2.5. Brain Inspector atlas. Rendered HERE rather than in the mixin
        # because the UI is passive - it places the texture, it does not draw
        # into GPU targets. Only while the window is open.
        self._render_brain_preview(ui_state)

        # 3. Process continuous input (camera movement)
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        process_camera_input(ui_state, self.window, self.ui.keybindings,
                             self.sim.view_tex, dt)
        process_controller_input(self.controller_cam, self.joystick_state, dt)

        # 3.2. Check if pending video should start
        cmd = self.command_handler
        if cmd.video_pending and self.sim.frame_count >= cmd.video_scheduled_start_frame:
            cmd.video_pending = False
            cmd.video_scheduled_start_frame = 0
            self.video_service.start()

        # 3.5. Screenshot state machine
        if self.screenshot_pending and not self.screenshot_in_progress:
            self.screenshot_pending = False
            self.screenshot_in_progress = True
            self.screenshot_saved_settings = {
                'speedmult': ui_state.preferences.speedmult,
                'blur_quality': ui_state.preferences.blur_quality,
                'motion_blur': ui_state.preferences.motion_blur,
                'going': ui_state.sim.going,
            }
            ui_state.preferences.speedmult = ui_state.preferences.motion_blur_samples
            ui_state.preferences.blur_quality = 1
            ui_state.preferences.motion_blur = True
            if not ui_state.sim.going:
                ui_state.sim.going = True

        # 4. Lock physics frequency to video recorder frequency if recording
        is_recording = self.video_service.is_active()

        if is_recording and not self.was_recording:
            self.user_speedmult = ui_state.preferences.speedmult
            self.user_motion_blur = ui_state.preferences.motion_blur
            self.user_blur_quality = ui_state.preferences.blur_quality
        elif not is_recording and self.was_recording:
            ui_state.preferences.speedmult = self.user_speedmult
            ui_state.preferences.motion_blur = self.user_motion_blur
            ui_state.preferences.blur_quality = self.user_blur_quality

        if is_recording:
            ui_state.preferences.speedmult = ui_state.preferences.motion_blur_samples
            ui_state.preferences.motion_blur = ui_state.preferences.recording_motion_blur
            ui_state.preferences.blur_quality = ui_state.preferences.recording_blur_quality

        self.was_recording = is_recording

        # 5. Apply state to components
        if ui_state.request_camera_reset:
            ui_state.camera.position[:] = [0.0, 0.0]
            ui_state.camera.zoom = 1.0
        self.sim.apply_state(ui_state.sim)
        self.sim.apply_camera_state(ui_state.camera)
        self.camera.apply_state(ui_state.camera)
        self.multi_load_service.apply_state(ui_state.multi_load)
        _auto_svc = self.auto_service
        # Auto and Explore are mutually exclusive; either counts as "running".
        _auto_on = ui_state.auto_tournament.enabled or ui_state.archive.enabled
        _tile_mut = (
            _auto_svc.tile_mutation_strength
            if (_auto_svc is not None
                and _auto_on
                and _auto_svc.tile_mutation_enabled)
            else 0.0
        )
        self.sim.apply_tournament(
            ui_state.tournament.enabled,
            grid=self.tournament_service.grid,
            mutation=_tile_mut,
            # Cohort colouring gives each tile a fixed palette by slot rather
            # than genome, which confounds ranking.
            plain_colour=(_auto_svc is not None and _auto_on),
            # Must cover every tile: a tile with no physics block written reads
            # a zeroed config and renders black (e.g. when the grid grows
            # between generations before tile_physics catches up).
            physics=(_auto_svc is not None
                     and _auto_on
                     and _auto_svc.physics_enabled
                     and len(_auto_svc.tile_physics) >= self.tournament_service.tiles),
        )
        if _auto_svc is not None and _auto_on:
            # z=0 must mean "the preset as loaded", not the midpoint of every
            # slider - the midpoint has no axial force and no drag.
            from services.physics_genome import PHYSICS_PARAMS
            _auto_svc.physics_origin = {
                n: float(getattr(ui_state.sim, n, 0.0))
                for n, _g, _lo, _hi in PHYSICS_PARAMS
            }
        if _tile_mut > 0.0:
            # Cohorts must be a multiple of the tile count or each tile holds
            # exactly one cohort and stays a monoculture. See cohort_tiling.
            from services.cohort_tiling import cohorts_for
            ui_state.sim.num_cohorts = cohorts_for(
                self.tournament_service.grid, _auto_svc.variants_per_tile
            )
        self.camera.BRIGHTNESS = ui_state.preferences.brightness
        self.camera.trail_overlay_strength = ui_state.preferences.trail_overlay_strength

        # 5.0.1 Force/Strafe field view modes: override view_tex with field texture
        if ui_state.sim.current_view_option in (4, 5):
            field_tex = self.advanced_drawing_processor.field_texture
            if field_tex is not None:
                self.sim.view_tex = field_tex
            else:
                # Field texture not initialized yet — fall back to canvas view
                ui_state.sim.current_view_option = 0

        # 5.1. Multi-load conflict prevention
        if ui_state.multi_load.multi_load_enabled:
            ui_state.sim.parameter_sweeps_enabled = False
            ui_state.preferences.mouse_mode = "Draw Trail"
            if self.param_lock_service.enabled:
                self.param_lock_service.reset()
                ui_state.preferences.parameter_locks_enabled = False

        # 5.1.5. Tournament conflict prevention
        # The tiled view replicates the canvas infinitely, which is incoherent with a
        # 16-tile grid and desyncs click->tile mapping. Force a plain camera view, and
        # frame the whole grid when tournament is first switched on.
        if ui_state.tournament.enabled:
            if ui_state.sim.current_view_option == 3:
                ui_state.sim.current_view_option = 2
                ui_state.camera.cam_brush_mode = True
                # Suppress the "leaving tiling mode" camera fmod below, which would
                # otherwise clobber the framing reset we are about to apply.
                self.prev_view_option = 2
            if not self._tournament_was_enabled:
                ui_state.camera.position[:] = [0.0, 0.0]
                ui_state.camera.zoom = 1.0
        self._tournament_was_enabled = ui_state.tournament.enabled

        # Auto mode drives the physics step count via speedmult, which the
        # SimulationRunner already honours.
        auto_running = ((auto.enabled or ui_state.archive.enabled)
                        and self.auto_service is not None
                        and ui_state.tournament.enabled)
        if auto_running:
            steps = self._drive_auto_tournament(ui_state)
            ui_state.preferences.speedmult = max(0, steps)

        # 5.2. Sync parameter lock master toggle
        self.param_lock_service.enabled = ui_state.preferences.parameter_locks_enabled

        # 5.5. Calculate sweep reticle info
        sweep_reticle_x, sweep_reticle_y, sweep_reticle_visible = self.sim.get_sweep_reticle_position()

        if is_recording or self.screenshot_in_progress:
            sweep_reticle_visible = False

        width, height = glfw.get_framebuffer_size(self.window)
        screen_aspect = width / height if height > 0 else 1.0

        if sweep_reticle_visible:
            screen_x, screen_y = self.camera.tex_to_screen(
                (sweep_reticle_x, sweep_reticle_y),
                self.sim.view_tex.size
            )
            sweep_reticle_x = screen_x / width
            sweep_reticle_y = screen_y / height

        sweep_mode = ui_state.sim.parameter_sweeps_enabled
        sweep_reticle_pos = (sweep_reticle_x, sweep_reticle_y)

        # Reposition camera when leaving tiling mode
        if self.prev_view_option == 3 and ui_state.sim.current_view_option != 3:
            ui_state.camera.position[0] = np.fmod(ui_state.camera.position[0] + 100.0, 2.0) - 1.0
            ui_state.camera.position[1] = np.fmod(ui_state.camera.position[1] + 100.0, 2.0) - 1.0
        self.prev_view_option = ui_state.sim.current_view_option

        # 6. Run simulation if going
        if ui_state.sim.going:
            self.sim_runner.run_simulation_frame(
                ui_state, sweep_mode, sweep_reticle_pos, sweep_reticle_visible,
                screen_aspect, ui_state.sim.watercolor_mode,
                tiling_mode=tiling_mode,
                screenshot_in_progress=self.screenshot_in_progress
            )

        # 6.5. Screenshot save and settings restoration
        if self.screenshot_in_progress:
            self._save_screenshot(ui_state)

        # 7. Render camera view
        self._render_camera_view(ui_state, sweep_mode, sweep_reticle_pos,
                                  sweep_reticle_visible, screen_aspect, tiling_mode)

        # 7.5. Render arrow debug overlay if enabled
        if ui_state.preferences.debug_arrows:
            width, height = glfw.get_framebuffer_size(self.window)
            adv_prefs = ui_state.preferences
            adv_active = adv_prefs.advanced_drawing_enabled
            field_tex = self.advanced_drawing_processor.field_texture
            # Use field_texture for force/strafe targets, canvas for trails
            if adv_active and not adv_prefs.advanced_draw_canvas and field_tex is not None:
                arrow_texture = field_tex
                arrow_resolution = field_tex.size
                use_zw = adv_prefs.advanced_draw_strafe_field
            else:
                arrow_texture = self.sim.can
                arrow_resolution = self.sim.can.size
                use_zw = False
            self.arrow_debug_service.render(
                canvas_texture=arrow_texture,
                cam_pos=tuple(self.camera.position),
                cam_zoom=self.camera.zoom,
                canvas_resolution=arrow_resolution,
                window_size=(width, height),
                arrow_sensitivity=ui_state.preferences.arrow_sensitivity,
                use_zw_channels=use_zw,
            )

        # 8. Update UI display info and render
        self.ui.update_display_info({
            'time': self.sim.time,
            'frame_count': self.sim.frame_count,
            'tex_size': self.sim.view_tex.size,
            'recording_active': self.video_service.is_active(),
            'video_pending': cmd.video_pending,
            'video_scheduled_start_frame': cmd.video_scheduled_start_frame,
        })
        self.ui.render()

    def _render_camera_view(self, ui_state, sweep_mode, sweep_reticle_pos,
                             sweep_reticle_visible, screen_aspect, tiling_mode):
        """Render the camera view to screen."""
        emboss_mode = ui_state.sim.emboss_mode
        if emboss_mode == 1:
            emboss_tex = self.sim.can
        elif emboss_mode == 2:
            emboss_tex = self.sim.brush_tex
        else:
            emboss_tex = None

        draw_trail_mode = ui_state.preferences.mouse_mode == "Draw Trail"

        width, height = glfw.get_framebuffer_size(self.window)
        mouse_x_norm = ui_state.mouse_pos[0] / width if width > 0 else 0.5
        mouse_y_norm = ui_state.mouse_pos[1] / height if height > 0 else 0.5
        mouse_screen_coords = (mouse_x_norm, mouse_y_norm)

        self.camera.render(
            sim_going=ui_state.sim.going,
            current_view_option=ui_state.sim.current_view_option,
            sweep_mode=sweep_mode,
            sweep_reticle_pos=sweep_reticle_pos,
            sweep_reticle_visible=sweep_reticle_visible,
            screen_aspect=screen_aspect,
            watercolor_mode=ui_state.sim.watercolor_mode,
            ink_weight=ui_state.sim.ink_weight,
            emboss_tex=emboss_tex,
            emboss_mode=emboss_mode,
            emboss_intensity=ui_state.sim.emboss_intensity,
            emboss_smoothness=ui_state.sim.emboss_smoothness,
            draw_trail_mode=draw_trail_mode,
            draw_size=ui_state.preferences.draw_size,
            mouse_screen_coords=mouse_screen_coords,
            exposure=ui_state.preferences.exposure,
            tiling_mode=tiling_mode,
            tonemap_softness=ui_state.preferences.tonemap_softness,
            bloom_enabled=ui_state.preferences.bloom_enabled,
            bloom_threshold=ui_state.preferences.bloom_threshold,
            bloom_intensity=ui_state.preferences.bloom_intensity,
            bloom_radius=ui_state.preferences.bloom_radius
        )

    def _save_screenshot(self, ui_state):
        """Save screenshot and restore settings."""
        if self.camera.assembled_texture is not None:
            from utilities.save_frame_gpu import save_frame_gpu
            import datetime
            import os
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = ui_state.preferences.filename_prefix or "screenshot"
            filename = save_frame_gpu(
                self.camera.assembled_texture,
                self.ctx,
                supersample_k=ui_state.preferences.supersample_k,
                return_array=False
            )
            if filename and os.path.exists(filename):
                screenshots_dir = get_screenshots_dir()
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                new_filename = screenshots_dir / f"{prefix}_{timestamp}.png"
                os.rename(filename, new_filename)
                print(f"Screenshot saved: {new_filename}")

        # Restore saved settings
        ui_state.preferences.speedmult = self.screenshot_saved_settings['speedmult']
        ui_state.preferences.blur_quality = self.screenshot_saved_settings['blur_quality']
        ui_state.preferences.motion_blur = self.screenshot_saved_settings['motion_blur']
        ui_state.sim.going = self.screenshot_saved_settings['going']
        self.screenshot_in_progress = False
        self.screenshot_saved_settings = {}

    def _save_auto_overrides(self, ui_state):
        """Remember the preferences an automatic mode is about to commandeer,
        then commandeer them. Shared by Auto and Explore - both drive the same
        rollout machine and need the same three conditions."""
        prefs = ui_state.preferences
        self._auto_prev_aspect = prefs.canvas_aspect_ratio
        self._auto_prev_speedmult = prefs.speedmult
        self._auto_prev_motion_blur = prefs.motion_blur

        # Tiles inherit the canvas aspect ratio, and a 16:9 tile cannot be
        # squared for CLIP without distortion, padding or lost content.
        if prefs.canvas_aspect_ratio != "1:1":
            prefs.canvas_aspect_ratio = "1:1"
            ui_state.request_world_size_change = True

        # Motion blur averages several renders into the texture _capture_tiles
        # grabs, smearing away the fine structure that distinguishes genomes.
        prefs.motion_blur = False

    def _undo_auto_overrides(self, ui_state):
        put_back_auto_overrides(
            ui_state, self._auto_prev_aspect, self._auto_prev_speedmult,
            self._auto_prev_motion_blur)

    def _restore_auto_overrides(self, ui_state):
        """Undo the transient auto/explore overrides before anything is
        persisted. Quitting while a mode is still enabled never crosses the
        mode->off edge that normally restores them, and a persisted
        speedmult of 0 would leave the next launch never stepping the sim.
        Checks both modes since either can commandeer the same preferences.
        """
        if not (ui_state.auto_tournament.enabled or ui_state.archive.enabled):
            return
        put_back_auto_overrides(
            ui_state, self._auto_prev_aspect, self._auto_prev_speedmult,
            self._auto_prev_motion_blur)

    @staticmethod
    def _step(label, fn, *args, **kwargs):
        """Run one shutdown step; log and continue if it raises, so a failing
        step (e.g. a lost GL context) doesn't skip the ones after it."""
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            print(f"[cleanup] {label} failed ({exc!r}); continuing")
            return None

    def cleanup(self):
        # Ordered by what is lost if a step doesn't run: archive writes first
        # (pure numpy/disk), GLFW teardown last (touches the context most
        # likely to be broken).
        ui_state = self._step("read ui state", self.ui.get_state)

        if self.archive is not None:
            self._step("flush archive", self.archive.maybe_flush, force=True)
        if self.goal_list is not None:
            self._step("save goals", self.goal_list.save)
        if ui_state is not None:
            # Before the store closes.
            self._step("save archive settings",
                       self._save_archive_settings, ui_state)
            self._step("restore auto overrides",
                       self._restore_auto_overrides, ui_state)
            self._step("save preferences", save_preferences, ui_state.preferences)

        # The scoring thread only reads its own copy of the frame buffer, so
        # closing it after the flush cannot race the archive.
        if self.auto_service is not None:
            self._step("close scoring thread", self.auto_service.close)
        if self.archive_store is not None:
            self._step("close archive store", self.archive_store.close)
        if self.thumb_cache is not None:
            self._step("release thumbnails", self.thumb_cache.release)

        self._step("advanced drawing", self.advanced_drawing_processor.cleanup)
        self._step("video", self.video_service.cleanup)
        self._step("ui", self.ui.cleanup)
        self._step("glfw", glfw.terminate)


if __name__ == "__main__":
    app = App()
    app.run()
