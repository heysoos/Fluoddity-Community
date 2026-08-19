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
from state.audio_in_state import to_dict as rig_to_dict
from command_handler import CommandHandler
from simulation_runner import SimulationRunner
from camera_input import process_camera_input
from controller_input import ControllerCam, process_controller_input
from utilities.advanced_drawing import AdvancedDrawingProcessor
from state import view_modes


# The ceiling an automatic mode holds the hue gain to while it is scoring.
AUTO_HUE_MAX = 0.12


def put_back_auto_overrides(ui_state, prev_aspect, prev_speedmult,
                            prev_motion_blur, prev_hue=None):
    """Restore the preferences an automatic mode commandeers.

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
    if prev_hue is not None:
        ui_state.sim.hue_sensitivity = prev_hue


def _release(cache):
    """Release a thumbnail cache if there is one. Both cleanup steps go
    through this so the lookup can sit inside the guarded step."""
    if cache is not None:
        cache.release()


def clamp_auto_hue(ui_state):
    """Hold the hue gain at or below AUTO_HUE_MAX while a mode is scoring.

    Called every frame rather than on the mode-enable edge: loading a preset
    mid-run writes that preset's own gain into SimState. See CLAUDE.md.
    """
    if not (ui_state.auto_tournament.enabled or ui_state.archive.enabled):
        return
    ui_state.sim.hue_sensitivity = min(
        ui_state.sim.hue_sensitivity, AUTO_HUE_MAX)


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
        self._restore_open_windows(loaded_prefs)
        self.ui._last_applied_world_size = loaded_prefs.world_size
        self.ui._last_applied_particle_density = loaded_prefs.particle_density

        # Create services (Orchestrator owns these)
        self.rule_manager = RuleManager()
        from services.undo_history import UndoHistory

        self.undo_history = UndoHistory()
        self._undo_preview_base = None
        self._undo_preview_showing = -1
        entity_stride = SIZE_OF_ENTITY_STRUCT // 4
        self.entity_picker = EntityPicker(self.sim.get_entity_buffer(), entity_stride)
        self.video_service = VideoRecorderService()
        self.config_saver = ConfigSaver()
        self.arrow_debug_service = ArrowDebugService(self.ctx)
        self.multi_load_service = MultiLoadService()
        self.tournament_service = TournamentService(grid=4)
        from services.audio_runtime import AudioRuntime
        self.audio_runtime = AudioRuntime()
        from services.audio_rig_io import load_rig
        load_rig(self.ui.state.audio)
        # The rig file is shared by every copy of Fluoddity, and the write at
        # exit used to be unconditional - so a second instance that never
        # touched audio wrote its empty rig over the one you had just built.
        # A session writes only what it changed. See _save_last_rig.
        self._rig_at_start = rig_to_dict(self.ui.state.audio)
        # Built lazily on first use of Auto mode - onnxruntime and cmaes must
        # never be imported at startup.
        self.auto_service = None
        self.tile_capture = None
        self.capture_blit = None
        self.capture_view = None
        self.vision_scorer = None
        self._auto_prev_aspect = None
        self._auto_prev_speedmult = None
        self._auto_prev_motion_blur = None
        self._auto_prev_hue = None
        self._auto_was_enabled = False
        self._last_crops = None
        # Explore (IMGEP) mode, also lazy - it needs the same vision scorer.
        self.imgep_driver = None
        self.prompt_driver = None
        self.archive = None
        self.archive_store = None
        self.goal_list = None
        self.map_layout_service = None
        self.thumb_cache = None
        self.atlas_cache = None
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
        # Discovered on the first frame that reads it, never here: the scan
        # initialises GLFW's joystick backend, which is a device enumeration.
        self.joystick_state = {'joystick_id': None, 'prev_buttons': []}

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

    def _restore_open_windows(self, prefs):
        """Reopen the windows that were on screen when the app last closed.

        Only the two windows whose flag lives outside PreferencesState need
        anything here. The browser's open is a one-shot request, asked for
        once at construction: repeating it would reload the archive every
        frame, and that reload rescores every entry.
        """
        self.ui.state.audio.show_window = prefs.show_audio_window
        if prefs.show_archive_browser:
            self.ui.state.archive.show_browser = True
            self.ui.state.archive.open_browser_requested = True

    # ------------------------------------------------------------------
    # Automatic (vision-guided) tournament
    # ------------------------------------------------------------------

    def _ensure_auto_service(self):
        """Build the vision scorer, capture buffer and service on first use.

        Imports stay lazy (onnxruntime/cmaes must not load at startup); Auto
        mode degrades to a message instead of crashing when they're absent.
        """
        if self.auto_service is not None:
            return True
        try:
            from services.auto_tournament_service import AutoTournamentService
            from services.capture_blit import CaptureBlit
            from services.capture_view import CaptureView
            from services.run_logger import RunLogger
            from services.tile_capture import TileCapture
            from services.vision_models import DEFAULT_KEY
            from services.vision_scorer import VisionScorer
            from tools.fetch_models import is_present
        except ImportError as exc:
            self.ui.auto_unavailable = f"missing package: {exc.name}"
            return False

        key = getattr(self.ui.get_state().auto_tournament, "model_key",
                      DEFAULT_KEY)
        if not is_present(key):
            self.ui.auto_unavailable = "model_missing"
            return False

        if not self._ensure_scorer(key):
            return False

        self.tile_capture = TileCapture(self.ctx, self.tournament_service.grid)
        self.capture_blit = CaptureBlit(self.ctx)
        self.capture_view = CaptureView(self.ctx, self.sim, self.camera)
        self.auto_service = AutoTournamentService(
            self.tournament_service,
            scorer=self.vision_scorer,
            logger=RunLogger(config={"grid": self.tournament_service.grid}),
        )
        self.command_handler.auto_service = self.auto_service
        self.ui.auto_service = self.auto_service
        self.ui.auto_unavailable = ""
        return True

    def _follow_auto_encoder(self, ui_state):
        """Make Auto's Encoder combo take effect while its tab is open.

        Per frame rather than on the enable edge, because the combo sits in a
        tab that is already open by the time it can be touched - read once, it
        is a control that does nothing. Costs one string compare until the key
        actually moves.

        Explore's encoder belongs to its archive and is a readout, so this
        stands down whenever Explore owns the driver.
        """
        svc = self.auto_service
        if svc is None or not ui_state.auto_tournament.enabled:
            return
        if self.imgep_driver is not None and svc.driver is self.imgep_driver:
            return
        want = ui_state.auto_tournament.model_key
        current = getattr(self.vision_scorer, "model", None)
        if current is not None and current.key == want:
            return

        from tools.fetch_models import is_present

        if not is_present(want):
            self.ui.auto_unavailable = "model_missing"
            return
        if not self._ensure_scorer(want):
            return
        self.ui.auto_unavailable = ""
        # The goal was embedded by the outgoing encoder, and the two spaces are
        # not comparable - at 512 against 768 the score is not even a shape
        # error until the first tile arrives.
        prompt = ui_state.auto_tournament.prompt.strip()
        if prompt:
            svc.set_prompt(prompt)
        # A generation half-scored in one space and half in another ranks
        # nothing, so the one in flight is thrown away rather than finished.
        svc.abort_generation()

    def _ensure_scorer(self, model_key: str) -> bool:
        """Make `model_key` the resident encoder. -> is it loaded?

        One at a time, replaced rather than stacked: keeping every encoder
        loaded would cost about a gigabyte of weights for a switch that happens
        once per archive. Auto and Explore are mutually exclusive, so the
        resident one follows whichever is scoring.

        Built before anything is reassigned, so a failure leaves the previous
        encoder in place rather than the app scoring with nothing.
        """
        import services.vision_scorer as vs

        current = getattr(self, "vision_scorer", None)
        if current is not None and current.model.key == model_key:
            return True
        try:
            built = vs.VisionScorer(model_key)
        except Exception as exc:
            self.ui.auto_unavailable = f"could not load encoder: {exc}"
            return False
        self.vision_scorer = built
        # EVERY driver by name. AutoTournamentService owns no scorer - its
        # `scorer` is a property forwarding to whichever driver is installed -
        # so assigning through the service reaches one of the two and leaves
        # the other holding the encoder that was just replaced. Which one
        # misses depends on the mode that happened to be running, and at equal
        # width (clip-b32 and clip-b16 are both 512) a foreign vector is
        # silently wrong rather than an error.
        for holder in (getattr(self, "prompt_driver", None),
                       getattr(self, "imgep_driver", None)):
            if holder is not None:
                holder.scorer = built
        if getattr(self, "auto_service", None) is not None:
            self.auto_service.scorer = built
        return True

    def _build_archive_set(self, path):
        """(Re)build everything that hangs off ONE archive directory, and point
        every holder at it. Keeps the existing ImgepDriver - its sigma, alpha
        and expedition cadence are the user's settings, not the archive's.
        """
        from services.archive import Archive
        from services.archive_io import ArchiveStore
        from services.goal_source import GoalList
        from services.map_layout_service import MapLayoutService
        from services.thumb_cache import ThumbCache, gl_loader
        from ui.archive_window import ATLAS_CACHE_CAPACITY, ATLAS_TEX_PX

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
        # The archive's own encoder, not whatever happens to be resident: its
        # stored vectors are only comparable to that one.
        archive = Archive(store=store, layout=layout, encoder=store.encoder)
        loaded, dropped = archive.load_from_store()
        print(f"[archive] {path.name}/{layout.signature()} [{store.encoder}]: "
              f"loaded {loaded} entries ({dropped} dropped)")

        goals = GoalList(store=store)
        goals.load()

        self.archive_store = store
        self.archive = archive
        self.goal_list = goals
        # ONE service across every switch - it owns a worker thread - and it
        # is rebound rather than rebuilt. Entry ids restart in each archive, so
        # bind() is what drops the previous archive's positions.
        if self.map_layout_service is None:
            self.map_layout_service = MapLayoutService()
        self.map_layout_service.bind(archive, path / "map_layout.npz",
                                     store.encoder)
        # Keyed by signature, because one archive holds every brain and each
        # of them has a 000000.jpg.
        self.thumb_cache = ThumbCache(gl_loader(self.ctx, archive.stores),
                                      capacity=256)
        # The map's own cache. Its textures are decoded small, so it can hold
        # a picture for EVERY cell where the gallery's 160px ones cannot -
        # which is the difference between an atlas that fills in and one that
        # goes sparse and re-decodes itself.
        self.atlas_cache = ThumbCache(
            gl_loader(self.ctx, archive.stores, max_px=ATLAS_TEX_PX),
            capacity=256, max_capacity=ATLAS_CACHE_CAPACITY)

        if self.imgep_driver is not None:
            self.imgep_driver.archive = archive
            self.imgep_driver.goals = goals
        self.ui.archive_obj = archive
        self.ui.archive_goals = goals
        self.ui.map_layout_service = self.map_layout_service
        self.ui.thumb_cache = self.thumb_cache
        self.ui.atlas_cache = self.atlas_cache
        self.command_handler.archive = archive
        self.command_handler.archive_store = store
        self.command_handler.goal_list = goals
        self.command_handler.map_layout_service = self.map_layout_service

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
        index.jsonl open for append. EVERY layout's store, not just the
        running one: an archive holds one per signature directory, and any one
        of them left open holds the whole folder. Flush before closing or
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
            self.archive.maybe_flush(force=True, closing=True)
        if self.goal_list is not None:
            self.goal_list.save()
        if self.archive is not None:
            self.archive.close()
        # Also directly: the archive may have failed to build, and this handle
        # is the app's own.
        if self.archive_store is not None:
            self.archive_store.close()
        # Both caches: entry ids restart at 0 in every archive and the key
        # derives from the id, so a kept texture shows the previous archive's
        # picture.
        _release(self.thumb_cache)
        _release(getattr(self, "atlas_cache", None))

    def _save_archive_settings(self, ui_state):
        """Persist the Explore settings into the archive's own folder.

        Called before anything lets go of an archive - a switch, and quitting.
        """
        if self.archive_store is None:
            return
        # From the SIM, not from whatever was last put in the field: this is a
        # readout of the brain the archive is on, and a stale one files the
        # layout it just left.
        layout = getattr(getattr(self, "sim", None), "brain_layout", None)
        if layout is not None:
            ui_state.archive.layout_signature = layout.signature()
        settings = ui_state.archive.to_settings()
        self.archive_store.save_settings(settings)
        # A change made after the last generation would otherwise never reach
        # the log, since the per-generation hook has stopped firing.
        if self.archive is not None:
            gen = int(getattr(self.imgep_driver, "gen", 0) or 0)
            self.archive.record_settings(settings, gen)

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

    def _restore_archive_layout(self, ui_state) -> bool:
        """Put back the brain this archive was last searched under. -> did it.

        Called only from the paths that OPEN an archive. Never from
        _apply_brain_layout, which WRITES the setting this reads - calling one
        from the other would put the outgoing layout straight back.

        A signature this build cannot rebuild keeps the current brain and says
        so: a plausible layout of the wrong width is worse than refusing.
        """
        from command_handler import CommandHandler
        from services.brains import layout_from_signature

        ast = ui_state.archive
        sig = str(ast.layout_signature or "")
        live = getattr(getattr(self, "sim", None), "brain_layout", None)
        if not sig or live is None or sig == live.signature():
            return False
        layout = layout_from_signature(sig)
        if layout is None:
            ast.warning = (
                f"this archive was searched under {sig}, which this build "
                f"cannot rebuild; the brain is unchanged.")
            return False
        # The WINDOW as well as the sim: _handle_brain_layout applies whatever
        # it finds in ui_state.brain every frame, so a restore that moves only
        # the sim is undone by the next one.
        CommandHandler._put_brain_window(layout, ui_state)
        self._apply_brain_layout(layout, ui_state)
        return True

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
        self._restore_archive_layout(ui_state)

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

        axes = AXES[min(bst.preview_axes, len(AXES) - 1)][1]
        # The sim's layout, not the window's: they agree except while a preview
        # borrows another brain's, and drawing slot 0 through the wrong field
        # map is a picture of a creature that is not running.
        layout = self.sim.brain_layout
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
                slot=self._brain_preview_slot(bst),
            )
        except Exception as exc:
            print(f"[brain] inspector render failed ({exc})")
            self.ui.brain_preview_tex = None
            self.brain_preview = False

    @staticmethod
    def _brain_preview_slot(bst) -> int:
        """Which slot of the flat brain buffer the Inspector draws.

        Slot 0 holds the loaded rule, and tile 0 under a tournament; the cohort
        slots hold one generated brain each when no rule is loaded, which is the
        only case where there is a choice to make.
        """
        from services.brains import COHORT_BRAIN_SLOT0, MAX_COHORT_BRAINS
        from ui.brain_window import BrainWindowMixin

        kind = BrainWindowMixin.source_kind(bst)
        i = max(0, int(bst.source_index))
        if kind == "cohort":
            return COHORT_BRAIN_SLOT0 + min(i, MAX_COHORT_BRAINS - 1)
        return i if kind == "tile" else 0

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

    def _apply_brain_layout(self, layout, ui_state, *,
                            keep_running: bool = False) -> bool:
        """Switch the brain layout. A hard reset of the SEARCH, never of the
        archive.

        One archive holds every layout, so this re-points it rather than
        switching to a sibling: the entries, their embeddings, their novelty
        and their thumbnails are all about pictures and survive the change.
        Only which rows are native moves. See
        docs/superpowers/specs/2026-08-17-brain-layout-search-design.md.

        `keep_running` is for a layout the SEARCH asked for. It moved its own
        space on purpose and has an expedition ready to start in the new one,
        so the pause and the optimizer reset a hand switch needs would cost the
        whole run.
        """
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

        # Said out loud because it redirects where a run's results are filed:
        # entries admitted after this land in a different signature directory
        # and the previous brain's stop being reachable as parents. The
        # archive's own settings record which brain it ended on, but not that
        # the change happened, nor when.
        running = (getattr(ui_state.archive, "running", False)
                   or getattr(ui_state.auto_tournament, "running", False))
        print(f"[brain] layout {current.signature()} -> {layout.signature()}"
              + (" WHILE A SEARCH IS RUNNING" if running else ""))

        # A layout change is an archive RE-POINT, not an archive switch: every
        # layout's entries are already in memory and only which of them are
        # native differs. The search still stops - its space just moved - but
        # the teardown that used to come with that does not.
        if not keep_running:
            ui_state.archive.running = False
            if self.auto_service is not None:
                self.auto_service.pause()

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
        self._refresh_driver_specs(layout, reset=not keep_running)

        if self.archive is not None:
            self.archive.retarget(layout)
        # AFTER the switch, so what is written names the layout the archive is
        # now on rather than the one it just left.
        self._save_archive_settings(ui_state)
        return True

    def _apply_requested_layout(self, ui_state) -> bool:
        """Honour a layout the SEARCH asked for, in the frame it asked.
        -> did anything move?

        The driver cannot switch brain itself - App owns the archive, the sim
        and the tournament - so it sets a one-shot and this reads it. The
        expedition the move was proposed for starts AFTERWARDS, because a
        genome of the child's width cannot be optimised under the parent's
        spec; begin_moved_expedition verifies the switch landed and drops the
        move if it did not.
        """
        from command_handler import CommandHandler

        drv = getattr(self, "imgep_driver", None)
        layout = getattr(drv, "requested_layout", None)
        if drv is None or layout is None:
            return False
        # Consumed whatever happens below: left standing it would be re-applied
        # every generation forever.
        drv.requested_layout = None
        applied = self._apply_brain_layout(layout, ui_state, keep_running=True)
        # The WINDOW as well as the sim: _handle_brain_layout applies whatever
        # it finds in ui_state.brain every frame, so a switch that moves only
        # the sim is undone by the next one.
        CommandHandler._put_brain_window(layout, ui_state)
        drv.begin_moved_expedition()
        return applied

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
        # After the settings, which is where layout_signature arrives.
        self._restore_archive_layout(ui_state)
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
        # The archive's stored vectors are only comparable to the encoder that
        # made them, so the search adopts it. Every call, not just the first:
        # switching archive can switch encoder. Deliberately NOT in
        # _build_archive_set, which the browser also reaches - opening the
        # gallery must not pay for an ONNX session. See CLAUDE.md.
        if self.archive_store is not None:
            from tools.fetch_models import is_present

            ui_state.archive.encoder_key = self.archive_store.encoder
            # Gated the same way Auto's combo is. Without it the ONE encoder
            # nobody chose - it comes off encoder.json - is the ONE that
            # answers a missing download with a raw ONNX error and no way out
            # of the tab.
            if not is_present(self.archive_store.encoder):
                self.ui.archive_unavailable = "model_missing"
                return False
            if not self._ensure_scorer(self.archive_store.encoder):
                self.ui.archive_unavailable = self.ui.auto_unavailable
                return False
        # Cleared HERE and not only where the driver is built, or an archive
        # whose encoder was missing leaves the banner - and so the dead tab -
        # standing after a switch to one whose encoder is on disk.
        self.ui.archive_unavailable = ""
        if self.imgep_driver is not None:
            return True

        from services.imgep_driver import ImgepDriver

        # The archive, its name and the settings are _open_archive's job.
        self.imgep_driver = ImgepDriver(
            self.tournament_service, self.vision_scorer,
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
                # Before the rest: everything below reports what is live NOW,
                # and a layout move that landed has changed it.
                self._apply_requested_layout(ui_state)
                self._after_generation(fit)
                self._record_settings_version(ui_state)
            return 0
        if action is Action.STEP:
            return max(1, int(svc.sim_steps_per_frame))
        return 1

    def _close_map_layout(self):
        svc = getattr(self, "map_layout_service", None)
        if svc is not None:
            svc.shutdown()

    def _update_map_layout(self, ui_state):
        """Decide whether the map needs laying out again, once per frame.

        Per frame rather than per generation: browsing an archive with no
        search running still has to draw a map, and the engine combo has to
        take effect when it is moved rather than at the next generation.
        """
        svc = self.map_layout_service
        if self.archive is None or svc is None:
            return
        svc.configure(ui_state.archive.map_layout)
        svc.update(self.archive)

    def _record_settings_version(self, ui_state):
        """One settings version per generation, and only if something moved.

        Per generation rather than per frame: the diff is cheap but a row per
        frame would bury the timeline it exists to make readable.
        """
        if self.archive is None:
            return
        gen = int(getattr(self.imgep_driver, "gen", 0) or 0)
        self.archive.record_settings(ui_state.archive.to_settings(), gen)

    def _after_generation(self, fit):
        """Periodic best-tile frame dump and checkpoint autosave."""
        from services.run_checkpoint import save_checkpoint

        svc = self.auto_service
        gen = svc.generation

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
        tiling_mode = (ui_state.sim.current_view_option == view_modes.CAMERA_TILED)

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
        self._follow_auto_encoder(ui_state)
        self._update_map_layout(ui_state)

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
                self.archive.maybe_flush(force=True, closing=True)
            self._undo_auto_overrides(ui_state)
        self._explore_was_enabled = expl.enabled

        # The encoder readout follows the archive, never the other way round.
        # Archive-level, so the count is every layout under it - which is what
        # load_from_store already loaded.
        if self.archive is not None:
            expl.archive_entry_count = len(self.archive)
            expl.encoder_key = self.archive.encoder
        # Read on the open edge only: the log grows with the run and this tab
        # redraws every frame.
        if expl.request_history_reload:
            expl.request_history_reload = False
            expl.history_rows = (self.archive_store.load_history()
                                 if self.archive_store is not None else [])

        # Undo/redo, before the commands: a restored preset must be applied in
        # the same frame the key was pressed. Before the soundtrack push too,
        # so a restored recording preference takes effect in its own frame.
        self._handle_undo(ui_state)

        # 1.9. The soundtrack choice, pushed BEFORE process_commands, which is
        # where the record toggle starts a take. Recording also starts from the
        # scheduled-start check below, which is why this is a per-frame push
        # rather than an argument at either call.
        self.video_service.configure(self.audio_runtime.capture,
                                     ui_state.preferences.record_audio,
                                     ui_state.preferences.record_audio_delay)

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
        # The controller camera is only read by the shader-driven field, so
        # that is the one condition worth a joystick scan for.
        adv = ui_state.preferences
        process_controller_input(
            self.controller_cam, self.joystick_state, dt,
            active=adv.advanced_drawing_enabled and adv.shader_driven_field)

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
            # With a soundtrack the physics rate is the user's, not the capture
            # frequency: blur samples ARE physics sub-steps, so forcing it up
            # would render far below real time. The audio still sets the file's
            # framerate, so sync holds at whatever rate is achieved - this only
            # decides whether the result is smooth or blurred.
            if not ui_state.preferences.record_audio:
                ui_state.preferences.speedmult = ui_state.preferences.motion_blur_samples
            ui_state.preferences.motion_blur = ui_state.preferences.recording_motion_blur
            ui_state.preferences.blur_quality = ui_state.preferences.recording_blur_quality

        if self.was_recording and not is_recording:
            message = self.video_service.take_message()
            if message:
                ui_state.preferences.record_notice = message

        self.was_recording = is_recording

        # 5. Apply state to components
        if ui_state.request_camera_reset:
            ui_state.camera.position[:] = [0.0, 0.0]
            ui_state.camera.zoom = 1.0
        # After process_commands, so a preset loaded this frame is capped in
        # the same frame it arrives.
        clamp_auto_hue(ui_state)
        # Audio modulates a COPY. ui_state.sim keeps what the user set, so the
        # sliders do not drift and Save writes slider values rather than
        # whatever the music was doing at that instant.
        _audio_sim, _audio_brain = self.audio_runtime.update(
            ui_state, dt, self.sim.brain_layout,
            self.rule_manager.get_current_rule())
        self.sim.apply_state(_audio_sim)
        if _audio_brain is not None:
            self.sim.apply_rule(_audio_brain)
        # The panel draws the modulation inside each slider's own track.
        self.ui.audio_overlays = self.audio_runtime.overlays(
            ui_state, _audio_sim)
        self.sim.apply_camera_state(ui_state.camera)
        self.camera.apply_state(ui_state.camera)
        self.multi_load_service.apply_state(ui_state.multi_load)
        # The step reflects the state the frame actually ran under. BEFORE the
        # preview, whose writes would otherwise commit as steps of their own.
        self._record_undo_step(ui_state)
        self._handle_undo_preview(ui_state)
        self._push_undo_rows(ui_state)
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
        if ui_state.sim.current_view_option in view_modes.FIELD_VIEWS:
            field_tex = self.advanced_drawing_processor.field_texture
            if field_tex is not None:
                self.sim.view_tex = field_tex
            else:
                # Field texture not initialized yet — fall back to canvas view
                ui_state.sim.current_view_option = view_modes.CANVAS

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
            if ui_state.sim.current_view_option == view_modes.CAMERA_TILED:
                ui_state.sim.current_view_option = view_modes.CAMERA
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
        if (self.prev_view_option == view_modes.CAMERA_TILED
                and ui_state.sim.current_view_option != view_modes.CAMERA_TILED):
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
        self._auto_prev_hue = ui_state.sim.hue_sensitivity

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
            self._auto_prev_motion_blur,
            prev_hue=getattr(self, "_auto_prev_hue", None))

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
            self._auto_prev_motion_blur,
            prev_hue=getattr(self, "_auto_prev_hue", None))

    def _save_last_rig(self, ui_state):
        """Write the last-used rig, but only if this session changed it.

        Several copies of the app share the one file, so an untouched instance
        must leave it exactly as it found it.
        """
        from services.audio_rig_io import rig_path, save_rig

        current = rig_to_dict(ui_state.audio)
        if current == getattr(self, "_rig_at_start", None):
            return False
        if save_rig(ui_state.audio):
            return True
        print(f"[cleanup] the audio rig could not be written to {rig_path()}")
        return False

    def _handle_undo(self, ui_state):
        """Ctrl+Z / Ctrl+Shift+Z, and a click in the history panel."""
        from services import undo_history as uh

        # A hover puts someone else's rule and physics on screen, and the
        # un-hover puts back what was there before it - wholesale. An undo
        # applied underneath one is therefore reverted a moment later with no
        # sign that anything happened, and the rebase below has already
        # overwritten the step it undid. Refuse, and say why.
        if self.command_handler.preview_active:
            if (ui_state.request_undo or ui_state.request_redo
                    or ui_state.undo_jump_index >= 0):
                ui_state.undo_notice = "Finish the preview first"
            ui_state.undo_jump_index = -1
            return

        target = None
        if ui_state.request_undo:
            target = self.undo_history.undo()
            if target is None:
                ui_state.undo_notice = "Nothing to undo"
        elif ui_state.request_redo:
            target = self.undo_history.redo()
            if target is None:
                ui_state.undo_notice = "Nothing to redo"
        elif ui_state.undo_jump_index >= 0:
            target = self.undo_history.jump(ui_state.undo_jump_index)
        ui_state.undo_jump_index = -1
        if target is None:
            return

        skipped = self.command_handler.apply_undo_snapshot(target, ui_state)
        ui_state.undo_notice = (
            f"{target.label} - settings only, the grid's owner keeps the brain"
            if skipped else target.label)
        # Applying a step ENDS any preview: the state is now deliberately this
        # one, so there is nothing to hand back, and restoring what was on
        # screen before the hover would silently undo the click.
        self._undo_preview_base = None
        self._undo_preview_showing = ui_state.undo_preview_index
        # Re-baseline, or the next frame reads this restore as a fresh change.
        self.undo_history.rebase(
            uh.capture(ui_state, self.rule_manager.get_current_rule(),
                       self.sim.brain_layout))

    def _preferences_are_borrowed(self, ui_state) -> bool:
        """Is something other than the user driving the render preferences?

        One predicate rather than a check per site, for the same reason
        CommandHandler.preview_active is one: a fourth borrower added later is
        covered by naming it HERE.
        """
        return bool(ui_state.auto_tournament.enabled
                    or ui_state.archive.enabled
                    or self.screenshot_in_progress
                    or self.video_service.is_active())

    def _record_undo_step(self, ui_state):
        """Commit a step if anything declared changed and no widget is active."""
        from services import undo_history as uh

        if ui_state.any_widget_active:
            return
        # A preview BORROWS a step; it is not a change, and recording it is a
        # runaway - the commit shifts the rows under the pointer, so the next
        # row previews and commits in turn. Covers the frame the preview is
        # handed back too, which is the one that clears the base.
        if self._undo_preview_base is not None:
            return
        # The undo panel is not the only thing that hovers. File > Load, the
        # archive browser and the clipboard all put a borrowed rule and its
        # physics on screen the same way.
        if self.command_handler.preview_active:
            return
        # Nor is a preference the user did not set. An automatic mode, a
        # recording and a screenshot each commandeer speedmult, motion_blur
        # and blur_quality - all three undoable - and put them back when they
        # finish, so the journal saw a slider move every time the state
        # machine changed phase. A generation deposits several, and the cap is
        # 200 steps: leaving Explore running discarded every real step the
        # user had made, and half the survivors restore speedmult = 0.
        if self._preferences_are_borrowed(ui_state):
            return
        snap = uh.capture(ui_state, self.rule_manager.get_current_rule(),
                          self.sim.brain_layout)
        held = self.undo_history.current()
        # Cleared either way past this point, and only past it: a frame that
        # commits nothing has not USED the tag, so a load deferred behind a
        # live widget keeps its name - while a load that moved nothing drops
        # it rather than naming whatever changes next.
        tag = ui_state.undo_tag
        if held is not None and uh.same(held, snap):
            ui_state.undo_tag = ""
            return
        ui_state.undo_tag = ""
        if tag:
            self.undo_history.tag(tag)
        self.undo_history.commit(snap)

    def _handle_undo_preview(self, ui_state):
        """Apply the hovered step, and put the live state back on un-hover.

        The state before the FIRST hover is what a restore returns to;
        re-hovering another row keeps it, because sliding down the list hovers
        several rows with no gap in between.
        """
        from services import undo_history as uh

        wanted = ui_state.undo_preview_index
        if wanted == self._undo_preview_showing:
            return
        if self._undo_preview_base is None and wanted >= 0:
            self._undo_preview_base = uh.capture(
                ui_state, self.rule_manager.get_current_rule(),
                self.sim.brain_layout)

        step = (self.undo_history.steps[wanted]
                if 0 <= wanted < len(self.undo_history.steps) else None)
        if step is None:
            if self._undo_preview_base is not None:
                self.command_handler.apply_undo_snapshot(
                    self._undo_preview_base, ui_state)
                self._undo_preview_base = None
        else:
            self.command_handler.apply_undo_snapshot(step, ui_state)
        self._undo_preview_showing = wanted

    def _push_undo_rows(self, ui_state):
        """Hand the panel its rows. The UI is passive and owns no journal."""
        history = self.undo_history
        self.ui.undo_steps = [
            (i, s.label, i > history.cursor)
            for i, s in enumerate(history.steps)]
        self.ui.undo_cursor = history.cursor

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
            self._step("flush archive", self.archive.maybe_flush,
                       force=True, closing=True)
        if self.goal_list is not None:
            self._step("save goals", self.goal_list.save)
        if ui_state is not None:
            # Before the store closes.
            self._step("save archive settings",
                       self._save_archive_settings, ui_state)
            self._step("restore auto overrides",
                       self._restore_auto_overrides, ui_state)
            self._step("save preferences", save_preferences, ui_state.preferences)
            # The lookup goes inside the lambda, as below: _step guards the
            # call, not the expression that produces it.
            self._step("save audio rig",
                       lambda: self._save_last_rig(ui_state))

        # The scoring thread only reads its own copy of the frame buffer, so
        # closing it after the flush cannot race the archive.
        if self.auto_service is not None:
            self._step("close scoring thread", self.auto_service.close)
        # getattr, and the check INSIDE the step: a raise out here would skip
        # every step below it, which is the whole reason each one is guarded.
        self._step("close map layout thread", self._close_map_layout)
        # Every layout's store, then the app's own handle. On quit this only
        # tidies up, but an archive left open is what stops the NEXT session
        # deleting the folder if the process lingers.
        if self.archive is not None:
            self._step("close archive layouts", self.archive.close)
        if self.archive_store is not None:
            self._step("close archive store", self.archive_store.close)
        # The lookup goes INSIDE the lambda, for the reason given below: a
        # raise out here skips every step under it.
        self._step("release thumbnails", lambda: _release(self.thumb_cache))
        self._step("release map thumbnails",
                   lambda: _release(getattr(self, "atlas_cache", None)))

        # The lookup goes INSIDE the lambda: _step guards the call, not the
        # expression that produces it, so a service that never got built would
        # otherwise raise here and skip every step below.
        self._step("audio", lambda: self.audio_runtime.close())
        self._step("advanced drawing", self.advanced_drawing_processor.cleanup)
        self._step("video", self.video_service.cleanup)
        self._step("ui", self.ui.cleanup)
        self._step("glfw", glfw.terminate)


if __name__ == "__main__":
    app = App()
    app.run()
