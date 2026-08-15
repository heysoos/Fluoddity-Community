from dataclasses import dataclass, field
from .sim_state import SimState
from .camera_state import CameraState
from .recording_state import RecordingState
from .preferences_state import PreferencesState
from .multi_load_state import MultiLoadState
from .tournament_state import TournamentState
from .auto_tournament_state import AutoTournamentState
from .archive_state import ArchiveState
from .brain_state import BrainState
from .audio_in_state import AudioInState


@dataclass
class UIState:
    """Aggregate state that UI exposes to Orchestrator each frame."""
    sim: SimState = field(default_factory=SimState)
    camera: CameraState = field(default_factory=CameraState)
    recording: RecordingState = field(default_factory=RecordingState)
    preferences: PreferencesState = field(default_factory=PreferencesState)
    multi_load: MultiLoadState = field(default_factory=MultiLoadState)
    tournament: TournamentState = field(default_factory=TournamentState)
    auto_tournament: AutoTournamentState = field(default_factory=AutoTournamentState)
    archive: ArchiveState = field(default_factory=ArchiveState)
    brain: BrainState = field(default_factory=BrainState)
    audio: AudioInState = field(default_factory=AudioInState)

    # Input state (updated by callbacks)
    keys_pressed: set = field(default_factory=set)
    mouse_pos: tuple = (0.0, 0.0)

    # One-shot click events (reset after get_state)
    left_click_this_frame: bool = False
    right_click_this_frame: bool = False

    # Any click events (includes clicks on imgui elements, for sweep restore)
    any_left_click_this_frame: bool = False
    any_right_click_this_frame: bool = False

    # Continuous mouse state (respects imgui capture)
    mouse_left_held: bool = False
    mouse_right_held: bool = False  # For right-click eraser in Draw Trail mode

    # Advanced drawing one-shot flags
    request_fill_operation: bool = False  # Fill entire canvas/field for one frame
    fill_direction_type: int = 0  # 0=fixed, 1=radial_in, 2=radial_out
    request_clear_force_field: bool = False  # Clear force field channels
    request_clear_strafe_field: bool = False  # Clear strafe field channels
    request_clear_canvas: bool = False  # Clear trails/canvas textures

    # Scroll input (for zoom-around-pointer)
    scroll_delta: float = 0.0

    # One-shot command flags (reset after get_state)
    request_reload: bool = False
    request_reset: bool = False
    request_full_reset: bool = False
    request_randomize_mutations: bool = False
    toggle_recording: bool = False
    request_screenshot: bool = False
    request_world_size_change: bool = False
    request_camera_reset: bool = False
    request_clear_canvas_and_fields: bool = False

    # Undo/redo. any_widget_active is written by the UI inside its own ImGui
    # frame; a capture is deferred while it holds, which is what makes a whole
    # drag one step. undo_jump_index is a one-shot; -1 is none.
    request_undo: bool = False
    request_redo: bool = False
    undo_jump_index: int = -1
    any_widget_active: bool = False
    undo_notice: str = ""
    # Which step the pointer is over, or -1. CONTINUOUS, not a one-shot: a
    # panel that stops being drawn would otherwise strand the preview.
    undo_preview_index: int = -1

    # Config save/load (Ctrl+C/Ctrl+V)
    request_save_config: bool = False
    request_load_config: bool = False
    clipboard_text: str = ""  # For passing clipboard content to orchestrator

    # File save/load/delete (menu bar)
    request_save_file: bool = False
    request_load_file: bool = False
    request_delete_file: bool = False
    save_filename: str = ""  # Filename to save to (without extension)
    # WHAT is being saved. "" is the live configuration (File > Save) and must
    # keep behaving exactly as before; the rest are tournament subjects. See
    # services/save_targets.
    save_kind: str = ""
    save_arg: int = -1       # tile index or archive entry id, per kind
    save_tiles: tuple = ()   # the selected tiles, for a manual tournament save
    load_filename: str = ""  # Filename to load from (without extension)
    load_category: str = ""  # Category for load operation (Core, Custom, Advanced)
    delete_filename: str = ""  # Filename to delete (without extension)
    delete_category: str = ""  # Category for delete operation (Core, Custom, Advanced)
    load_watercolor_override: bool | None = None  # Override watercolor mode when loading

    # Config preview (for Load submenu hover)
    request_preview_config: bool = False  # Push rule for preview
    request_clear_preview: bool = False  # Pop preview rule
    preview_filename: str = ""  # Filename to preview
    preview_category: str = ""  # Category for preview operation

    # Field loader (load image as force/strafe field)
    request_load_force_field_image: bool = False
    request_load_strafe_field_image: bool = False
    field_load_image_path: str = ""

    # Config clipboard flags
    request_preview_clipboard_config: bool = False
    request_clear_clipboard_preview: bool = False
    request_load_clipboard_config: bool = False
    request_delete_clipboard_config: bool = False
    request_import_clipboard_to_multiload: bool = False
    clipboard_config_index: int = -1
