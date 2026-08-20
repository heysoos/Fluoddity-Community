from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from utilities.paths import get_user_preferences_path


@dataclass
class PreferencesState:
    """User preferences that persist between program sessions."""

    # Camera/rendering preferences
    speedmult: int = 5
    motion_blur: bool = True
    blur_quality: int = 2  # Motion blur render cadence (1 = every frame, 2 = every 2 frames, etc.)
    world_size: float = 0.40  # World size multiplier (affects entity count and canvas dimensions)
    canvas_aspect_ratio: str = "1:1"  # Canvas aspect ratio (e.g. "1:1", "16:9", "3:4")
    rule_seed: float = 0.0
    brightness: float = 3.0  # Global brightness multiplier
    tonemap_softness: float = 2.5  # Asinh tonemap stretch (higher = more highlight compression)
    exposure: float = 0.0  # Frame blending for motion blur effect (0=disabled, 1=long exposure)
    bloom_enabled: bool = True  # Whether bloom post-processing is active
    bloom_threshold: float = 0.11  # Brightness threshold for bloom extraction
    bloom_intensity: float = 0.23  # Bloom contribution strength
    bloom_radius: float = 1.0  # Bloom blur spread

    # UI preferences
    show_preferences_window: bool = True  # Whether preferences window is visible
    show_controls_window: bool = False  # Help controls window
    show_parameter_sweeps_window: bool = False  # Help parameter sweeps window
    show_tutorial_window: bool = True  # Help tutorial window
    show_performance_window: bool = False  # Help performance window
    physics_tooltips_enabled: bool = True
    debug_arrows: bool = False  # Visual debug overlay for velocity field
    arrow_sensitivity: float = 15.0  # Velocity scale for debug arrows (pow(2, x))
    mouse_mode: str = "Select Particle"  # "Select Particle" or "Draw Trail"
    draw_size: float = 0.031  # Gaussian kernel width for trail drawing
    draw_power: float = 1.0  # Velocity strength when drawing trails
    menu_close_threshold: float = 80.0  # Distance in pixels before menus auto-close

    # Advanced Drawing preferences
    advanced_drawing_enabled: bool = False  # Whether Advanced Drawing window is shown
    advanced_draw_canvas: bool = True  # "Trails / Canvas (Default)" checkbox
    advanced_draw_force_field: bool = False  # "Force Field" checkbox
    advanced_draw_strafe_field: bool = False  # "Strafe Field" checkbox
    brush_mode: int = 0  # 0=Mouse Direction, 1=Inverse, 2=Fixed, 3=Attract, 4=Repel
    fixed_direction_heading: float = 0.0  # Range -PI to PI, heading for Fixed Direction mode
    force_field_strength: float = 1.0  # Multiplier for force field effect (log scale 0.0001-10.0)
    strafe_field_strength: float = 1.0  # Multiplier for strafe field effect (log scale 0.0001-10.0)
    draw_target_overlay_opacity: float = 0.0  # Opacity of draw target field overlay in frame assembly (0-1)
    shader_driven_field: bool = False  # Use a frag shader to override the field texture
    field_override_shader: str = "march.frag"  # Currently selected field override shader filename

    # Spout field source (shaders/field_override/spout.frag, needs --spout-in)
    spout_field_mode: int = 1  # 0=channels (rg/ba), 1=luminance gradient, 2=perpendicular
    spout_field_scale: float = 1.0  # Force field strength from the incoming texture
    spout_strafe_scale: float = 0.0  # Strafe field strength (0 = force only)

    # Palette: bend particle hue toward an external colour (the VJ rig's
    # `edgecol`), so the particles sit in the show's colour family instead of
    # running their own full-spectrum rainbow. Driven over OSC in live use.
    # These live here rather than on SimState deliberately: SimState's
    # appearance fields are saved into physics configs and overwritten on load
    # (config_saver.py), and a preset recall must not change the show's colour.
    palette_mix: float = 0.0  # 0 = untouched simulation hues (default, a no-op)
    palette_hue: float = 0.0  # Target hue, 0..1 around the wheel
    palette_sat: float = 0.8  # Target saturation
    palette_value: float = 1.0  # Target brightness
    palette_spread: float = 0.15  # Hue band half-width in turns (0.5 = whole wheel)
    palette_stops: int = 0  # 0/1 = continuous band; 2+ quantizes it into that many hues

    # Activity mask: how much each region of the canvas gets to do. A bias,
    # not a stencil -- mask_floor sets the activity level outside the mask, so
    # empty areas stay alive at a controlled level rather than going dead.
    # Source is the incoming Spout texture and/or a built-in vignette.
    mask_ink: float = 0.0  # How much particle alpha follows the mask (0 = off)
    mask_force: float = 0.0  # How much particle force follows the mask (0 = off)
    mask_pull: float = 0.0  # Containment pull up the mask gradient
    mask_floor: float = 0.3  # Activity level where the mask is dark
    mask_gamma: float = 1.0  # Contrast on the mask (>1 tightens toward bright areas)
    mask_blur: float = 2.0  # Blur radius in mask texels -- softens shapes into regions
    mask_vignette: float = 0.0  # Radial falloff strength
    mask_vignette_softness: float = 0.5  # Radial falloff width
    mask_source: int = 0  # 0=feed, 1=vignette, 2=max(feed,vignette), 3=feed*vignette

    # Datamosh: the particles displace the incoming texture's pixels instead of
    # being drawn over it. The flow is the velocity map the simulation already
    # writes, so this reads the simulation and changes nothing about it.
    # Off by default, and off releases the buffers -- a true no-op.
    mosh_enabled: bool = False  # Master switch for datamosh mode
    mosh_source: int = 0  # 0=Spout feed, 1=particle frame, 2=feed+particles
    mosh_amount: float = 0.02  # Max displacement per displayed frame, in uv
    mosh_contrast: float = 1.0  # Response selectivity around the frame's own average
    mosh_scale: float = 0.0  # Stroke size, as a mip level of the flow field
    mosh_flow_mix: float = 0.0  # 0 = trail map (smooth), 1 = this frame's splat (sharp)
    mosh_swirl: float = 0.0  # Rotates the shift, +-90 degrees at +-1
    mosh_refresh: float = 0.08  # Live source returning per frame (1 = no accumulation)
    mosh_block: float = 0.0  # Macroblock size in px for the flow lookup (0 = off)
    mosh_chroma: float = 0.0  # Per-channel displacement spread
    mosh_ink: float = 0.0  # Crisp particles added back on top after moshing

    # Physics slider group collapsed states (True = expanded/open, False = collapsed)
    physics_group_basics: bool = True  # Default: open (trail sensors + mutation)
    physics_group_forces: bool = True  # Default: open (global force mult, drag)
    physics_group_advanced: bool = False  # Default: collapsed
    physics_group_additional: bool = False  # Default: collapsed
    physics_group_notes: bool = False  # Default: collapsed

    # Load menu collapsed states (True = expanded/open, False = collapsed)
    load_menu_core_open: bool = True  # Default: open
    load_menu_custom_open: bool = True  # Default: open
    load_menu_advanced_open: bool = False  # Default: collapsed

    # Recording preferences
    max_frames: int = 1800  # 150 * 12
    motion_blur_samples: int = 12
    supersample_k: int = 1
    filename_prefix: str = ""
    recording_motion_blur: bool = True  # Motion blur setting used during video recording
    recording_blur_quality: int = 1  # Blur quality setting used during video recording
    video_end_frame: int = 0  # Target frame for video to end on (0 = disabled, start immediately)

    # Simulation determinism
    strong_determinism: bool = False  # Double-buffer canvas for fully deterministic simulation

    # Parameter locks
    parameter_locks_enabled: bool = False  # Master toggle for parameter lock feature


def save_preferences(prefs: PreferencesState, filepath: Path | str = None) -> None:
    """Save preferences to a JSON file."""
    if filepath is None:
        filepath = get_user_preferences_path()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(prefs)
    filepath.write_text(json.dumps(data, indent=2))


def load_preferences(filepath: Path | str = None) -> PreferencesState:
    """Load preferences from a JSON file. Returns default preferences if file doesn't exist."""
    if filepath is None:
        filepath = get_user_preferences_path()
    filepath = Path(filepath)
    if not filepath.exists():
        return PreferencesState()

    try:
        data = json.loads(filepath.read_text())
        # Filter out any fields that are no longer in PreferencesState (backward compat)
        valid_fields = set(PreferencesState.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return PreferencesState(**filtered_data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Failed to load preferences from {filepath}: {e}")
        print("Using default preferences")
        return PreferencesState()
