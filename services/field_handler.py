"""Field texture handler: manages field save/load/preview/cache for configs.

Owns all GPU field texture state transitions (snapshot, restore, write, clear)
and the LRU cache for field PNGs. CommandHandler delegates field operations here.
"""
import numpy as np
from services.field_texture_cache import FieldTextureCache
from utilities.field_texture_io import is_field_nonzero, save_field_png as _save_field_png

MAX_FIELD_SNAPSHOTS = 20  # Max clipboard entries with non-None field snapshots

# The legacy field texture packed force in .xy and strafe in .zw of one buffer,
# so a brush layer reads the pair its destination used to occupy.
_BRUSH_CHANNELS = {"force": "xy", "strafe": "zw"}


def ensure_brush_layer(stack, destination: str):
    """Add a brush layer for `destination` if the stack has none.

    Selecting Force Field in Drawing Controls and painting has to put paint
    somewhere; without a brush layer the stroke would reach no destination.
    """
    for layer in stack.layers:
        if layer.source == "brush" and layer.destination == destination:
            return layer
    from state.field_stack import FieldLayer
    layer = FieldLayer(
        source="brush", destination=destination, mapping="rg_direct",
        blend="replace", params={"_channels": _BRUSH_CHANNELS[destination]})
    stack.layers.append(layer)
    return layer


class FieldHandler:
    """Manages field texture persistence across config save/load/preview/clipboard.

    Operates on the BRUSH SOURCE's buffer, which is the one field-injection
    source with memory. All methods are no-ops when no brush layer exists.
    """

    def __init__(self, field_bus, sim, param_lock_service=None):
        self.field_bus = field_bus
        self.sim = sim
        self.param_lock_service = param_lock_service
        self.cache = FieldTextureCache(max_size=20)

        # File preview cached state
        self._cached_field_data = None
        self._cached_field_strengths = None

        # Clipboard preview cached state
        self._clipboard_cached_field_data = None
        self._clipboard_cached_field_strengths = None

        # Last Ctrl+C field cache (for Ctrl+V restore)
        self._last_copied_field_data = None  # np.ndarray or None
        self._last_copied_field_strengths = None  # (force, strafe) or None

    @property
    def _brush(self):
        """The brush source's buffer, or None when no brush layer exists."""
        return self.field_bus.brush_source() if self.field_bus is not None else None

    @property
    def _has_field_tex(self):
        brush = self._brush
        return brush is not None and brush.snapshot() is not None

    def _mark_dirty(self):
        if self.field_bus is not None:
            self.field_bus.mark_dirty()

    def _write_field_with_locks(self, new_data):
        """Write field data to GPU, respecting force/strafe field locks.

        If both fields locked, skips write. If one locked, preserves its
        channels (force=XY 0:2, strafe=ZW 2:4) from existing GPU data.
        """
        pls = self.param_lock_service
        block_force = pls and pls.should_block_force_field()
        block_strafe = pls and pls.should_block_strafe_field()

        if block_force and block_strafe:
            return  # Both locked, write nothing

        if not block_force and not block_strafe:
            self._brush.write(new_data); self._mark_dirty()
            return

        # Partial lock: preserve locked channels from existing GPU state
        existing = self._brush.snapshot()
        if existing is None:
            self._brush.write(new_data); self._mark_dirty()
            return

        merged = new_data.copy()
        if block_force:
            merged[:, :, 0:2] = existing[:, :, 0:2]
        else:
            merged[:, :, 2:4] = existing[:, :, 2:4]
        self._brush.write(merged); self._mark_dirty()

    def _should_skip_clear(self):
        """Whether clear_fields should be skipped due to field locks."""
        pls = self.param_lock_service
        return pls and (pls.should_block_force_field() or pls.should_block_strafe_field())

    def _write_field_strengths(self, ui_state, force_val, strafe_val):
        """Write field strength scalars, respecting individual param locks."""
        pls = self.param_lock_service
        if not (pls and pls.is_locked('force_field_strength')):
            ui_state.preferences.force_field_strength = force_val
        if not (pls and pls.is_locked('strafe_field_strength')):
            ui_state.preferences.strafe_field_strength = strafe_val

    # --- Snapshot / query ---

    def snapshot_with_strengths(self, ui_state):
        """Snapshot current field texture and strengths if non-zero.

        Also caches the result for Ctrl+V restore. Each Ctrl+C overwrites
        the previous cache (None if field is zero or uninitialized).

        Returns:
            (field_snapshot, field_strengths) where field_snapshot is np.ndarray
            or None, and field_strengths is (force, strafe) tuple or None.
        """
        if not self._has_field_tex:
            self._last_copied_field_data = None
            self._last_copied_field_strengths = (
                ui_state.preferences.force_field_strength,
                ui_state.preferences.strafe_field_strength,
            )
            return None, None

        field_data = self._brush.snapshot()
        if field_data is not None and is_field_nonzero(field_data):
            field_strengths = (
                ui_state.preferences.force_field_strength,
                ui_state.preferences.strafe_field_strength,
            )
            self._last_copied_field_data = field_data
            self._last_copied_field_strengths = field_strengths
            return field_data, field_strengths

        self._last_copied_field_data = None
        self._last_copied_field_strengths = (
            ui_state.preferences.force_field_strength,
            ui_state.preferences.strafe_field_strength,
        )
        return None, None

    # --- File save/load ---

    def save_field_png(self, field_data, filename, configs_dir):
        """Save field PNG alongside config, or remove stale PNG.

        Args:
            field_data: np.ndarray from snapshot_with_strengths(), or None.
            filename: Config filename (without .json extension).
            configs_dir: Directory containing config files.
        """
        fields_png_path = configs_dir / f"{filename}_fields.png"
        if field_data is not None:
            _save_field_png(field_data, fields_png_path)
        elif fields_png_path.exists():
            fields_png_path.unlink()

    def delete_field_png(self, filepath):
        """Delete companion _fields.png and invalidate cache."""
        fields_path = filepath.with_name(filepath.stem + "_fields.png")
        if fields_path.exists():
            fields_path.unlink()
        self.cache.invalidate(filepath)

    def invalidate_cache(self, filepath):
        """Invalidate cached field data for a given config filepath."""
        self.cache.invalidate(filepath)

    def apply_for_config(self, config, json_filepath, ui_state):
        """Install the config's layer stack and its brush buffer.

        A config with no stack but a companion _fields.png predates the stack
        and becomes two brush layers over that texture, so every preset
        already on disk opens with the field it was saved with.
        """
        from services.config_saver import legacy_brush_stack
        from state.field_stack import stack_from_dict

        canvas_dim_x, canvas_dim_y = self.sim.get_canvas_dimensions()
        field_data = self.cache.get(json_filepath, canvas_dim_y, canvas_dim_x)

        stack_dict = getattr(config, "field_stack", {}) or {}
        if not stack_dict and field_data is not None:
            stack_dict = legacy_brush_stack()

        # A preset must never silently delete a setup that took a long time to
        # build. It carries no stack of its own - which is every preset saved
        # before this feature existed - so there is nothing to install and the
        # current one stays. The lock covers the other case, where the preset
        # does carry one.
        if stack_dict and not ui_state.field_stack.locked:
            incoming = stack_from_dict(stack_dict)
            ui_state.field_stack.layers = list(incoming.layers)
            ui_state.field_stack.enabled = incoming.enabled
            self._mark_dirty()

        if field_data is not None:
            # The stack was installed a moment ago, so the brush's buffer does
            # not exist until the next rebuild - and the paint has to land now.
            if self.field_bus is not None:
                self.field_bus.ensure_brush_source(ui_state.field_stack)
            if self._brush is not None:
                self._write_field_with_locks(field_data)
        else:
            if self._has_field_tex and not self._should_skip_clear():
                self._brush.clear()
                self._mark_dirty()

        # Apply field strengths from config (respects locks)
        if config.force_field_strength is not None:
            self._write_field_strengths(
                ui_state, config.force_field_strength, config.strafe_field_strength)
        elif field_data is None:
            self._write_field_strengths(ui_state, 1.0, 1.0)

    def apply_last_copied(self, ui_state):
        """Apply the field state cached from the most recent Ctrl+C.

        If the cached field data is None (zero/uninitialized at copy time),
        clears the field texture. If the field texture isn't initialized and
        cached data is None, does nothing (no need to init for all-zeros).
        Respects field locks.
        """
        if self._last_copied_field_data is not None:
            if self._brush is not None:
                self._write_field_with_locks(self._last_copied_field_data)
        else:
            # Cached field is None (all zeros) - clear if initialized, skip if not
            if self._has_field_tex and not self._should_skip_clear():
                self._brush.clear(); self._mark_dirty()

        # Restore field strengths (respects locks)
        if self._last_copied_field_strengths is not None:
            self._write_field_strengths(
                ui_state,
                self._last_copied_field_strengths[0],
                self._last_copied_field_strengths[1])

    def clear_fields(self):
        """Clear the GPU field texture to zeros (if initialized)."""
        if self._has_field_tex:
            self._brush.clear(); self._mark_dirty()

    # --- File preview cache/restore ---

    def cache_for_preview(self, ui_state):
        """Cache current field state before starting file preview."""
        if self._has_field_tex:
            self._cached_field_data = self._brush.snapshot()
        else:
            self._cached_field_data = None
        self._cached_field_strengths = (
            ui_state.preferences.force_field_strength,
            ui_state.preferences.strafe_field_strength,
        )

    def restore_from_preview(self, ui_state):
        """Restore cached field state when clearing file preview."""
        if self._has_field_tex and self._cached_field_data is not None:
            self._brush.write(self._cached_field_data); self._mark_dirty()

        if self._cached_field_strengths is not None:
            ui_state.preferences.force_field_strength = self._cached_field_strengths[0]
            ui_state.preferences.strafe_field_strength = self._cached_field_strengths[1]

    def discard_preview_cache(self):
        """Discard cached field state (preview was finalized via load)."""
        self._cached_field_data = None
        self._cached_field_strengths = None

    # --- Clipboard preview cache/restore ---

    def cache_for_clipboard_preview(self, ui_state):
        """Cache current field state before starting clipboard preview."""
        if self._has_field_tex:
            self._clipboard_cached_field_data = self._brush.snapshot()
        else:
            self._clipboard_cached_field_data = None
        self._clipboard_cached_field_strengths = (
            ui_state.preferences.force_field_strength,
            ui_state.preferences.strafe_field_strength,
        )

    def restore_from_clipboard_preview(self, ui_state):
        """Restore cached field state when clearing clipboard preview."""
        if self._has_field_tex and self._clipboard_cached_field_data is not None:
            self._brush.write(self._clipboard_cached_field_data); self._mark_dirty()
        self._clipboard_cached_field_data = None

        if self._clipboard_cached_field_strengths is not None:
            ui_state.preferences.force_field_strength = self._clipboard_cached_field_strengths[0]
            ui_state.preferences.strafe_field_strength = self._clipboard_cached_field_strengths[1]
            self._clipboard_cached_field_strengths = None

    def discard_clipboard_preview_cache(self):
        """Discard clipboard cached field state (preview was finalized via click)."""
        self._clipboard_cached_field_data = None
        self._clipboard_cached_field_strengths = None

    # --- Clipboard field snapshot ---

    def apply_snapshot(self, field_snapshot, config, ui_state):
        """Apply a field snapshot from a clipboard entry. Respects field locks.

        Args:
            field_snapshot: np.ndarray or None from clipboard tuple.
            config: PhysicsConfig to read field_strengths from.
            ui_state: For writing force/strafe field strength preferences.
        """
        if field_snapshot is not None:
            if self._brush is not None:
                self._write_field_with_locks(field_snapshot)
        else:
            if self._has_field_tex and not self._should_skip_clear():
                self._brush.clear(); self._mark_dirty()

        if config.force_field_strength is not None:
            self._write_field_strengths(
                ui_state, config.force_field_strength, config.strafe_field_strength)
        elif field_snapshot is None:
            self._write_field_strengths(ui_state, 1.0, 1.0)

    def load_field_from_image(self, filepath: str, target: str):
        """Load an image file and write its polar-to-cartesian data to a field.

        Args:
            filepath: Path to the PNG/JPEG image file.
            target: "force" to write .xy channels, "strafe" to write .zw channels.
        """
        from pathlib import Path
        from utilities.field_texture_io import load_image_as_polar_field

        canvas_dim_x,canvas_dim_y = self.sim.get_canvas_dimensions()
        cartesian = load_image_as_polar_field(Path(filepath), canvas_dim_y, canvas_dim_x)
        if cartesian is None:
            print(f"Failed to load field image: {filepath}")
            return

        if self._brush is None:
            print("Warning: no brush layer to load a field image into")
            return

        existing = self._brush.snapshot()
        if existing is None:
            existing = np.zeros((canvas_dim_y, canvas_dim_x, 4), dtype=np.float32)

        if target == "force":
            existing[:, :, 0] = cartesian[:, :, 0]
            existing[:, :, 1] = cartesian[:, :, 1]
        elif target == "strafe":
            existing[:, :, 2] = cartesian[:, :, 0]
            existing[:, :, 3] = cartesian[:, :, 1]

        self._brush.write(existing); self._mark_dirty()
        print(f"Loaded {target} field from image: {filepath}")

    def enforce_snapshot_cap(self, config_clipboard):
        """Null out oldest field snapshots if >MAX_FIELD_SNAPSHOTS entries have data."""
        entries_with_fields = []
        for i, entry in enumerate(config_clipboard):
            _config, _label, field_snapshot = entry
            if field_snapshot is not None:
                entries_with_fields.append(i)

        while len(entries_with_fields) > MAX_FIELD_SNAPSHOTS:
            oldest_idx = entries_with_fields.pop(0)
            config, label, _ = config_clipboard[oldest_idx]
            config_clipboard[oldest_idx] = (config, label, None)
