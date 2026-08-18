import math

import numpy as np
import moderngl


class EntityPicker:
    """Handles entity selection from screen coordinates."""

    def __init__(self, entity_buffer: moderngl.Buffer, entity_stride: int):
        """Initialize EntityPicker.

        Args:
            entity_buffer: GPU buffer containing entity data
            entity_stride: Number of floats per entity (8: pos:2 + vel:2 + size:1 + hue:1 + sat:1 + padding:1)
        """
        self.entity_buffer = entity_buffer
        self.entity_stride = entity_stride

    def update_buffer(self, entity_buffer: moderngl.Buffer):
        """Update the entity buffer reference.

        Call this when the buffer is reallocated (e.g., world size change).

        Args:
            entity_buffer: New GPU buffer containing entity data
        """
        self.entity_buffer = entity_buffer

    def find_nearest_entity(self, tex_coords: tuple[float, float],
                            canvas_aspect: float,
                            active_count: int) -> tuple[int, tuple[float, float], float]:
        """Find the entity closest to given texture coordinates.

        Args:
            tex_coords: (x, y) in texture space where (0,0) is top-left
            canvas_aspect: Canvas width / height. Required, not defaulted: a
                caller that omits it silently picks the wrong particle
                everywhere except the centre of a square canvas.
            active_count: Number of active entities. The cohort is a pure
                function of the index and is no longer stored, so this is what
                recovers it.

        Returns:
            Tuple of (entity_index, (pos_x, pos_y), cohort_normalized)
            - entity_index: Index of the nearest entity
            - (pos_x, pos_y): World-space position of the entity (in [-1, 1] range)
            - cohort_normalized: Normalized cohort value (0-1) for parameter sweep calculations
        """
        ent_cache = np.frombuffer(self.entity_buffer.read(), dtype=np.float32)

        # Extract positions (every Nth float starting at 0 and 1)
        xs = ent_cache[0::self.entity_stride].copy()
        ys = ent_cache[1::self.entity_stride].copy()

        # Entity space is stretched onto the canvas area-preservingly, the same
        # way get_canvas_dimensions stretches the canvas, so undo that before
        # measuring against a click. This is the whole of the aspect fix.
        factor = math.sqrt(canvas_aspect)

        # Convert from entity space to [0,1] texture space
        xs_tex = xs / factor / 2.0 + 0.5
        ys_tex = ys * factor / 2.0 + 0.5

        # Compute squared distances
        dx = xs_tex - tex_coords[0]
        dy = ys_tex - tex_coords[1]
        distances_sq = dx * dx + dy * dy

        nearest_idx = int(distances_sq.argmin())

        # Extract position and cohort for the nearest entity
        pos_x = float(xs[nearest_idx])  # Already in world space [-1, 1]
        pos_y = float(ys[nearest_idx])
        # The struct no longer carries a cohort. reset() used to store
        # get_cohort(index)/cohorts, which is exactly index/ACTIVE_COUNT - a
        # pure function of the index, so it is recomputed rather than read.
        cohort_normalized = float(nearest_idx) / float(max(active_count, 1))

        return (nearest_idx, (pos_x, pos_y), cohort_normalized)
