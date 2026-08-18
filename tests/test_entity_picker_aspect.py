"""Picking must measure distance where the particle is DRAWN.

Entity space is scaled to the canvas the same way `get_canvas_dimensions` scales
the canvas itself - area-preserving, so x spreads by sqrt(aspect) and y by its
reciprocal. Comparing raw entity coordinates against a click ignores that, and
on any non-square canvas the search picks a particle that is not the one under
the pointer. The error is zero at the centre and grows toward the edges, which
is why a square canvas cannot show it.
"""
import struct

import numpy as np
import pytest

from services.entity_picker import EntityPicker

STRIDE = 12          # pos2 vel2 size1 cohort1 pad2 color4


class FakeBuffer:
    """Stands in for the GPU buffer; the picker only ever calls read()."""

    def __init__(self, positions):
        rows = []
        for i, (x, y) in enumerate(positions):
            row = [0.0] * STRIDE
            row[0], row[1] = x, y
            row[5] = float(i)          # cohort
            rows.append(row)
        self._blob = struct.pack(f"<{len(rows) * STRIDE}f",
                                 *[v for row in rows for v in row])

    def read(self):
        return self._blob


# On a 4:1 canvas x is spread by 2 and y squeezed by 2. These two particles and
# this click are chosen so the two mappings disagree about which is nearest.
WIDE = [(0.0, 0.4), (1.0, 0.0)]
CLICK = (0.72, 0.62)


def test_a_wide_canvas_picks_the_particle_under_the_pointer():
    picker = EntityPicker(FakeBuffer(WIDE), STRIDE)
    idx, _, _ = picker.find_nearest_entity(CLICK, 4.0)
    assert idx == 1


def test_a_square_canvas_is_unchanged():
    picker = EntityPicker(FakeBuffer(WIDE), STRIDE)
    idx, _, _ = picker.find_nearest_entity(CLICK, 1.0)
    assert idx == 0


def test_the_position_returned_is_world_space_not_the_scaled_probe():
    """The caller feeds this straight to the sliders, which are in world space."""
    picker = EntityPicker(FakeBuffer(WIDE), STRIDE)
    _, pos, _ = picker.find_nearest_entity(CLICK, 4.0)
    assert pos == pytest.approx((1.0, 0.0))
