"""Tile and cohort indexing, mirroring entity_update.glsl exactly.

Both `index_home_tile(index)` and `get_cohort(index)` are slices of the
same particle numbering, so they nest: with COHORTS = k * tiles, each tile
contains exactly k cohorts, and each cohort gets its own deterministic tweak of
that tile's genome.

If COHORTS == tiles, each tile holds exactly one cohort - one tweak applied to
every particle in it - so the tile stays uniform, just displaced. That is the
trap this module exists to prevent.
"""
from __future__ import annotations

import math

# The Number of Cohorts slider is capped at 144 in ui/physics_window.py.
MAX_COHORTS = 144


def tile_of(index: int, active: int, grid: int) -> int:
    """Mirrors index_home_tile() in entity_update.glsl."""
    n = grid * grid
    tile = int(math.floor(float(index) / float(active) * float(n)))
    return max(0, min(tile, n - 1))


def cohort_of(index: int, active: int, cohorts: int) -> int:
    """Mirrors floor(get_cohort(index)) in entity_update.glsl."""
    return int(math.floor(float(cohorts) * float(index) / float(active)))


def max_variants(grid: int) -> int:
    """Largest variants-per-tile that keeps COHORTS within the 144 cap."""
    return max(1, MAX_COHORTS // (grid * grid))


def cohorts_for(grid: int, variants: int) -> int:
    """Cohort count that gives exactly `variants` distinct cohorts per tile."""
    return variants * grid * grid


def box_grid(cohorts: int) -> tuple[int, int]:
    """(boxes across, boxes up) for one box per cohort.

    As near square as the count allows, so a count that is not a product of
    two close factors gets blank cells rather than a strip: 13 cohorts is 4x4
    with three blanks, never 13x1.
    """
    n = max(1, int(cohorts))
    gx = math.ceil(math.sqrt(n))
    gy = -(-n // gx)                              # ceil division
    return gx, gy
