from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(p):
    return (ROOT / p).read_text()


def test_entity_update_has_tournament_hooks():
    src = read("shaders/entity_update.glsl")
    assert "uniform int TOURNAMENT_MODE;" in src
    assert "uniform int TOURNAMENT_GRID;" in src
    assert "tournament_home_tile" in src
    assert "tournament_tile_box" in src


def test_brush_clips_deposits_to_home_tile():
    """A particle sprite has ~1px extent; without clipping it deposits trail into
    the neighbouring tile, which that tile's sensors then read (measured as a ~25x
    energy spike at the seam before this clip was added)."""
    vert = read("shaders/brush.vert")
    frag = read("shaders/brush.frag")
    assert "TOURNAMENT_MODE" in vert and "tile_lo" in vert and "tile_hi" in vert
    assert "frag_world" in vert
    assert "TOURNAMENT_MODE" in frag
    i = frag.index("TOURNAMENT_MODE == 1")
    assert "discard" in frag[i:i + 300], "deposits are not clipped to the home tile"


def test_sensors_are_confined_by_the_boundary_condition():
    """Both sensors go through confine_sample, which applies the world's OWN
    boundary condition to the tile instead of clamping unconditionally.

    Clamping was the anisotropy: a particle heading into a wall had both
    sensors past it, both clamped to the same coordinate, and therefore no
    steering differential at all in a band around every tile. The half-texel
    inset that stops the bilinear tap straddling the seam now lives inside
    confine_sample; tests/test_tile_isolation_gl.py measures both on the GPU.
    """
    src = read("shaders/entity_update.glsl")
    assert "vec2 confine_sample(" in src
    assert "lsample = confine_sample(" in src
    assert "rsample = confine_sample(" in src
    assert "clamp(lsample" not in src, "the unconditional clamp is back"


def test_the_tile_is_the_particles_whole_world():
    """One box drives every boundary decision. The old code applied the canvas
    boundary and then bolted a per-tile bounce on afterwards, which made a tile
    a walled box no matter which boundary condition the preset asked for."""
    src = read("shaders/entity_update.glsl")
    assert "void particle_world_box(" in src
    assert "particle_world_box(index, wlo, whi)" in src
    # The old bounce's giveaway, and not a phrase any comment would contain.
    assert "abs(e.vel.x)" not in src, "the post-hoc tile bounce is back"


def test_canvas_frag_has_tournament_isolation():
    src = read("shaders/canvas.frag")
    assert "uniform int TOURNAMENT_MODE;" in src
    assert "uniform int TOURNAMENT_GRID;" in src


def test_diffusion_taps_are_bounded_by_the_tile_box():
    """Not by a tile INDEX: a probe that walks off the canvas clamped back to
    the same index, so the guard passed and the tap fell through to a sampler
    with repeat set. Measured, a lit tile put 64% of its brightness into the
    tile on the opposite side of the canvas."""
    src = read("shaders/canvas.frag")
    assert "void tournament_tile_uv_box(" in src
    assert "tile_tap(" in src
    assert "tournament_tile_uv(np)" not in src, "the index comparison is back"
