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


def test_sensor_clamp_is_inset_by_a_texel():
    """get_can() samples bilinearly, so clamping exactly to the seam would still
    blend texels from the neighbouring tile."""
    src = read("shaders/entity_update.glsl")
    i = src.index("lsample = clamp(lsample")
    assert "texel" in src[max(0, i - 400):i], "sensor clamp is not inset"


def test_canvas_frag_has_tournament_isolation():
    src = read("shaders/canvas.frag")
    assert "uniform int TOURNAMENT_MODE;" in src
    assert "uniform int TOURNAMENT_GRID;" in src
