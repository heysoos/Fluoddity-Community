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


def test_canvas_frag_has_tournament_isolation():
    src = read("shaders/canvas.frag")
    assert "uniform int TOURNAMENT_MODE;" in src
    assert "uniform int TOURNAMENT_GRID;" in src
