"""Sampling helpers shared by the measurement tools."""
import numpy as np
from PIL import Image

from tools.archive_sample import load_thumbs, nn_distance, recent_layouts


def _layout(root, archive, layout, n, px=160):
    d = root / archive / layout
    (d / "thumbs").mkdir(parents=True)
    (d / "index.jsonl").write_text("", encoding="utf-8")
    for i in range(n):
        Image.fromarray(np.full((px, px, 3), (i * 7) % 256, np.uint8)).save(
            d / "thumbs" / f"{i:06d}.jpg")
    return d


def _empty_layout(root, archive, layout):
    d = root / archive / layout
    (d / "thumbs").mkdir(parents=True)
    (d / "index.jsonl").write_text("", encoding="utf-8")
    return d


def test_an_empty_sibling_layout_is_never_ranked(tmp_path):
    """Switching brain creates an empty sibling and writes to it, so the most
    recently touched directory is reliably one nothing was explored in."""
    _layout(tmp_path, "a", "fourier-n10", 3)
    _empty_layout(tmp_path, "a", "gabor-n7")
    assert [p.name for p in recent_layouts(tmp_path, 3)] == ["fourier-n10"]


def test_layouts_from_several_archives_are_ranked_together(tmp_path):
    _layout(tmp_path, "a", "fourier-n10", 2)
    _layout(tmp_path, "b", "mlp-n16-a0", 2)
    assert len(recent_layouts(tmp_path, 5)) == 2


def test_only_the_most_recent_n_come_back(tmp_path):
    for i in range(4):
        _layout(tmp_path, f"a{i}", "fourier-n10", 2)
    assert len(recent_layouts(tmp_path, 2)) == 2


def test_a_cleared_snapshot_is_skipped(tmp_path):
    """A cleared archive is the user's undo, not a population to measure."""
    _layout(tmp_path, "live", "fourier-n10", 2)
    _layout(tmp_path, "live.cleared-1786485038", "fourier-n10", 2)
    got = recent_layouts(tmp_path, 5)
    assert [p.parent.name for p in got] == ["live"]


def test_a_missing_root_is_an_empty_list_not_an_error(tmp_path):
    assert recent_layouts(tmp_path / "nope", 3) == []


def test_thumbs_are_resized_to_the_requested_input(tmp_path):
    d = _layout(tmp_path, "a", "fourier-n10", 4)
    out = load_thumbs(d, 4, 224, np.random.default_rng(0))
    assert out.shape == (4, 224, 224, 3)
    assert out.dtype == np.uint8


def test_sampling_never_returns_more_than_asked(tmp_path):
    d = _layout(tmp_path, "a", "fourier-n10", 10)
    assert len(load_thumbs(d, 3, 224, np.random.default_rng(0))) == 3


def test_an_empty_layout_samples_to_nothing(tmp_path):
    d = _empty_layout(tmp_path, "a", "gabor-n7")
    assert len(load_thumbs(d, 4, 224, np.random.default_rng(0))) == 0


def test_nn_distance_never_matches_a_row_with_itself():
    e = np.eye(4, dtype=np.float32)
    assert np.allclose(nn_distance(e), 1.0)


def test_nn_distance_finds_the_closest_other_row():
    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    got = nn_distance(e)
    assert got[0] == 0.0 and got[1] == 0.0
    assert got[2] == 1.0


def test_nn_distance_blocks_without_changing_the_answer():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(70, 8)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    assert np.allclose(nn_distance(e, block=8), nn_distance(e, block=1024))
