"""The picture goal's CROP is the user's, and the optimizer can be asked to
score DENSITY alone.

The crop is a square slid along the picture's long side and shrunk by a zoom;
the default reproduces the centre crop exactly. Greyscale forces saturation to
zero in the capture's particle pass, so what the encoder sees is particle
density with no colour term at all, and the reference picture is read by luma
to match.
"""
import numpy as np
import pytest
from PIL import Image

from services.run_checkpoint import load_checkpoint, save_checkpoint
from services.vision_scorer import load_goal_image, to_grayscale
from state.auto_tournament_state import AutoTournamentState
from tests.test_image_goal import (
    FakeScorer,
    _FakeSvc,
    _handler,
    _png,
    _scorer,
    _service,
    _tiles,
    _ui,
)
from ui.auto_tournament_window import (
    crop_after_drag,
    crop_square_px,
    preview_layout,
)


# ---- the crop ------------------------------------------------------------

def _bands(tmp_path):
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:, :100] = (255, 0, 0)
    img[:, 100:200] = (0, 255, 0)
    img[:, 200:] = (0, 0, 255)
    p = tmp_path / "bands.png"
    Image.fromarray(img).save(p)
    return str(p)


def test_the_crop_slides_along_the_long_side(tmp_path):
    p = _bands(tmp_path)
    left = load_goal_image(p, 32, crop=(0.0, 0.5, 1.0))
    right = load_goal_image(p, 32, crop=(1.0, 0.5, 1.0))
    assert np.array_equal(left[0, 16, 16], (255, 0, 0))
    assert np.array_equal(right[0, 16, 16], (0, 0, 255))
    assert np.array_equal(load_goal_image(p, 32)[0, 16, 16], (0, 255, 0))


def test_zoom_crops_tighter_from_the_same_centre(tmp_path):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[100:200, 100:200] = (0, 255, 0)
    p = tmp_path / "square.png"
    Image.fromarray(img).save(p)
    out = load_goal_image(str(p), 32, crop=(0.5, 0.5, 3.0))
    assert np.array_equal(out[0, 1, 1], (0, 255, 0))
    assert np.array_equal(out[0, 30, 30], (0, 255, 0))


def test_the_state_carries_the_crop():
    s = AutoTournamentState()
    assert s.goal_crop() == (0.5, 0.5, 1.0)
    s.goal_crop_x, s.goal_crop_y, s.goal_crop_zoom = 0.1, 0.9, 2.0
    assert s.goal_crop() == (0.1, 0.9, 2.0)


def test_the_driver_remembers_the_crop_and_a_checkpoint_carries_it(tmp_path):
    path = _png(tmp_path)
    svc = _service(FakeScorer())
    svc.set_image_goal(path, crop=(0.2, 0.9, 2.0))
    assert svc.goal_crop == (0.2, 0.9, 2.0)
    ck = tmp_path / "ck.npz"
    save_checkpoint(ck, svc.checkpoint_state())
    other = _service(FakeScorer())
    other.restore(load_checkpoint(ck))
    assert other.goal_crop == pytest.approx((0.2, 0.9, 2.0))


def test_the_handler_passes_the_crop(tmp_path):
    svc = _FakeSvc()
    ch = _handler(svc)
    path = _png(tmp_path)
    ui, ats = _ui(goal_kind="image", goal_image=path, goal_changed=True,
                  goal_crop_x=0.1, goal_crop_y=0.7, goal_crop_zoom=2.0)
    ch._handle_auto_tournament(ui)
    assert svc.crops == [(0.1, 0.7, 2.0)]


def test_loading_a_checkpoint_puts_the_crop_on_screen(tmp_path):
    path = _png(tmp_path)
    saver = _service(FakeScorer())
    saver.set_image_goal(path, crop=(0.3, 0.6, 1.5))
    ck = tmp_path / "ck.npz"
    save_checkpoint(ck, saver.checkpoint_state())
    svc = _service(FakeScorer())
    ch = _handler(svc)
    ui, ats = _ui()
    ats.load_checkpoint_path = str(ck)
    ch._load_auto_checkpoint(svc, ats)
    assert ats.goal_crop() == pytest.approx((0.3, 0.6, 1.5))


def test_the_preview_keeps_the_pictures_aspect():
    assert preview_layout((800, 400), 160) == (160, 80)
    assert preview_layout((400, 800), 160) == (80, 160)
    assert preview_layout((50, 50), 160) == (160, 160)


def test_the_crop_square_follows_the_alignment():
    assert crop_square_px(160, 80, (0.0, 0.5, 1.0)) == (0, 0, 80)
    assert crop_square_px(160, 80, (1.0, 0.5, 1.0)) == (80, 0, 80)
    assert crop_square_px(160, 80, (0.5, 0.5, 2.0)) == (60, 20, 40)


def test_a_drag_moves_the_square_only_where_there_is_room():
    cx, cy, zoom = crop_after_drag((0.5, 0.5, 1.0), 20, 40, 160, 80)
    assert cx == pytest.approx(0.75)
    assert cy == 0.5, "no slack on the short side, so no movement"
    assert zoom == 1.0
    assert crop_after_drag((0.5, 0.5, 1.0), 1000, 0, 160, 80)[0] == 1.0
    assert crop_after_drag((0.5, 0.5, 1.0), -1000, 0, 160, 80)[0] == 0.0


# ---- grayscale -----------------------------------------------------------

def test_to_grayscale_makes_every_channel_equal_by_luma():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)
    g = to_grayscale(img)
    assert g.shape == img.shape and g.dtype == np.uint8
    assert np.array_equal(g[..., 0], g[..., 1])
    assert np.array_equal(g[..., 1], g[..., 2])
    green = np.zeros((1, 4, 4, 3), np.uint8)
    green[..., 1] = 255
    blue = np.zeros((1, 4, 4, 3), np.uint8)
    blue[..., 2] = 255
    assert to_grayscale(green)[0, 0, 0, 0] > to_grayscale(blue)[0, 0, 0, 0]


def test_a_grayscale_goal_embeds_grey_pictures_and_grey_distractors():
    s = _scorer()
    seen = []
    real = s._embed_images

    def spy(crops):
        seen.append(np.asarray(crops))
        return real(crops)

    s._embed_images = spy
    colour = np.random.default_rng(1).integers(0, 256, (1, 224, 224, 3),
                                                dtype=np.uint8)
    s.set_image_goal(colour, distractors=True, grayscale=True)
    assert seen
    for c in seen:
        assert np.array_equal(c[..., 0], c[..., 1])
        assert np.array_equal(c[..., 1], c[..., 2])


def test_grey_and_colour_distractors_are_cached_apart():
    s = _scorer()
    s.set_image_goal(_tiles(1), distractors=True, grayscale=True)
    s.set_image_goal(_tiles(1), distractors=True, grayscale=False)
    s.set_image_goal(_tiles(1), distractors=True, grayscale=True)
    assert sum(s._vision.batch_sizes) == (4 + 1) + (4 + 1) + 1


def test_the_state_has_a_grayscale_switch_that_is_off():
    assert AutoTournamentState().grayscale is False


def test_the_service_hands_its_grayscale_setting_to_the_picture(tmp_path):
    sc = FakeScorer()
    svc = _service(sc)
    svc.configure(grayscale=True)
    svc.set_image_goal(_png(tmp_path))
    assert sc.image_calls[-1][2] is True


def test_grayscale_rides_in_the_checkpoint(tmp_path):
    path = _png(tmp_path)
    saver = _service(FakeScorer())
    saver.configure(grayscale=True)
    saver.set_image_goal(path)
    ck = tmp_path / "ck.npz"
    save_checkpoint(ck, saver.checkpoint_state())
    sc = FakeScorer()
    other = _service(sc)
    other.restore(load_checkpoint(ck))
    assert other.grayscale is True
    assert sc.image_calls[-1][2] is True, "the picture is re-embedded grey"


def test_the_handler_configures_grayscale_and_loads_it_back(tmp_path):
    svc = _FakeSvc()
    ch = _handler(svc)
    ui, ats = _ui(grayscale=True)
    ch._handle_auto_tournament(ui)
    assert svc.configured["grayscale"] is True

    saver = _service(FakeScorer())
    saver.configure(grayscale=True)
    saver.set_prompt("coral")
    ck = tmp_path / "ck.npz"
    save_checkpoint(ck, saver.checkpoint_state())
    real = _service(FakeScorer())
    ui, ats = _ui()
    ats.load_checkpoint_path = str(ck)
    _handler(real)._load_auto_checkpoint(real, ats)
    assert ats.grayscale is True


def test_the_capture_is_asked_for_grey_when_the_service_says_so():
    from main import App

    asked = {}

    class _CV:
        def render(self, ui_state, kwargs, side, grayscale=False):
            asked["grayscale"] = grayscale
            return None

    app = type("A", (), {})()
    app.sim_runner = type("R", (), {"last_assemble_kwargs": {}})()
    app.tournament_service = type("T", (), {"grid": 2})()
    app.auto_service = type("S", (), {"grayscale": True})()
    app.capture_view = _CV()
    assert App._capture_tiles(app, object()) is None
    assert asked["grayscale"] is True
