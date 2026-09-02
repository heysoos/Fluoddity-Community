"""An Auto-mode goal can be a PICTURE, scored image-to-image.

Text or image, never both: the scorer holds ONE goal - a target, a possibly
empty reference set and the scale that goes with them - and score() is one
function over either. The image path never touches the text distractor list.
"""
import numpy as np
import pytest
from PIL import Image

from services.vision_models import get
from services.vision_scorer import (
    VisionScorer,
    distractor_images,
    load_goal_image,
)

B32 = get("clip-b32")


# ---- the picture on disk -------------------------------------------------

def test_a_wide_picture_is_centre_cropped_then_resized(tmp_path):
    """Three vertical bands; the centre crop keeps the middle one."""
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:, :100] = (255, 0, 0)
    img[:, 100:200] = (0, 255, 0)
    img[:, 200:] = (0, 0, 255)
    p = tmp_path / "bands.png"
    Image.fromarray(img).save(p)

    out = load_goal_image(str(p), 224)
    assert out.shape == (1, 224, 224, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[0, 112, 112], (0, 255, 0))
    assert np.array_equal(out[0, 5, 5], (0, 255, 0))


def test_a_tall_picture_is_centre_cropped_too(tmp_path):
    img = np.zeros((300, 100, 3), dtype=np.uint8)
    img[100:200] = (0, 255, 0)
    p = tmp_path / "tall.png"
    Image.fromarray(img).save(p)
    out = load_goal_image(str(p), 64)
    assert out.shape == (1, 64, 64, 3)
    assert np.array_equal(out[0, 32, 32], (0, 255, 0))


def test_a_greyscale_or_alpha_picture_comes_back_rgb(tmp_path):
    p = tmp_path / "grey.png"
    Image.fromarray(np.full((50, 50), 200, dtype=np.uint8), "L").save(p)
    out = load_goal_image(str(p), 32)
    assert out.shape == (1, 32, 32, 3)
    assert np.array_equal(out[0, 0, 0], (200, 200, 200))


# ---- the synthesized references -----------------------------------------

def test_the_distractors_are_black_white_grey_and_noise():
    d = distractor_images(224)
    assert d.shape == (4, 224, 224, 3)
    assert d.dtype == np.uint8
    flats = {int(f.min()) for f in d if f.min() == f.max()}
    assert flats == {0, 128, 255}
    noise = [f for f in d if f.min() != f.max()]
    assert len(noise) == 1 and noise[0].std() > 40


def test_the_distractors_are_the_same_every_time():
    assert np.array_equal(distractor_images(32), distractor_images(32))


# ---- scoring under an image goal ----------------------------------------

class _StubSession:
    """Every image embeds to the same unit vector, so what the score does with
    the goal is visible without a model."""

    def __init__(self):
        self.batch_sizes = []

    def get_inputs(self):
        return [type("I", (), {"name": "pixel_values",
                               "type": "tensor(float)"})()]

    def get_outputs(self):
        return [type("O", (), {"name": "image_embeds"})()]

    def run(self, _out_names, feed):
        n = next(iter(feed.values())).shape[0]
        self.batch_sizes.append(n)
        emb = np.zeros((n, 4), dtype=np.float32)
        emb[:, 0] = 1.0
        return [emb]


def _scorer():
    s = VisionScorer.__new__(VisionScorer)
    s._vision = _StubSession()
    s._vision_in = "pixel_values"
    s._vision_out = "image_embeds"
    s._vision_dtype = np.float32
    s._goal_emb = None
    s._goal_scale = None
    s._prompt = ""
    s._n_views = 3
    s._rng = np.random.default_rng(0)
    s._model = B32
    s._available = True
    return s


def _tiles(n=3):
    return np.zeros((n, 224, 224, 3), dtype=np.uint8)


def test_an_image_goal_scores_the_target_among_the_synthesized_pictures():
    s = _scorer()
    s.set_image_goal(_tiles(1), distractors=True)
    # target + 4 distractors, all identical under the stub -> uniform softmax
    out = s.score(_tiles(3))
    assert out.shape == (3,)
    assert np.allclose(out, 1.0 / 5.0, atol=1e-5)


def test_the_reference_is_the_untouched_frame_not_three_views():
    s = _scorer()
    s.set_image_goal(_tiles(1), distractors=True)
    assert s._goal_emb.shape == (5, 4)


def test_an_image_goal_uses_the_image_logit_scale():
    """A reference 0.01 of cosine behind the target: the softmax gap is the
    scale times that, so the value says which scale was used."""
    s = _scorer()
    s.set_image_goal(_tiles(1), distractors=True)
    ref = np.array([0.99, np.sqrt(1 - 0.99 ** 2), 0, 0], dtype=np.float32)
    s._goal_emb = np.stack([np.array([1, 0, 0, 0], dtype=np.float32), ref])
    want = 1.0 / (1.0 + np.exp(-B32.image_logit_scale * 0.01))
    assert s.score(_tiles(1))[0] == pytest.approx(want, abs=1e-4)
    assert want != pytest.approx(
        1.0 / (1.0 + np.exp(-B32.text_logit_scale * 0.01)), abs=1e-3)


def test_an_image_goal_without_distractors_is_the_plain_cosine():
    s = _scorer()
    s.set_image_goal(_tiles(1), distractors=False)
    assert s._goal_emb.shape == (1, 4)
    s._goal_emb = np.array([[0.6, 0.8, 0, 0]], dtype=np.float32)
    # A softmax over one logit would say 1.0; the cosine says 0.6.
    assert s.score(_tiles(2)) == pytest.approx([0.6, 0.6], abs=1e-5)


def test_the_distractors_are_embedded_once_per_scorer():
    s = _scorer()
    s.set_image_goal(_tiles(1), distractors=True)
    s.set_image_goal(_tiles(1), distractors=True)
    assert sum(s._vision.batch_sizes) == 4 + 1 + 1


def test_setting_an_image_goal_clears_the_prompt():
    s = _scorer()
    s._prompt = "a cat"
    s.set_image_goal(_tiles(1))
    assert s.prompt == ""


def test_score_before_any_goal_names_both_setters():
    s = _scorer()
    with pytest.raises(RuntimeError, match="set_image_goal"):
        s.score(_tiles(1))


# ---- the driver and the service -----------------------------------------

import types

from services.auto_tournament_service import AutoTournamentService
from services.prompt_driver import PromptDriver
from services.run_checkpoint import load_checkpoint, save_checkpoint
from services.tournament_service import TournamentService


class FakeScorer:
    """Records what it was handed; scores nothing."""

    def __init__(self, px=32):
        self.model = types.SimpleNamespace(px=px)
        self.prompt = ""
        self.image_calls = []

    def set_prompt(self, text, distractors=None):
        self.prompt = text

    def set_image_goal(self, image, distractors=True):
        self.image_calls.append((np.asarray(image), bool(distractors)))
        self.prompt = ""

    def score(self, images):
        return np.zeros(len(images), dtype=np.float32)


def _png(tmp_path, name="cat.png", px=40):
    p = tmp_path / name
    Image.fromarray(np.full((px, px, 3), 90, dtype=np.uint8)).save(p)
    return str(p)


def _driver(scorer=None):
    ts = TournamentService(grid=2)
    ts.init_population()
    return PromptDriver(ts, scorer if scorer is not None else FakeScorer())


def test_the_driver_loads_the_picture_at_the_encoders_size(tmp_path):
    sc = FakeScorer(px=32)
    d = _driver(sc)
    path = _png(tmp_path)
    d.set_image_goal(path, distractors=True)
    assert len(sc.image_calls) == 1
    img, distractors = sc.image_calls[0]
    assert img.shape == (1, 32, 32, 3) and distractors is True
    assert d.goal_kind == "image"
    assert d.goal_image == path
    assert d.goal_distractors is True
    assert d.prompt == ""


def test_the_status_label_names_the_picture_and_the_number(tmp_path):
    d = _driver()
    path = _png(tmp_path, "reef.jpg")
    d.set_image_goal(path, distractors=True)
    assert d.status()["prompt"] == "image: reef.jpg"
    assert d.status()["score_label"] == "fitness"
    d.set_image_goal(path, distractors=False)
    assert d.status()["score_label"] == "cosine"


def test_a_prompt_puts_the_driver_back_on_text(tmp_path):
    sc = FakeScorer()
    d = _driver(sc)
    d.set_image_goal(_png(tmp_path))
    d.set_prompt("a cat")
    assert d.goal_kind == "text"
    assert d.goal_image == ""
    assert d.status()["prompt"] == "a cat"
    assert sc.prompt == "a cat"


def test_an_unreadable_picture_raises_and_leaves_the_goal_alone(tmp_path):
    sc = FakeScorer()
    d = _driver(sc)
    d.set_prompt("a cat")
    with pytest.raises(OSError):
        d.set_image_goal(str(tmp_path / "missing.png"))
    assert d.goal_kind == "text"
    assert sc.prompt == "a cat"
    assert sc.image_calls == []


def test_a_driver_without_a_scorer_still_remembers_the_goal(tmp_path):
    ts = TournamentService(grid=2)
    ts.init_population()
    d = PromptDriver(ts, None)
    path = _png(tmp_path)
    d.set_image_goal(path, distractors=False)
    assert d.goal_kind == "image" and d.goal_image == path


def _service(scorer):
    ts = TournamentService(grid=2)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=scorer, logger=None)
    svc.configure(steps_per_gen=100, snapshots_per_gen=2,
                  sim_steps_per_frame=10)
    return svc


def test_the_checkpoint_round_trips_the_picture_goal(tmp_path):
    path = _png(tmp_path)
    svc = _service(FakeScorer())
    svc.set_image_goal(path, distractors=False)
    assert svc.goal_kind == "image"
    ck = tmp_path / "ck.npz"
    save_checkpoint(ck, svc.checkpoint_state())

    sc = FakeScorer()
    other = _service(sc)
    other.set_prompt("stale")
    other.restore(load_checkpoint(ck))
    assert other.goal_kind == "image"
    assert other.goal_image == path
    assert other.goal_distractors is False
    assert sc.image_calls and sc.image_calls[-1][1] is False


def test_an_old_checkpoint_restores_as_a_text_goal(tmp_path):
    svc = _service(FakeScorer())
    svc.set_prompt("glowing coral")
    state = svc.checkpoint_state()
    for k in ("goal_kind", "goal_image", "goal_distractors"):
        state.pop(k, None)
    ck = tmp_path / "old.npz"
    save_checkpoint(ck, state)

    sc = FakeScorer()
    other = _service(sc)
    other.restore(load_checkpoint(ck))
    assert other.goal_kind == "text"
    assert other.prompt == "glowing coral"
    assert sc.prompt == "glowing coral"


def test_a_checkpoint_whose_picture_is_gone_keeps_the_optimizer(tmp_path):
    path = _png(tmp_path, "gone.png")
    svc = _service(FakeScorer())
    svc.set_image_goal(path)
    ck = tmp_path / "ck.npz"
    save_checkpoint(ck, svc.checkpoint_state())
    (tmp_path / "gone.png").unlink()

    sc = FakeScorer()
    other = _service(sc)
    other.set_prompt("before")
    with pytest.raises(OSError):
        other.restore(load_checkpoint(ck))
    assert other.generation == svc.generation
    assert other.goal_kind == "text" and sc.prompt == "before"
