"""Requires a downloaded encoder. Run with: pytest -m gpu"""
import numpy as np
import pytest

from services.vision_models import DEFAULT_KEY
from tools.fetch_models import is_present

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def scorer():
    if not is_present(DEFAULT_KEY):
        pytest.skip("encoder not downloaded")
    from services.vision_scorer import VisionScorer

    return VisionScorer()


def _square(color):
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[56:168, 56:168] = color
    return img


def test_red_square_prefers_its_own_prompt(scorer):
    red = _square((220, 30, 30))[None]
    scorer.set_prompt("a red square")
    red_on_red = scorer.score(red)[0]
    scorer.set_prompt("a blue circle")
    red_on_blue = scorer.score(red)[0]
    assert red_on_red > red_on_blue


def test_blank_image_scores_lower_than_content(scorer):
    blank = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    red = _square((220, 30, 30))[None]
    scorer.set_prompt("a red square")
    assert scorer.score(red)[0] > scorer.score(blank)[0]


def test_cpu_provider_loads(scorer):
    """Spec 11.1 requires a working CPU fallback when DirectML is absent.
    The default ORT_ENABLE_ALL crashes here, which is why sessions pin BASIC."""
    from services.vision_scorer import VisionScorer

    cpu = VisionScorer(providers=["CPUExecutionProvider"])
    cpu.set_prompt("a red square")
    assert np.isfinite(cpu.score(_square((220, 30, 30))[None])[0])


def test_black_canvas_is_not_a_degenerate_attractor(scorer):
    """A dead simulation must not outscore a live one.

    Measured in the Task 3 gate: with only the original six distractors, pure
    black scored 0.37 on 'flowing water' and outranked 31 of 32 real tiles.
    """
    black = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    for prompt in ("tree branches", "flowing water", "a spider web"):
        scorer.set_prompt(prompt)
        assert scorer.score(black)[0] < 0.10, (
            f"black scores too well on {prompt!r}; the optimizer would "
            "drive toward an empty canvas"
        )
