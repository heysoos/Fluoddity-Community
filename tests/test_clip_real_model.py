"""Requires the downloaded CLIP model. Run with: pytest -m gpu"""
import numpy as np
import pytest

from tools.fetch_clip_onnx import MODEL_DIR, is_present

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def scorer():
    if not is_present(MODEL_DIR):
        pytest.skip("CLIP model not downloaded")
    from services.clip_scorer import CLIPScorer

    return CLIPScorer(MODEL_DIR)


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
    from services.clip_scorer import CLIPScorer

    cpu = CLIPScorer(MODEL_DIR, providers=["CPUExecutionProvider"])
    cpu.set_prompt("a red square")
    assert np.isfinite(cpu.score(_square((220, 30, 30))[None])[0])
