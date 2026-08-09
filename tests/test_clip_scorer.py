import numpy as np
import pytest

from services.clip_scorer import (
    CLIP_MEAN,
    CLIP_STD,
    DEFAULT_DISTRACTORS,
    LOGIT_SCALE,
    CLIPScorer,
    augment,
    preprocess,
)


def test_preprocess_shape_dtype_and_contiguity():
    crops = np.full((2, 224, 224, 3), 255, dtype=np.uint8)
    x = preprocess(crops)
    assert x.shape == (2, 3, 224, 224)
    # The fp16 exports declare tensor(float) I/O despite fp16 weights, so
    # float32 is the default; the dtype is driven by the session descriptor.
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"], "ONNX Runtime needs a contiguous NCHW buffer"


def test_preprocess_honours_an_explicit_dtype():
    crops = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    assert preprocess(crops, dtype=np.float16).dtype == np.float16


def test_preprocess_applies_clip_normalisation():
    crops = np.full((1, 224, 224, 3), 255, dtype=np.uint8)
    x = preprocess(crops)
    expected = (1.0 - CLIP_MEAN) / CLIP_STD
    for c in range(3):
        assert np.allclose(x[0, c], expected[c], atol=1e-3)


def test_augment_produces_n_views_per_image():
    rng = np.random.default_rng(0)
    crops = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    out = augment(crops, n_views=3, rng=rng)
    assert out.shape == (9, 224, 224, 3)
    assert out.dtype == np.uint8


def test_augment_first_view_is_the_untouched_frame():
    rng = np.random.default_rng(0)
    crops = np.random.default_rng(1).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)
    out = augment(crops, n_views=3, rng=rng)
    # views are laid out image-major: [img0 v0, img0 v1, img0 v2, img1 v0, ...]
    assert np.array_equal(out[0], crops[0])
    assert np.array_equal(out[3], crops[1])


class _StubSession:
    """Stands in for an onnxruntime InferenceSession."""

    def __init__(self, kind):
        self.kind = kind
        self.batch_sizes = []

    def get_inputs(self):
        name = "pixel_values" if self.kind == "vision" else "input_ids"
        return [type("I", (), {"name": name, "type": "tensor(float)"})()]

    def get_outputs(self):
        name = "image_embeds" if self.kind == "vision" else "text_embeds"
        return [type("O", (), {"name": name})()]

    def run(self, _out_names, feed):
        arr = next(iter(feed.values()))
        n = arr.shape[0]
        self.batch_sizes.append(n)
        emb = np.zeros((n, 4), dtype=np.float32)
        emb[:, 0] = 1.0
        return [emb]


def _scorer_with_stubs():
    s = CLIPScorer.__new__(CLIPScorer)
    s._vision = _StubSession("vision")
    s._text = _StubSession("text")
    s._vision_in = "pixel_values"
    s._vision_out = "image_embeds"
    s._vision_dtype = np.float32
    s._text_in = "input_ids"
    s._text_out = "text_embeds"
    s._tokenizer = None
    s._text_emb = None
    s._prompt = ""
    s._n_views = 3
    s._rng = np.random.default_rng(0)
    s._available = True
    return s


def test_score_returns_probability_per_image():
    s = _scorer_with_stubs()
    # 1 target + 6 distractors, all identical -> uniform softmax = 1/7
    s._text_emb = np.tile(np.array([[1.0, 0, 0, 0]], dtype=np.float32), (7, 1))
    out = s.score(np.zeros((4, 224, 224, 3), dtype=np.uint8))
    assert out.shape == (4,)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0 / 7.0, atol=1e-5)


def test_score_target_is_index_zero():
    s = _scorer_with_stubs()
    emb = np.zeros((3, 4), dtype=np.float32)
    emb[0] = [1.0, 0, 0, 0]
    emb[1] = [0, 1.0, 0, 0]
    emb[2] = [0, 0, 1.0, 0]
    s._text_emb = emb
    out = s.score(np.zeros((1, 224, 224, 3), dtype=np.uint8))
    assert out[0] > 0.99, "an image aligned with prompt 0 must score near 1"


def test_score_chunks_large_batches():
    s = _scorer_with_stubs()
    s._text_emb = np.tile(np.array([[1.0, 0, 0, 0]], dtype=np.float32), (7, 1))
    # 64 tiles x 3 views = 192 images -> must not be one 192-image call
    s.score(np.zeros((64, 224, 224, 3), dtype=np.uint8))
    assert max(s._vision.batch_sizes) <= CLIPScorer.MAX_CHUNK
    assert sum(s._vision.batch_sizes) == 192


def test_score_before_set_prompt_raises():
    s = _scorer_with_stubs()
    with pytest.raises(RuntimeError, match="set_prompt"):
        s.score(np.zeros((1, 224, 224, 3), dtype=np.uint8))


def test_default_distractors_include_abstract_texture():
    """Without this one, every trail pattern scores highly on every prompt."""
    assert "an abstract texture" in DEFAULT_DISTRACTORS


def test_logit_scale_matches_clip():
    assert LOGIT_SCALE == pytest.approx(100.0)


# --- embeddings, for the exploration archive -------------------------------


def test_embed_returns_l2_normalised_rows():
    s = _scorer_with_stubs()
    e = s.embed(np.zeros((3, 224, 224, 3), dtype=np.uint8), n_views=1)
    assert e.shape == (3, 4)
    assert e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


def test_embed_with_one_view_skips_augmentation():
    s = _scorer_with_stubs()
    s.embed(np.zeros((5, 224, 224, 3), dtype=np.uint8), n_views=1)
    assert sum(s._vision.batch_sizes) == 5, "n_views=1 must not expand the batch"


def test_embed_defaults_to_the_instance_view_count():
    s = _scorer_with_stubs()          # _n_views == 3
    e = s.embed(np.zeros((2, 224, 224, 3), dtype=np.uint8))
    assert len(e) == 6


def test_embed_chunks_large_batches():
    s = _scorer_with_stubs()
    s.embed(np.zeros((200, 224, 224, 3), dtype=np.uint8), n_views=1)
    assert max(s._vision.batch_sizes) <= CLIPScorer.MAX_CHUNK
    assert sum(s._vision.batch_sizes) == 200


def test_score_is_recoverable_from_embed():
    """score() must be exactly the softmax over embed(); if these ever disagree,
    the archive and the prompt objective are measuring different things."""
    s = _scorer_with_stubs()
    emb = np.zeros((3, 4), dtype=np.float32)
    emb[0] = [1.0, 0, 0, 0]
    emb[1] = [0, 1.0, 0, 0]
    emb[2] = [0, 0, 1.0, 0]
    s._text_emb = emb

    images = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    s._rng = np.random.default_rng(7)
    got = s.score(images)

    s._rng = np.random.default_rng(7)
    e = s.embed(images, s._n_views)
    logits = LOGIT_SCALE * (e @ emb.T)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    expected = p[:, 0].reshape(4, -1).mean(axis=1)

    assert np.allclose(got, expected, atol=1e-6)


def test_embed_text_does_not_disturb_the_cached_prompt_embedding():
    s = _scorer_with_stubs()

    class _Tok:
        def encode_batch(self, prompts):
            return [type("E", (), {"ids": [0] * 77, "attention_mask": [1] * 77})()
                    for _ in prompts]

    s._tokenizer = _Tok()
    cached = np.full((7, 4), 0.5, dtype=np.float32)
    s._text_emb = cached
    out = s.embed_text(["a", "b"])
    assert out.shape == (2, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert s._text_emb is cached, "embed_text must not overwrite the score() cache"


# ---- averaged views ----------------------------------------------------

def test_embed_mean_returns_one_row_per_image():
    """embed() returns B*v rows and does no reduction. Handing those to a
    novelty archive would make three sub-crops of one tile into three
    competing descriptors for one behaviour."""
    s = _scorer_with_stubs()
    out = s.embed_mean(np.zeros((4, 224, 224, 3), dtype=np.uint8), n_views=3)
    assert out.shape[0] == 4 and out.ndim == 2


def test_embed_mean_output_is_renormalised():
    """A mean of unit vectors is not a unit vector, and everything downstream
    treats a dot product as a cosine."""
    s = _scorer_with_stubs()
    out = s.embed_mean(np.zeros((3, 224, 224, 3), dtype=np.uint8), n_views=3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_mean_at_one_view_is_the_plain_embedding():
    """1 view must stay the raw frame and cost one pass, so turning the
    setting down really does turn the feature off."""
    s = _scorer_with_stubs()
    imgs = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    a = s.embed_mean(imgs, n_views=1)
    assert a.shape[0] == 5 and a.ndim == 2
    assert sum(s._vision.batch_sizes) == 5, "one view must not expand the batch"


def test_embed_mean_actually_runs_every_view():
    s = _scorer_with_stubs()
    s.embed_mean(np.zeros((5, 224, 224, 3), dtype=np.uint8), n_views=3)
    assert sum(s._vision.batch_sizes) == 15
