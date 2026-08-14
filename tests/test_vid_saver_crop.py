"""VidSaver decides the video size once, then holds it for the whole take."""
from __future__ import annotations

import numpy as np
import pytest

from utilities import vid_saver as vs


class _Tex:
    def __init__(self, size):
        self.size = size


class _FakeRecorder:
    instances: list = []

    def __init__(self, width, height, fps=50, output_path=None, **kw):
        self.input_width, self.input_height = width, height
        self.fps, self.output_path = fps, output_path
        self.frames = 0
        self.closed = False
        _FakeRecorder.instances.append(self)

    def write_frame_from_array(self, arr):
        assert arr.shape == (self.input_height, self.input_width, 3), (
            "the array handed to ffmpeg must match the size it was opened with")
        self.frames += 1

    def close(self):
        self.closed = True


class _FakeView:
    """Stands in for the GL crop; records the rects it was asked for."""

    def __init__(self, ctx, size):
        self.size = tuple(size)
        self.rects: list = []
        self.released = False

    def crop(self, src_tex, view_rect):
        self.rects.append(view_rect)
        return _Tex(self.size)

    def release(self):
        self.released = True


@pytest.fixture
def saver(monkeypatch):
    _FakeRecorder.instances = []
    views: list = []

    def make_view(ctx, size):
        v = _FakeView(ctx, size)
        views.append(v)
        return v

    monkeypatch.setattr(vs, "FFmpegVideoRecorder", _FakeRecorder)
    monkeypatch.setattr(vs, "RecordView", make_view)
    monkeypatch.setattr(vs, "reset_gpu_frame_counter", lambda: None)
    monkeypatch.setattr(
        vs, "save_frame_gpu",
        lambda tex, ctx, supersample_k=1, return_array=True: np.zeros(
            (tex.size[1] // supersample_k, tex.size[0] // supersample_k, 3),
            dtype=np.uint8))
    s = vs.VidSaver()
    s.active = True
    s.views = views
    return s


def test_the_size_is_frozen_even_though_the_rect_keeps_moving(saver):
    """The rect is re-derived per frame so the world stays framed, but the
    encoder cannot take a dimension change - so a zoom must rescale."""
    tex = _Tex((1920, 1080))
    saver.frame(None, tex, ssk_w=1, view_rect=((0.25, 0.0), (0.75, 1.0)))
    saver.frame(None, tex, ssk_w=1, view_rect=((0.1, 0.2), (0.9, 0.8)))
    saver.frame(None, tex, ssk_w=1, view_rect=((0.0, 0.0), (1.0, 1.0)))

    assert len(_FakeRecorder.instances) == 1, "the take must not be split"
    rec = _FakeRecorder.instances[0]
    assert (rec.input_width, rec.input_height) == (960, 1080)
    assert rec.frames == 3


def test_every_frame_is_cropped_with_its_own_rect(saver):
    tex = _Tex((1920, 1080))
    first = ((0.25, 0.0), (0.75, 1.0))
    second = ((0.1, 0.2), (0.9, 0.8))
    saver.frame(None, tex, ssk_w=1, view_rect=first)
    saver.frame(None, tex, ssk_w=1, view_rect=second)
    assert saver.views[0].rects == [first, second]


def test_a_window_resize_no_longer_splits_a_cropped_take(saver):
    """Output dimensions stop tracking the window once the crop target is
    fixed, so a resize only changes sampling resolution."""
    saver.frame(None, _Tex((1920, 1080)), ssk_w=1,
                view_rect=((0.25, 0.0), (0.75, 1.0)))
    saver.frame(None, _Tex((1600, 900)), ssk_w=1,
                view_rect=((0.25, 0.0), (0.75, 1.0)))
    assert len(_FakeRecorder.instances) == 1


def test_without_a_rect_the_recording_is_unchanged(saver):
    """The raw canvas views pass no rect and must behave exactly as before."""
    saver.frame(None, _Tex((800, 600)), ssk_w=2)
    rec = _FakeRecorder.instances[0]
    assert (rec.input_width, rec.input_height) == (400, 300)
    assert saver.views == [], "no crop target should be built without a rect"


def test_without_a_rect_a_resize_still_splits_the_take(saver):
    """That check is what stops ffmpeg being handed the wrong frame size."""
    saver.frame(None, _Tex((800, 600)), ssk_w=1)
    saver.frame(None, _Tex((640, 480)), ssk_w=1)
    assert len(_FakeRecorder.instances) == 2


def test_supersampling_is_applied_to_the_cropped_target(saver):
    saver.frame(None, _Tex((1920, 1080)), ssk_w=2,
                view_rect=((0.0, 0.0), (1.0, 1.0)))
    rec = _FakeRecorder.instances[0]
    assert (rec.input_width, rec.input_height) == (960, 540)
    assert saver.views[0].size == (1920, 1080)


def test_finishing_releases_the_crop_target(saver):
    saver.frame(None, _Tex((1920, 1080)), ssk_w=1,
                view_rect=((0.25, 0.0), (0.75, 1.0)))
    view = saver.views[0]
    saver.finish()
    assert view.released, "the FBO would leak once per take otherwise"
    assert not saver.active


def test_a_second_take_builds_a_fresh_target(saver):
    saver.frame(None, _Tex((1920, 1080)), ssk_w=1,
                view_rect=((0.25, 0.0), (0.75, 1.0)))
    saver.finish()
    saver.active = True
    saver.frame(None, _Tex((1920, 1080)), ssk_w=1,
                view_rect=((0.0, 0.0), (1.0, 1.0)))
    assert len(saver.views) == 2
    assert saver.views[1].size == (1920, 1080)
