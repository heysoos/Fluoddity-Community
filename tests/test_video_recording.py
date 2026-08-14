"""The recording path, which used to run entirely on the frame loop.

Three separable costs, one test group each: the readback must not stall, the
bytes must reach ffmpeg without a host pass over them, and `stdin.write` must
never happen on the caller's thread. See the recording caveat in CLAUDE.md.
"""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np
import pytest

from utilities import ffmpeg_recorder
from utilities.ffmpeg_recorder import FFmpegVideoRecorder


# --- what ffmpeg is asked for ------------------------------------------------

def test_the_encoder_is_fed_rgba_so_no_host_pass_drops_the_alpha(monkeypatch):
    """The GPU writes RGBA8; asking ffmpeg for rgb24 would put a repack of the
    whole image back on the frame loop."""
    seen = {}
    monkeypatch.setattr(ffmpeg_recorder, "find_ffmpeg", lambda: "ffmpeg")

    def fake_popen(argv, **kw):
        seen["argv"] = argv
        return _FakeProc(_FakeStdin())

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    rec = FFmpegVideoRecorder(width=4, height=2, output_path="x.mp4")
    argv = seen["argv"]
    assert argv[argv.index("-pixel_format") + 1] == "rgba"
    assert argv[argv.index("-preset") + 1] == ffmpeg_recorder.PRESET
    assert rec.frame_bytes == 4 * 2 * 4


def test_the_soundtrack_still_decides_the_real_rate():
    """The -framerate above is nominal. A take with audio is retimed at the
    mux from frames divided by the soundtrack's own length, so the encoder's
    header is not what sync depends on."""
    fps, warn = ffmpeg_recorder.recording_fps(300, 6.0, 6.0, 50.0)
    assert fps == pytest.approx(50.0) and warn == ""
    slow, warn = ffmpeg_recorder.recording_fps(300, 12.0, 12.0, 50.0)
    assert slow == pytest.approx(25.0), "a slow render must stretch, not sync"


def test_the_preset_is_one_that_can_keep_up():
    """`slow` drains under the sim's framerate at 1080p, so the queue only
    postpones the stall."""
    assert ffmpeg_recorder.PRESET in ("ultrafast", "superfast", "veryfast")


# --- the writer thread -------------------------------------------------------

class _FakeStdin:
    """Blocks like a full pipe until it is let go."""

    def __init__(self):
        self.gate = threading.Event()
        self.written = []

    def write(self, data):
        self.gate.wait(timeout=5.0)
        self.written.append(bytes(data))

    def flush(self):
        pass

    def close(self):
        pass


class _FakeProc:
    returncode = None

    def __init__(self, stdin):
        self.stdin = stdin

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def _recorder(monkeypatch, w=4, h=2, stdin=None):
    stdin = stdin or _FakeStdin()
    monkeypatch.setattr(ffmpeg_recorder, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **kw: _FakeProc(stdin))
    return FFmpegVideoRecorder(width=w, height=h, output_path="x.mp4"), stdin


def test_a_blocked_pipe_does_not_block_the_caller(monkeypatch):
    """The whole point: ffmpeg's pace must not become the frame loop's."""
    rec, stdin = _recorder(monkeypatch)
    frame = b"\x00" * rec.frame_bytes

    t = time.perf_counter()
    rec.write_frame(frame)
    rec.write_frame(frame)
    elapsed = time.perf_counter() - t

    assert elapsed < 0.5, "write_frame waited on the stalled pipe"
    assert stdin.written == [], "nothing should have reached a blocked pipe"
    stdin.gate.set()


def test_the_queue_is_bounded_so_a_backlog_cannot_eat_memory(monkeypatch):
    rec, stdin = _recorder(monkeypatch)
    assert rec._queue.maxsize >= ffmpeg_recorder._QUEUE_MIN
    assert rec._queue.maxsize <= ffmpeg_recorder._QUEUE_MAX
    stdin.gate.set()


def test_a_full_queue_blocks_rather_than_dropping_a_frame(monkeypatch):
    """Every frame the sim produced belongs in the file."""
    rec, stdin = _recorder(monkeypatch)
    frame = b"\x00" * rec.frame_bytes
    done = threading.Event()

    def fill():
        for _ in range(rec._queue.maxsize + 3):
            rec.write_frame(frame)
        done.set()

    want = rec._queue.maxsize + 3
    threading.Thread(target=fill, daemon=True).start()
    assert not done.wait(timeout=0.3), "the queue accepted more than its cap"
    stdin.gate.set()
    assert done.wait(timeout=5.0), "the backlog never drained"
    # done fires when the last frame is QUEUED; join waits for it to be written.
    rec._queue.join()
    assert len(stdin.written) == want


def test_everything_queued_is_written_before_close_returns(monkeypatch):
    rec, stdin = _recorder(monkeypatch)
    stdin.gate.set()
    for _ in range(5):
        rec.write_frame(b"\x00" * rec.frame_bytes)
    rec.close()
    assert len(stdin.written) == 5


def test_a_writer_failure_reaches_the_caller(monkeypatch):
    """An exception on the writer thread would otherwise be silent."""
    class _Broken(_FakeStdin):
        def write(self, data):
            raise BrokenPipeError("pipe closed")

    rec, _ = _recorder(monkeypatch, stdin=_Broken())
    rec.write_frame(b"\x00" * rec.frame_bytes)
    for _ in range(100):
        if rec._error is not None:
            break
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="Failed to write frame"):
        rec.write_frame(b"\x00" * rec.frame_bytes)


def test_odd_dimensions_are_refused_rather_than_padded(monkeypatch):
    """H.264 will not take them, and the reader already rounds down; padding
    per frame was a host-side rebuild of the whole image."""
    monkeypatch.setattr(ffmpeg_recorder, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **kw: _FakeProc(_FakeStdin()))
    with pytest.raises(ValueError, match="even dimensions"):
        FFmpegVideoRecorder(width=647, height=480, output_path="x.mp4")


def test_a_wrong_sized_frame_is_refused(monkeypatch):
    rec, stdin = _recorder(monkeypatch)
    with pytest.raises(ValueError, match="expected"):
        rec.write_frame(b"\x00" * (rec.frame_bytes - 4))
    stdin.gate.set()


# --- the GPU half ------------------------------------------------------------

moderngl = pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU on this machine
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def _source(ctx, w, h, colour=(1.0, 0.0, 0.0, 1.0)):
    tex = ctx.texture((w, h), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    fbo.clear(*colour)
    return tex, fbo


def test_the_reader_rounds_the_output_down_to_even(ctx):
    from utilities.save_frame_gpu import AsyncFrameReader
    r = AsyncFrameReader()
    tex, fbo = _source(ctx, 647, 481)
    r.submit(ctx, tex, 1)
    assert r.size == (646, 480)
    r.release()
    fbo.release()
    tex.release()


def test_the_first_submit_yields_nothing_and_the_next_yields_the_first(ctx):
    """One frame of latency is what buys the readback its asynchrony."""
    from utilities.save_frame_gpu import AsyncFrameReader
    r = AsyncFrameReader()
    tex, fbo = _source(ctx, 64, 64, (1.0, 0.0, 0.0, 1.0))

    first, _ = r.submit(ctx, tex, 1)
    assert first is None

    fbo.use()
    fbo.clear(0.0, 1.0, 0.0, 1.0)
    second, _ = r.submit(ctx, tex, 1)
    assert second is not None
    assert bytes(second[:4]) == bytes((255, 0, 0, 255)), "got the wrong frame"

    tail = r.drain()
    assert bytes(tail[:4]) == bytes((0, 255, 0, 255))
    assert r.drain() is None
    r.release()
    fbo.release()
    tex.release()


def test_the_readback_is_rgba_bytes_of_exactly_the_right_length(ctx):
    from utilities.save_frame_gpu import AsyncFrameReader
    r = AsyncFrameReader()
    tex, fbo = _source(ctx, 128, 64)
    r.submit(ctx, tex, 1)
    data = r.drain()
    w, h = r.size
    assert len(data) == w * h * 4
    r.release()
    fbo.release()
    tex.release()


def test_the_shader_flips_so_no_host_pass_has_to(ctx):
    """GL reads bottom-up; the encoder wants top-down. Doing it in the vertex
    shader is what removes np.flipud and its copy from every frame."""
    from utilities.save_frame_gpu import AsyncFrameReader

    w = h = 64
    tex = ctx.texture((w, h), 4, dtype="f4")
    # Bottom half red, top half green, in GL's own orientation (v=0 at bottom).
    px = np.zeros((h, w, 4), dtype=np.float32)
    px[: h // 2] = (1.0, 0.0, 0.0, 1.0)
    px[h // 2:] = (0.0, 1.0, 0.0, 1.0)
    tex.write(px.tobytes())

    r = AsyncFrameReader()
    r.submit(ctx, tex, 1)
    out = np.frombuffer(r.drain(), dtype=np.uint8).reshape(h, w, 4)
    # Row 0 of the readback must be the IMAGE's top row, which is GL's top.
    assert tuple(out[0, 0, :3]) == (0, 255, 0)
    assert tuple(out[-1, 0, :3]) == (255, 0, 0)
    r.release()
    tex.release()


def test_the_screenshot_path_agrees_with_the_recorder_on_which_way_is_up(ctx):
    """Both go through the same shader. The flip moved INTO it, so a leftover
    np.flipud here would put screenshots upside down and nothing would say so."""
    from utilities.save_frame_gpu import (AsyncFrameReader, save_frame_gpu,
                                          cleanup_gpu_supersampling)

    w = h = 64
    tex = ctx.texture((w, h), 4, dtype="f4")
    px = np.zeros((h, w, 4), dtype=np.float32)
    px[: h // 2] = (1.0, 0.0, 0.0, 1.0)
    px[h // 2:] = (0.0, 1.0, 0.0, 1.0)
    tex.write(px.tobytes())

    shot = save_frame_gpu(tex, ctx, supersample_k=1, return_array=True)
    assert shot.shape == (h, w, 3)
    assert shot.dtype == np.uint8

    r = AsyncFrameReader()
    r.submit(ctx, tex, 1)
    recorded = np.frombuffer(r.drain(), dtype=np.uint8).reshape(h, w, 4)
    assert np.array_equal(shot, recorded[:, :, :3]), "the two paths disagree"

    r.release()
    cleanup_gpu_supersampling()
    tex.release()


def test_supersampling_still_shrinks_by_k(ctx):
    from utilities.save_frame_gpu import AsyncFrameReader
    r = AsyncFrameReader()
    tex, fbo = _source(ctx, 256, 128)
    r.submit(ctx, tex, 2)
    assert r.size == (128, 64)
    r.release()
    fbo.release()
    tex.release()


def test_a_resize_mid_recording_is_reported_so_the_file_can_be_restarted(ctx):
    from utilities.save_frame_gpu import AsyncFrameReader
    r = AsyncFrameReader()
    small, fbo_a = _source(ctx, 64, 64)
    r.submit(ctx, small, 1)
    big, fbo_b = _source(ctx, 128, 128)
    data, resized = r.submit(ctx, big, 1)
    assert resized is True
    assert data is None, "a frame of the old size must not reach the new file"
    r.release()
    for o in (fbo_a, fbo_b, small, big):
        o.release()
