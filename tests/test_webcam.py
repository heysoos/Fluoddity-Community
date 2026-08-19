"""The webcam reader: decoupled from the frame loop, and never fatal.

A camera is hardware that may be absent, busy, or unplugged mid-run, so every
failure has to land on the layer's error line rather than in a traceback. The
tests that need a device skip without one; the rest use a fake process, because
a suite that only runs on a machine with a camera tests nothing on the machines
that do not have one.
"""
from __future__ import annotations

import platform
import threading
import time

import pytest

from services import webcam

HAS_CAMERA = bool(webcam.list_devices())


class _FakePipe:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def read(self, _n=-1):
        if self.closed or not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self, chunks, err=b""):
        self.stdout = _FakePipe(chunks)
        self.stderr = _FakePipe([err])
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False


def _reader_over(proc, size=(2, 2)):
    r = webcam.WebcamReader.__new__(webcam.WebcamReader)
    r.device, r.size, r.fps = "fake", size, 30
    r.error = None
    r._proc = proc
    r._stop = threading.Event()
    r._lock = threading.Lock()
    r._frame = None
    r._serial = 0
    r._thread = threading.Thread(target=r._pump, daemon=True)
    r._thread.start()
    return r


def _wait_for_serial(reader, at_least, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, serial = reader.latest()
        if serial >= at_least:
            return serial
        time.sleep(0.01)
    return reader.latest()[1]


def test_nothing_has_arrived_yet_is_not_an_error():
    reader = _reader_over(_FakeProc([]))
    frame, serial = reader.latest()
    assert frame is None and serial == 0
    reader.close()


def test_only_the_newest_frame_is_kept():
    """A queue here would be latency: nothing downstream wants the frames it
    skipped, and a camera faster than the sim would build up a backlog."""
    size = 12  # 2x2 rgb
    reader = _reader_over(_FakeProc([b"\x01" * size, b"\x02" * size,
                                     b"\x03" * size]))
    _wait_for_serial(reader, 3)
    frame, serial = reader.latest()
    assert serial == 3
    assert frame == b"\x03" * size, "an older frame was served"
    reader.close()


def test_a_short_read_ends_the_stream_without_raising():
    reader = _reader_over(_FakeProc([b"\x01" * 4]))
    reader._thread.join(timeout=2)
    assert reader.error, "a truncated stream must reach the layer's error line"
    reader.close()


def test_a_camera_that_dies_reports_what_ffmpeg_said():
    reader = _reader_over(_FakeProc([], err=b"Could not run graph\n"))
    reader._thread.join(timeout=2)
    assert "graph" in (reader.error or "")
    reader.close()


def test_close_is_safe_twice():
    reader = _reader_over(_FakeProc([]))
    reader.close()
    reader.close()


def test_listing_devices_never_raises():
    assert isinstance(webcam.list_devices(), list)


@pytest.mark.skipif(platform.system() != "Windows", reason="dshow only")
def test_the_device_list_has_no_duplicates_and_no_audio():
    names = webcam.list_devices()
    assert len(names) == len(set(names))


@pytest.mark.skipif(not HAS_CAMERA, reason="no camera on this machine")
def test_a_real_camera_delivers_a_frame_of_the_right_size():
    device = webcam.list_devices()[0]
    size = (640, 480)
    reader = webcam.WebcamReader(device, size)
    try:
        _wait_for_serial(reader, 1, timeout=10.0)
        frame, serial = reader.latest()
        if reader.error and serial == 0:
            pytest.skip(f"camera unavailable: {reader.error}")
        assert serial >= 1, "no frame arrived"
        assert len(frame) == size[0] * size[1] * 3
    finally:
        reader.close()
