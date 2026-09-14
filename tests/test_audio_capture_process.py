"""The parent half of the capture: a child that owns the device, a reader
thread that owns the analyser and the tap.

The child is the real `serve()` on a thread with a fake PortAudio, so the
pipes, the replies and the reader are the ones the app runs.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from services import audio_capture
from tests.test_audio_worker import _FakePa
from utilities import audio_worker


class _ThreadChild:
    """What `_spawn` hands back in these tests: serve() on a thread."""

    def __init__(self, pa):
        self.pa = pa
        self.ctrl, ctrl_child = mp.Pipe()
        self.data, data_child = mp.Pipe(duplex=False)
        self.proc = None
        self.thread = threading.Thread(
            target=audio_worker.serve, args=(ctrl_child, data_child, pa),
            daemon=True)
        self.thread.start()
        self.killed = False

    @property
    def alive(self):
        return self.thread.is_alive()

    def request(self, msg, timeout):
        return audio_capture._Child.request(self, msg, timeout)

    def kill(self):
        self.killed = True
        for conn in (self.ctrl, self.data):
            try:
                conn.close()
            except OSError:
                pass

    def stream(self):
        return self.pa.instances[-1].streams[-1]


@pytest.fixture
def cap(monkeypatch):
    monkeypatch.setattr(audio_capture, "is_available", lambda: True)
    pa = _FakePa()
    spawned = []

    def spawn():
        child = _ThreadChild(pa)
        spawned.append(child)
        return child

    c = audio_capture.AudioCapture(spawn=spawn)
    c.spawned = spawned
    c.pa = pa
    yield c
    c.shutdown()


def _wait(pred, timeout=5.0):
    t0 = time.perf_counter()
    while not pred():
        if time.perf_counter() - t0 > timeout:
            return False
        time.sleep(0.005)
    return True


def _block(channels=2, frames=512, value=0.0):
    return np.full(frames * channels, value, dtype="f4").tobytes()


def test_start_opens_the_device_and_reads_its_rate(cap):
    assert cap.start(3, auto_gain=False) is True
    assert cap.status == "active"
    assert cap.sample_rate == 48000.0 and cap.channels == 2


def test_blocks_from_the_child_reach_the_tap_verbatim_and_in_order(cap):
    cap.start(3)
    got = []
    cap.tap = got.append
    blocks = [_block(value=i) for i in range(4)]
    for b in blocks:
        cap.spawned[-1].stream().feed(b)
    assert _wait(lambda: len(got) == 4)
    assert got == blocks


def test_blocks_from_the_child_are_analysed_into_a_snapshot(cap):
    cap.start(3, auto_gain=False)
    assert cap.snapshot() is None
    cap.spawned[-1].stream().feed(_block(value=0.1))
    assert _wait(lambda: cap.snapshot() is not None)


def test_a_drop_marker_is_counted_and_never_reaches_the_tap(cap):
    cap.start(3)
    got = []
    cap.tap = got.append
    stream = cap.spawned[-1].stream()
    stream.feed(_block(), status=_FakePa.paInputOverflow)
    assert _wait(lambda: len(got) == 1)
    assert cap.dropped == 1
    assert b"" not in got


def test_start_resets_the_drop_count(cap):
    cap.start(3)
    cap.spawned[-1].stream().feed(_block(), status=_FakePa.paInputOverflow)
    assert _wait(lambda: cap.dropped == 1)
    cap.start(3)
    assert cap.dropped == 0


def test_a_device_the_child_cannot_open_is_an_error_not_an_exception(cap):
    cap.pa.fail_open = True
    assert cap.start(3) is False
    assert cap.status == "error" and "busy" in cap.last_error


def test_stop_closes_the_device_but_keeps_the_child_warm(cap):
    cap.start(3)
    stream = cap.spawned[-1].stream()
    cap.stop()
    assert cap.status == "idle"
    assert _wait(lambda: stream.closed)
    assert cap.spawned[-1].alive and len(cap.spawned) == 1
    assert cap.start(3) is True
    assert len(cap.spawned) == 1, "a warm child is reused"


def test_a_block_arriving_after_stop_writes_nothing(cap):
    """The tap and the analyser are dropped BEFORE the child is told, so a
    block already in the pipe lands on nothing."""
    cap.start(3)
    got = []
    cap.tap = got.append
    cap.stop()
    assert cap.tap is None and cap._analyzer is None
    assert cap.snapshot() is None


def test_prewarm_launches_the_child_and_start_reuses_it(cap):
    cap.prewarm()
    assert len(cap.spawned) == 1 and cap.status == "idle"
    cap.start(3)
    assert len(cap.spawned) == 1


def test_a_dead_child_is_replaced_at_the_next_start(cap):
    cap.start(3)
    first = cap.spawned[-1]
    first.ctrl.send(None)               # the child exits on its own
    assert _wait(lambda: not first.alive)
    assert cap.start(3) is True
    assert len(cap.spawned) == 2 and cap.spawned[-1] is not first


def test_shutdown_ends_the_child(cap):
    cap.start(3)
    child = cap.spawned[-1]
    cap.shutdown()
    assert _wait(lambda: not child.alive)
    assert cap.status == "idle"


def test_a_child_that_never_answers_is_an_error_not_a_hang(monkeypatch):
    monkeypatch.setattr(audio_capture, "is_available", lambda: True)
    monkeypatch.setattr(audio_capture, "START_TIMEOUT", 0.2)

    class _Mute:
        alive = True

        def __init__(self):
            self.ctrl, _ = mp.Pipe()
            self.data, _ = mp.Pipe(duplex=False)
            self.proc = None
            self.killed = False

        def request(self, msg, timeout):
            return audio_capture._Child.request(self, msg, timeout)

        def kill(self):
            self.killed = True

    mutes = []

    def spawn():
        mutes.append(_Mute())
        return mutes[-1]

    cap = audio_capture.AudioCapture(spawn=spawn)
    t0 = time.perf_counter()
    assert cap.start(3) is False
    assert time.perf_counter() - t0 < 5.0
    assert cap.status == "error" and "did not answer" in cap.last_error
    assert mutes[0].killed, "a silent child is not kept"


def test_the_real_child_is_a_separate_process():
    """The whole reason this exists: the device is read where the frame
    loop's interpreter cannot starve it."""
    child = audio_capture._spawn_child()
    try:
        assert child.proc.pid != mp.current_process().pid
        assert child.alive
    finally:
        child.kill()
