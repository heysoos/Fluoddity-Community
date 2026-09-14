"""The capture child: a dumb pump from the device to a pipe.

Driven in-process on a thread with a fake PortAudio, over real pipes, so
every message the parent will see is the one under test.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from utilities import audio_worker


class _FakeStream:
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        self.closed = False

    def start_stream(self):
        self.running = True

    def stop_stream(self):
        self.running = False

    def close(self):
        self.closed = True

    def feed(self, data: bytes, status: int = 0):
        """What the device does: hand the callback one block."""
        return self.callback(data, len(data) // 4, None, status)


class _FakePyAudio:
    def __init__(self, module):
        self.module = module
        self.streams: list[_FakeStream] = []
        self.terminated = False

    def get_default_input_device_info(self):
        return self.module.devices[0]

    def get_device_info_by_index(self, index):
        for d in self.module.devices:
            if d["index"] == index:
                return d
        raise OSError(f"Invalid device index {index}")

    def open(self, **kw):
        if self.module.fail_open:
            raise OSError("device is busy")
        self.module.opened.append(kw)
        stream = _FakeStream(kw["stream_callback"])
        self.streams.append(stream)
        return stream

    def terminate(self):
        self.terminated = True


class _FakePa:
    """The slice of pyaudiowpatch the worker touches."""
    paFloat32 = 1
    paContinue = 0
    paInputOverflow = 2

    def __init__(self, devices=None, fail_open=False):
        self.devices = devices or [
            {"index": 3, "name": "Mic", "defaultSampleRate": 48000.0,
             "maxInputChannels": 2}]
        self.fail_open = fail_open
        self.opened: list[dict] = []
        self.instances: list[_FakePyAudio] = []

    def PyAudio(self):
        inst = _FakePyAudio(self)
        self.instances.append(inst)
        return inst


class _Child:
    """serve() on a thread, with the parent's ends of both pipes."""

    def __init__(self, pa: _FakePa):
        self.pa = pa
        self.ctrl, ctrl_child = mp.Pipe()
        self.data, data_child = mp.Pipe(duplex=False)
        self.thread = threading.Thread(
            target=audio_worker.serve, args=(ctrl_child, data_child, pa),
            daemon=True)
        self.thread.start()

    def request(self, msg, timeout=5.0):
        self.ctrl.send(msg)
        assert self.ctrl.poll(timeout), f"no reply to {msg!r}"
        return self.ctrl.recv()

    def stream(self) -> _FakeStream:
        return self.pa.instances[-1].streams[-1]

    def finish(self, timeout=5.0):
        try:
            self.ctrl.send(None)
        except (OSError, ValueError):
            pass
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the worker did not exit"


@pytest.fixture
def child():
    pa = _FakePa()
    c = _Child(pa)
    yield c
    c.finish()


def _wait(pred, timeout=5.0):
    t0 = time.perf_counter()
    while not pred():
        if time.perf_counter() - t0 > timeout:
            return False
        time.sleep(0.005)
    return True


def test_opening_a_device_reports_its_rate_and_channels(child):
    assert child.request(("open", 3, 512)) == ("opened", 48000.0, 2)


def test_the_default_device_is_index_none(child):
    assert child.request(("open", None, 512))[0] == "opened"
    assert child.pa.opened[-1]["input_device_index"] == 3


def test_the_block_size_asked_for_is_the_one_opened(child):
    child.request(("open", 3, 1024))
    assert child.pa.opened[-1]["frames_per_buffer"] == 1024


def test_a_device_that_cannot_open_is_an_error_reply_not_a_dead_child(child):
    child.pa.fail_open = True
    kind, message = child.request(("open", 3, 512))
    assert kind == "error" and "busy" in message
    child.pa.fail_open = False
    assert child.request(("open", 3, 512))[0] == "opened"


def test_an_unknown_index_is_an_error_reply(child):
    kind, message = child.request(("open", 999999, 512))
    assert kind == "error" and message


def test_device_blocks_reach_the_data_pipe_verbatim_and_in_order(child):
    child.request(("open", 3, 512))
    blocks = [np.full(1024, i, dtype="f4").tobytes() for i in range(5)]
    for b in blocks:
        child.stream().feed(b)
    got = []
    while len(got) < 5 and _wait(child.data.poll, timeout=2.0):
        got.append(child.data.recv_bytes())
    assert got == blocks


def test_an_overflow_flag_becomes_an_empty_marker_before_the_block(child):
    child.request(("open", 3, 512))
    child.stream().feed(b"\x00" * 64, status=_FakePa.paInputOverflow)
    assert _wait(child.data.poll)
    assert child.data.recv_bytes() == b""
    assert _wait(child.data.poll)
    assert child.data.recv_bytes() == b"\x00" * 64


def test_the_callback_never_blocks_on_a_parent_that_is_not_reading(child):
    """The whole point: the parent stalls, the child keeps taking blocks."""
    child.request(("open", 3, 512))
    stream = child.stream()
    block = b"\x00" * 4096
    t0 = time.perf_counter()
    for _ in range(2000):                   # ~8 MB nobody is reading
        stream.feed(block)
    assert time.perf_counter() - t0 < 2.0
    n = 0
    while _wait(child.data.poll, timeout=1.0) and n < 2000:
        assert child.data.recv_bytes() == block
        n += 1
    assert n == 2000, "blocks were lost while the parent was away"


def test_the_queue_is_capped_and_a_dropped_block_is_marked(child, monkeypatch):
    monkeypatch.setattr(audio_worker, "MAX_QUEUED_BLOCKS", 8)
    child.request(("open", 3, 512))
    stream = child.stream()
    # Hold the pump so the queue fills: the parent is not reading and the
    # pipe itself absorbs some, so feed well past both.
    for i in range(400):
        stream.feed(bytes([i % 256]) * 4096)
    got = []
    while _wait(child.data.poll, timeout=0.5):
        got.append(child.data.recv_bytes())
    assert len([g for g in got if g]) < 400
    assert b"" in got, "a drop must be reported, never silent"


def test_closing_stops_the_stream_and_reopening_makes_a_new_one(child):
    child.request(("open", 3, 512))
    first = child.stream()
    assert child.request(("close",)) == ("closed",)
    assert not first.running and first.closed
    child.request(("open", 3, 512))
    assert child.stream() is not first and child.stream().running


def test_opening_over_an_open_stream_closes_the_old_one(child):
    child.request(("open", 3, 512))
    first = child.stream()
    child.request(("open", 3, 512))
    assert first.closed and child.stream() is not first


def test_closing_when_nothing_is_open_is_harmless(child):
    assert child.request(("close",)) == ("closed",)


def test_losing_the_parent_stops_the_device_and_ends_the_worker():
    """An orphan must not keep the microphone open in the background."""
    pa = _FakePa()
    c = _Child(pa)
    c.request(("open", 3, 512))
    stream = c.stream()
    c.ctrl.close()
    c.data.close()
    c.thread.join(5.0)
    assert not c.thread.is_alive()
    assert stream.closed and pa.instances[-1].terminated


def test_the_stop_message_closes_everything(child):
    child.request(("open", 3, 512))
    stream = child.stream()
    child.finish()
    assert stream.closed and child.pa.instances[-1].terminated
