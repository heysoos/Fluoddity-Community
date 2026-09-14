"""The audio device is read in a CHILD PROCESS, and this is the child.

A PortAudio callback has to run Python for every block, so a frame loop that
holds the interpreter starves it and the driver throws the audio away. The
child has an interpreter of its own: its callback only queues the block, a
pump thread feeds the pipe, and a parent that stalls finds everything it
missed waiting for it. Nothing here is analysed - the parent keeps the
analyser and the recording tap, so the child needs no settings at all.

Kept free of the `services` package on purpose: that package's import order
is the app's, and the child wants only pyaudiowpatch.
"""
from __future__ import annotations

import queue
import threading

# Blocks queued while the parent is away, past which the oldest are dropped
# and the drop is marked. At the default block that is about a minute.
MAX_QUEUED_BLOCKS = 6000

# An empty message on the data pipe: the device overflowed, or the queue did.
DROP_MARKER = b""


def _pump(q: queue.Queue, data) -> None:
    while True:
        item = q.get()
        if item is None:
            return
        try:
            data.send_bytes(item)
        except (EOFError, OSError, ValueError, BrokenPipeError):
            return


def _reply(ctrl, msg) -> None:
    try:
        ctrl.send(msg)
    except (EOFError, OSError, ValueError, BrokenPipeError):
        pass


def serve(ctrl, data, pa_mod=None) -> None:
    """Answer `("open", index, block)` and `("close",)` on `ctrl`; stream
    device blocks down `data`. Ends on `None` or when the parent is gone."""
    if pa_mod is None:
        import pyaudiowpatch as pa_mod
    pa = pa_mod.PyAudio()
    q: queue.Queue = queue.Queue()
    pump = threading.Thread(target=_pump, args=(q, data), daemon=True,
                            name="fluoddity-audio-pump")
    pump.start()
    stream = None

    def close_stream() -> None:
        nonlocal stream
        if stream is None:
            return
        s, stream = stream, None
        for step in (s.stop_stream, s.close):
            try:
                step()
            except Exception:           # noqa: BLE001 - it is going anyway
                pass

    def on_block(in_data, _frames, _time_info, status):
        # The parent reads a marker as one lost block; it cannot know the
        # length either way, and the count is what it reports.
        if status:
            q.put(DROP_MARKER)
        if q.qsize() >= MAX_QUEUED_BLOCKS:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put(DROP_MARKER)
        q.put(bytes(in_data))
        return (None, pa_mod.paContinue)

    try:
        while True:
            try:
                msg = ctrl.recv()
            except (EOFError, OSError):
                return
            if msg is None:
                return
            if msg[0] == "open":
                close_stream()
                try:
                    _, index, block = msg
                    if index is None:
                        info = pa.get_default_input_device_info()
                    else:
                        info = pa.get_device_info_by_index(int(index))
                    rate = float(info["defaultSampleRate"])
                    channels = max(1, int(info.get("maxInputChannels", 1)))
                    stream = pa.open(
                        format=pa_mod.paFloat32, channels=channels,
                        rate=int(rate), input=True,
                        input_device_index=int(info["index"]),
                        frames_per_buffer=int(block),
                        stream_callback=on_block)
                    stream.start_stream()
                    reply = ("opened", rate, channels)
                except Exception as exc:    # noqa: BLE001 - the parent says why
                    close_stream()
                    reply = ("error", str(exc) or repr(exc))
                _reply(ctrl, reply)
            elif msg[0] == "close":
                close_stream()
                _reply(ctrl, ("closed",))
    finally:
        close_stream()
        q.put(None)
        try:
            pa.terminate()
        except Exception:               # noqa: BLE001
            pass
