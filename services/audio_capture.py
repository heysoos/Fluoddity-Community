"""The only module here that performs IO.

The device is read by a child process; a reader thread here analyses each
block and publishes an immutable snapshot, which the frame loop reads without
blocking. See `utilities/audio_worker.py` for why it is a process.
"""
from __future__ import annotations

import threading

import numpy as np

from services.audio_analysis import FFT_SIZE, HOP, Analyzer, SignalSnapshot

# The analyser derives its smoothing from this, so the two must agree.
_BLOCK = HOP


def _pyaudio():
    """Imported here, never at module scope - it is an optional dependency."""
    import pyaudiowpatch
    return pyaudiowpatch


def is_available() -> bool:
    try:
        _pyaudio()
        return True
    except Exception:
        return False


DEFAULT_DEVICE_NAME = "(Default)"

# MME cuts a device name to this many characters, so a name at exactly this
# length may be a longer one cut short.
_MME_NAME_LIMIT = 31

# A host API this build does not rank: preferred over the legacy wrappers,
# never over WASAPI.
_UNRANKED_API = 1


def _api_ranks(pa_mod) -> dict[int, int]:
    """How much each host API is preferred, lowest first.

    Windows offers one row per host API, so the same microphone is listed
    several times. WASAPI is the only one carrying loopback endpoints, and it
    reports a device's real sample rate where MME reports a fixed one.
    """
    return {pa_mod.paWASAPI: 0, pa_mod.paDirectSound: 2, pa_mod.paMME: 3}


def _device_row(info, apis: dict, loopback: bool) -> dict:
    api, rank = apis.get(int(info.get("hostApi", -1)), ("", _UNRANKED_API))
    return {"index": int(info["index"]), "name": str(info["name"]),
            "loopback": loopback, "rate": float(info["defaultSampleRate"]),
            "api": api, "rank": rank}


def _same_device(kept: dict, other: dict) -> bool:
    """Do two rows name one device, allowing for MME's truncation?

    `kept` is the better-ranked row, so it holds the full name. A name under
    the MME limit was never cut and must match in full - a prefix test over
    those pairs two devices that merely start alike.
    """
    if kept["loopback"] != other["loopback"]:
        return False
    full, short = kept["name"], other["name"]
    if full == short:
        return True
    return (len(short) == _MME_NAME_LIMIT and len(full) > len(short)
            and full.startswith(short))


def dedupe_devices(devices: list[dict]) -> list[dict]:
    """One row per device: loopbacks first, then the best-ranked host API.

    A device that no better-ranked API offers is kept as it stands.
    """
    kept: list[dict] = []
    for d in sorted(devices, key=lambda d: (not d["loopback"], d["rank"],
                                            d["index"])):
        if not any(_same_device(k, d) for k in kept):
            kept.append(d)
    return kept


def list_devices() -> list[dict]:
    """Inputs and loopbacks, one row per device, or [] if the package is absent.

    The selection is keyed by NAME, in the combo and again at Start, so a
    repeated name makes the second row unselectable - see `dedupe_devices`.
    """
    if not is_available():
        return []
    pa_mod = _pyaudio()
    pa = None
    out: list[dict] = []
    try:
        pa = pa_mod.PyAudio()
        ranks = _api_ranks(pa_mod)
        apis = {}
        for a in range(pa.get_host_api_count()):
            try:
                api = pa.get_host_api_info_by_index(a)
            except Exception:
                continue
            apis[a] = (str(api["name"]), ranks.get(int(api["type"]),
                                                   _UNRANKED_API))
        seen = set()
        try:
            for info in pa.get_loopback_device_info_generator():
                out.append(_device_row(info, apis, loopback=True))
                seen.add(int(info["index"]))
        except Exception:
            pass
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if i in seen or int(info.get("maxInputChannels", 0)) < 1:
                continue
            out.append(_device_row(info, apis, loopback=False))
    except Exception:
        pass
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
    return dedupe_devices(out)


def device_choices() -> list[dict]:
    """What the Device combo offers: (Default) first, then the devices.

    (Default) carries no index - it is whichever input Windows calls the
    default at the moment Start is pressed.
    """
    return [{"index": None, "name": DEFAULT_DEVICE_NAME, "loopback": False,
             "rate": 0.0, "api": "", "rank": 0}] + list_devices()


# How long Start waits for the child to open a device. A cold child is a
# fresh interpreter importing PortAudio, which a scanned disk can make slow.
START_TIMEOUT = 20.0
STOP_TIMEOUT = 3.0


class _Child:
    """The capture process and the parent's ends of its two pipes.

    Spawned once and kept across Start/Stop, so the launch is paid at most
    once a session. `alive` is False once it has died or been killed.
    """

    def __init__(self, ctrl, data, proc) -> None:
        self.ctrl = ctrl
        self.data = data
        self.proc = proc

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def request(self, msg, timeout: float):
        """Send one command and wait for its reply; None on silence."""
        try:
            self.ctrl.send(msg)
            if not self.ctrl.poll(timeout):
                return None
            return self.ctrl.recv()
        except (EOFError, OSError, ValueError, BrokenPipeError):
            return None

    def kill(self) -> None:
        for conn in (self.ctrl, self.data):
            try:
                conn.close()
            except OSError:
                pass
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()


def _spawn_child() -> _Child:
    """A real child process running `utilities.audio_worker.serve`."""
    import multiprocessing as mp
    from utilities import audio_worker
    ctx = mp.get_context("spawn")
    ctrl, ctrl_child = ctx.Pipe()
    data, data_child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=audio_worker.serve,
                       args=(ctrl_child, data_child),
                       daemon=True, name="fluoddity-audio")
    proc.start()
    ctrl_child.close()
    data_child.close()
    return _Child(ctrl, data, proc)


class AudioCapture:
    """Owns the analyser and the child that owns the device.

    Never raises into the frame loop. The device is read in a separate
    process (`utilities/audio_worker.py`) so a frame that holds the
    interpreter cannot starve the driver; blocks arrive on a reader thread
    here, which runs the analyser and the recording tap in device order.
    """

    def __init__(self, spawn=None) -> None:
        self.status = "idle"
        self.last_error = ""
        self.sample_rate = 0.0
        self._spawn = spawn or _spawn_child
        self._child: _Child | None = None
        self._reader: threading.Thread | None = None
        self._open = False
        self._analyzer: Analyzer | None = None
        self._snapshot: SignalSnapshot | None = None
        self._lock = threading.Lock()
        self._tail = np.zeros(FFT_SIZE, dtype=np.float32)
        self._channels = 1
        # Set while a recording wants the soundtrack; receives each block's
        # bytes verbatim. Read once into a local by the reader and cleared
        # before teardown, exactly as _analyzer is.
        self.tap = None
        # Blocks the device or the child had to throw away since Start. A
        # take that saw any is reported, since its soundtrack has a hole.
        self.dropped = 0
        # Remembered so a Start picks up settings pushed while nothing ran.
        self._bands: dict = {}
        self._release: float | None = None

    @property
    def channels(self) -> int:
        """How the tap's bytes are interleaved; the sidecar carries no header."""
        return self._channels

    def snapshot(self) -> SignalSnapshot | None:
        with self._lock:
            return self._snapshot

    def set_auto_gain(self, on: bool) -> None:
        """Retune a RUNNING analyser. A no-op when nothing is capturing."""
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.set_auto_gain(on)

    def set_bands(self, bands: dict | None) -> None:
        """The per-band measure and dB window, pushed while blocks arrive."""
        self._bands = dict(bands or {})
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.set_bands(self._bands)

    def set_release(self, seconds: float) -> None:
        self._release = max(0.0, float(seconds))
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.set_release(self._release)

    def _on_block(self, in_data, _frame_count, _time_info, _status):
        tap = self.tap
        if tap is not None:
            # Guarded separately from the analysis: the signals driving the sim
            # must not stop because a recording's disk filled up.
            try:
                tap(in_data)
            except Exception as exc:
                self.last_error = repr(exc)
        try:
            analyzer = self._analyzer
            if analyzer is not None:
                samples = np.frombuffer(in_data, dtype=np.float32)
                if self._channels > 1:
                    samples = samples.reshape(-1, self._channels).mean(axis=1)
                # Keep one FFT window of history so a small callback still
                # analyses a full block.
                n = min(samples.size, FFT_SIZE)
                if n:
                    self._tail = np.roll(self._tail, -n)
                    self._tail[FFT_SIZE - n:] = samples[-n:]
                snap = analyzer.process(self._tail)
                with self._lock:
                    self._snapshot = snap
        except Exception as exc:            # never kill the audio thread
            self.last_error = repr(exc)

    def _read(self, child: _Child) -> None:
        """The reader thread: one per child, ends when its pipe does."""
        data = child.data
        while True:
            try:
                block = data.recv_bytes()
            except (EOFError, OSError, ValueError):
                return
            if not block:
                self.dropped += 1
                continue
            self._on_block(block, 0, None, 0)

    def prewarm(self) -> None:
        """Launch the child now, so Start pays only the device open."""
        try:
            self._ensure_child()
        except Exception as exc:            # noqa: BLE001 - Start retries
            self.last_error = str(exc) or repr(exc)

    def _ensure_child(self) -> _Child:
        child = self._child
        if child is not None and child.alive:
            return child
        if child is not None:
            child.kill()
        child = self._spawn()
        self._child = child
        self._reader = threading.Thread(target=self._read, args=(child,),
                                        daemon=True,
                                        name="fluoddity-audio-reader")
        self._reader.start()
        return child

    def start(self, device_index: int | None, auto_gain: bool = True) -> bool:
        self.stop()
        if not is_available():
            self.status = "error"
            self.last_error = ("PyAudioWPatch is not installed, so audio input "
                               "is unavailable.")
            return False
        try:
            child = self._ensure_child()
            reply = child.request(("open", device_index, _BLOCK),
                                  START_TIMEOUT)
        except Exception as exc:            # noqa: BLE001
            reply = ("error", str(exc) or repr(exc))
        if reply is None:
            # A child that does not answer is not one to keep.
            self._drop_child()
            reply = ("error", "the audio process did not answer")
        if reply[0] != "opened":
            self.status = "error"
            self.last_error = str(reply[1]) if len(reply) > 1 else "unknown"
            return False
        _, rate, channels = reply
        self.sample_rate = float(rate)
        self._channels = max(1, int(channels))
        self._tail = np.zeros(FFT_SIZE, dtype=np.float32)
        self.dropped = 0
        # Installed once the device is open, so no block is analysed under
        # the previous device's rate.
        self._analyzer = Analyzer(self.sample_rate, auto_gain=auto_gain,
                                  bands=self._bands, release=self._release)
        self._open = True
        self.status = "active"
        self.last_error = ""
        return True

    def fail(self, message: str) -> None:
        """Report a device the caller could not resolve, with none to open."""
        self.stop()
        self.status = "error"
        self.last_error = message

    def stop(self) -> None:
        # The analyser and the tap go first, so a block still in the pipe
        # publishes nothing into a stream that is being torn down and writes
        # nothing into a file the main thread is closing.
        self._analyzer = None
        self.tap = None
        child = self._child
        if self._open and child is not None and child.alive:
            if child.request(("close",), STOP_TIMEOUT) is None:
                self._drop_child()
        self._open = False
        with self._lock:
            self._snapshot = None
        self.status = "idle"

    def shutdown(self) -> None:
        """Stop, and end the child. The last thing the runtime does."""
        self.stop()
        child = self._child
        if child is not None:
            try:
                child.ctrl.send(None)
            except (EOFError, OSError, ValueError, BrokenPipeError):
                pass
            if child.proc is not None:
                child.proc.join(STOP_TIMEOUT)
        self._drop_child()

    def _drop_child(self) -> None:
        child, self._child = self._child, None
        if child is not None:
            child.kill()
