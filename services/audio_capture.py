"""The only module here that performs IO.

Analysis runs on the capture callback thread and publishes an immutable
snapshot; the frame loop reads the newest one and never blocks. A dropped audio
buffer therefore costs a stale signal rather than a stalled frame.
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


def list_devices() -> list[dict]:
    """Inputs and loopbacks, or an empty list if the package is absent."""
    if not is_available():
        return []
    pa_mod = _pyaudio()
    pa = None
    out: list[dict] = []
    try:
        pa = pa_mod.PyAudio()
        seen = set()
        try:
            for info in pa.get_loopback_device_info_generator():
                out.append({"index": int(info["index"]),
                            "name": str(info["name"]),
                            "loopback": True,
                            "rate": float(info["defaultSampleRate"])})
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
            out.append({"index": i, "name": str(info["name"]),
                        "loopback": False,
                        "rate": float(info["defaultSampleRate"])})
    except Exception:
        return out
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
    return out


class AudioCapture:
    """Owns the stream and the analyser. Never raises into the frame loop."""

    def __init__(self) -> None:
        self.status = "idle"
        self.last_error = ""
        self.sample_rate = 0.0
        self._pa = None
        self._stream = None
        self._analyzer: Analyzer | None = None
        self._snapshot: SignalSnapshot | None = None
        self._lock = threading.Lock()
        self._tail = np.zeros(FFT_SIZE, dtype=np.float32)
        self._channels = 1
        # Set while a recording wants the soundtrack; receives each block's
        # bytes verbatim. Read once into a local by the callback and cleared
        # before teardown, exactly as _analyzer is.
        self.tap = None
        # Captured at start, because the callback must not import anything.
        self._continue = None
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
        return (None, self._continue)

    def start(self, device_index: int | None, auto_gain: bool = True) -> bool:
        self.stop()
        if not is_available():
            self.status = "error"
            self.last_error = ("PyAudioWPatch is not installed, so audio input "
                               "is unavailable.")
            return False
        pa_mod = _pyaudio()
        try:
            self._continue = pa_mod.paContinue
            self._pa = pa_mod.PyAudio()
            if device_index is None:
                info = self._pa.get_default_input_device_info()
                device_index = int(info["index"])
            else:
                info = self._pa.get_device_info_by_index(int(device_index))

            self.sample_rate = float(info["defaultSampleRate"])
            self._channels = max(1, int(info.get("maxInputChannels", 1)))
            self._analyzer = Analyzer(self.sample_rate, auto_gain=auto_gain,
                                      bands=self._bands,
                                      release=self._release)
            self._tail = np.zeros(FFT_SIZE, dtype=np.float32)

            self._stream = self._pa.open(
                format=pa_mod.paFloat32,
                channels=self._channels,
                rate=int(self.sample_rate),
                input=True,
                input_device_index=int(device_index),
                frames_per_buffer=_BLOCK,
                stream_callback=self._on_block,
            )
            self._stream.start_stream()
            self.status = "active"
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc) or repr(exc)
            self.stop()
            self.status = "error"
            return False

    def stop(self) -> None:
        # The analyser and the tap go first, so a callback still in flight
        # publishes nothing into a stream that is being torn down and writes
        # nothing into a file the main thread is closing.
        self._analyzer = None
        self.tap = None
        for close in (self._close_stream, self._close_pa):
            try:
                close()
            except Exception:
                pass
        self._stream = None
        self._pa = None
        with self._lock:
            self._snapshot = None
        self.status = "idle"

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()

    def _close_pa(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
