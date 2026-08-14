"""The soundtrack sidecar for a recording.

Raw interleaved float32, exactly as the device delivered it - ffmpeg reads it
back with `-f f32le -ar <rate> -ac <channels>`, so converting anything here
would only be a chance to get it wrong.

Written from the audio callback thread, which must never see an exception: a
failure costs the soundtrack and leaves the recording alone.
"""
from __future__ import annotations

BYTES_PER_SAMPLE = 4        # float32


class AudioTrackWriter:
    """Appends captured PCM to one file. Not thread-safe by design.

    Only the capture callback writes, and only the main thread opens and
    closes, so the two never touch it at once.
    """

    def __init__(self, path: str, sample_rate: float, channels: int) -> None:
        self.path = path
        self.sample_rate = float(sample_rate)
        self.channels = max(1, int(channels))
        self.bytes_written = 0
        self.last_error = ""
        self._sink = None
        try:
            handle = open(path, "wb")
        except Exception as exc:
            self.ok = False
            self.last_error = str(exc) or repr(exc)
            return
        self._handle = handle
        self._sink = handle.write
        self.ok = True

    @property
    def duration_seconds(self) -> float:
        """How much audio this file holds - the recording's clock."""
        per_second = BYTES_PER_SAMPLE * self.channels * self.sample_rate
        if per_second <= 0.0:
            return 0.0
        return self.bytes_written / per_second

    def write(self, data: bytes) -> None:
        sink = self._sink
        if sink is None:
            return
        try:
            sink(data)
        except Exception as exc:
            # Losing the soundtrack is survivable; killing the audio thread is
            # not, and it would take the analyser down with it.
            self._sink = None
            self.ok = False
            self.last_error = str(exc) or repr(exc)
            return
        self.bytes_written += len(data)

    def close(self) -> None:
        self._sink = None
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        self._handle = None
        try:
            handle.close()
        except Exception as exc:
            self.last_error = str(exc) or repr(exc)
