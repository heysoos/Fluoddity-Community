"""The soundtrack for one take: tap while it runs, mux when it ends.

A take is expensive, so every path through finish() leaves the output file in
place - a soundtrack is worth strictly less than the recording it belongs to.
"""
from __future__ import annotations

import os
import subprocess

from services.audio_track import AudioTrackWriter
from utilities.ffmpeg_recorder import find_ffmpeg, mux_command, recording_fps


def run_mux(cmd, out_path):
    """(ok, error). Patched out in tests; nothing else here runs a process."""
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    except Exception as exc:
        return False, str(exc) or repr(exc)
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return False, tail.splitlines()[-1] if tail else \
            f"ffmpeg exited with {proc.returncode}"
    return True, ""


class RecordingAudio:
    """Owns the sidecar and the tap for the length of one recording."""

    def __init__(self, capture, sidecar_path: str,
                 nominal_fps: float = 50.0) -> None:
        self.capture = capture
        self.sidecar_path = sidecar_path
        self.nominal_fps = float(nominal_fps)
        self.track: AudioTrackWriter | None = None

    def start(self) -> bool:
        """Open the sidecar and install the tap; False if there is no audio."""
        cap = self.capture
        if cap is None or getattr(cap, "status", "") != "active":
            return False
        track = AudioTrackWriter(self.sidecar_path, cap.sample_rate,
                                 getattr(cap, "channels", 1))
        if not track.ok:
            return False
        self.track = track
        cap.tap = track.write
        return True

    def finish(self, video_path, out_path, frame_count, wall_seconds) -> str:
        """Mux the take. `out_path` exists when this returns, always."""
        track = self._detach()
        if track is None or track.bytes_written <= 0:
            _replace(video_path, out_path)
            _discard(self.sidecar_path)
            return ""

        fps, warning = recording_fps(frame_count, track.duration_seconds,
                                     wall_seconds, self.nominal_fps)
        cmd = mux_command(find_ffmpeg(), video_path, self.sidecar_path,
                          out_path, self.nominal_fps / fps if fps > 0 else 1.0,
                          track.sample_rate, track.channels)
        ok, error = run_mux(cmd, out_path)
        if not ok:
            # The video is finished and complete on its own; keep it.
            _replace(video_path, out_path)
            _discard(self.sidecar_path)
            return f"Audio could not be added ({error}); saved without sound."
        _discard(video_path)
        _discard(self.sidecar_path)
        return warning

    def _detach(self):
        """Remove the tap before the file closes under a callback in flight."""
        track, self.track = self.track, None
        if self.capture is not None and self.capture.tap is not None:
            self.capture.tap = None
        if track is not None:
            track.close()
        return track


def _replace(src, dst) -> None:
    try:
        os.replace(src, dst)
    except Exception:
        pass


def _discard(path) -> None:
    try:
        os.remove(path)
    except Exception:
        pass
