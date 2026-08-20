"""A webcam, read through ffmpeg, on a worker thread.

ffmpeg is already a dependency reached by subprocess, so this needs no new
package. Reading is DECOUPLED from the frame loop: the worker keeps only the
newest frame and the main thread takes whatever is there, so a camera slower
than the sim drops frames instead of stalling it, and a camera faster than the
sim never queues up latency.

Nothing here touches GL. The caller uploads `latest()` to a texture.
"""
from __future__ import annotations

import platform
import re
import subprocess
import threading

from utilities.ffmpeg_recorder import find_ffmpeg

# Windows is the platform this ships on; the others are here so a device list
# on a developer's machine is an empty list rather than an exception.
_INPUT_FORMAT = {"Windows": "dshow", "Darwin": "avfoundation", "Linux": "v4l2"}

DEFAULT_SIZE = (640, 480)


def _creationflags() -> int:
    """Keep the console window from flashing up on Windows."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def input_format() -> str:
    return _INPUT_FORMAT.get(platform.system(), "v4l2")


def list_devices(timeout: float = 6.0) -> list[str]:
    """Camera names, newest ffmpeg first. Empty where none can be listed."""
    system = platform.system()
    if system != "Windows":
        # avfoundation and v4l2 enumerate differently; unsupported rather than
        # guessed, so the combo says so instead of offering a device that fails.
        return []
    try:
        exe = find_ffmpeg()
    except Exception:
        return []
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-list_devices", "true", "-f", "dshow",
             "-i", "dummy"],
            capture_output=True, text=True, timeout=timeout,
            creationflags=_creationflags())
    except Exception:
        return []
    # ffmpeg writes the list to stderr and exits non-zero; that is normal.
    names = []
    video_section = True
    for line in (proc.stderr or "").splitlines():
        if "DirectShow audio devices" in line:
            video_section = False
        elif "DirectShow video devices" in line:
            video_section = True
        match = re.search(r'"([^"]+)"', line)
        if match and video_section and "Alternative name" not in line:
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


class WebcamReader:
    """One camera. Start it, poll `latest()`, `close()` when done."""

    def __init__(self, device: str, size=DEFAULT_SIZE, fps: int = 30):
        self.device = device
        self.size = size
        self.fps = fps
        self.error: str | None = None
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._serial = 0
        self._start()

    # -- lifecycle ------------------------------------------------------

    def _command(self, exe):
        w, h = self.size
        target = (f"video={self.device}" if platform.system() == "Windows"
                  else self.device)
        return [exe, "-hide_banner", "-loglevel", "error",
                "-f", input_format(),
                "-framerate", str(self.fps),
                "-video_size", f"{w}x{h}",
                "-i", target,
                # rawvideo rows arrive top-down and a GL texture is bottom-up,
                # so the camera would be mirrored on the particles and in the
                # preview. Flipping here costs nothing and keeps every reader
                # of the texture on one convention.
                "-vf", "vflip",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    def _start(self) -> None:
        try:
            exe = find_ffmpeg()
        except Exception as exc:
            self.error = f"ffmpeg not found: {exc}"
            return
        try:
            self._proc = subprocess.Popen(
                self._command(exe), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, creationflags=_creationflags())
        except Exception as exc:
            self.error = f"could not open {self.device}: {exc}"
            return
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        w, h = self.size
        need = w * h * 3
        stream = self._proc.stdout
        while not self._stop.is_set():
            try:
                chunk = stream.read(need)
            except Exception:
                break
            if not chunk or len(chunk) < need:
                break
            # Only the NEWEST frame is kept: a queue here would be latency,
            # since nothing downstream wants the frames it skipped.
            with self._lock:
                self._frame = chunk
                self._serial += 1
        if not self._stop.is_set():
            self.error = self._read_error() or "the camera stopped sending"

    def _read_error(self) -> str:
        """ffmpeg's last line. Reads the handle it was given, since close()
        may have cleared `_proc` while this thread was still unwinding."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            text = proc.stderr.read().decode("utf-8", "replace").strip()
        except Exception:
            return ""
        return text.splitlines()[-1] if text else ""

    # -- reading --------------------------------------------------------

    def latest(self):
        """(rgb bytes, serial), or (None, serial) before the first frame."""
        with self._lock:
            return self._frame, self._serial

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        """Terminate, JOIN, then close - in that order, and never another.

        The pump thread blocks inside `read()`. Closing the pipe under it from
        another thread is what a reader closed and reopened a few times does
        not survive; terminating first makes that read return EOF, so the
        thread leaves on its own and the pipes are closed by nobody else.
        """
        self._stop.set()
        proc, self._proc = self._proc, None
        thread, self._thread = self._thread, None

        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass

        if thread is not None:
            thread.join(timeout=3)

        if proc is not None:
            for pipe in (proc.stdout, proc.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except Exception:
                    pass
