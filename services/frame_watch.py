"""Where a slow frame went.

The frame loop marks its phases; a frame over the threshold is kept for the
Performance window and appended to perf.log in the user data folder, so a
hang seen on stage can be read back afterwards with its cause attached.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Stall:
    when: float
    total_ms: float
    phases: tuple
    flags: str

    def line(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.when))
        parts = " ".join(f"{name} {ms:.0f}" for name, ms in self.phases
                         if ms >= 1.0)
        tail = f"  [{self.flags}]" if self.flags else ""
        return f"{stamp}  {self.total_ms:5.0f} ms  {parts}{tail}"


class FrameWatch:
    """begin() at the top of a frame, mark(name) after each phase, end() last.

    `path` is the log file, or None to keep stalls in memory only. `clock`
    is injectable for tests.
    """

    def __init__(self, threshold_ms: float = 60.0, path=None, keep: int = 30,
                 clock=time.perf_counter):
        self.threshold_ms = float(threshold_ms)
        self.path = path
        self.keep = int(keep)
        self._clock = clock
        self.recent: list[Stall] = []
        self.frames = 0
        self.stalls = 0
        self._t0 = None
        self._last = None
        self._phases: list = []

    def begin(self) -> None:
        self._t0 = self._last = self._clock()
        self._phases = []

    def mark(self, name: str) -> None:
        if self._last is None:
            return
        now = self._clock()
        self._phases.append((name, (now - self._last) * 1000.0))
        self._last = now

    def end(self, flags: str = "") -> Stall | None:
        if self._t0 is None:
            return None
        total = (self._clock() - self._t0) * 1000.0
        self.frames += 1
        self._t0 = self._last = None
        if total < self.threshold_ms:
            return None
        stall = Stall(time.time(), total, tuple(self._phases), flags)
        self.stalls += 1
        self.recent.append(stall)
        del self.recent[:-self.keep]
        self._write(stall)
        return stall

    def _write(self, stall: Stall) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(stall.line() + "\n")
        except OSError:
            pass                        # a log must never cost the frame
