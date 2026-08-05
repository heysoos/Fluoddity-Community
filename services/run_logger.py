"""Per-generation metrics for an automatic tournament run.

JSONL rather than CSV because the schema will grow (physics blocks, novelty
terms), and appending a field must not break older files or readers.

All values are FITNESS, higher is better. Never 'loss'.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

_HISTORY_KEYS = ("gen", "fit_best", "fit_mean", "sigma")


def new_run_id(now: _dt.datetime | None = None) -> str:
    return (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")


class RunLogger:
    def __init__(self, root="runs", run_id: str | None = None,
                 config: dict | None = None):
        self.root = Path(root)
        self.enabled = True
        self._fh = None
        self._config = dict(config or {})
        self._n_logged = 0
        self._history: dict[str, list] = {k: [] for k in _HISTORY_KEYS}
        self._open(run_id or new_run_id())

    def _open(self, run_id: str) -> None:
        self.run_id = run_id
        self.dir = self.root / run_id
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "config.json").write_text(
                json.dumps(self._config, indent=2))
            self._fh = open(self.dir / "log.jsonl", "a", encoding="utf-8")
        except OSError as exc:
            self.enabled = False
            print(f"[RunLogger] logging disabled ({exc}); the run continues")

    def start_new_run(self) -> None:
        """Begin a fresh run: clears the plotted history and moves to a new
        folder so the previous search's generations are not appended to.

        A reset that logged nothing reuses the current run rather than leaving
        an empty folder behind - Reset is easy to click twice.
        """
        self._history = {k: [] for k in _HISTORY_KEYS}
        if not self.enabled or self._n_logged == 0:
            return
        self.close()
        self._n_logged = 0
        # new_run_id() has one-second resolution; two resets inside the same
        # second must not land in the same folder.
        base = new_run_id()
        run_id, n = base, 2
        while (self.root / run_id).exists():
            run_id, n = f"{base}-{n}", n + 1
        self._open(run_id)

    def log_generation(self, rec: dict) -> None:
        if not self.enabled:
            return
        for k in _HISTORY_KEYS:
            if k in rec:
                self._history[k].append(rec[k])
        self._n_logged += 1
        try:
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()  # a crash must lose at most one generation
        except OSError as exc:
            self.enabled = False
            print(f"[RunLogger] logging disabled mid-run ({exc})")

    def history(self) -> dict[str, list]:
        return {k: list(v) for k, v in self._history.items()}

    def load_history(self, hist: dict) -> None:
        """Restore the sparkline after a checkpoint load."""
        for k in _HISTORY_KEYS:
            if k in (hist or {}):
                self._history[k] = list(hist[k])

    def save_frame(self, img: np.ndarray, gen: int) -> None:
        """Periodic best-tile PNG. A run folder of these assembles directly
        into a timelapse of the evolution."""
        if not self.enabled:
            return
        try:
            from PIL import Image

            d = self.dir / "frames"
            d.mkdir(exist_ok=True)
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(
                d / f"gen_{gen:06d}.png"
            )
        except (OSError, ValueError) as exc:
            print(f"[RunLogger] frame save failed ({exc})")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
