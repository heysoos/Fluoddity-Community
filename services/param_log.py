"""One row per recorded frame: the physics the sim actually ran under.

Fed from the frame loop with audio's modulated copy of the sim state, so a
take driven by music records the path the music drew and not the sliders.
Written beside the video as `<take>_params.csv` when the take finishes.
"""
from __future__ import annotations

import csv
from pathlib import Path


class ParamLog:
    def __init__(self) -> None:
        self._rows: list[tuple[dict, dict]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows = []

    def row(self, values: dict, signals: dict) -> None:
        self._rows.append((dict(values), dict(signals)))

    def write(self, video_path: str, frames: int) -> Path | None:
        """Writes rows 0..frames-1 beside the video; None when there is
        nothing to write. Empties the log either way."""
        rows, self._rows = self._rows[:max(0, int(frames))], []
        if not rows:
            return None
        out = Path(video_path).with_name(Path(video_path).stem + "_params.csv")
        keys = list(rows[0][0])
        sig = list(rows[0][1])
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", *keys, *(f"sig:{s}" for s in sig)])
            for i, (values, signals) in enumerate(rows):
                w.writerow([i, *(values.get(k, "") for k in keys),
                            *(signals.get(s, "") for s in sig)])
        return out
