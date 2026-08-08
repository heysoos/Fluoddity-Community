"""The interface AutoTournamentService searches through.

The service owns the rollout state machine - WRITE_RULES, STEP, CAPTURE x S,
SCORE, spread across real application frames, with abort-on-resize. That machine
is identical for every search. What differs is only two decisions: which genomes
to run next, and what to do with the resulting images.

A driver receives RAW CROPS rather than scores, which is what lets PromptDriver
call scorer.score() while ImgepDriver calls scorer.embed(). The consequence is
that the service no longer imports or knows about CLIP at all.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class SearchDriver(Protocol):
    name: str

    def set_spec(self, spec) -> None:
        """The genome layout changed (physics search toggled). Any optimizer
        state is for a different dimension and must be discarded. Must not
        raise."""

    def ask(self, n: int) -> np.ndarray:
        """(n, spec.dim) float32 - one search vector per tile."""

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray]) -> np.ndarray:
        """snapshots is the service's crop buffer verbatim: a list of S arrays
        of shape (n, 224, 224, 3) uint8, in capture order. May be empty when a
        generation produced no captures.

        Returns (n,) float32 - a per-tile score for the UI and the log ONLY.
        Whatever learning the driver does, it does internally."""

    def status(self) -> dict:
        """Free-form, for the UI status line and the JSONL record."""

    def checkpoint_state(self) -> dict: ...

    def restore(self, d: dict) -> None: ...

    def reset(self) -> None:
        """Discard all learned state. The archive, if any, survives."""
