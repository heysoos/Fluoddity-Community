"""Audio modulation of a brain's decode scales.

A scale change is a re-decode. Every modality already has a correct
encode()/decode() pair, so nothing here is modality-specific.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout


class BrainModulator:
    """Holds the search vector for the current brain.

    encode() clips at the rails and is lossy at the extremes, so the vector is
    computed once when the base brain changes and re-decoded every frame after.
    Re-encoding per frame would drift.
    """

    def __init__(self) -> None:
        self._z = None
        self._modality = None
        self._layout: BrainLayout | None = None

    def clear(self) -> None:
        self._z = None
        self._modality = None
        self._layout = None

    def set_base(self, params, modality, layout: BrainLayout) -> None:
        """Adopt a decoded brain. A wrong-width array is refused."""
        self.clear()
        if params is None:
            return
        arr = np.asarray(params, dtype=np.float32).reshape(-1)
        if arr.size != layout.length:
            return
        try:
            z, _clamped = modality.encode(arr, layout)
        except Exception:
            return
        self._z = np.asarray(z, dtype=np.float32)
        self._modality = modality
        self._layout = layout

    def base_scales(self) -> dict[str, float]:
        if self._layout is None:
            return {}
        return {k: float(v) for k, v in self._layout.scales}

    def modulated(self, scales: dict[str, float]):
        """The brain decoded under `scales`, or None if there is no base.

        Keys absent from `scales` keep the base layout's value.
        """
        if self._z is None or self._layout is None:
            return None
        merged = self.base_scales()
        for k, v in (scales or {}).items():
            if k in merged:
                merged[k] = float(v)
        layout = BrainLayout(self._layout.modality, self._layout.shape,
                             self._layout.length,
                             scales=tuple(merged.items()))
        try:
            return self._modality.decode(self._z, layout)
        except Exception:
            return None
