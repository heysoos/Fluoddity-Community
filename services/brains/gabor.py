"""Gabor brain: a Gaussian envelope wrapped around the oscillation.

    g_i(x) = exp(-||x - c_i||^2 / 2 sigma_i^2) * cos(dot(x, f_i) + phi_i)
    out    = sum_i a_i * g_i(x)

The envelope is the point. A unit responds near its centre and is silent
elsewhere, so a brain is a set of localised "when I see roughly this, do that"
rules rather than one global interference pattern. It also damps the drift
measured on the Fourier runs: a high frequency inside a narrow envelope makes a
locally intricate response, not global chaos.

Layout, 14 floats per filter: centre(4), frequency(4), amplitude(4), sigma, phase.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout, Setting, register

FLOATS_PER_FILTER = 14
CENTER_SCALE = 2.0
FREQ_SCALE = 3.0
AMP_SCALE = 1.0
# Strictly positive and bounded. The GPU divides by sigma^2, and mutation scales
# it further, so the floor is what keeps a filter from becoming a delta spike.
SIGMA_MIN = 0.15
SIGMA_MAX = 2.0
EPS = 1e-4


def _squash(z, scale):
    return scale * np.tanh(z)


class GaborModality:
    name = "gabor"
    modality_id = 1
    glsl_file = "shaders/brains/gabor.glsl"

    def settings_schema(self) -> list[Setting]:
        # 36 filters * 14 floats = 504, just inside MAX_BRAIN_FLOATS.
        return [
            Setting("filters", "Filters", "int", 4, 36, 12),
            Setting("freq_scale", "Freq Scale", "float", 0.5, 6.0, FREQ_SCALE),
            Setting("envelope_width", "Envelope Width", "float", 0.2, 2.0, 1.0),
            Setting("phase_spread", "Phase Spread", "float", 0.0, 3.1416, 3.1416),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("filters", 12))
        return BrainLayout("gabor", (n,), FLOATS_PER_FILTER * n, scales=(
            ("freq_scale", float(s.get("freq_scale", FREQ_SCALE))),
            ("envelope_width", float(s.get("envelope_width", 1.0))),
            ("phase_spread", float(s.get("phase_spread", np.pi))),
            ("center_scale", float(s.get("center_scale", CENTER_SCALE))),
        ))

    @staticmethod
    def _scales(layout: BrainLayout):
        """The decode scales this layout was built with. Envelope Width scales
        the sigma BAND rather than replacing it, so the floor stays positive."""
        cs = layout.scale("center_scale", CENTER_SCALE)
        fs = layout.scale("freq_scale", FREQ_SCALE)
        ew = layout.scale("envelope_width", 1.0)
        ps = layout.scale("phase_spread", np.pi)
        return cs, fs, SIGMA_MIN * ew, SIGMA_MAX * ew, max(ps, 1e-6)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n, FLOATS_PER_FILTER)
        out = np.empty_like(z)
        cs, fs, s_lo, s_hi, ps = self._scales(layout)
        out[:, 0:4] = _squash(z[:, 0:4], cs)
        out[:, 4:8] = _squash(z[:, 4:8], fs)
        out[:, 8:12] = _squash(z[:, 8:12], AMP_SCALE)
        half = 0.5 * (s_hi - s_lo)
        out[:, 12] = s_lo + half * (1.0 + np.tanh(z[:, 12]))
        out[:, 13] = _squash(z[:, 13], ps)
        return out.reshape(-1).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(n, FLOATS_PER_FILTER)
        raw = np.empty_like(p)
        cs, fs, s_lo, s_hi, ps = self._scales(layout)
        raw[:, 0:4] = p[:, 0:4] / cs
        raw[:, 4:8] = p[:, 4:8] / fs
        raw[:, 8:12] = p[:, 8:12] / AMP_SCALE
        half = 0.5 * (s_hi - s_lo)
        raw[:, 12] = (p[:, 12] - s_lo) / half - 1.0
        raw[:, 13] = p[:, 13] / ps
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        """No hand-tuned prior in this iteration (see the spec). A plain
        Gaussian in z, decoded through the same squash the search uses, so a
        random brain and a searched one are drawn from the same family."""
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        return self.decode(z, layout)


register(GaborModality())
