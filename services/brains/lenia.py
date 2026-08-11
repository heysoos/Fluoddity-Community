"""Lenia bumps: the growth function, a Gaussian band rather than an oscillation.

    u_i = dot(x, w_i)
    G_i = 2 * exp(-(u_i - mu_i)^2 / 2 sigma_i^2) - 1
    out = sum_i a_i * G_i

The -1 is essential. Response is positive inside a narrow band of sensor values
and negative everywhere else - the "thrive at this density, die away from it"
rule that gives Lenia its membranes. Non-oscillatory and smooth; expected to be
the best-conditioned landscape of the four.

On priors: canonical Lenia parameters (orbium and friends) are tuned for a
continuous CA whose input is a kernel-weighted neighbourhood sum over a grid.
Here the input is two sensor taps scaled by sqrt(ws)*38.855*SENSOR_GAIN. Those
numbers DO NOT transfer and must not be copied in. What transfers is scale-free:
Lenia growth bands are consistently narrow, roughly sigma/mu ~ 0.1, which is what
SIGMA_MAX encodes relative to MU_SCALE.

Layout, 10 floats per bump: projection(4), amplitude(4), mu, sigma.
"""
from __future__ import annotations

import numpy as np

from services.brains import (BrainLayout, Setting, register,
                             unit_scale_mask)

FLOATS_PER_BUMP = 10
W_SCALE = 3.0
AMP_SCALE = 1.0
MU_SCALE = 2.0
# Narrow relative to MU_SCALE - that ratio is the part of Lenia that transfers.
SIGMA_MIN = 0.02
SIGMA_MAX = 0.6
EPS = 1e-4


def growth(u, mu, sigma):
    """Lenia's growth mapping, vectorised. Peaks at +1 on the band, -1 off it."""
    return 2.0 * np.exp(-((u - mu) ** 2) / (2.0 * sigma ** 2)) - 1.0


class LeniaModality:
    name = "lenia"
    modality_id = 2
    glsl_file = "shaders/brains/lenia.glsl"

    def settings_schema(self) -> list[Setting]:
        # Projection Scale before the two band settings, because it sets the
        # AXIS they sit on: u = dot(x, w), and mu and sigma are positions and
        # widths along u.
        #
        # It is a CEILING on a trained value, not a trained value itself - the
        # projection is 4 searchable floats per bump and the optimizer moves
        # them freely inside +/- this. Wide range because the effect is mild:
        # measured over 0.5 to 12 against a realistic input, in-band coverage
        # runs 27.4% to 17.4% and output p50 moves 18%. A narrow range would
        # be a slider that does nothing.
        return [
            Setting("bumps", "Bumps", "int", 4, 48, 12),
            Setting("w_scale", "Projection Scale", "float", 0.5, 12.0, W_SCALE),
            Setting("mu_scale", "Band Center", "float", 0.5, 4.0, MU_SCALE),
            Setting("sigma_max", "Band Width", "float", 0.05, 1.0, SIGMA_MAX),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("bumps", 12))
        return BrainLayout("lenia", (n,), FLOATS_PER_BUMP * n, scales=(
            ("mu_scale", float(s.get("mu_scale", MU_SCALE))),
            ("sigma_max", float(s.get("sigma_max", SIGMA_MAX))),
            ("w_scale", float(s.get("w_scale", W_SCALE))),
        ))

    # One bump is 10 floats: projection(4), amplitude(4), mu, sigma. The
    # projection and sigma SCALE - see lenia.glsl's lenia_param_at. mu does NOT:
    # it is a LOCATION on the u axis, and scaling would pin a band centred near
    # zero at zero forever.
    UNIT_FLOATS = FLOATS_PER_BUMP
    SCALE_OFFSETS = frozenset({0, 1, 2, 3, 9})

    def unit_floats(self, layout: BrainLayout):
        return self.UNIT_FLOATS

    def scale_mask(self, layout: BrainLayout):
        return unit_scale_mask(layout, self.UNIT_FLOATS, self.SCALE_OFFSETS)

    @staticmethod
    def _scales(layout: BrainLayout):
        return (layout.scale("w_scale", W_SCALE),
                layout.scale("mu_scale", MU_SCALE),
                max(layout.scale("sigma_max", SIGMA_MAX), SIGMA_MIN * 2.0))

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n, FLOATS_PER_BUMP)
        out = np.empty_like(z)
        ws, mus, s_hi = self._scales(layout)
        out[:, 0:4] = ws * np.tanh(z[:, 0:4])
        out[:, 4:8] = AMP_SCALE * np.tanh(z[:, 4:8])
        out[:, 8] = mus * np.tanh(z[:, 8])
        half = 0.5 * (s_hi - SIGMA_MIN)
        out[:, 9] = SIGMA_MIN + half * (1.0 + np.tanh(z[:, 9]))
        return out.reshape(-1).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(n, FLOATS_PER_BUMP)
        raw = np.empty_like(p)
        ws, mus, s_hi = self._scales(layout)
        raw[:, 0:4] = p[:, 0:4] / ws
        raw[:, 4:8] = p[:, 4:8] / AMP_SCALE
        raw[:, 8] = p[:, 8] / mus
        half = 0.5 * (s_hi - SIGMA_MIN)
        raw[:, 9] = (p[:, 9] - SIGMA_MIN) / half - 1.0
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        return self.decode(z, layout)


register(LeniaModality())
