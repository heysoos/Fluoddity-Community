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

from services.brains import (AUDIO_INPUTS_SETTING, AUDIO_SCALE_SETTING,
                             BrainLayout, Setting, audio_inputs_of, register,
                             unit_scale_mask)

# Before any audio weights, which extend the projection.
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
        # them freely inside +/- this. Wide range because the effect is mild
        # across the slider; a narrow one would be a control that does nothing.
        return [
            Setting("bumps", "Bumps", "int", 4, 48, 12),
            Setting("w_scale", "Projection Scale", "float", 0.5, 12.0, W_SCALE),
            Setting("mu_scale", "Band Center", "float", 0.5, 4.0, MU_SCALE),
            Setting("sigma_max", "Band Width", "float", 0.05, 1.0, SIGMA_MAX),
            AUDIO_INPUTS_SETTING,
            AUDIO_SCALE_SETTING,
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("bumps", 12))
        k = audio_inputs_of(s)
        return BrainLayout("lenia", (n,), (FLOATS_PER_BUMP + k) * n, scales=(
            ("mu_scale", float(s.get("mu_scale", MU_SCALE))),
            ("sigma_max", float(s.get("sigma_max", SIGMA_MAX))),
            ("w_scale", float(s.get("w_scale", W_SCALE))),
            ("audio_scale", float(s.get("audio_scale",
                                        AUDIO_SCALE_SETTING.default))),
        ), audio_inputs=k)

    # One bump is 10 floats: projection(4), amplitude(4), mu, sigma, then the
    # audio weights. The projection, sigma and audio SCALE - see lenia.glsl's
    # lenia_param_at. mu does NOT: it is a LOCATION on the u axis, and scaling
    # would pin a band centred near zero at zero forever.
    UNIT_FLOATS = FLOATS_PER_BUMP
    SCALE_OFFSETS = frozenset({0, 1, 2, 3, 9})
    # The OUTGOING half - see fourier.py. projection(4) comes first.
    AMPLITUDE_SLICE = (4, 8)

    def unit_floats(self, layout: BrainLayout):
        return FLOATS_PER_BUMP + layout.audio_inputs

    def scale_mask(self, layout: BrainLayout):
        stride = self.unit_floats(layout)
        return unit_scale_mask(layout, stride, self.SCALE_OFFSETS
                               | set(range(FLOATS_PER_BUMP, stride)))

    def audio_weight_index(self, layout: BrainLayout) -> np.ndarray:
        n, k = layout.shape[0], layout.audio_inputs
        stride = FLOATS_PER_BUMP + k
        return (np.arange(n)[:, None] * stride + FLOATS_PER_BUMP
                + np.arange(k)[None, :]).reshape(-1)

    @staticmethod
    def _scales(layout: BrainLayout):
        return (layout.scale("w_scale", W_SCALE),
                layout.scale("mu_scale", MU_SCALE),
                max(layout.scale("sigma_max", SIGMA_MAX), SIGMA_MIN * 2.0))

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n, self.unit_floats(layout))
        out = np.empty_like(z)
        ws, mus, s_hi = self._scales(layout)
        out[:, 0:4] = ws * np.tanh(z[:, 0:4])
        out[:, 4:8] = AMP_SCALE * np.tanh(z[:, 4:8])
        out[:, 8] = mus * np.tanh(z[:, 8])
        half = 0.5 * (s_hi - SIGMA_MIN)
        out[:, 9] = SIGMA_MIN + half * (1.0 + np.tanh(z[:, 9]))
        out[:, 10:] = ws * layout.scale("audio_scale", 1.0) * np.tanh(z[:, 10:])
        return out.reshape(-1).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(n, self.unit_floats(layout))
        raw = np.empty_like(p)
        ws, mus, s_hi = self._scales(layout)
        raw[:, 0:4] = p[:, 0:4] / ws
        raw[:, 4:8] = p[:, 4:8] / AMP_SCALE
        raw[:, 8] = p[:, 8] / mus
        half = 0.5 * (s_hi - SIGMA_MIN)
        raw[:, 9] = (p[:, 9] - SIGMA_MIN) / half - 1.0
        raw[:, 10:] = p[:, 10:] / (ws * max(layout.scale("audio_scale", 1.0),
                                            EPS))
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        return self.decode(z, layout)


register(LeniaModality())
