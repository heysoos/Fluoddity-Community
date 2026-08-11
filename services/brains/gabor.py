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

from services.brains import (BrainLayout, Setting, register,
                             unit_scale_mask)

FLOATS_PER_FILTER = 14

# INPUT SCALE - the typical magnitude of one sensor component, and the only
# quantity in this modality that has units. Everything spatial is expressed as a
# multiple of it: a centre lives in the input's own space, a sigma is a distance
# in that space, and a frequency is one over it.
#
# It has to be a setting because it is not a property of the brain, it is a
# property of the PRESET. Measured over all 23 of physics_configs/Core by
# tools/brain_input_scale.py, the median |input| per preset runs 0.0016
# (Searching) to 1.43 (Bubbles) - a spread of about 900x - with a median of
# 0.057 and a median p90 of 0.36. No constant can be right for all of them.
#
# 0.5 covers the p90-p99 band of the median preset. The old value was 2.0, which
# put every centre 10x further out than the input ever reached: the envelope was
# then near-constant over everything a particle actually reads, so it did
# nothing and a Gabor filter degenerated into a plain oscillation. Measured as
# the correlation between a unit's response and the same unit with its envelope
# removed - 1.0 means it IS a Fourier - the median preset scored 0.85 at 2.0
# against 0.74 at 0.35, and the presets with room to work moved much further
# (Salt 0.84 -> 0.61, Bubbles 0.71 -> 0.35).
#
# Lenia deliberately does NOT get this treatment. Its bump compares w.x, not x,
# and the projection amplifies by W_SCALE and sums four terms, which lands it
# near its own mu range already - which is why its Inspector atlas showed the
# narrow bands its spec asks for while Gabor's showed broad plane waves.
INPUT_SCALE = 0.5

# Dimensionless shape, in multiples of INPUT_SCALE. At the default they come out
# as sigma in [0.1, 1.0] and frequency 3.0, so changing Input Scale rescales a
# filter without reshaping it.
FREQ_CYCLES = 1.5           # cycles per input scale -> freq = FREQ_CYCLES/S
SIGMA_MIN_REL = 0.2
SIGMA_MAX_REL = 2.0
AMP_SCALE = 1.0
# The GPU divides by sigma^2 and mutation scales it further, so an absolute
# floor is still needed however small Input Scale is set.
SIGMA_FLOOR = 1e-3
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
            # The measured per-preset range is 0.0016 to 1.43, so the slider has
            # to reach both ends. Read the preset's own figure off
            # tools/brain_input_scale.py, or just watch the Inspector: at the
            # right setting the tiles show blobs, at the wrong one, plane waves.
            Setting("input_scale", "Input Scale", "float", 0.01, 4.0, INPUT_SCALE),
            Setting("freq_cycles", "Freq Cycles", "float", 0.25, 6.0, FREQ_CYCLES),
            Setting("envelope_width", "Envelope Width", "float", 0.2, 2.0, 1.0),
            Setting("phase_spread", "Phase Spread", "float", 0.0, 3.1416, 3.1416),
        ]

    # One filter is 14 floats: centre(4), frequency(4), amplitude(4), sigma,
    # phase. Frequency and sigma SCALE - see gabor.glsl's gabor_param_at.
    UNIT_FLOATS = FLOATS_PER_FILTER
    SCALE_OFFSETS = frozenset({4, 5, 6, 7, 12})

    def unit_floats(self, layout: BrainLayout):
        return self.UNIT_FLOATS

    def scale_mask(self, layout: BrainLayout):
        return unit_scale_mask(layout, self.UNIT_FLOATS, self.SCALE_OFFSETS)

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("filters", 12))
        return BrainLayout("gabor", (n,), FLOATS_PER_FILTER * n, scales=(
            ("input_scale", float(s.get("input_scale", INPUT_SCALE))),
            ("freq_cycles", float(s.get("freq_cycles", FREQ_CYCLES))),
            ("envelope_width", float(s.get("envelope_width", 1.0))),
            ("phase_spread", float(s.get("phase_spread", np.pi))),
        ))

    @staticmethod
    def _scales(layout: BrainLayout):
        """Absolute decode scales, derived from Input Scale.

        Centre and sigma are proportional to it and frequency inversely so, so
        moving one slider rescales a filter to a different preset without
        reshaping it. Envelope Width scales the sigma BAND rather than replacing
        it, and the floor is absolute, so no combination reaches zero - the GPU
        divides by sigma^2.
        """
        s = max(layout.scale("input_scale", INPUT_SCALE), 1e-4)
        ew = layout.scale("envelope_width", 1.0)
        fs = layout.scale("freq_cycles", FREQ_CYCLES) / s
        ps = layout.scale("phase_spread", np.pi)
        s_lo = max(SIGMA_MIN_REL * s * ew, SIGMA_FLOOR)
        s_hi = max(SIGMA_MAX_REL * s * ew, s_lo + SIGMA_FLOOR)
        return s, fs, s_lo, s_hi, max(ps, 1e-6)

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
