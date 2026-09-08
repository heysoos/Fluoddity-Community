"""Gabor brain: a Gaussian envelope wrapped around the oscillation.

    g_i(x) = exp(-||x - c_i||^2 / 2 sigma_i^2) * cos(dot(x, f_i) + phi_i)
    out    = sum_i a_i * g_i(x)

The envelope is the point. A unit responds near its centre and is silent
elsewhere, so a brain is a set of localised "when I see roughly this, do that"
rules rather than one global interference pattern. It also damps the frequency
drift the Fourier runs show: a high frequency inside a narrow envelope makes a
locally intricate response, not global chaos.

Layout, 14 floats per filter: centre(4), frequency(4), amplitude(4), sigma, phase.
"""
from __future__ import annotations

import numpy as np

from services.brains import (AUDIO_INPUTS_SETTING, AUDIO_SCALE_SETTING,
                             BrainLayout, Setting, audio_inputs_of, register,
                             unit_scale_mask)

# Before any audio weights, which extend the FREQUENCY and never the centre:
# the envelope measures distance from the centre, and an audio centre would
# make silence read as a distance.
FLOATS_PER_FILTER = 14

# INPUT SCALE - the typical magnitude of one sensor component, and the only
# quantity in this modality that has units. Everything spatial is expressed as a
# multiple of it: a centre lives in the input's own space, a sigma is a distance
# in that space, and a frequency is one over it.
#
# It has to be a setting because it is a property of the PRESET, not of the
# brain, and no constant suits all of them. Read a preset's figure off
# tools/brain_input_scale.py; see the brain input scale caveat in CLAUDE.md for
# the spread and for why Lenia does not need this.
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
            AUDIO_INPUTS_SETTING,
            AUDIO_SCALE_SETTING,
        ]

    # One filter is 14 floats: centre(4), frequency(4), amplitude(4), sigma,
    # phase, then the audio weights. Frequency, sigma and audio SCALE - see
    # gabor.glsl's gabor_param_at.
    UNIT_FLOATS = FLOATS_PER_FILTER
    SCALE_OFFSETS = frozenset({4, 5, 6, 7, 12})
    # The OUTGOING half - see fourier.py. centre(4), frequency(4) come first.
    AMPLITUDE_SLICE = (8, 12)

    def unit_floats(self, layout: BrainLayout):
        return FLOATS_PER_FILTER + layout.audio_inputs

    def scale_mask(self, layout: BrainLayout):
        stride = self.unit_floats(layout)
        return unit_scale_mask(layout, stride, self.SCALE_OFFSETS
                               | set(range(FLOATS_PER_FILTER, stride)))

    def audio_weight_index(self, layout: BrainLayout) -> np.ndarray:
        n, k = layout.shape[0], layout.audio_inputs
        stride = FLOATS_PER_FILTER + k
        return (np.arange(n)[:, None] * stride + FLOATS_PER_FILTER
                + np.arange(k)[None, :]).reshape(-1)

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("filters", 12))
        k = audio_inputs_of(s)
        return BrainLayout("gabor", (n,), (FLOATS_PER_FILTER + k) * n, scales=(
            ("input_scale", float(s.get("input_scale", INPUT_SCALE))),
            ("freq_cycles", float(s.get("freq_cycles", FREQ_CYCLES))),
            ("envelope_width", float(s.get("envelope_width", 1.0))),
            ("phase_spread", float(s.get("phase_spread", np.pi))),
            ("audio_scale", float(s.get("audio_scale",
                                        AUDIO_SCALE_SETTING.default))),
        ), audio_inputs=k)

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
        z = np.asarray(z, dtype=np.float32).reshape(n, self.unit_floats(layout))
        out = np.empty_like(z)
        cs, fs, s_lo, s_hi, ps = self._scales(layout)
        out[:, 0:4] = _squash(z[:, 0:4], cs)
        out[:, 4:8] = _squash(z[:, 4:8], fs)
        out[:, 8:12] = _squash(z[:, 8:12], AMP_SCALE)
        half = 0.5 * (s_hi - s_lo)
        out[:, 12] = s_lo + half * (1.0 + np.tanh(z[:, 12]))
        out[:, 13] = _squash(z[:, 13], ps)
        out[:, 14:] = _squash(z[:, 14:], fs * layout.scale("audio_scale", 1.0))
        return out.reshape(-1).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(n, self.unit_floats(layout))
        raw = np.empty_like(p)
        cs, fs, s_lo, s_hi, ps = self._scales(layout)
        raw[:, 0:4] = p[:, 0:4] / cs
        raw[:, 4:8] = p[:, 4:8] / fs
        raw[:, 8:12] = p[:, 8:12] / AMP_SCALE
        half = 0.5 * (s_hi - s_lo)
        raw[:, 12] = (p[:, 12] - s_lo) / half - 1.0
        raw[:, 13] = p[:, 13] / ps
        raw[:, 14:] = p[:, 14:] / (fs * max(layout.scale("audio_scale", 1.0),
                                            EPS))
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        """No hand-tuned prior in this iteration (see the spec). A plain
        Gaussian in z, decoded through the same squash the search uses, so a
        random brain and a searched one are drawn from the same family."""
        from services.brains import audio_aware_normal

        z = audio_aware_normal(rng, layout, 0.5)
        return self.decode(z, layout)


register(GaborModality())
