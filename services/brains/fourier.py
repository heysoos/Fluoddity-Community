"""The original brain: a random-Fourier-features net, 10 units, linear in the
amplitudes.

    out = sum_i amp_i * basis(dot(x, freq_i) + offset_i)

Global and periodic - every reading excites every unit in all directions, with
no 'off' state. Kept bit-exact at the default layout so archived genomes are
unchanged.
"""
from __future__ import annotations

import numpy as np

from services.brains import (BrainLayout, Setting, register,
                             unit_scale_mask)

FREQ_SCALE = 3.0
AMP_SCALE = 1.0
EPS = 1e-4


class FourierModality:
    name = "fourier"
    modality_id = 0
    glsl_file = "shaders/brains/fourier.glsl"

    def settings_schema(self) -> list[Setting]:
        return [
            Setting("centers", "Centers", "int", 4, 48, 10),
            # The legacy decode used a flat 3.0 with no low-frequency bias,
            # while random_genome biases low. Exposing both as settings makes
            # that a choice rather than an accident.
            Setting("freq_scale", "Freq Scale", "float", 0.5, 4.0, FREQ_SCALE),
            Setting("low_freq_bias", "Low-Freq Bias", "float", 0.0, 1.0, 0.0),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("centers", 10))
        return BrainLayout("fourier", (n,), 8 * n, scales=(
            ("freq_scale", float(s.get("freq_scale", FREQ_SCALE))),
            ("low_freq_bias", float(s.get("low_freq_bias", 0.0))),
        ))

    # One centre is 8 floats: frequency(4) then amplitude(4). Frequency SCALES -
    # see fourier.glsl's fourier_param_at, which is the same split per particle.
    UNIT_FLOATS = 8
    SCALE_OFFSETS = frozenset({0, 1, 2, 3})

    def unit_floats(self, layout: BrainLayout):
        return self.UNIT_FLOATS

    def scale_mask(self, layout: BrainLayout):
        return unit_scale_mask(layout, self.UNIT_FLOATS, self.SCALE_OFFSETS)

    @staticmethod
    def _bias(t, b):
        """Bend the frequency squash toward low frequencies.

        t is tanh(z) in [-1,1]; the result is t*(1-b) + t*|t|*b, so b=0 is the
        identity - the legacy decode, bit for bit - and b=1 is the square law
        that random() already uses to bias its generated frequencies low. The
        setting existed and was wired to nothing until a schema-derived test
        went looking; it is the same defect the scale wiring had.
        """
        return t if b == 0.0 else t * ((1.0 - b) + b * np.abs(t))

    @staticmethod
    def _unbias(y, b):
        """Inverse of _bias. Solves b|t|^2 + (1-b)|t| - |y| = 0 for |t|."""
        if b == 0.0:
            return y
        a = np.abs(y)
        root = np.sqrt((1.0 - b) ** 2 + 4.0 * b * a)
        return np.sign(y) * (root - (1.0 - b)) / (2.0 * b)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        """(8N,) search vector -> (8N,) flat params in GPU order.

        z is ordered [all frequencies, all amplitudes]; the GPU wants them
        interleaved per center. Preserved from genome_spec.decode.
        """
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n * 2, 4)
        b = float(np.clip(layout.scale("low_freq_bias", 0.0), 0.0, 1.0))
        freq = (layout.scale("freq_scale", FREQ_SCALE)
                * self._bias(np.tanh(z[:n]), b))
        amp = AMP_SCALE * np.tanh(z[n:])
        return np.concatenate([freq, amp], axis=1).astype(np.float32).reshape(-1)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        g = np.asarray(params, dtype=np.float32).reshape(n, 8)
        fs = layout.scale("freq_scale", FREQ_SCALE)
        b = float(np.clip(layout.scale("low_freq_bias", 0.0), 0.0, 1.0))
        raw = np.concatenate([self._unbias(g[:, :4] / fs, b),
                              g[:, 4:] / AMP_SCALE], axis=0)
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        """Mirrors services.genome.random_genome, which biases frequencies low
        'for smoother base behaviors'."""
        n = layout.shape[0]
        freq_scale = 1.0 + 2.0 * rng.random((n, 4)) ** 2
        freq = (rng.random((n, 4)) * 2.0 - 1.0) * freq_scale
        amp = rng.random((n, 4)) * 2.0 - 1.0
        return np.concatenate([freq, amp], axis=1).astype(np.float32).reshape(-1)


register(FourierModality())
