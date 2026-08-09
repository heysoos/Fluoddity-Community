"""The original brain: a random-Fourier-features net, 10 units, linear in the
amplitudes.

    out = sum_i amp_i * basis(dot(x, freq_i) + offset_i)

Global and periodic - every reading excites every unit in all directions, with
no 'off' state. Kept bit-exact at the default layout so archived genomes are
unchanged.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout, Setting, register

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
            # while random_genome biases low. Measured over five runs, evolved
            # frequencies drift to ~2x the generator's mean. Exposing both as
            # settings makes that a choice rather than an accident.
            Setting("freq_scale", "Freq Scale", "float", 0.5, 4.0, FREQ_SCALE),
            Setting("low_freq_bias", "Low-Freq Bias", "float", 0.0, 1.0, 0.0),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("centers", 10))
        return BrainLayout("fourier", (n,), 8 * n)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        """(8N,) search vector -> (8N,) flat params in GPU order.

        z is ordered [all frequencies, all amplitudes]; the GPU wants them
        interleaved per center. Preserved from genome_spec.decode.
        """
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n * 2, 4)
        freq = FREQ_SCALE * np.tanh(z[:n])
        amp = AMP_SCALE * np.tanh(z[n:])
        return np.concatenate([freq, amp], axis=1).astype(np.float32).reshape(-1)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        g = np.asarray(params, dtype=np.float32).reshape(n, 8)
        raw = np.concatenate([g[:, :4] / FREQ_SCALE, g[:, 4:] / AMP_SCALE], axis=0)
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
