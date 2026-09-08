"""The original brain: a random-Fourier-features net, 10 units, linear in the
amplitudes.

    out = sum_i amp_i * basis(dot(x, freq_i) + offset_i)

Global and periodic - every reading excites every unit in all directions, with
no 'off' state. Kept bit-exact at the default layout so archived genomes are
unchanged.
"""
from __future__ import annotations

import numpy as np

from services.brains import (AUDIO_INPUTS_SETTING, AUDIO_SCALE_SETTING,
                             BrainLayout, Setting, audio_inputs_of, register,
                             unit_scale_mask)

FREQ_SCALE = 3.0
AMP_SCALE = 1.0
EPS = 1e-4
# One centre: frequency(4), amplitude(4), then its audio weights, which
# extend the frequency vector and decode as frequencies do.
BASE_FLOATS = 8


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
            AUDIO_INPUTS_SETTING,
            AUDIO_SCALE_SETTING,
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("centers", 10))
        k = audio_inputs_of(s)
        return BrainLayout("fourier", (n,), (BASE_FLOATS + k) * n, scales=(
            ("freq_scale", float(s.get("freq_scale", FREQ_SCALE))),
            ("low_freq_bias", float(s.get("low_freq_bias", 0.0))),
            ("audio_scale", float(s.get("audio_scale",
                                        AUDIO_SCALE_SETTING.default))),
        ), audio_inputs=k)

    # One centre is 8 floats: frequency(4) then amplitude(4), then the audio
    # weights. Frequency and audio SCALE - see fourier.glsl's
    # fourier_param_at, which is the same split per particle.
    UNIT_FLOATS = BASE_FLOATS
    SCALE_OFFSETS = frozenset({0, 1, 2, 3})
    # The OUTGOING half: the evaluation is `out += amplitude * basis`, so a
    # centre with zero amplitude contributes nothing and a grown brain is
    # bit-identical to its parent at birth.
    AMPLITUDE_SLICE = (4, 8)

    def unit_floats(self, layout: BrainLayout):
        return BASE_FLOATS + layout.audio_inputs

    def scale_mask(self, layout: BrainLayout):
        stride = self.unit_floats(layout)
        return unit_scale_mask(layout, stride, self.SCALE_OFFSETS
                               | set(range(BASE_FLOATS, stride)))

    def audio_weight_index(self, layout: BrainLayout) -> np.ndarray:
        n, k = layout.shape[0], layout.audio_inputs
        stride = BASE_FLOATS + k
        return (np.arange(n)[:, None] * stride + BASE_FLOATS
                + np.arange(k)[None, :]).reshape(-1)

    def audio_z_index(self, layout: BrainLayout) -> np.ndarray:
        """z is [all frequencies, all amplitudes, all audio weights]."""
        n, k = layout.shape[0], layout.audio_inputs
        return BASE_FLOATS * n + np.arange(n * k)

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
        n, k = layout.shape[0], layout.audio_inputs
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        zb = z[:BASE_FLOATS * n].reshape(n * 2, 4)
        b = float(np.clip(layout.scale("low_freq_bias", 0.0), 0.0, 1.0))
        fs = layout.scale("freq_scale", FREQ_SCALE)
        freq = fs * self._bias(np.tanh(zb[:n]), b)
        amp = AMP_SCALE * np.tanh(zb[n:])
        parts = [freq, amp]
        if k:
            za = z[BASE_FLOATS * n:].reshape(n, k)
            parts.append(fs * layout.scale("audio_scale", 1.0)
                         * self._bias(np.tanh(za), b))
        return np.concatenate(parts, axis=1).astype(np.float32).reshape(-1)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n, k = layout.shape[0], layout.audio_inputs
        g = np.asarray(params, dtype=np.float32).reshape(n, BASE_FLOATS + k)
        fs = layout.scale("freq_scale", FREQ_SCALE)
        b = float(np.clip(layout.scale("low_freq_bias", 0.0), 0.0, 1.0))
        parts = [self._unbias(g[:, :4] / fs, b), g[:, 4:8] / AMP_SCALE]
        if k:
            # Floored, because the scale legitimately reaches zero.
            a_s = max(layout.scale("audio_scale", 1.0), EPS)
            parts.append(self._unbias(g[:, BASE_FLOATS:] / (fs * a_s), b))
        raw = np.concatenate([p.reshape(-1) for p in parts])
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random_audio(self, rng, layout: BrainLayout) -> np.ndarray:
        """The audio weights alone, the same prior as random(), drawn PER
        UNIT so a larger brain keeps a smaller one's."""
        n, k = layout.shape[0], layout.audio_inputs
        r = rng.random((n * k, 2))
        a_scale = 1.0 + 2.0 * r[:, 0] ** 2
        return ((r[:, 1] * 2.0 - 1.0) * a_scale
                * layout.scale("audio_scale", 1.0)).astype(np.float32)

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        """Mirrors services.genome.random_genome, which biases frequencies low
        'for smoother base behaviors'. The audio weights are drawn the same
        way AFTER the historical draws, so a K=0 brain is the one this
        always produced."""
        n, k = layout.shape[0], layout.audio_inputs
        freq_scale = 1.0 + 2.0 * rng.random((n, 4)) ** 2
        freq = (rng.random((n, 4)) * 2.0 - 1.0) * freq_scale
        amp = rng.random((n, 4)) * 2.0 - 1.0
        parts = [freq, amp]
        if k:
            a_scale = 1.0 + 2.0 * rng.random((n, k)) ** 2
            parts.append((rng.random((n, k)) * 2.0 - 1.0) * a_scale
                         * layout.scale("audio_scale", 1.0))
        return np.concatenate(parts, axis=1).astype(np.float32).reshape(-1)


register(FourierModality())
