"""A one-hidden-layer MLP: out = W2 . act(W1 x + b1) + b2.

Activation is part of the LAYOUT, not a free parameter - tanh and sin are
different function families, and a genome evolved under one means nothing under
the other. That is why activation appears in the signature, and so in the
archive directory: the two must never share stored genomes.

Layout, 9H + 4 floats: W1 (H rows of 4), b1 (H), W2 (4 rows of H), b2 (4).

W2 is stored OUTPUT-MAJOR (4 rows of H) rather than hidden-major. The shader
accumulates one hidden unit at a time into a vec4, so it reads the four weights
of hidden unit j strided by H. Hidden-major would let it read a vec4 contiguously
but would force it to hold the whole hidden vector first - a local float[48] per
invocation, which spills to scratch memory for every particle every frame.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout, Setting, register

ACTIVATIONS = ("tanh", "sin", "gelu")
W_SCALE = 2.0
B_SCALE = 1.0
EPS = 1e-4


class MLPModality:
    name = "mlp"
    modality_id = 3
    glsl_file = "shaders/brains/mlp.glsl"

    def settings_schema(self) -> list[Setting]:
        # 9*48 + 4 = 436, inside MAX_BRAIN_FLOATS.
        return [
            Setting("hidden", "Hidden Width", "int", 4, 48, 16),
            Setting("activation", "Activation", "choice", 0, 2, 0,
                    choices=ACTIVATIONS),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        h = int(s.get("hidden", 16))
        act = int(s.get("activation", 0))
        return BrainLayout("mlp", (h, act), 9 * h + 4)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        h = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        out = np.empty_like(z)
        out[: 4 * h] = W_SCALE * np.tanh(z[: 4 * h])              # W1
        out[4 * h: 5 * h] = B_SCALE * np.tanh(z[4 * h: 5 * h])    # b1
        out[5 * h: 9 * h] = W_SCALE * np.tanh(z[5 * h: 9 * h])    # W2
        out[9 * h:] = B_SCALE * np.tanh(z[9 * h:])                # b2
        return out.astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        h = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        raw = np.empty_like(p)
        raw[: 4 * h] = p[: 4 * h] / W_SCALE
        raw[4 * h: 5 * h] = p[4 * h: 5 * h] / B_SCALE
        raw[5 * h: 9 * h] = p[5 * h: 9 * h] / W_SCALE
        raw[9 * h:] = p[9 * h:] / B_SCALE
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        return self.decode(z, layout)


register(MLPModality())
