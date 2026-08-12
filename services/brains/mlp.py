"""An MLP of one or more hidden layers: out = W_out . act_k(...act_1(W1 x + b1)...) + b_out.

Activation is part of the LAYOUT, not a free parameter - tanh and sin are
different function families, and a genome evolved under one means nothing under
the other. That is why every layer's activation appears in the signature, and so
in the archive directory: two stacks that differ only there must never share
stored genomes.

Layout, packed by `layer_spans`:

    for each hidden layer l:  W_l (w_l rows of fan_in_l), then b_l (w_l)
    then W_out (4 rows of w_k), then b_out (4)

At one hidden layer that is W1 (H rows of 4), b1 (H), W_out (4 rows of H),
b_out (4) - 9H + 4 floats, byte for byte what this modality has always
produced. Nothing about a depth-1 brain changes, which is what lets every
existing genome, archive and config keep working with no migration.

W_out is stored OUTPUT-MAJOR (4 rows of w_k) rather than unit-major. The shader
accumulates one final-layer unit at a time into a vec4, so it reads the four
weights of unit j strided by w_k. Unit-major would let it read a vec4
contiguously but would force it to hold the final hidden vector first.
"""
from __future__ import annotations

import numpy as np

from services.brains import (BrainLayout, Setting, register,
                             unit_scale_mask)

ACTIVATIONS = ("tanh", "sin", "gelu")
W_SCALE = 2.0
B_SCALE = 1.0
EPS = 1e-4

# What a layer's Reroll draws from, BEFORE the decode - so a reroll is always
# representable and always lands inside the rails. `normal` is what random()
# uses. `sparse` sets most of z to zero, and z = 0 decodes to exactly 0, so
# sparsity in z is sparsity in the weights.
DISTRIBUTIONS = ("normal", "uniform", "sparse", "heavy-tail")
SPARSITY = 0.8
CAUCHY_SCALE = 0.25

IN_DIM = 4          # the four sensor taps
OUT_DIM = 4         # force xy and strafe xy

MIN_WIDTH = 1
# This modality's historical Hidden Width maximum, so every depth-1 genome ever
# written is still reachable and still decodes to the same brain.
MAX_WIDTH = 48
# The cap on EVERY layer of a stack deeper than one, and much lower, because it
# sizes the ping-pong locals mlp.glsl needs to hold a layer's activations - a
# cost the depth-1 path pays too, without ever using them. It is MEASURED; see
# the variable-depth MLP caveat in CLAUDE.md and tools/measure_brain_depth.py.
# Mirrored in mlp.glsl as MAX_MLP_WIDTH.
MAX_DEEP_WIDTH = 8
MAX_DEPTH = 8


def layer_spans(shape):
    """-> (hidden, out_w, out_b, length) for an interleaved `shape`.

    `hidden` is one (w_off, b_off, fan_in, width) per hidden layer. The offsets
    are what mlp.glsl recomputes, so the two must agree - that is what
    tests/test_mlp_forward_gl.py compares, rather than a second reading of this
    comment.
    """
    off = 0
    hidden = []
    fan_in = IN_DIM
    for i in range(len(shape) // 2):
        w = int(shape[2 * i])
        w_off = off
        off += fan_in * w
        b_off = off
        off += w
        hidden.append((w_off, b_off, fan_in, w))
        fan_in = w
    out_w = off
    off += OUT_DIM * fan_in
    out_b = off
    off += OUT_DIM
    return hidden, out_w, out_b, off


def _weight_mask(shape) -> np.ndarray:
    """True where a float is a WEIGHT (W_SCALE), False where it is a bias."""
    hidden, out_w, out_b, n = layer_spans(shape)
    m = np.zeros(n, dtype=bool)
    for w_off, b_off, _fan_in, _w in hidden:
        m[w_off:b_off] = True
    m[out_w:out_b] = True
    return m


def _shape_from_layers(layers) -> tuple[int, ...]:
    """An interleaved shape from [[width, activation], ...], clamped.

    Clamps rather than raises, because the input may be a config written by a
    build with wider limits. A clamped stack no longer matches the signature it
    came from, and layout_from_signature refuses on exactly that mismatch - a
    refusal being the one outcome better than a plausible layout of the wrong
    width.
    """
    kept = list(layers)[:MAX_DEPTH]
    hi = MAX_WIDTH if len(kept) <= 1 else MAX_DEEP_WIDTH
    shape: list[int] = []
    for pair in kept:
        w, a = (list(pair) + [0, 0])[:2]
        shape.append(int(np.clip(int(w), MIN_WIDTH, hi)))
        shape.append(int(np.clip(int(a), 0, len(ACTIVATIONS) - 1)))
    return tuple(shape) or (16, 0)


def _draw_z(rng, n: int, dist: int) -> np.ndarray:
    if dist == 1:
        return rng.uniform(-1.0, 1.0, n)
    if dist == 2:
        z = rng.normal(0.0, 1.0, n)
        z[rng.random(n) < SPARSITY] = 0.0
        return z
    if dist == 3:
        return rng.standard_cauchy(n) * CAUCHY_SCALE
    return rng.normal(0.0, 0.5, n)          # what random() draws


class MLPModality:
    name = "mlp"
    modality_id = 3
    glsl_file = "shaders/brains/mlp.glsl"

    def settings_schema(self) -> list[Setting]:
        # lo/hi bound each layer's width and `choices` names the activations;
        # the DEPTH is bounded by MAX_DEPTH and by the float budget, both of
        # which layout_from_settings enforces - see the UI's `+` button, which
        # asks whether adding a layer changes the layout rather than
        # re-deriving either limit.
        return [
            Setting("layers", "Layers", "layers", MIN_WIDTH, MAX_WIDTH, 16,
                    choices=ACTIVATIONS),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        layers = s.get("layers")
        if not layers:
            # A config written before the stack existed. Accepted forever: this
            # is the whole of the migration.
            layers = [[int(s.get("hidden", 16)), int(s.get("activation", 0))]]
        shape = _shape_from_layers(layers)
        return BrainLayout("mlp", shape, layer_spans(shape)[3])

    # ---- structure the generic helpers cannot derive --------------------

    def signature_of(self, layout: BrainLayout) -> str:
        """mlp-n<w1>[.<w2>...]-a<a1>[.<a2>...].

        Depth 1 emits mlp-n16-a0, character for character what the positional
        form produced, so every archive directory and config stamp on disk still
        names the layout it always did. Dots are legal in directory names on
        both platforms and the string always ends in a digit.
        """
        w = ".".join(str(int(v)) for v in layout.shape[0::2])
        a = ".".join(str(int(v)) for v in layout.shape[1::2])
        return f"mlp-n{w}-a{a}"

    def settings_from_signature(self, sig: str):
        parts = sig.split("-")
        if len(parts) != 3 or not parts[1].startswith("n") \
                or not parts[2].startswith("a"):
            return None
        widths, acts = parts[1][1:].split("."), parts[2][1:].split(".")
        if len(widths) != len(acts) or not widths:
            return None
        if not all(t.isdigit() for t in widths + acts):
            return None
        return {"layers": [[int(w), int(a)] for w, a in zip(widths, acts)]}

    def settings_of(self, layout: BrainLayout) -> dict:
        return {"layers": [[int(w), int(a)] for w, a
                           in zip(layout.shape[0::2], layout.shape[1::2])]}

    def layout_uniforms(self, layout: BrainLayout) -> dict:
        """The stack, for mlp.glsl. `shape` verbatim and zero-padded, so there
        is exactly ONE encoding and the GPU never re-derives what the host
        already knows."""
        flat = [int(v) for v in layout.shape[: 2 * MAX_DEPTH]]
        flat += [0] * (2 * MAX_DEPTH - len(flat))
        return {"BRAIN_DEPTH": len(layout.shape) // 2, "BRAIN_LAYERS": flat}

    def unit_count(self, layout: BrainLayout) -> int:
        """The LAST hidden layer's units, the only ones that decompose
        additively into the output. An earlier layer's unit reaches the output
        through further nonlinearities and has no contribution to show."""
        return int(layout.shape[-2]) if len(layout.shape) >= 2 else 0

    # ---- per-layer operations -------------------------------------------

    distributions = DISTRIBUTIONS

    def layer_parts(self, layout: BrainLayout, i: int) -> dict:
        """{'weights': (lo, hi), 'biases': (lo, hi)} for hidden layer i.

        What the Brain window's right-click menu edits. Half-open, in floats,
        into the decoded genome. The output layer has no row of its own, so it
        is not addressable here - rerolling the last hidden layer leaves the
        output weights alone, which is what makes the op readable on screen.
        """
        hidden, _out_w, _out_b, _n = layer_spans(layout.shape)
        w_off, b_off, _fan_in, w = hidden[i]
        return {"weights": (w_off, b_off), "biases": (b_off, b_off + w)}

    def layer_reroll(self, rng, layout: BrainLayout, i: int, part: str,
                     dist: int) -> np.ndarray:
        """Fresh DECODED values for one part of one layer.

        Drawn in z and then squashed, exactly as decode() would, so a reroll is
        always a genome the search could also have produced.
        """
        lo, hi = self.layer_parts(layout, i)[part]
        scale = W_SCALE if part == "weights" else B_SCALE
        return (scale * np.tanh(_draw_z(rng, hi - lo, int(dist)))
                ).astype(np.float32)

    # ---- mutation and crossover ----------------------------------------

    # No unit and no scales, both for the same reason: a hidden unit's input
    # weights, bias and output column are three separate regions of the buffer
    # (W_out is output-major), so nothing is contiguous to cross over, and every
    # float is an independent weight or bias - the standard perturbation.
    # mlp.glsl's mlp_param_at is brain_add for every index.
    UNIT_FLOATS = None
    SCALE_OFFSETS = frozenset()

    def unit_floats(self, layout: BrainLayout):
        return self.UNIT_FLOATS

    def scale_mask(self, layout: BrainLayout):
        return unit_scale_mask(layout, self.UNIT_FLOATS, self.SCALE_OFFSETS)

    # ---- decode / encode -----------------------------------------------

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        """FLAT, in every layer. The full +/-W_SCALE range is reachable
        everywhere, by the search and by hand alike, so a rail means one thing
        rather than a different thing per layer. A fan-in gain folded in here
        would not be a normalisation but a CAP - see random()."""
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        scale = np.where(_weight_mask(layout.shape), W_SCALE, B_SCALE)
        return (scale * np.tanh(z)).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        scale = np.where(_weight_mask(layout.shape), W_SCALE, B_SCALE)
        raw = p / scale
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        """A fresh brain, fan-in normalised at INITIALISATION only.

        A hidden->hidden layer sums w_{l-1} terms rather than 4, so at the plain
        draw its pre-activations grow like sqrt(w_{l-1}) and a deep brain would
        be born saturated - a sign function with a flat landscape around it.
        Only the hidden->hidden weights are scaled, so a depth-1 draw has no
        such slice and is bit-identical to what this modality always produced.

        A starting point, not a limit: the search and any later edit can take a
        layer anywhere inside the rails.
        """
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        hidden, _out_w, _out_b, _n = layer_spans(layout.shape)
        for w_off, b_off, fan_in, _w in hidden[1:]:
            z[w_off:b_off] *= np.sqrt(IN_DIM / float(fan_in))
        return self.decode(z, layout)


register(MLPModality())
