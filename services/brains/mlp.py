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

from services.brains import (AUDIO_INPUTS_SETTING, AUDIO_SCALE_SETTING,
                             MAX_BRAIN_FLOATS, BrainLayout, Setting,
                             audio_inputs_of, register, unit_scale_mask)
from services.brains import settings_of as registry_settings_of

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
# written is still reachable and still decodes to the same brain. Every layer
# of every stack may reach it; what actually stops a stack growing is
# MAX_BRAIN_FLOATS, which BrainLayout enforces.
MAX_WIDTH = 48
MAX_DEPTH = 8
# A new layer's width. The `layers` Setting's own default, so the search adds
# what the `+ Add layer` button adds.
DEFAULT_NEW_LAYER_WIDTH = 16

# What mlp.glsl's ping-pong locals must hold, and the ONLY thing the compiled
# shader is sized by. Bucketed so a width drag lands on a handful of variants
# rather than one per value, and floored at 4 because mlp_hidden seeds `cur`
# with the four sensor taps before it looks at any width. A DEPTH-1 stack takes
# the floor: it never enters the deep path, and the arrays it does not use are
# exactly what used to tax it. Cost rises steeply with the bucket - see the
# variable-depth MLP caveat in CLAUDE.md and tools/measure_brain_depth.py.
SCRATCH_BUCKETS = (4, 8, 16, 24, 32, 48)


def scratch_width(shape, audio_inputs: int = 0) -> int:
    """-> the MAX_MLP_WIDTH bucket an interleaved `shape` needs.

    The scratch array is seeded with every input, so it must hold the four
    taps plus the audio channels whatever the depth."""
    widths = [int(v) for v in shape[0::2]]
    want = max(widths) if len(widths) > 1 else 0
    want = max(want, IN_DIM + int(audio_inputs))
    for b in SCRATCH_BUCKETS:
        if want <= b:
            return b
    return SCRATCH_BUCKETS[-1]


def layer_spans(shape, audio_inputs: int = 0):
    """-> (hidden, out_w, out_b, length) for an interleaved `shape`.

    `hidden` is one (w_off, b_off, fan_in, width) per hidden layer. The offsets
    are what mlp.glsl recomputes, so the two must agree - that is what
    tests/test_mlp_forward_gl.py compares, rather than a second reading of this
    comment. The first layer's fan-in is the four taps plus the audio
    channels, whose columns come LAST in each row.
    """
    off = 0
    hidden = []
    fan_in = IN_DIM + int(audio_inputs)
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


def _weight_mask(shape, audio_inputs: int = 0) -> np.ndarray:
    """True where a float is a WEIGHT (W_SCALE), False where it is a bias."""
    hidden, out_w, out_b, n = layer_spans(shape, audio_inputs)
    m = np.zeros(n, dtype=bool)
    for w_off, b_off, _fan_in, _w in hidden:
        m[w_off:b_off] = True
    m[out_w:out_b] = True
    return m


def _audio_index(shape, audio_inputs: int) -> np.ndarray:
    """Flat indices of the first layer's audio columns."""
    k = int(audio_inputs)
    hidden, _ow, _ob, _n = layer_spans(shape, k)
    if not k or not hidden:
        return np.zeros(0, dtype=np.int64)
    w_off, _b_off, fan_in, w = hidden[0]
    return (w_off + np.arange(w)[:, None] * fan_in + IN_DIM
            + np.arange(k)[None, :]).reshape(-1)


def _shape_from_layers(layers, audio_inputs: int = 0) -> tuple[int, ...]:
    """An interleaved shape from [[width, activation], ...], clamped.

    Clamps rather than raises, because the input may be a config written by a
    build with wider limits. A clamped stack no longer matches the signature it
    came from, and layout_from_signature refuses on exactly that mismatch - a
    refusal being the one outcome better than a plausible layout of the wrong
    width.

    The FLOAT BUDGET is clamped here too, and it is the only limit a deep stack
    meets before MAX_WIDTH: nothing else stops [48]x8, which is sixteen times
    over. Applied as one cap across the stack, largest that fits, so a narrow
    layer is left alone. It always terminates - every layer at MIN_WIDTH is 27
    floats at the deepest.
    """
    kept = list(layers)[:MAX_DEPTH]
    shape: list[int] = []
    for pair in kept:
        w, a = (list(pair) + [0, 0])[:2]
        shape.append(int(np.clip(int(w), MIN_WIDTH, MAX_WIDTH)))
        shape.append(int(np.clip(int(a), 0, len(ACTIVATIONS) - 1)))
    if not shape:
        return (16, 0)

    def capped(c):
        return tuple(min(v, c) if i % 2 == 0 else v
                     for i, v in enumerate(shape))

    def fits(c):
        return layer_spans(capped(c), audio_inputs)[3] <= MAX_BRAIN_FLOATS

    if fits(MAX_WIDTH):
        return tuple(shape)
    lo, hi = MIN_WIDTH, MAX_WIDTH
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return capped(lo)


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
        #
        # The two scales are what every other modality already declares: they
        # change what a z MEANS without changing how many floats it has, so
        # they neither split the archive nor reset a search, and they are the
        # only thing here a search or an audio mapping can move continuously.
        # Their ranges are measured; see the MLP scale caveat in CLAUDE.md.
        return [
            Setting("layers", "Layers", "layers", MIN_WIDTH, MAX_WIDTH, 16,
                    choices=ACTIVATIONS),
            Setting("w_scale", "Weight Scale", "float", 0.25, 8.0, W_SCALE),
            Setting("b_scale", "Bias Scale", "float", 0.0, 4.0, B_SCALE),
            AUDIO_INPUTS_SETTING,
            AUDIO_SCALE_SETTING,
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        layers = s.get("layers")
        if not layers:
            # A config written before the stack existed. Accepted forever: this
            # is the whole of the migration.
            layers = [[int(s.get("hidden", 16)), int(s.get("activation", 0))]]
        k = audio_inputs_of(s)
        shape = _shape_from_layers(layers, k)
        return BrainLayout("mlp", shape, layer_spans(shape, k)[3], scales=(
            ("w_scale", float(s.get("w_scale", W_SCALE))),
            ("b_scale", float(s.get("b_scale", B_SCALE))),
            ("audio_scale", float(s.get("audio_scale",
                                        AUDIO_SCALE_SETTING.default))),
        ), audio_inputs=k)

    def audio_weight_index(self, layout: BrainLayout) -> np.ndarray:
        return _audio_index(layout.shape, layout.audio_inputs)

    @staticmethod
    def _scales(layout: BrainLayout):
        """(weight, bias). The weight scale is floored: at zero every weight is
        zero, which is a brain with no input at all rather than a quiet one."""
        return (max(layout.scale("w_scale", W_SCALE), EPS),
                layout.scale("b_scale", B_SCALE))

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

    def layout_moves(self, layout: BrainLayout, bounds):
        """The stack's own operators. -> [(operator, settings), ...]

        Proposals only: layout_moves.candidate_moves rebuilds each one and
        rejects any that _shape_from_layers clamped, which is the check that
        makes a bounded proposal safe.

        Growth is by ONE unit and a new layer is APPENDED. An inserted layer
        would renumber every layer after it, leaving nothing for the transfer
        to carry across; a large jump is a restart wearing a growth move's
        name.
        """
        # The REGISTRY's, not this class's: it carries the scales and the
        # audio input count, and candidate_moves compares a rebuilt child
        # against exactly that dict.
        base = registry_settings_of(layout)
        layers = [list(p) for p in base["layers"]]
        depth = len(layers)
        max_depth = (MAX_DEPTH if bounds.max_depth is None
                     else min(MAX_DEPTH, int(bounds.max_depth)))
        max_width = (MAX_WIDTH if bounds.max_width is None
                     else min(MAX_WIDTH, int(bounds.max_width)))
        out = []

        def with_layers(new):
            return {**base, "layers": new}

        for i in range(depth):
            w, a = layers[i]
            if w + 1 <= max_width:
                grown = [list(p) for p in layers]
                grown[i][0] = w + 1
                out.append(("grow", with_layers(grown)))
            if w - 1 >= MIN_WIDTH:
                shrunk = [list(p) for p in layers]
                shrunk[i][0] = w - 1
                out.append(("shrink", with_layers(shrunk)))
            # The LAST hidden layer may not go: a brain with none is not a
            # smaller brain, it is a different model.
            if depth > 1:
                dropped = [list(p) for j, p in enumerate(layers) if j != i]
                out.append(("drop_layer", with_layers(dropped)))
            for act in range(len(ACTIVATIONS)):
                if act != a:
                    swapped = [list(p) for p in layers]
                    swapped[i][1] = act
                    out.append(("activation", with_layers(swapped)))

        if depth < max_depth:
            width = min(int(DEFAULT_NEW_LAYER_WIDTH), max_width)
            out.append(("add_layer",
                        with_layers([list(p) for p in layers] + [[width, 0]])))
        return out

    def transfer_genome(self, params, parent: BrainLayout,
                        child: BrainLayout, rng) -> np.ndarray:
        """Repack a decoded stack into `child`'s shape.

        A hidden unit is three regions - input weights, bias, and a COLUMN of
        the next matrix - and W_out is OUTPUT-MAJOR, so that column is one
        strided entry per output rather than a contiguous append. Every offset
        comes from layer_spans, which is the definition mlp.glsl is checked
        against.

        Whatever a layer does not inherit is DRAWN, then its outgoing column is
        zeroed: the child evaluates exactly as its parent did at birth, while a
        new unit still has an incoming half to contribute the moment the search
        moves its output weight.
        """
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        out = np.asarray(self.random(rng, child),
                         dtype=np.float32).reshape(-1)
        p_hidden, p_ow, p_ob, _pn = layer_spans(parent.shape,
                                                parent.audio_inputs)
        c_hidden, c_ow, c_ob, _cn = layer_spans(child.shape,
                                                child.audio_inputs)

        # Layer l of the child inherits from layer l of the parent, which is
        # what makes APPENDING a layer the only safe way to deepen a stack.
        for li, (w_off, b_off, fan_in, w) in enumerate(c_hidden):
            if li >= len(p_hidden):
                break
            pw_off, pb_off, p_fan, p_w = p_hidden[li]
            rows, cols = min(w, p_w), min(fan_in, p_fan)
            src = p[pw_off:pb_off].reshape(p_w, p_fan)
            dst = out[w_off:b_off].reshape(w, fan_in)
            dst[:rows, :cols] = src[:rows, :cols]
            # A widened fan-in means the layer BELOW grew: those columns are
            # the new unit's outgoing weights and start silent.
            dst[:rows, cols:] = 0.0
            out[b_off:b_off + rows] = p[pb_off:pb_off + rows]

        # W_out, output-major (OUT_DIM, fan_in of the last hidden layer).
        p_fan = p_hidden[-1][3] if p_hidden else IN_DIM + parent.audio_inputs
        c_fan = c_hidden[-1][3] if c_hidden else IN_DIM + child.audio_inputs
        src = p[p_ow:p_ob].reshape(OUT_DIM, p_fan)
        dst = out[c_ow:c_ob].reshape(OUT_DIM, c_fan)
        cols = min(c_fan, p_fan)
        # Only when the LAST layer is the same one: a dropped or added layer
        # changes which units W_out reads, and those weights mean nothing.
        if len(c_hidden) == len(p_hidden):
            dst[:, :cols] = src[:, :cols]
            dst[:, cols:] = 0.0
            out[c_ob:c_ob + OUT_DIM] = p[p_ob:p_ob + OUT_DIM]
        return out.astype(np.float32)

    def layout_uniforms(self, layout: BrainLayout) -> dict:
        """The stack, for mlp.glsl. `shape` verbatim and zero-padded, so there
        is exactly ONE encoding and the GPU never re-derives what the host
        already knows."""
        flat = [int(v) for v in layout.shape[: 2 * MAX_DEPTH]]
        flat += [0] * (2 * MAX_DEPTH - len(flat))
        return {"BRAIN_DEPTH": len(layout.shape) // 2, "BRAIN_LAYERS": flat}

    def shader_defines(self, layout: BrainLayout) -> dict:
        """What mlp.glsl must be COMPILED with while `layout` runs, NOT uniforms.

        A uniform cannot size a local array, so the scratch width has to be
        baked in - which is what makes the entity-update program per-layout and
        why sim.py caches one per distinct set of these.

        Answered for ANY layout, because mlp.glsl is compiled into the same
        program as the other three and its arrays are allocated whether a
        particle runs an MLP or not. Someone else's layout gets the floor,
        which is what keeps a wide stack's cost off every other brain - it is
        the whole reason this is a define rather than a raised constant.
        """
        shape = layout.shape if layout.modality == self.name else ()
        return {"MAX_MLP_WIDTH": scratch_width(shape, layout.audio_inputs)}

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
        hidden, _out_w, _out_b, _n = layer_spans(layout.shape,
                                                 layout.audio_inputs)
        w_off, b_off, _fan_in, w = hidden[i]
        return {"weights": (w_off, b_off), "biases": (b_off, b_off + w)}

    def layer_reroll(self, rng, layout: BrainLayout, i: int, part: str,
                     dist: int) -> np.ndarray:
        """Fresh DECODED values for one part of one layer.

        Drawn in z and then squashed, exactly as decode() would, so a reroll is
        always a genome the search could also have produced.
        """
        lo, hi = self.layer_parts(layout, i)[part]
        w, b = self._scales(layout)
        scale = w if part == "weights" else b
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
        w, b = self._scales(layout)
        scale = np.where(_weight_mask(layout.shape, layout.audio_inputs), w, b)
        audio = _audio_index(layout.shape, layout.audio_inputs)
        scale[audio] *= layout.scale("audio_scale", 1.0)
        return (scale * np.tanh(z)).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        w, b = self._scales(layout)
        scale = np.where(_weight_mask(layout.shape, layout.audio_inputs),
                         w, max(b, EPS))
        audio = _audio_index(layout.shape, layout.audio_inputs)
        scale[audio] *= max(layout.scale("audio_scale", 1.0), EPS)
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
        from services.brains import audio_aware_normal

        def normalise(z0):
            hidden, _out_w, _out_b, _n = layer_spans(layout.shape, 0)
            for w_off, b_off, fan_in, _w in hidden[1:]:
                z0[w_off:b_off] *= np.sqrt(IN_DIM / float(fan_in))
            return z0

        z = audio_aware_normal(rng, layout, 0.5, normalise)
        return self.decode(z, layout)


register(MLPModality())
