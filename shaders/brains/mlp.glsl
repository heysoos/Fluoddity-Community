// out = W_out . act_k(...act_1(W1 x + b1)...) + b_out, one or more hidden layers.
//
// Packing, which services/brains/mlp.py's layer_spans() defines and this file
// recomputes - the tile_lo_texel() situation, guarded by a GPU-versus-NumPy
// parity test rather than by a second reading of this comment:
//
//     for each hidden layer l:  W_l (w_l rows of fan_in_l), then b_l (w_l)
//     then W_out (4 rows of w_k), then b_out (4)
//
// W_out is OUTPUT-MAJOR, so final-layer unit j's four output weights sit at
// stride w_k. That is what lets the depth-1 loop finish with each unit
// completely - activation, output column, accumulate - and never hold the
// hidden vector at all.
//
// PURE: reads only brain_params (through the mutation helpers), base, x,
// BRAIN_SHAPE, BRAIN_DEPTH and BRAIN_LAYERS.

// MAX_MLP_WIDTH sizes the ping-pong locals the deep path needs to hold a
// layer's activations, and THAT COST IS PAID BY EVERY BRAIN IN THIS FILE - the
// arrays are allocated per invocation whatever the uniform branch does, so a
// Fourier preset's step time rises with this number without a line of MLP ever
// running. It is therefore COMPILED PER LAYOUT: sim.py prepends the width the
// live stack actually needs, and this default only covers a build that does
// not (the Inspector, the compile check, the GPU tests). Never raise the
// default to cover a wider stack - that taxes every other layout instead. See
// the caveat in CLAUDE.md and tools/measure_brain_depth.py.
//
// The floor is 4: mlp_hidden seeds `cur` with the four sensor taps before it
// looks at any width. MAX_MLP_DEPTH mirrors services/brains/mlp.py's MAX_DEPTH
// and sizes a uniform array, which costs nothing per invocation.
#ifndef MAX_MLP_WIDTH
#define MAX_MLP_WIDTH 48
#endif
#define MAX_MLP_DEPTH 8

// layout.shape verbatim, zero-padded: (w1, a1, w2, a2, ...). ONE encoding, so
// the GPU never re-derives what the host already knows.
uniform int BRAIN_DEPTH;
uniform int BRAIN_LAYERS[2 * MAX_MLP_DEPTH];

int mlp_width(int l)  { return BRAIN_LAYERS[2 * l]; }
int mlp_act_of(int l) { return BRAIN_LAYERS[2 * l + 1]; }

// Activation is part of the LAYOUT, not a parameter: tanh and sin are different
// function families and a genome evolved under one means nothing under the
// other.
float mlp_act_v(int a, float v) {
    if (a == 0) return tanh(v);
    if (a == 1) return sin(v);
    // GELU, the usual tanh approximation.
    return 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
}

float mlp_act(float v) {
    return mlp_act_v(BRAIN_SHAPE.y, v);
}

// Every weight and bias is an independent scalar, so they all OFFSET - the
// standard weight perturbation. Nothing here is a width or a direction.
float mlp_param_at(uint base, int i) {
    return brain_add(base, i);
}

// ---- offsets --------------------------------------------------------------

// Layer l's weight block, its bias block, and its fan-in.
void mlp_layer_off(int l, out int w_off, out int b_off, out int fan) {
    int off = 0;
    fan = 4;
    for (int i = 0; i < l; i++) {
        int w = mlp_width(i);
        off += fan * w + w;
        fan = w;
    }
    w_off = off;
    b_off = off + fan * mlp_width(l);
}

// W_out's offset, which is everything the hidden layers occupy.
int mlp_out_off() {
    int w_off, b_off, fan;
    mlp_layer_off(BRAIN_DEPTH - 1, w_off, b_off, fan);
    return b_off + mlp_width(BRAIN_DEPTH - 1);
}

// ---- the deep path --------------------------------------------------------

// Runs hidden layers [0, stop) into `cur`, and returns the width left there.
// stop == 0 leaves x itself, which is what makes layer 0 need no special case.
int mlp_hidden(uint base, vec4 x, int stop, inout float cur[MAX_MLP_WIDTH]) {
    float nxt[MAX_MLP_WIDTH];
    cur[0] = x.x; cur[1] = x.y; cur[2] = x.z; cur[3] = x.w;
    int fan = 4;
    int off = 0;
    for (int l = 0; l < stop; l++) {
        int w = mlp_width(l);
        int b_off = off + fan * w;
        int act = mlp_act_of(l);
        for (int j = 0; j < w; j++) {
            float s = mlp_param_at(base, b_off + j);
            int r = off + j * fan;
            for (int i = 0; i < fan; i++) {
                s += cur[i] * mlp_param_at(base, r + i);
            }
            nxt[j] = mlp_act_v(act, s);
        }
        for (int j = 0; j < w; j++) cur[j] = nxt[j];
        off = b_off + w;
        fan = w;
    }
    return fan;
}

// One final-layer unit's four output weights. `c` is its column in W_out.
vec4 mlp_out_column(uint base, int out_off, int c, int wk) {
    return vec4(mlp_param_at(base, out_off + c),
                mlp_param_at(base, out_off + c + wk),
                mlp_param_at(base, out_off + c + 2 * wk),
                mlp_param_at(base, out_off + c + 3 * wk));
}

vec4 mlp_out_bias(uint base, int out_off, int wk) {
    int b = out_off + 4 * wk;
    return vec4(mlp_param_at(base, b),     mlp_param_at(base, b + 1),
                mlp_param_at(base, b + 2), mlp_param_at(base, b + 3));
}

// ---- the contract ---------------------------------------------------------

// ONE final-layer unit's contribution: its activation times its output column.
// Note a unit is NOT contiguous in the buffer - its input weights, bias and
// output column are three different regions - so the Inspector cannot isolate
// it by shifting `base`, only by calling this.
vec4 mlp_unit(uint base, int j, vec4 x) {
    if (BRAIN_DEPTH <= 1) {
        // Today's path, verbatim. A uniform branch, so no divergence, and a
        // depth-1 brain cannot regress.
        int h = BRAIN_SHAPE.x;
        int r = j * 4;
        vec4 w1 = vec4(mlp_param_at(base, r),     mlp_param_at(base, r + 1),
                       mlp_param_at(base, r + 2), mlp_param_at(base, r + 3));
        float a = mlp_act(dot(x, w1) + mlp_param_at(base, 4 * h + j));
        int c = 5 * h + j;
        return a * vec4(mlp_param_at(base, c),
                        mlp_param_at(base, c + h),
                        mlp_param_at(base, c + 2 * h),
                        mlp_param_at(base, c + 3 * h));
    }
    int k = BRAIN_DEPTH - 1;
    float cur[MAX_MLP_WIDTH];
    int fan = mlp_hidden(base, x, k, cur);
    int w_off, b_off, unused;
    mlp_layer_off(k, w_off, b_off, unused);
    float s = mlp_param_at(base, b_off + j);
    int r = w_off + j * fan;
    for (int i = 0; i < fan; i++) {
        s += cur[i] * mlp_param_at(base, r + i);
    }
    float a = mlp_act_v(mlp_act_of(k), s);
    return a * mlp_out_column(base, mlp_out_off(), j, mlp_width(k));
}

vec4 brain_mlp(uint base, vec4 x) {
    if (BRAIN_DEPTH <= 1) {
        int h = BRAIN_SHAPE.x;
        int b2 = 9 * h;
        // The output bias belongs to no hidden unit, so it is added here and
        // the Inspector's per-unit tiles do not include it.
        vec4 result = vec4(mlp_param_at(base, b2),     mlp_param_at(base, b2 + 1),
                           mlp_param_at(base, b2 + 2), mlp_param_at(base, b2 + 3));
        for (int j = 0; j < h; j++) {
            result += mlp_unit(base, j, x);
        }
        return result;
    }
    float cur[MAX_MLP_WIDTH];
    int wk = mlp_hidden(base, x, BRAIN_DEPTH, cur);
    int out_off = mlp_out_off();
    vec4 result = mlp_out_bias(base, out_off, wk);
    for (int j = 0; j < wk; j++) {
        result += cur[j] * mlp_out_column(base, out_off, j, wk);
    }
    return result;
}
