// out = W2 . act(W1 x + b1) + b2, one hidden layer.
//
// Layout, 9H + 4 floats: W1 (H rows of 4), b1 (H), W2 (4 rows of H), b2 (4).
//
// W2 is stored OUTPUT-MAJOR, so hidden unit j's four output weights sit at
// stride H. That is deliberate: it lets the loop finish with each hidden unit
// completely - compute its activation, scale its output column, accumulate -
// and never hold the hidden vector. Hidden-major would read a contiguous vec4
// but force a local float[48] per invocation, which spills to scratch memory
// for every particle every frame.
//
// PURE: reads only brain_params (through the mutation helpers), base, x and
// BRAIN_SHAPE.

// Activation is part of the LAYOUT (BRAIN_SHAPE.y), not a parameter: tanh and
// sin are different function families and a genome evolved under one means
// nothing under the other.
float mlp_act(float v) {
    int a = BRAIN_SHAPE.y;
    if (a == 0) return tanh(v);
    if (a == 1) return sin(v);
    // GELU, the usual tanh approximation.
    return 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
}

// Every weight and bias is an independent scalar, so they all OFFSET - the
// standard weight perturbation. Nothing here is a width or a direction.
float mlp_param_at(uint base, int i) {
    return brain_add(base, i);
}

// ONE hidden unit's contribution: its activation times its output column.
// Note a hidden unit is NOT contiguous in the buffer - its input weights, bias
// and output column are three different regions - so the Inspector cannot
// isolate it by shifting `base`, only by calling this.
vec4 mlp_unit(uint base, int j, vec4 x) {
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

vec4 brain_mlp(uint base, vec4 x) {
    int h = BRAIN_SHAPE.x;
    int b2 = 9 * h;
    // The output bias belongs to no hidden unit, so it is added here and the
    // Inspector's per-unit tiles do not include it.
    vec4 result = vec4(mlp_param_at(base, b2),     mlp_param_at(base, b2 + 1),
                       mlp_param_at(base, b2 + 2), mlp_param_at(base, b2 + 3));
    for (int j = 0; j < h; j++) {
        result += mlp_unit(base, j, x);
    }
    return result;
}
