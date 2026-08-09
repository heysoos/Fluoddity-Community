// Gaussian envelope around an oscillation. Localised: a filter responds near
// its centre and is silent elsewhere, so a brain is a set of "when I see
// roughly this, do that" rules rather than one global interference pattern.
//
// Layout, 14 floats: centre(4), frequency(4), amplitude(4), sigma, phase.
//
// PURE: reads only brain_params (through the mutation helpers), base, x and
// BRAIN_SHAPE. The Brain Inspector calls this from a fragment pass.

// The i-th float as the particle sees it. THE definition of this modality's
// mutation - the evaluation below reads through it rather than duplicating the
// field mapping, because two copies of that mapping is precisely how the
// Fourier writeback came to disagree with its own evaluation.
float gabor_param_at(uint base, int i) {
    int k = i % 14;
    // Frequency and envelope width SCALE; centre, amplitude and phase OFFSET.
    // Offsetting the width would let mutation walk it through zero, and the
    // envelope divides by its square.
    if ((k >= 4 && k < 8) || k == 12) return brain_mul(base, i);
    return brain_add(base, i);
}

vec4 gabor_v4(uint base, int i) {
    return vec4(gabor_param_at(base, i),     gabor_param_at(base, i + 1),
                gabor_param_at(base, i + 2), gabor_param_at(base, i + 3));
}

// ONE filter's contribution. The Brain Inspector draws exactly this.
vec4 gabor_unit(uint base, int i, vec4 x) {
    int o = i * 14;
    vec4 c = gabor_v4(base, o);
    vec4 f = gabor_v4(base, o + 4);
    vec4 a = gabor_v4(base, o + 8);
    // decode() floors sigma at 0.15, but mutation scales it afterwards, so the
    // guard is here rather than only on the host. The sign is irrelevant - only
    // sigma^2 is used - but zero is a division by zero.
    float sg = max(abs(gabor_param_at(base, o + 12)), 1e-3);
    float ph = gabor_param_at(base, o + 13);
    vec4 d = x - c;
    float env = exp(-dot(d, d) / (2.0 * sg * sg));
    return a * (env * cos(dot(x, f) + ph));
}

vec4 brain_gabor(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    for (int i = 0; i < n; i++) {
        result += gabor_unit(base, i, x);
    }
    return result;
}
