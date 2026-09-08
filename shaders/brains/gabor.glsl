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
// One filter: centre(4), frequency(4), amplitude(4), sigma, phase, then its
// audio weights, which extend the FREQUENCY and never the centre.
int gabor_stride() { return 14 + BRAIN_AUDIO_IN; }

// Where float i sits in the DEAF brain, for the mutation's hash.
int gabor_deaf_index(int i) {
    int s = gabor_stride();
    int u = i / s, k = i % s;
    return (k < 14) ? u * 14 + k : -(1 + u * BRAIN_AUDIO_IN + (k - 14));
}

float gabor_param_at(uint base, int i) {
    int k = i % gabor_stride();
    int j = gabor_deaf_index(i);
    // Frequency, envelope width and audio SCALE; centre, amplitude and phase
    // OFFSET. Offsetting the width would let mutation walk it through zero,
    // and the envelope divides by its square.
    if ((k >= 4 && k < 8) || k == 12 || k >= 14) return brain_mul_at(base, i, j);
    return brain_add_at(base, i, j);
}

float gabor_audio_phase(uint base, int o) {
    float s = 0.0;
    for (int k = 0; k < BRAIN_AUDIO_IN; k++) s += g_audio[k] * gabor_param_at(base, o + k);
    return s;
}

vec4 gabor_v4(uint base, int i) {
    return vec4(gabor_param_at(base, i),     gabor_param_at(base, i + 1),
                gabor_param_at(base, i + 2), gabor_param_at(base, i + 3));
}

// ONE filter's contribution. The Brain Inspector draws exactly this.
vec4 gabor_unit(uint base, int i, vec4 x) {
    int o = i * gabor_stride();
    vec4 c = gabor_v4(base, o);
    vec4 f = gabor_v4(base, o + 4);
    vec4 a = gabor_v4(base, o + 8);
    // decode() already floors sigma, but the floor is proportional to Input
    // Scale (0.1 at the defaults, 0.002 at the slider's minimum) and mutation
    // scales it further, so the guard belongs here too. The sign is irrelevant -
    // only sigma^2 is used - but zero is a division by zero.
    float sg = max(abs(gabor_param_at(base, o + 12)), 1e-3);
    float ph = gabor_param_at(base, o + 13);
    vec4 d = x - c;
    float env = exp(-dot(d, d) / (2.0 * sg * sg));
    float arg = dot(x, f);
    if (BRAIN_AUDIO_IN > 0) arg += gabor_audio_phase(base, o + 14);
    return a * (env * cos(arg + ph));
}

vec4 brain_gabor(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    for (int i = 0; i < n; i++) {
        result += gabor_unit(base, i, x);
    }
    return result;
}
