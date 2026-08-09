// Lenia growth bands. The -1 is what makes this Lenia: response is positive
// inside a narrow band of sensor values and NEGATIVE everywhere else, the
// "thrive at this density, die away from it" rule that produces membranes.
// Drop it and the response is a smooth blur.
//
// Layout, 10 floats: projection(4), amplitude(4), mu, sigma.
//
// PURE: reads only brain_params (through the mutation helpers), base, x and
// BRAIN_SHAPE.

// The i-th float as the particle sees it - this modality's mutation, and the
// single source the evaluation reads through.
float lenia_param_at(uint base, int i) {
    int k = i % 10;
    // The projection and the band width SCALE; amplitude and the band centre
    // OFFSET. mu is a LOCATION on the u axis, so an offset moves the band -
    // scaling it would pin a band centred near zero at zero forever. sigma is a
    // WIDTH, and the growth term divides by its square.
    if (k < 4 || k == 9) return brain_mul(base, i);
    return brain_add(base, i);
}

vec4 lenia_v4(uint base, int i) {
    return vec4(lenia_param_at(base, i),     lenia_param_at(base, i + 1),
                lenia_param_at(base, i + 2), lenia_param_at(base, i + 3));
}

vec4 brain_lenia(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    for (int i = 0; i < n; i++) {
        int o = i * 10;
        vec4 w = lenia_v4(base, o);
        vec4 a = lenia_v4(base, o + 4);
        float mu = lenia_param_at(base, o + 8);
        // Bands are narrow (decode floors sigma at 0.02); mutation scales it
        // further, so guard against a zero denominator here too.
        float sg = max(abs(lenia_param_at(base, o + 9)), 1e-3);
        float u = dot(x, w) - mu;
        float g = 2.0 * exp(-(u * u) / (2.0 * sg * sg)) - 1.0;
        result += a * g;
    }
    return result;
}
