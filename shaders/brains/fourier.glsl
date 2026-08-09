// Random-Fourier-features brain. Bit-exact with the original fourier_noise()
// at 10 centers: same basis, same index-derived phase offsets.
//
// Layout, 8 floats per centre: frequency(4), amplitude(4).
//
// PURE: reads only brain_params (via brain_at), base, x and BRAIN_SHAPE. The
// Brain Inspector calls this from a fragment pass, so it must not touch entity
// state.
vec4 brain_fourier(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    for (int i = 0; i < n; i++) {
        vec4 f = brain_at4(base, i * 8);
        vec4 a = brain_at4(base, i * 8 + 4);
        float phase = dot(x, f);
        float po = 2.0 * float(i) * 0.6283 + a.w * 3.14159;
        vec4 basis = vec4(
            sin(phase + po),
            cos(phase + po * 0.7),
            sin(phase * 2.0 + po * 1.3),
            cos(phase * 2.0 + po * 0.5)
        );
        result += a * basis;
    }
    return result;
}
