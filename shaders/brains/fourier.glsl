// Random-Fourier-features brain. Bit-exact with the original fourier_noise()
// at 10 centers: same basis, same index-derived phase offsets.
//
// Layout, 8 floats per centre: frequency(4), amplitude(4).
//
// MUTATION LIVES HERE, not in the generic brain_add/brain_mul helpers, because
// the original is STRUCTURED in ways a per-float helper cannot express:
//
//   - ONE scalar scales all four frequency components of a centre, so the
//     frequency VECTOR keeps its direction and only changes magnitude.
//     Jittering each component independently rotates it as well, which is a
//     different function, not a rounding difference - see the brain mutation
//     caveat in CLAUDE.md.
//   - amplitudes take a vec4 from ONE hash4() per centre, not four hash().
//   - the seed is derived from the rule's CONTENT, so two cohorts holding
//     different rules diverge even at the same cohort index.
//
// A modality owns its mutation the same way it owns decode().
//
// PURE: reads only brain_params, base, x and BRAIN_SHAPE. The Brain Inspector
// calls this from a fragment pass, so it must not touch entity state.

// The seed the original hashed: centers[4].frequency.xy + centers[7].amplitude.yx
// + centers[1].frequency.zw, plus the cohort. The modulo keeps a short layout
// (Centers < 8) in range; at the legacy 10 centres it is the identity.
float fourier_mut_seed(uint base, int n) {
    int m = max(n, 1);
    uint c4 = uint(4 % m) * 8u;
    uint c7 = uint(7 % m) * 8u;
    uint c1 = uint(1 % m) * 8u;
    vec2 a = vec2(brain_params[base + c4 + 0u], brain_params[base + c4 + 1u]);
    vec2 b = vec2(brain_params[base + c7 + 5u], brain_params[base + c7 + 4u]);
    vec2 c = vec2(brain_params[base + c1 + 2u], brain_params[base + c1 + 3u]);
    return hash(a + b + c) + g_brain_cohort;
}

// One centre's mutation. Shared by the SSBO path and the writeback so the two
// cannot drift apart - they must agree exactly or click-to-adopt copies a rule
// the particle was never running.
void fourier_mutate(inout vec4 freq, inout vec4 amp, int i, float mseed) {
    amp += g_brain_mut * (-1.0 + 2.0 * hash4(-.5 + vec2(float(-i) + mseed, float(i))));
    freq *= 1.0 + g_brain_mut * 0.5 * (hash(vec2(mseed, float(i))) - .5);
}

void fourier_load(uint base, int i, out vec4 freq, out vec4 amp) {
    uint o = base + uint(i * 8);
    freq = vec4(brain_params[o + 0u], brain_params[o + 1u],
                brain_params[o + 2u], brain_params[o + 3u]);
    amp  = vec4(brain_params[o + 4u], brain_params[o + 5u],
                brain_params[o + 6u], brain_params[o + 7u]);
}

// The basis, over a centre held in registers rather than read from the buffer.
//
// The phase offset is derived from the centre INDEX, which is why a unit cannot
// be previewed by pointing `base` at it and setting n=1: centre 5 seen at index
// 0 is a different function from the one the particles run. `i` is therefore a
// parameter rather than something the caller can fake.
vec4 fourier_eval(vec4 f, vec4 a, int i, vec4 x) {
    float phase = dot(x, f);
    float po = 2.0 * float(i) * 0.6283 + a.w * 3.14159;
    vec4 basis = vec4(
        sin(phase + po),
        cos(phase + po * 0.7),
        sin(phase * 2.0 + po * 1.3),
        cos(phase * 2.0 + po * 0.5)
    );
    return a * basis;
}

vec4 fourier_unit_at(uint base, int i, vec4 x, float mseed) {
    vec4 f, a;
    fourier_load(base, i, f, a);
    if (g_brain_mut != 0.0) fourier_mutate(f, a, i, mseed);
    return fourier_eval(f, a, i, x);
}

// The same unit, for a caller with no seed to hand (the Brain Inspector).
vec4 fourier_unit(uint base, int i, vec4 x) {
    float mseed = (g_brain_mut == 0.0) ? 0.0 : fourier_mut_seed(base, BRAIN_SHAPE.x);
    return fourier_unit_at(base, i, x, mseed);
}

vec4 brain_fourier(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    // Computed once from the UNMUTATED rule, as the original did before its
    // loop - recomputing it per unit is six SSBO reads and a hash each time.
    float mseed = (g_brain_mut == 0.0) ? 0.0 : fourier_mut_seed(base, n);
    for (int i = 0; i < n; i++) {
        result += fourier_unit_at(base, i, x, mseed);
    }
    return result;
}

// Bulk writeback: the whole brain as the particle sees it, mutation included.
//
// The seed and each centre's jitter are computed ONCE. Driving this from
// fourier_param_at() per float instead recomputed fourier_mut_seed() - six SSBO
// reads and a hash - plus the entire centre mutation for all 80 floats of every
// one of 600k particles. Measured click cost 13.0 ms against the original's
// 2.2 ms, which is the hitch you can feel; per centre it is back to 2.3 ms.
void fourier_write(uint base, uint out_base) {
    int n = BRAIN_SHAPE.x;
    float mseed = (g_brain_mut == 0.0) ? 0.0 : fourier_mut_seed(base, n);
    for (int i = 0; i < n; i++) {
        if (i * 8 + 7 >= BRAIN_LEN) break;
        vec4 f, a;
        fourier_load(base, i, f, a);
        if (g_brain_mut != 0.0) fourier_mutate(f, a, i, mseed);
        uint o = out_base + uint(i * 8);
        particle_brains[o + 0u] = f.x;
        particle_brains[o + 1u] = f.y;
        particle_brains[o + 2u] = f.z;
        particle_brains[o + 3u] = f.w;
        particle_brains[o + 4u] = a.x;
        particle_brains[o + 5u] = a.y;
        particle_brains[o + 6u] = a.z;
        particle_brains[o + 7u] = a.w;
    }
}

// The i-th float of this brain as the particle sees it, mutation included.
// The readable definition of what fourier_write() emits in bulk; the mutation
// tests assert against this one, and the Brain Inspector reads single floats.
float fourier_param_at(uint base, int i) {
    int c = i / 8;
    int k = i - c * 8;
    vec4 f, a;
    fourier_load(base, c, f, a);
    if (g_brain_mut != 0.0) {
        fourier_mutate(f, a, c, fourier_mut_seed(base, BRAIN_SHAPE.x));
    }
    return (k < 4) ? f[k] : a[k - 4];
}
