// Shared brain declarations. Prepended BEFORE every brain_*.glsl.
//
// One flat float array holds every brain. std430 gives a float array a 4-byte
// stride with no padding, so packing is a straight memcpy and there is no
// struct alignment to get wrong. Manual mode uses slot 0; tournament mode uses
// the tile index; multi-load uses the config index.
#define MAX_BRAIN_FLOATS 1024

// Where the per-cohort generated brains start in the flat buffer, in SLOTS.
// The slots below this belong to multi-load configs and tournament tiles.
#define COHORT_BRAIN_SLOT0 64
#define MAX_COHORT_BRAINS 144

layout(std430, binding = 4) buffer BrainBuffer {
    float brain_params[];
};

// The adopted brain, written only when WRITE_RULES is set (click-to-adopt).
// ONE brain of BRAIN_LEN floats, at offset 0: the writeback is scoped to the
// single particle being read back, so a row per particle would be bytes nobody
// looks at.
//
// Declared here rather than in entity_update.glsl so brain_write() can live
// beside the modalities - it is prepended ahead of the dispatch, whereas
// entity_update's own declarations come after it.
layout(std430, binding = 2) buffer BrainReadbackBuffer {
    float particle_brains[];
};

uniform int   BRAIN_MODALITY;   // which brain_* function to call
uniform int   BRAIN_LEN;        // active floats per brain; bounds every loop
uniform ivec4 BRAIN_SHAPE;      // structural ints (n_centers, hidden width, ...)

// Per-particle mutation, as invocation-local globals rather than extra
// function parameters, so every brain_*() keeps the same signature and the
// Brain Inspector can call it unchanged (both stay 0 there).
//
// Applied on READ. The obvious alternative - copy the brain into a local
// float[MAX_BRAIN_FLOATS], mutate, then evaluate - is kilobytes per invocation
// and spills to local memory for every particle every frame.
// Set by the host when no rule is loaded: every cohort then reads its OWN
// generated brain out of the cohort slots instead of sharing slot 0.
uniform int BRAIN_PER_COHORT;

float g_brain_mut = 0.0;
float g_brain_cohort = 0.0;

// Per-cohort mutation jitter for one float index, in [-amount, +amount].
// Deterministic in (cohort, index), so a cohort's variant is stable frame to
// frame.
float brain_jit(int i) {
    if (g_brain_mut == 0.0) return 0.0;
    return g_brain_mut * (2.0 * hash(vec2(g_brain_cohort, float(i))) - 1.0);
}

// Generic per-float mutation, for modalities whose parameters mutate
// INDEPENDENTLY of one another. Two modes, because a modality knows which of
// its floats are SCALES and which are OFFSETS: jittering a near-zero frequency
// additively drags a smooth centre into a chaotic one, whereas scaling leaves
// it near zero.
//
// A modality whose mutation is STRUCTURED must not use these - it implements
// its own and exposes a <modality>_param_at() for the writeback. Fourier does:
// one scalar scales a whole frequency VECTOR, so the four components are not
// independent, and mutating them separately is a measurably different function.
float brain_add(uint base, int i) {   // offsets: amplitudes, biases, centres
    return brain_params[base + uint(i)] + brain_jit(i);
}

float brain_mul(uint base, int i) {   // scales: widths, frequencies
    return brain_params[base + uint(i)] * (1.0 + brain_jit(i));
}

vec4 brain_add4(uint base, int i) {
    return vec4(brain_add(base, i), brain_add(base, i + 1),
                brain_add(base, i + 2), brain_add(base, i + 3));
}

vec4 brain_mul4(uint base, int i) {
    return vec4(brain_mul(base, i), brain_mul(base, i + 1),
                brain_mul(base, i + 2), brain_mul(base, i + 3));
}
