// Shared brain declarations. Prepended BEFORE every brain_*.glsl.
//
// One flat float array holds every brain. std430 gives a float array a 4-byte
// stride with no padding, so packing is a straight memcpy and there is no
// struct alignment to get wrong. Manual mode uses slot 0; tournament mode uses
// the tile index; multi-load uses the config index.
#define MAX_BRAIN_FLOATS 512

layout(std430, binding = 4) buffer BrainBuffer {
    float brain_params[];
};

uniform int   BRAIN_MODALITY;   // which brain_* function to call
uniform int   BRAIN_LEN;        // active floats per brain; bounds every loop
uniform ivec4 BRAIN_SHAPE;      // structural ints (n_centers, hidden width, ...)

// Per-particle mutation, as invocation-local globals rather than extra
// function parameters, so every brain_*() keeps the same signature and the
// Brain Inspector can call it unchanged (both stay 0 there).
//
// Applied on READ. The obvious alternative - copy the brain into a local
// float[MAX_BRAIN_FLOATS], mutate, then evaluate - is 2 KB per invocation and
// spills to local memory for every particle every frame.
float g_brain_mut = 0.0;
float g_brain_cohort = 0.0;

// Safety net, preserved from the old all-zero Rule check: when no brain has
// been uploaded - manual mode with no rule loaded, which is the startup state,
// or a buffer realloc not yet followed by an upload - fall back to a per-cohort
// random Fourier rule so the tile still runs instead of freezing on silence.
// Set per invocation in main(); always false in the Brain Inspector.
bool  g_brain_fallback = false;
float g_brain_seed = 0.0;

// Per-cohort mutation jitter for one float index, in [-amount, +amount].
// Deterministic in (cohort, index), so a cohort's variant is stable frame to
// frame.
float brain_jit(int i) {
    if (g_brain_mut == 0.0) return 0.0;
    return g_brain_mut * (2.0 * hash(vec2(g_brain_cohort, float(i))) - 1.0);
}

// Two mutation modes, because a modality knows which of its floats are SCALES
// and which are OFFSETS, and the distinction matters: jittering a near-zero
// frequency additively drags a smooth centre into a chaotic one, which the
// original mutate_rule() avoided by scaling frequencies and offsetting
// amplitudes. Each modality picks per field.
float brain_add(uint base, int i) {   // offsets: amplitudes, biases, centres
    return brain_params[base + uint(i)] + brain_jit(i);
}

float brain_mul(uint base, int i) {   // scales: frequencies
    // 0.25 reproduces the original `freq *= 1 + amount*0.5*(hash-0.5)`.
    return brain_params[base + uint(i)] * (1.0 + 0.25 * brain_jit(i));
}

vec4 brain_add4(uint base, int i) {
    return vec4(brain_add(base, i), brain_add(base, i + 1),
                brain_add(base, i + 2), brain_add(base, i + 3));
}

vec4 brain_mul4(uint base, int i) {
    return vec4(brain_mul(base, i), brain_mul(base, i + 1),
                brain_mul(base, i + 2), brain_mul(base, i + 3));
}

// Kept so the generic writeback can dump a brain as the particle sees it.
float brain_at(uint base, int i) { return brain_add(base, i); }
