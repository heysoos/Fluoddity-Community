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

// Read one float of a brain with this particle's cohort mutation applied.
// Deterministic in (cohort, index), so a cohort's variant is stable frame to
// frame. Additive on every float: the old mutate_rule() split additive
// (amplitude) from multiplicative (frequency), which cannot be expressed
// without knowing the layout.
float brain_at(uint base, int i) {
    float v = brain_params[base + uint(i)];
    if (g_brain_mut == 0.0) return v;
    return v + g_brain_mut * (hash(vec2(g_brain_cohort, float(i))) - 0.5);
}

vec4 brain_at4(uint base, int i) {
    return vec4(brain_at(base, i), brain_at(base, i + 1),
                brain_at(base, i + 2), brain_at(base, i + 3));
}
