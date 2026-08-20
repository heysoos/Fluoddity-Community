// Per-cohort audio modulation. Prepended to entity_update.glsl.
//
// The host reduces every masked mapping into one gain and one offset per
// cohort, because the whole modulation chain is affine in the base value.
// Applied AFTER calculate_setting, never inside it: that function must stay
// character-identical across entity_update.glsl, canvas.frag and sim.py, and
// canvas.frag has no cohort to index with.
//
// Row order is mirrored in services/cohort_audio.COHORT_AUDIO_PARAMS and
// compared by a test. A row here naming a different parameter than the host
// thinks reads the wrong modulation with nothing raising.
#define CA_SENSOR_GAIN 0
#define CA_SENSOR_ANGLE 1
#define CA_SENSOR_DISTANCE 2
#define CA_MUTATION_SCALE 3
#define CA_GLOBAL_FORCE_MULT 4
#define CA_DRAG 5
#define CA_AXIAL_FORCE 6
#define CA_LATERAL_FORCE 7
#define CA_STRAFE_POWER 8
#define CA_HAZARD_RATE 9

// One entry per cohort, then one more holding that row's (lo, hi) clamp.
#define CA_SLOTS 144
#define CA_STRIDE 145

layout(std430, binding = 5) buffer CohortAudioBuffer {
    vec2 cohort_audio_ga[];   // x = gain, y = offset; at CA_SLOTS, (lo, hi)
};

uniform bool COHORT_AUDIO_ACTIVE;

float cohort_audio(float v, int row, float cohort) {
    if (!COHORT_AUDIO_ACTIVE) return v;
    int c = clamp(int(floor(cohort)), 0, CA_SLOTS - 1);
    vec2 ga = cohort_audio_ga[row * CA_STRIDE + c];
    vec2 lohi = cohort_audio_ga[row * CA_STRIDE + CA_SLOTS];
    return clamp(v * ga.x + ga.y, lohi.x, lohi.y);
}
