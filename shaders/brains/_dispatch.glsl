// Modality dispatch. Prepended AFTER every brain_*.glsl, so all four are
// declared. Shared by entity_update.glsl and brain_preview.frag, which is what
// keeps "adding a modality touches only its own two files" true.
//
// The branch is UNIFORM across the whole dispatch - every particle takes the
// same path - so there is no warp divergence, and switching modality costs one
// uniform write rather than a shader recompile.

// The generated rule a particle runs when no brain has been uploaded, WITH the
// cohort mutation applied - the original mutated the generated rule too, and
// skipping that made Mutation Scale a no-op on the startup population.
//
// The seed is hashed from the centres just GENERATED, not from the blank buffer
// that triggered the fallback. Reusing the fallback's own generation seed
// instead correlates the jitter with the rule it is jittering, which amplifies
// the spread rather than exploring around it.
FourierCenter[10] fallback_centers() {
    FourierCenter[10] fc = generate_random_centers(g_brain_seed);
    if (g_brain_mut == 0.0) return fc;
    float mseed = hash(fc[4].frequency.xy + fc[7].amplitude.yx + fc[1].frequency.zw)
                + g_brain_cohort;
    for (int c = 0; c < 10; c++) {
        fourier_mutate(fc[c].frequency, fc[c].amplitude, c, mseed);
    }
    return fc;
}

vec4 eval_brain(uint base, vec4 x) {
    if (g_brain_fallback) {
        return fourier_noise(fallback_centers(), x);
    }
    if (BRAIN_MODALITY == 0) return brain_fourier(base, x);
    if (BRAIN_MODALITY == 1) return brain_gabor(base, x);
    if (BRAIN_MODALITY == 2) return brain_lenia(base, x);
    return brain_mlp(base, x);
}

// The i-th float of the active brain as the particle sees it, mutation
// included. The writeback emits this, so an adopted rule is the one that was
// running. A modality that mutates structurally (fourier scales a frequency
// VECTOR by one scalar) cannot be read back with the generic per-float helper.
float brain_param_at(uint base, int i) {
    if (BRAIN_MODALITY == 0) return fourier_param_at(base, i);
    return brain_add(base, i);
}
