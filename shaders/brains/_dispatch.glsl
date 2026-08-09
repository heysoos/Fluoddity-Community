// Modality dispatch. Prepended AFTER every brain_*.glsl, so all four are
// declared. Shared by entity_update.glsl and brain_preview.frag, which is what
// keeps "adding a modality touches only its own two files" true.
//
// The branch is UNIFORM across the whole dispatch - every particle takes the
// same path - so there is no warp divergence, and switching modality costs one
// uniform write rather than a shader recompile.
// The generated rule a particle runs when no brain has been uploaded, WITH its
// cohort mutation applied. The old code mutated the generated rule too, and
// skipping that made Mutation Scale a no-op on the startup population.
// Indexing matches the flat layout so the writeback and the evaluation agree.
FourierCenter[10] fallback_centers() {
    FourierCenter[10] fc = generate_random_centers(g_brain_seed);
    if (g_brain_mut == 0.0) return fc;
    for (int c = 0; c < 10; c++) {
        for (int k = 0; k < 4; k++) {
            fc[c].frequency[k] *= 1.0 + 0.25 * brain_jit(c * 8 + k);
            fc[c].amplitude[k] += brain_jit(c * 8 + 4 + k);
        }
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
