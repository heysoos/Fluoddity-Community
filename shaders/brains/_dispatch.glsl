// Modality dispatch. Prepended AFTER every brain_*.glsl, so all four are
// declared. Shared by entity_update.glsl and brain_preview.frag, which is what
// keeps "adding a modality touches only its own two files" true.
//
// The branch is UNIFORM across the whole dispatch - every particle takes the
// same path - so there is no warp divergence, and switching modality costs one
// uniform write rather than a shader recompile.
vec4 eval_brain(uint base, vec4 x) {
    if (g_brain_fallback) {
        return fourier_noise(generate_random_centers(g_brain_seed), x);
    }
    if (BRAIN_MODALITY == 0) return brain_fourier(base, x);
    if (BRAIN_MODALITY == 1) return brain_gabor(base, x);
    if (BRAIN_MODALITY == 2) return brain_lenia(base, x);
    return brain_mlp(base, x);
}
