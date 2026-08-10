// Modality dispatch. Prepended AFTER every brain_*.glsl, so all four are
// declared. Shared by entity_update.glsl and brain_preview.frag, which is what
// keeps "adding a modality touches only its own two files" true.
//
// The branch is UNIFORM across the whole dispatch - every particle takes the
// same path - so there is no warp divergence, and switching modality costs one
// uniform write rather than a shader recompile.

// There is always a brain in the buffer, so there is nothing to fall back to.
vec4 eval_brain(uint base, vec4 x) {
    if (BRAIN_MODALITY == 0) return brain_fourier(base, x);
    if (BRAIN_MODALITY == 1) return brain_gabor(base, x);
    if (BRAIN_MODALITY == 2) return brain_lenia(base, x);
    return brain_mlp(base, x);
}

// ONE unit's contribution to the response - one Fourier centre, one Gabor
// filter, one Lenia bump, one MLP hidden unit. The Brain Inspector draws this,
// and it is the SAME function the summing loop uses, so a tile shows what the
// particles actually compute rather than a reimplementation of it.
vec4 eval_brain_unit(uint base, int i, vec4 x) {
    if (BRAIN_MODALITY == 0) return fourier_unit(base, i, x);
    if (BRAIN_MODALITY == 1) return gabor_unit(base, i, x);
    if (BRAIN_MODALITY == 2) return lenia_unit(base, i, x);
    return mlp_unit(base, i, x);
}

// The i-th float of the active brain as the particle sees it. Each modality
// owns this because each decides which of its floats are SCALES and which are
// OFFSETS - a width that offsets can be walked through zero, and a direction
// that offsets per component is rotated rather than resized.
float brain_param_at(uint base, int i) {
    if (BRAIN_MODALITY == 0) return fourier_param_at(base, i);
    if (BRAIN_MODALITY == 1) return gabor_param_at(base, i);
    if (BRAIN_MODALITY == 2) return lenia_param_at(base, i);
    return mlp_param_at(base, i);
}

// The whole active brain, mutation included, so an adopted rule is the one that
// was running.
//
// Fourier takes a bulk path because its mutation is structured: the seed is
// hashed from the rule's own content, so a per-float loop would re-derive it 80
// times per particle. The others mutate each float independently, which makes
// the per-float loop the same work either way.
void brain_write(uint base, uint out_base) {
    if (BRAIN_MODALITY == 0)  { fourier_write(base, out_base);    return; }
    for (int i = 0; i < BRAIN_LEN; i++) {
        particle_brains[out_base + uint(i)] = brain_param_at(base, i);
    }
}
