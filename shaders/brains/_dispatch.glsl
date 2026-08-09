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

// The generated rule, written out as the particle is running it. brain_params
// still holds the blank buffer that triggered the fallback; writing THAT made
// click-to-adopt copy zeros, which re-blanked slot 0 on apply and flipped every
// cohort onto its own random rule - most sluggish, a few lively.
void fourier_write_fallback(uint out_base) {
    FourierCenter[10] fc = fallback_centers();
    for (int c = 0; c < 10; c++) {
        if (c * 8 + 7 >= BRAIN_LEN) break;
        uint o = out_base + uint(c * 8);
        particle_brains[o + 0u] = fc[c].frequency.x;
        particle_brains[o + 1u] = fc[c].frequency.y;
        particle_brains[o + 2u] = fc[c].frequency.z;
        particle_brains[o + 3u] = fc[c].frequency.w;
        particle_brains[o + 4u] = fc[c].amplitude.x;
        particle_brains[o + 5u] = fc[c].amplitude.y;
        particle_brains[o + 6u] = fc[c].amplitude.z;
        particle_brains[o + 7u] = fc[c].amplitude.w;
    }
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
    if (g_brain_fallback)     { fourier_write_fallback(out_base); return; }
    if (BRAIN_MODALITY == 0)  { fourier_write(base, out_base);    return; }
    for (int i = 0; i < BRAIN_LEN; i++) {
        particle_brains[out_base + uint(i)] = brain_param_at(base, i);
    }
}
