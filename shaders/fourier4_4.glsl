// Shared hashing. Prepended FIRST, ahead of the brain header, because
// brain_jit() and every modality's mutation are built on hash()/hash4().
//
// The name is historical: this file used to hold the Fourier feature network
// too - struct FourierCenter, fourier_noise(), generate_random_centers() and
// two wrappers, 65 of its 91 lines. Those were reached only by the GPU-side
// fallback brain, which is gone; the brain itself lives in shaders/brains/.

// PCG hash - bit-exact across all platforms
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float hash(vec2 co){
    uvec2 u = uvec2(floatBitsToUint(co.x), floatBitsToUint(co.y));
    uint h = pcg_hash(u.x ^ pcg_hash(u.y));
    return float(h) / float(0xffffffffu);
}

vec4 hash4(vec2 co){
    return vec4(
        hash(co),
        hash(co*-1+5),
        hash(co.yx-100),
        hash(co.yx*-1 + 25)
    );
}
