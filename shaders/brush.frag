#version 430

in vec2 uv;
in vec4 pos_vel;
in vec4 view_col;
in vec2 frag_world;
flat in vec2 tile_lo;
flat in vec2 tile_hi;
out vec4 brush_out;

uniform int frame_count;
uniform int TOURNAMENT_MODE;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float gaussian(vec2 pos, float sigma) {
    float sigma2 = sigma * sigma;
    float norm = 1.0 / (2.0 * 3.14159265359 * sigma2);
    float exponent = -(dot(pos, pos)) / (2.0 * sigma2);
    return norm * exp(exponent);
}

void main() {
    // Clear to black on frame 0 to prevent garbage data
    if (frame_count == 0) {
        brush_out = vec4(0, 0, 0, 1);
        return;
    }

    // Tournament: a particle sprite has ~1px extent, so without this clip a
    // particle near a seam deposits trail into the neighbouring tile, which that
    // tile's sensors then read. Keeps tiles genuinely independent.
    if (TOURNAMENT_MODE == 1) {
        if (any(lessThan(frag_world, tile_lo)) || any(greaterThan(frag_world, tile_hi))) {
            discard;
        }
    }

    float kernel_func = gaussian(uv - .5, .163);
    if (length(uv - .5) > .5 || view_col.w == 0) { discard; }
    vec2 vel = pos_vel.zw;
    // The target is RG32F, so only xy are stored. w still matters: the blend
    // is SRC_ALPHA, ONE, so it is this deposit's weight.
    brush_out = vec4(vel, 0, 1) * kernel_func;
}
