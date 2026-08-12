#version 430
// Draws one brain unit's response over a 2D slice of the 4D sensor space.
//
// The brain sources are PREPENDED into this file by services/brain_preview.py,
// exactly as sim.py prepends them into entity_update.glsl, and it binds the
// same parameter buffer. So a tile shows what the particles actually compute
// rather than a reimplementation of it - which is the whole reason the brain
// functions are pure.
//
// g_brain_mut stays 0 here: the Inspector shows the STORED brain, not one
// cohort's mutated variant of it.

in vec2 texcoord;
out vec4 frag;

uniform int   PREVIEW_UNIT;     // -1 = the whole brain, >=0 = that unit alone
uniform int   PREVIEW_CHANNEL;  // 0..3 = one output, 4 = |force|, 5 = |strafe|,
                                // 6 = project onto PREVIEW_OUT,
                                // 7 = project onto PREVIEW_RGB, one per colour
#define PREVIEW_CHANNEL_RGB 7

// A random unit direction in the 4D OUTPUT space, for PREVIEW_CHANNEL 6. Same
// reasoning as the input plane: the four outputs are (force.xy, strafe.xy) in
// the particle's own frame, and a unit that acts diagonally across them shows
// in none of the four single-channel views.
uniform vec4  PREVIEW_OUT;

// Three ORTHONORMAL output directions, into R, G and B. One picture instead of
// four, so a unit acting across the outputs reads as a HUE rather than having
// to be hunted for one channel at a time.
uniform vec4  PREVIEW_RGB[3];
uniform float PREVIEW_RANGE;    // half-extent of the swept plane
uniform float PREVIEW_GAIN;     // display scale only; does not touch the brain

// The plane to sweep, as two directions in the 4D sensor space. An axis-aligned
// slice is just the case where these are two of the standard basis vectors, so
// there is no separate code path for it - x = U*p.x + V*p.y reproduces the old
// "set two components, leave the rest at zero" exactly.
//
// They are supplied ORTHONORMAL. A random plane spanned by two directions that
// are merely random is skewed and unequally scaled, so PREVIEW_RANGE would mean
// a different distance along each one and the picture would be sheared.
uniform vec4  PREVIEW_U;
uniform vec4  PREVIEW_V;

// Blue for negative, near-black at zero, orange for positive. Diverging rather
// than a single ramp because the SIGN is the whole point - a Lenia bump is
// positive on its band and negative everywhere else, and a ramp would hide it.
vec3 diverging(float v) {
    float m = clamp(abs(v), 0.0, 1.0);
    vec3 cold = vec3(0.25, 0.55, 1.00);
    vec3 warm = vec3(1.00, 0.62, 0.20);
    vec3 c = (v < 0.0) ? cold : warm;
    // sqrt lifts the low end so weak structure is visible without blowing out
    // the strong parts.
    return c * sqrt(m);
}

void main() {
    vec2 p = (texcoord * 2.0 - 1.0) * PREVIEW_RANGE;
    vec4 x = PREVIEW_U * p.x + PREVIEW_V * p.y;

    vec4 r = (PREVIEW_UNIT < 0) ? eval_brain(0u, x)
                                : eval_brain_unit(0u, PREVIEW_UNIT, x);

    vec3 rgb;
    if (PREVIEW_CHANNEL == PREVIEW_CHANNEL_RGB) {
        vec3 v3 = vec3(dot(r, PREVIEW_RGB[0]),
                       dot(r, PREVIEW_RGB[1]),
                       dot(r, PREVIEW_RGB[2])) * PREVIEW_GAIN;
        // Mid-grey at zero, so the SIGN of each direction still reads; the same
        // sqrt lift diverging() uses, so Contrast means the same thing here.
        rgb = 0.5 + 0.5 * sign(v3) * sqrt(clamp(abs(v3), 0.0, 1.0));
    } else {
        float v;
        if (PREVIEW_CHANNEL == 4)      v = length(r.xy);   // force magnitude
        else if (PREVIEW_CHANNEL == 5) v = length(r.zw);   // strafe magnitude
        else if (PREVIEW_CHANNEL == 6) v = dot(r, PREVIEW_OUT);
        else                           v = r[PREVIEW_CHANNEL];
        rgb = diverging(v * PREVIEW_GAIN);
    }

    // Axes through the origin, so "which way is zero" is readable at a glance.
    vec2 d = abs(p) / PREVIEW_RANGE;
    if (min(d.x, d.y) < 0.004) rgb = mix(rgb, vec3(0.35), 0.5);

    frag = vec4(rgb, 1.0);
}
