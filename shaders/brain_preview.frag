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
uniform int   PREVIEW_CHANNEL;  // 0..3 = one output, 4 = |force|, 5 = |strafe|
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

    float v;
    if (PREVIEW_CHANNEL == 4)      v = length(r.xy);   // force magnitude
    else if (PREVIEW_CHANNEL == 5) v = length(r.zw);   // strafe magnitude
    else                           v = r[PREVIEW_CHANNEL];

    vec3 rgb = diverging(v * PREVIEW_GAIN);

    // Axes through the origin, so "which way is zero" is readable at a glance.
    vec2 d = abs(p) / PREVIEW_RANGE;
    if (min(d.x, d.y) < 0.004) rgb = mix(rgb, vec3(0.35), 0.5);

    frag = vec4(rgb, 1.0);
}
