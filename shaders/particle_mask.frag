#version 430

// Build the per-region activity mask.
//
// Run once per rendered frame into a small (canvas/4) RGBA16F target, then
// sampled per particle by entity_update.glsl. Small on purpose: the mask is a
// low-frequency field, so a quarter-res target keeps the blur taps cheap and
// gives entity_update a single fetch that already carries its gradient.
//
// The point is a *bias*, not a stencil. mask_floor sets how much activity
// survives outside the mask, so the empty parts of the frame stay alive at a
// controlled level instead of going dead. Feeding a vvvv visual in here should
// enrich it, not trace it -- which is what mask_blur is for: enough blur turns
// hard-edged source imagery into soft regions rather than outlines.
//
// Output channels:
//   r  activity = mix(floor, 1.0, weight)   what the simulation scales by
//   g  weight                               the raw 0..1 mask, for display
//   ba gradient of activity                 containment direction, free here

uniform sampler2D external_tex;
uniform vec2 mask_resolution;
uniform bool mask_connected;    // false when no Spout sender is attached
uniform float mask_floor;
uniform float mask_gamma;
uniform float mask_blur;        // radius in mask texels
uniform float mask_vignette;
uniform float mask_vignette_softness;
uniform int mask_source;        // 0=feed, 1=vignette, 2=max, 3=multiply

in vec2 texcoord;
out vec4 fragColor;

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Sample the feed with the edges held, never wrapped. The mask does not own
// the incoming texture, so it must not depend on how that texture was
// configured -- a repeating source would otherwise let the blur drag content
// in from the opposite side of the frame, which is always wrong here.
//
// The inset is half a source texel, not zero: clamping the coordinate to
// exactly 0 still lets a LINEAR+repeat sampler blend the far edge texel in,
// because the bilinear footprint straddles the boundary. This is what
// CLAMP_TO_EDGE does internally.
float feed_luma(vec2 uv) {
    vec2 half_texel = 0.5 / max(vec2(textureSize(external_tex, 0)), vec2(1.0));
    return luma(texture(external_tex,
                        clamp(uv, half_texel, 1.0 - half_texel)).rgb);
}

// Separable-ish 3x3 tent, sampled at `radius` texels. One tap set is enough
// because the target is already quarter-res and bilinear filtering does the
// rest; wider softening comes from raising the radius, not the tap count.
float blurred_luma(vec2 uv, float radius) {
    if (radius <= 0.0) {
        return feed_luma(uv);
    }
    vec2 t = radius / max(mask_resolution, vec2(1.0));
    float sum = 0.0;
    float wsum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            // Tent weights: 4 centre, 2 edge, 1 corner.
            float w = (3.0 - float(abs(x)) - float(abs(y)));
            w = w * w * 0.25;
            sum += w * feed_luma(uv + vec2(x, y) * t);
            wsum += w;
        }
    }
    return sum / max(wsum, 1e-6);
}

// Radial falloff, usable with no incoming texture at all. Measured in the
// canvas's own aspect so it stays circular on a 16:9 world.
float vignette_at(vec2 uv) {
    vec2 p = (uv - 0.5) * 2.0;
    float aspect = mask_resolution.x / max(mask_resolution.y, 1.0);
    if (aspect > 1.0) p.x *= aspect; else p.y /= max(aspect, 1e-6);
    float r = length(p) / max(1.0, aspect > 1.0 ? aspect : 1.0 / aspect);
    float soft = max(mask_vignette_softness, 0.01);
    float v = 1.0 - smoothstep(1.0 - soft, 1.0, r);
    // mask_vignette scales how much of the falloff is applied at all, so 0
    // leaves a flat field of 1 rather than a dark frame.
    return mix(1.0, v, clamp(mask_vignette, 0.0, 1.0));
}

// The 0..1 mask before the floor is applied.
float weight_at(vec2 uv) {
    float feed = mask_connected ? blurred_luma(uv, mask_blur) : 1.0;
    feed = clamp(feed, 0.0, 1.0);
    float vign = vignette_at(uv);

    float w;
    if (mask_source == 1)      w = vign;
    else if (mask_source == 2) w = max(feed, vign);
    else if (mask_source == 3) w = feed * vign;
    else                       w = feed;

    return pow(clamp(w, 0.0, 1.0), max(mask_gamma, 0.01));
}

void main(void) {
    vec2 uv = texcoord;
    float w = weight_at(uv);
    float floor_level = clamp(mask_floor, 0.0, 1.0);
    float activity = mix(floor_level, 1.0, w);

    // Central-difference gradient of activity, in UV space. Computed here
    // rather than in entity_update so each particle pays for one fetch
    // instead of four.
    vec2 t = 1.0 / max(mask_resolution, vec2(1.0));
    float wl = mix(floor_level, 1.0, weight_at(uv - vec2(t.x, 0.0)));
    float wr = mix(floor_level, 1.0, weight_at(uv + vec2(t.x, 0.0)));
    float wd = mix(floor_level, 1.0, weight_at(uv - vec2(0.0, t.y)));
    float wu = mix(floor_level, 1.0, weight_at(uv + vec2(0.0, t.y)));
    vec2 grad = vec2(wr - wl, wu - wd) * 0.5;

    // Same guard the field shaders use: a NaN reaching a texture the
    // simulation samples poisons particle positions permanently.
    if (isnan(activity) || !(activity <= 1.0e6)) activity = 1.0;
    if (isnan(grad.x + grad.y) || !(length(grad) < 8.0)) grad = vec2(0.0);

    fragColor = vec4(activity, w, grad);
}
