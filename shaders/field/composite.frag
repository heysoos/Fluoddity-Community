#version 430

// Maps a source texture into a destination's channel pair. The one composite
// shader in the system: blending and channel masking are GL state, not
// branches here.
//
// Both channel pairs are written with the same value; the colour mask decides
// which pair lands.

in vec2 texcoord;
out vec4 fragColor;

uniform sampler2D src;
// Index into state/field_stack.MAPPINGS: 0 rg_direct, 1 polar, 2 gradient,
// 3 curl, 4 luminance, 5 rg_signed, 6 edge, 7 edge_flow.
uniform int   mapping;
uniform float strength;
uniform float sign_mul;   // +1 attract, -1 repel
uniform float blur_lod;   // mip level of the pre-filter
uniform vec2  texel;
uniform bool  scalar_out; // destination is scalar: contribute the magnitude
uniform int   src_channels; // 0 = read .xy / .rgb, 1 = read .zw
// Source aspect over destination aspect. A source shaped differently from the
// bus is scaled to COVER and centre-cropped, never squeezed to fit.
uniform float aspect_ratio;
// Mirrors the picture left to right, before anything reads it.
uniform bool  flip_x;

vec2 cover(vec2 uv){
    if (flip_x) uv.x = 1.0 - uv.x;
    if (aspect_ratio == 1.0) return uv;
    vec2 scale = (aspect_ratio > 1.0) ? vec2(1.0 / aspect_ratio, 1.0)
                                      : vec2(1.0, aspect_ratio);
    return (uv - 0.5) * scale + 0.5;
}

vec4 fetch(vec2 uv){
    vec4 c = textureLod(src, cover(uv), blur_lod);
    return (src_channels == 1) ? vec4(c.zw, 0.0, 1.0) : c;
}

float scalar_at(vec2 uv){
    return dot(fetch(uv).rgb, vec3(0.2126, 0.7152, 0.0722));
}

float lum_at(float dx, float dy){
    return scalar_at(texcoord + vec2(dx, dy) * texel);
}

// A step from black to white scores 4 under these weights, so the divide puts
// the strongest edge a source can hold at 1.
const float SOBEL_FULL = 4.0;

// Sobel, direction and strength kept apart: the direction is normalised, so a
// soft ramp steers as firmly as a hard edge, and the strength is the edge
// itself rather than the slope's raw height.
void sobel(out vec2 dir, out float mag){
    float tl = lum_at(-1.0,  1.0), tc = lum_at(0.0,  1.0), tr = lum_at(1.0,  1.0);
    float ml = lum_at(-1.0,  0.0),                         mr = lum_at(1.0,  0.0);
    float bl = lum_at(-1.0, -1.0), bc = lum_at(0.0, -1.0), br = lum_at(1.0, -1.0);
    vec2 g = vec2((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl),
                  (tl + 2.0 * tc + tr) - (bl + 2.0 * bc + br));
    float len = length(g);
    dir = (len > 1e-5) ? g / len : vec2(0.0);
    mag = clamp(len / SOBEL_FULL, 0.0, 1.0);
}

void main(){
    vec2 v;
    if (mapping == 0) {
        v = fetch(texcoord).rg;
    } else if (mapping == 1) {
        // Hue as angle, value as magnitude. The historical PNG field encoding.
        vec4 c = fetch(texcoord);
        float a = c.r * 6.28318530718;
        v = vec2(cos(a), sin(a)) * c.b;
    } else if (mapping == 4) {
        v = vec2(scalar_at(texcoord), 0.0);
    } else if (mapping == 5) {
        // A camera's channels are all positive, so read straight they carry a
        // constant push into one corner. Mid grey is the rest position.
        v = (fetch(texcoord).rg - 0.5) * 2.0;
    } else if (mapping == 6 || mapping == 7) {
        vec2 dir;
        float mag;
        sobel(dir, mag);
        v = ((mapping == 6) ? dir : vec2(-dir.y, dir.x)) * mag * sign_mul;
    } else {
        float l = scalar_at(texcoord - vec2(texel.x, 0.0));
        float r = scalar_at(texcoord + vec2(texel.x, 0.0));
        float d = scalar_at(texcoord - vec2(0.0, texel.y));
        float u = scalar_at(texcoord + vec2(0.0, texel.y));
        vec2 g = 0.5 * vec2(r - l, u - d);
        // curl is the gradient turned a quarter turn, so it is divergence-free
        // and particles circulate along contours instead of piling into peaks.
        v = (mapping == 2) ? g : vec2(-g.y, g.x);
        v *= sign_mul;
    }

    // A mirrored picture holds mirrored vectors. The derivative mappings get
    // that from the flipped sample offsets; the direct reads do not.
    if (flip_x && (mapping == 0 || mapping == 1 || mapping == 5)) v.x = -v.x;

    v *= strength;
    if (scalar_out) v = vec2(length(v), 0.0);
    // A source may produce a non-finite value; the particles must never see one.
    if (any(isnan(v)) || any(isinf(v))) v = vec2(0.0);

    fragColor = vec4(v, v);
}
