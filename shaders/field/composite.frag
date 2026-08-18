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
uniform int   mapping;    // 0 rg_direct, 1 polar, 2 gradient, 3 curl, 4 luminance
uniform float strength;
uniform float sign_mul;   // +1 attract, -1 repel
uniform float blur_lod;   // mip level of the pre-filter
uniform vec2  texel;
uniform bool  scalar_out; // destination is scalar: contribute the magnitude
uniform int   src_channels; // 0 = read .xy / .rgb, 1 = read .zw

vec4 fetch(vec2 uv){
    vec4 c = textureLod(src, uv, blur_lod);
    return (src_channels == 1) ? vec4(c.zw, 0.0, 1.0) : c;
}

float scalar_at(vec2 uv){
    return dot(fetch(uv).rgb, vec3(0.2126, 0.7152, 0.0722));
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

    v *= strength;
    if (scalar_out) v = vec2(length(v), 0.0);
    // A source may produce a non-finite value; the particles must never see one.
    if (any(isnan(v)) || any(isinf(v))) v = vec2(0.0);

    fragColor = vec4(v, v);
}
