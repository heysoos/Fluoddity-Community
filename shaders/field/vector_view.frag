#version 430

// Renders a force/strafe field so a person can read it: direction as hue,
// magnitude as brightness. Raw RG shows a red-green wash in which two opposite
// vectors look much the same.

in vec2 texcoord;
out vec4 fragColor;

uniform sampler2D src;
uniform int   pair;   // 0 = read .xy (force), 1 = read .zw (strafe)
uniform float gain;

vec3 hsv(vec3 c){
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){
    vec4 c = texture(src, texcoord);
    vec2 v = (pair == 1) ? c.zw : c.xy;
    float m = length(v) * gain;
    if (m <= 0.0){
        // A dim grid, so an empty field reads as empty rather than as broken.
        vec2 g = step(0.97, fract(texcoord * 8.0));
        float line = max(g.x, g.y) * 0.10;
        fragColor = vec4(vec3(line), 1.0);
        return;
    }
    float angle = atan(v.y, v.x) / 6.28318530718 + 0.5;
    fragColor = vec4(hsv(vec3(angle, 1.0, clamp(m, 0.0, 1.0))), 1.0);
}
