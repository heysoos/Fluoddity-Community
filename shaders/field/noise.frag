#version 430

in vec2 texcoord;
out vec4 fragColor;

uniform vec2  canvas_resolution;
uniform float time;

uniform float scale;    // 0.5..32 = 4.0   "Scale"
uniform int   octaves;  // 1..8 = 4        "Octaves"
uniform float speed;    // 0..4 = 0.4      "Speed"
uniform float warp;     // 0..2 = 0.0      "Domain Warp"

float hash(vec2 p){
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float vnoise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

float fbm(vec2 p){
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < octaves; i++){
        sum += amp * vnoise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

void main(){
    vec2 p = texcoord * scale + vec2(0.0, time * speed);
    if (warp > 0.0){
        p += warp * vec2(fbm(p + 7.3), fbm(p - 3.1));
    }
    float v = fbm(p);
    fragColor = vec4(v, v, v, 1.0);
}
