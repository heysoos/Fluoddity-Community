#version 430

// Value-noise fBm. Time is the THIRD axis, so Speed churns the field in place
// rather than sliding it across the canvas.

in vec2 texcoord;
out vec4 fragColor;

uniform vec2  canvas_resolution;
uniform float time;

uniform float scale;    // 0.5..32 = 4.0   "Scale"
uniform int   octaves;  // 1..8 = 4        "Octaves"
uniform float speed;    // 0..4 = 0.4      "Speed"
uniform float warp;     // 0..2 = 0.0      "Domain Warp"

float hash(vec3 p){
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

float vnoise(vec3 p){
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash(i + vec3(0, 0, 0));
    float n100 = hash(i + vec3(1, 0, 0));
    float n010 = hash(i + vec3(0, 1, 0));
    float n110 = hash(i + vec3(1, 1, 0));
    float n001 = hash(i + vec3(0, 0, 1));
    float n101 = hash(i + vec3(1, 0, 1));
    float n011 = hash(i + vec3(0, 1, 1));
    float n111 = hash(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

float fbm(vec3 p){
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < octaves; i++){
        sum += amp * vnoise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

void main(){
    vec3 p = vec3(texcoord * scale, time * speed);
    if (warp > 0.0){
        p.xy += warp * vec2(fbm(p + 7.3), fbm(p - 3.1));
    }
    float v = fbm(p);
    fragColor = vec4(v, v, v, 1.0);
}
