#version 430

in vec2 texcoord;
out vec4 fragColor;

uniform vec2 canvas_resolution;

uniform int   shape;   // 0..1 = 0        "Radial"
uniform float angle;   // -3.15..3.15 = 0 "Angle"
uniform vec2  centre;  // 0..1 = .5,.5    "Centre"
uniform float falloff; // 0.1..4 = 1.0    "Falloff"

void main(){
    float v;
    if (shape == 0){
        v = dot(texcoord - centre, vec2(sin(angle), cos(angle))) + 0.5;
    } else {
        v = 1.0 - clamp(length(texcoord - centre) * 2.0, 0.0, 1.0);
    }
    v = pow(clamp(v, 0.0, 1.0), falloff);
    fragColor = vec4(v, v, v, 1.0);
}
