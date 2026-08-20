#version 430

// Adds an injected field into the sim's own trail canvas, which is what the
// brains sense. Blended ONE,ONE into the canvas framebuffer, so this is a
// deposit alongside the particles' own - not a force on them.

in vec2 texcoord;
out vec4 fragColor;

uniform sampler2D field;
// The trail is a moving average whose decay is already on the clock, so the
// deposit has to be too: a slower sim must lay down less per step.
uniform float amount;

// Strength is a deposit RATE, and the trail keeps accumulating it, so full
// strength has to land near the level a busy sim already runs at rather than
// at the mapping's own full scale. See the field injection caveats.
const float TRAIL_GAIN = 0.1;

void main(){
    vec2 v = texture(field, texcoord).xy * amount * TRAIL_GAIN;
    if (any(isnan(v)) || any(isinf(v))) v = vec2(0.0);
    fragColor = vec4(v, 0.0, 1.0);
}
