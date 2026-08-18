#version 150

// Drive the force/strafe field from a Spout texture (e.g. a vvvv/HLSL shader).
//
// Selected like any other override shader: enable Advanced Drawing, tick
// "Shader Driven Field", and pick spout.frag from the shader dropdown. It only
// does anything when the app was started with --spout-in <name>; with no sender
// connected the field is left at zero, which is a no-op for the simulation.
//
// Three interpretations of the incoming image, chosen with Spout Field Mode:
//   0  channels  - r,g are force xy and b,a are strafe zw, each mapped x*2-1.
//                  Exact and direct; best when vvvv is rendering a purpose-made
//                  vector field (a flow map, a normal map, a signed noise).
//   1  gradient  - force follows the luminance gradient (Sobel), so particles
//                  climb toward bright regions. Reads naturally from arbitrary
//                  imagery, which is what you usually have.
//   2  gradient perpendicular - as above, rotated 90 degrees, so particles
//                  circulate along contours instead of climbing them. Gives
//                  swirling motion around bright shapes.

uniform sampler2D external_tex;
uniform vec2 canvas_resolution;
uniform int spout_field_mode;
uniform float spout_field_scale;
uniform float spout_strafe_scale;
uniform bool spout_connected;

in vec2 texcoord;
out vec4 fragColor;

float luma(vec2 uv) {
    vec3 c = texture(external_tex, uv).rgb;
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main(void) {
    if (!spout_connected) {
        fragColor = vec4(0.0);
        return;
    }

    vec2 uv = texcoord;
    vec2 force;
    vec2 strafe;

    if (spout_field_mode == 0) {
        vec4 c = texture(external_tex, uv);
        force  = (c.rg * 2.0 - 1.0);
        strafe = (c.ba * 2.0 - 1.0);
    } else {
        // Sobel on luminance. Step by one canvas texel so the gradient scale is
        // independent of the sender's resolution.
        vec2 t = 1.0 / max(canvas_resolution, vec2(1.0));
        float tl = luma(uv + vec2(-t.x,  t.y));
        float l  = luma(uv + vec2(-t.x,  0.0));
        float bl = luma(uv + vec2(-t.x, -t.y));
        float tr = luma(uv + vec2( t.x,  t.y));
        float r  = luma(uv + vec2( t.x,  0.0));
        float br = luma(uv + vec2( t.x, -t.y));
        float tt = luma(uv + vec2( 0.0,  t.y));
        float bb = luma(uv + vec2( 0.0, -t.y));

        vec2 g = vec2((tr + 2.0 * r + br) - (tl + 2.0 * l + bl),
                      (tl + 2.0 * tt + tr) - (bl + 2.0 * bb + br));

        if (spout_field_mode == 2) {
            g = vec2(-g.y, g.x);   // rotate 90 degrees: circulate, don't climb
        }
        force  = g;
        strafe = g;
    }

    force  *= spout_field_scale;
    strafe *= spout_strafe_scale;

    // Same guard march.frag uses. A NaN reaching the field texture poisons
    // particle positions permanently, and the field persists across frames.
    if (isnan(force.x + force.y) || !(length(force) < 8.0)) { force = vec2(0.0); }
    if (isnan(strafe.x + strafe.y) || !(length(strafe) < 8.0)) { strafe = vec2(0.0); }

    fragColor = vec4(force, strafe);
}
