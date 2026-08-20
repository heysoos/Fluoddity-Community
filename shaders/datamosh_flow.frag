#version 430

// Condense the simulation's two velocity maps into the one field the datamosh
// pass reads, already converted to canvas-uv displacement.
//
// The reason this is its own pass rather than two taps inside datamosh.frag is
// the mip chain built on top of it. That chain does two jobs at once:
//
//   * its top level is the frame's mean displacement magnitude, which is what
//     lets the response curve be calibrated against the picture instead of
//     against a number that moves with world size and particle count;
//   * its lower levels are a ready-made blur of the field, so "how big are the
//     strokes" becomes a level-of-detail choice rather than a tap count.
//
// Output channels:
//   xy  displacement in canvas uv per frame
//   z   its magnitude, so the mip chain averages a magnitude rather than a
//       vector -- averaged vectors cancel, and a mean of zero would be no use
//       as a reference
//   w   unused

uniform sampler2D flow_can_tex;    // persistent trail map (velocity)
uniform sampler2D flow_brush_tex;  // this frame's splat (velocity)
uniform vec2  canvas_edges;        // (sqrt(ca), 1/sqrt(ca)) -- entity-space extent
uniform float mosh_flow_mix;       // 0 = trails, 1 = instantaneous splat

in vec2 texcoord;
out vec4 fragColor;

void main() {
    vec2 trails = texture(flow_can_tex, texcoord).xy;
    vec2 splat  = texture(flow_brush_tex, texcoord).xy;
    vec2 vel = mix(trails, splat, mosh_flow_mix);

    // Entity-space velocity -> canvas uv. The canvas spans +-canvas_edges in
    // entity space (see brush.vert), so half that per uv unit.
    vec2 duv = vel * (0.5 / canvas_edges);

    fragColor = vec4(duv, length(duv), 1.0);
}
