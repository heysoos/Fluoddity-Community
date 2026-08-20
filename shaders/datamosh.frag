#version 430

// Datamosh: shift the incoming texture's pixels along the particle flow.
//
// The simulation already maintains the motion field this needs. brush.frag
// splats `velocity * gaussian` additively into brush_tex, and the canvas is
// that same quantity accumulated under Trail Persistence / Trail Diffusion.
// Both are RG32F velocity vectors at canvas resolution, so no extra pass over
// the particles is needed -- this reads what the sim writes anyway.
// datamosh_flow.frag condenses the two into one mipped field; see there for
// why the mip chain matters.
//
// Displacement is applied BACKWARDS (sample at uv - offset): pixels are pulled
// from where the flow came from, which leaves no holes.
//
// The mode is a post-process on the assembled frame, run once per displayed
// frame from SimulationRunner._process_assembled_frame. It deliberately ignores
// camera pan/zoom -- the output has to stay pixel-aligned with the incoming
// feed for the rig to composite it.

uniform sampler2D prev_tex;      // last output, the feedback buffer
uniform sampler2D source_tex;    // incoming Spout frame
uniform sampler2D particle_tex;  // assembled particle frame
uniform sampler2D flow_tex;      // datamosh_flow.frag's field, with a mip chain

uniform vec2  source_resolution;
uniform vec2  flow_cover;        // aspect correction, flow uv -> output uv
uniform float flow_max_lod;      // top of the mip chain: the whole-frame average
uniform bool  source_connected;  // false when no Spout sender is attached

uniform int   mosh_source;       // 0=feed, 1=particles, 2=feed+particles
uniform float mosh_amount;       // displacement per frame, in output uv
uniform float mosh_contrast;     // how selective the response is (1 = neutral)
uniform float mosh_scale;        // flow mip level: how broad the strokes are
uniform float mosh_swirl;        // -1..1, rotates the shift up to +-90 degrees
uniform float mosh_refresh;      // how much live source returns each frame
uniform float mosh_block;        // macroblock size in pixels, 0 = off
uniform float mosh_chroma;       // per-channel displacement spread
uniform float mosh_ink;          // crisp particles added back on top

in vec2 texcoord;
out vec4 fragColor;

// Output uv -> flow uv, cover fit: fill the frame, crop the field. A stretch
// would skew displacement angles whenever a 1:1 canvas drives a 16:9 feed --
// diagonals would come out at the wrong angle, which is exactly the thing the
// eye notices in a directional effect.
vec2 to_flow_uv(vec2 uv) {
    return (uv - 0.5) * flow_cover + 0.5;
}

vec2 read_flow(vec2 uv) {
    vec2 fuv = to_flow_uv(uv);

    // Quantize the LOOKUP, not the result: whole macroblocks then shift
    // together, which is what reads as a codec artefact rather than a smooth
    // displacement map.
    if (mosh_block >= 1.0) {
        vec2 grid = source_resolution / mosh_block;
        fuv = (floor(fuv * grid) + 0.5) / grid;
    }

    // Mip level as stroke size. Averaging the field lets opposing directions
    // cancel, so a higher level leaves only the large-scale coherent motion --
    // regions rather than individual streaks.
    return textureLod(flow_tex, fuv, clamp(mosh_scale, 0.0, flow_max_lod)).xy;
}

vec3 read_source(vec2 uv) {
    vec3 feed = source_connected ? texture(source_tex, uv).rgb : vec3(0.0);
    vec3 ink  = texture(particle_tex, uv).rgb;

    // With no sender, the feed modes fall back to the particle frame rather
    // than moshing black -- a silent black output is indistinguishable from a
    // broken bridge.
    if (mosh_source == 1 || !source_connected) return ink;
    if (mosh_source == 2) return feed + ink;
    return feed;
}

void main() {
    vec2 uv = texcoord;

    vec2 duv = read_flow(uv);
    float m = length(duv);

    // The reference is the whole frame's mean displacement magnitude, read off
    // the top of the flow field's mip chain. Normalizing by it is what makes
    // the response independent of world size, particle count and config: x = 1
    // is "average activity for this picture, right now", whatever that is in
    // absolute terms.
    float reference = textureLod(flow_tex, vec2(0.5), flow_max_lod).z;
    float x = m / max(reference, 1e-9);

    // Contrast pivots around that average: amp is 0.5 at x = 1 for every
    // setting, so this changes how selective the effect is without changing
    // its overall strength. Below 1 the curve flattens and the whole frame
    // drifts together in broad strokes; above 1 it sharpens toward a threshold
    // where only the busiest streaks move at all.
    float xg = pow(max(x, 0.0), max(mosh_contrast, 0.0));
    float amp = xg / (1.0 + xg);

    vec2 dir = m > 1e-9 ? duv / m : vec2(0.0);

    float a = mosh_swirl * 1.57079633;
    dir = vec2(dir.x * cos(a) - dir.y * sin(a),
               dir.x * sin(a) + dir.y * cos(a));

    vec2 offset = dir * amp * mosh_amount;

    // Both the feedback buffer AND the live source are sampled at the shifted
    // coordinate. Displacing only the buffer would make mosh_refresh = 1 a
    // straight passthrough -- the whole range has to warp, and refresh only
    // decides how much of the shift is inherited from previous frames.
    vec3 col;
    float refresh = clamp(mosh_refresh, 0.0, 1.0);

    if (mosh_chroma > 0.0) {
        vec2 off_r = offset * (1.0 + mosh_chroma);
        vec2 off_b = offset * (1.0 - mosh_chroma);
        col.r = mix(texture(prev_tex, uv - off_r).r, read_source(uv - off_r).r, refresh);
        col.g = mix(texture(prev_tex, uv - offset).g, read_source(uv - offset).g, refresh);
        col.b = mix(texture(prev_tex, uv - off_b).b, read_source(uv - off_b).b, refresh);
    } else {
        vec2 suv = uv - offset;
        col = mix(texture(prev_tex, suv).rgb, read_source(suv), refresh);
    }

    // Added after the mosh, at the undisplaced coordinate, so these particles
    // stay crisp -- distinct from source mode 2, where they are smeared into
    // the image along with it.
    if (mosh_ink > 0.0) {
        col += mosh_ink * texture(particle_tex, uv).rgb;
    }

    fragColor = vec4(col, 1.0);
}
