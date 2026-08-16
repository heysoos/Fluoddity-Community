#version 430

in vec2 texcoord;

uniform sampler2D brush_tex;
uniform sampler2D can_tex;
out vec4 can_out;

// Draw trail mode uniforms
uniform bool draw_mode;
uniform vec2 mouse;
uniform vec2 previous_mouse;
uniform float draw_size;
uniform float draw_power;

// Advanced drawing uniforms
// brush_mode codes: 0=mouse_dir, 1=inverse, 2=fixed, 3=attract, 4=repel
uniform int brush_mode;
uniform float fixed_direction_heading;
uniform bool erase_mode;       // Right-click eraser active
uniform bool fill_mode;        // Fill entire canvas for one frame
uniform int fill_direction_type; // 0=fixed, 1=radial_in, 2=radial_out
uniform bool canvas_draw_active; // Whether canvas is a draw target

// Boundary conditions
uniform int BOUNDARY_CONDITIONS_MODE; //0-1-2 == BOUNCE-RESET-WRAP

// Tiling mode
uniform bool tiling_mode;

// Tournament isolation
uniform int TOURNAMENT_MODE;   // 0 = off, 1 = on
uniform int TOURNAMENT_GRID;   // grid side length

// Canvas dimensions for aspect correction
uniform vec2 canvas_resolution;

// SYNCHRONIZED: This struct must match entity_update.glsl
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag
struct PhysicsSetting {
    float slider_value;
    float min_value;
    float max_value;
    float x_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float y_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float cohort_sweep; // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float jitter;       // 0.0 = off, higher = more randomness (proportional to result)
};

uniform PhysicsSetting TRAIL_PERSISTENCE_SETTING;
uniform PhysicsSetting TRAIL_DIFFUSION_SETTING;
//Must match entity_update.glsl's, or the trail and the particles keep
//different clocks. 1.0 leaves the arithmetic below untouched.
uniform float TIME_SCALE = 1.0;

uniform int frame_count;

#define COHORTS 64//TODO THIS IS A HACK, OFTEN WRONG. Cohort sweeps a little broken

// SYNCHRONIZED: This function must match entity_update.glsl and sim.py::calculate_setting
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag, sim.py
float calculate_setting(PhysicsSetting setting, vec2 pos, float cohort){
    //if no sweep modes are active and no jitter, just return slider value
    if(setting.y_sweep == 0.0 && setting.cohort_sweep == 0.0 && setting.x_sweep == 0.0 && setting.jitter == 0.0)
        {return setting.slider_value;}
    //otherwise calculate parameter sweeps
    pos = (pos+1)/2.;//convert to 0..1 for use as a mix coefficient
    cohort = cohort / COHORTS; //convert to 0..1 for mixing

    // Count active sweeps and accumulate results
    float result = 0;
    int active_sweeps = 0;
    if(setting.x_sweep != 0.0) {
        // For inverse sweep (x_sweep < 0), swap min and max
        if(setting.x_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, pos.x);
        } else {
            result += mix(setting.max_value, setting.min_value, pos.x);
        }
        active_sweeps++;
    }
    if(setting.y_sweep != 0.0) {
        // For inverse sweep (y_sweep < 0), swap min and max
        if(setting.y_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, pos.y);
        } else {
            result += mix(setting.max_value, setting.min_value, pos.y);
        }
        active_sweeps++;
    }
    if(setting.cohort_sweep != 0.0) {
        // For inverse sweep (cohort_sweep < 0), swap min and max
        if(setting.cohort_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, cohort);
        } else {
            result += mix(setting.max_value, setting.min_value, cohort);
        }
        active_sweeps++;
    }

    // Average the results or use slider_value if no sweeps
    result = active_sweeps > 0 ? result / float(active_sweeps) : setting.slider_value;

    // Apply jitter: random variation proportional to the result value
    // Uses a simple hash function since canvas.frag doesn't have access to entity_update's hash()
    if(setting.jitter != 0.0) {
        float random = fract(sin(dot(pos + float(frame_count) * 0.01, vec2(12.9898, 78.233))) * 43758.5453) * 2.0 - 1.0;
        result += setting.jitter * result * random;
    }

    return result;
}

// The trail's half of the same statement, and it must agree with
// entity_update.glsl's sense_off_world() about where the world ends, or the
// particles and their trail disagree.
//
// Wrap is a torus. Anything else has an OPEN edge: a diffusion tap beyond it
// contributes nothing, so trail reaching the border leaves and is gone. It used
// to fall through to the sampler, and repeat_x/repeat_y carried it clean across
// the canvas to the opposite edge - measured at 60 steps, 15.5% of a blob on
// the left edge arrived at the right one.
//
// Open, not sealed. A zero-flux mirror is the tidier boundary and it is the
// wrong one here: measured over physics_configs/Core it pushes LavaLamp's
// border/interior from 10.6x to 14.2x, because the leak it replaces was acting
// as a SINK draining the bright edge. Absorbing keeps that drain, locally,
// without teleporting anything - LavaLamp 7.5x, Streamers 2.9x -> 0.4x. Mass is
// therefore NOT conserved at the border, on purpose.
vec4 getCan(vec2 p, sampler2D sam) {
    if(BOUNDARY_CONDITIONS_MODE == 2) return texture(sam, fract(p));
    if(any(lessThan(p, vec2(0.0))) || any(greaterThan(p, vec2(1.0))))
        return vec4(0.0);
    vec2 res = vec2(textureSize(sam, 0));
    return texture(sam, clamp(p, 0.5 / res, 1.0 - 0.5 / res));
}

// Tournament tiles, in TEXEL INDICES and INTEGER arithmetic.
//
// SYNCHRONIZED with tile_lo_texel() in entity_update.glsl and brush.vert. All
// three have to agree on where a seam is to the last bit, or particles,
// deposits and trails disagree about which tile a texel is in.
//
// Two separate things forced this, both measured 2026-08-09 at grid 8, wrap,
// 400 diffusion steps, on the shipped shader:
//
//  1. A tile must own a WHOLE NUMBER of texels. The diffusion is a discrete
//     5-tap stencil, and the canvas is 647 texels wide at the default
//     world_size of 0.40 - 647/8 = 80.875, so evenly divided seams ran through
//     the middle of a texel. The middle column retained 25.7% of its own trail
//     and 39 of 64 tiles lit a tile they could not legally reach. At 1024,
//     where 1024/8 = 128, every tile was exact. The grid slider is 2..8, so no
//     canvas size makes this divide for every setting.
//
//  2. The arithmetic must be exact AT the seam. GLSL does not require division
//     to be correctly rounded, and a seam is decided by the last bit: 647*4/8
//     is exactly 323.5, and this GPU evaluated floor((323.5/647)*8) as 3 where
//     the true value is 4. Texel 323 therefore sat outside its own tile's box
//     and bridged tiles 3 and 4 - which made snapping alone WORSE, not better
//     (the middle column fell to 4.6%). Integers cannot do that.
int tile_of_texel(int t, int g, int res){
    // Which tile owns texel t: the tile its CENTRE (t + 0.5) falls in, i.e.
    // floor((2t+1)*g / 2res) done exactly.
    return clamp(((2 * t + 1) * g) / (2 * res), 0, g - 1);
}
int tile_lo_texel(int k, int g, int res){
    // First texel of tile k: the smallest t with (2t+1)*g >= 2*k*res.
    if(k <= 0) return 0;
    if(k >= g) return res;
    int b = 2 * g;
    return (2 * k * res - g + b - 1) / b;        // ceil division, exact
}
void tournament_tile_texel_box(ivec2 t, ivec2 res, out ivec2 lo, out ivec2 hi){
    int g = TOURNAMENT_GRID;
    ivec2 k = ivec2(tile_of_texel(t.x, g, res.x), tile_of_texel(t.y, g, res.y));
    lo = ivec2(tile_lo_texel(k.x,     g, res.x), tile_lo_texel(k.y,     g, res.y));
    hi = ivec2(tile_lo_texel(k.x + 1, g, res.x), tile_lo_texel(k.y + 1, g, res.y));
}

// One diffusion tap, kept inside the tile the centre sample belongs to.
// texelFetch, not texture(): a discrete stencil wants the texel itself, and no
// filtering means no way to blend in a neighbour that belongs to another tile.
vec4 tile_tap(ivec2 t, ivec2 lo, ivec2 hi, vec4 centre, sampler2D sam){
    if(all(greaterThanEqual(t, lo)) && all(lessThan(t, hi)))
        return texelFetch(sam, t, 0);
    // Wrap makes the tile a torus, so the tap comes from the opposite side of
    // the SAME tile. Under bounce or reset the seam is a wall, and substituting
    // the centre value is zero net flux across it.
    if(BOUNDARY_CONDITIONS_MODE == 2){
        // Folded by hand, NOT with %: GLSL leaves % undefined when either
        // operand is negative, and the south and west probes are always at
        // lo - 1. Measured, that cost the tile 84% of its trail at res 647 -
        // and nothing at all at 1024, where the tile is 128 wide and the
        // compiler can implement % as a bitmask that happens to be right for
        // negatives. One step always suffices: the probe is one texel out.
        ivec2 w = hi - lo;
        ivec2 q = t - lo;
        q += ivec2(lessThan(q, ivec2(0))) * w;
        q -= ivec2(greaterThanEqual(q, w)) * w;
        return texelFetch(sam, lo + q, 0);
    }
    return centre;
}

vec4 getBlur(vec2 pos, sampler2D sam,float diffusion_constant) {
    ivec2 imsz = textureSize(sam, 0);
    vec3 off = vec3(1. / vec2(imsz), 0);
    vec4 cc = getCan(pos, sam);
    vec4 nc, sc, wc, ec;
    if(TOURNAMENT_MODE == 1){
        // The tile's own texel box decides what is out of bounds, never a tile
        // index derived from clamp(uv). A probe that walked off the canvas
        // clamped back into the same tile, so the comparison silently passed
        // and the tap fell through to the sampler - which has repeat_x/y set,
        // and duly returned the OPPOSITE EDGE OF THE CANVAS.
        ivec2 t = ivec2(pos * vec2(imsz));
        ivec2 tlo, thi; tournament_tile_texel_box(t, imsz, tlo, thi);
        nc = tile_tap(t + ivec2(0, 1), tlo, thi, cc, sam);
        sc = tile_tap(t - ivec2(0, 1), tlo, thi, cc, sam);
        wc = tile_tap(t - ivec2(1, 0), tlo, thi, cc, sam);
        ec = tile_tap(t + ivec2(1, 0), tlo, thi, cc, sam);
    } else {
        nc = getCan(pos + off.zy, sam);
        sc = getCan(pos - off.zy, sam);
        wc = getCan(pos - off.xz, sam);
        ec = getCan(pos + off.xz, sam);
    }
    float K = diffusion_constant;
    return (cc * K + nc + sc + wc + ec) / (4. + K);
}

// Aspect-correct UV delta so length() is isotropic in entity space
vec2 aspect_correct_uv(vec2 uv_delta) {
    float ca = canvas_resolution.x / canvas_resolution.y;
    return uv_delta * vec2(sqrt(ca), 1.0/sqrt(ca));
}

// Calculate draw vector based on brush mode
vec2 calculate_draw_vector(int mode, vec2 mouse_vel, float heading,
                           vec2 pixel_pos, vec2 mouse_p) {
    if (mode == 0) {
        return mouse_vel;                       // Mouse Direction
    } else if (mode == 1) {
        return -mouse_vel;                      // Inverse Mouse Direction
    } else if (mode == 2) {
        return .01*vec2(sin(heading), cos(heading)); // Fixed Direction (0 = up)
    } else if (mode == 3) {
        // In - Attract (toward mouse)
        vec2 to_mouse = mouse_p - pixel_pos;
        vec2 corrected = aspect_correct_uv(to_mouse);
        float len = length(corrected);
        return len > 0.0 ? .01*corrected / len : vec2(0.0);
    } else if (mode == 4) {
        // Out - Repel (away from mouse)
        vec2 from_mouse = pixel_pos - mouse_p;
        vec2 corrected = aspect_correct_uv(from_mouse);
        float len = length(corrected);
        return len > 0.0 ? .01*corrected / len : vec2(0.0);
    }
    return vec2(0.0);
}

// Gaussian kernel for draw trail mode
float draw_kernel(float distance, float size) {
    // Gaussian: exp(-distance^2 / (2 * sigma^2))
    // Using size as sigma
    float sigma = size;
    return exp(-distance * distance / (2.0 * sigma * sigma));
}

void main() {
    // Clear to black on frame 0 to prevent garbage data
    if (frame_count == 0) {
        can_out = vec4(0, 0, 0, 1);
        return;
    }

    vec4 brush_color = texture(brush_tex, texcoord);
    vec4 can_color;
    // Map texcoord to entity space for parameter sweeps
    float _ca = canvas_resolution.x / canvas_resolution.y;
    vec2 entity_space_pos = (texcoord * 2.0 - 1.0) * vec2(sqrt(_ca), 1.0/sqrt(_ca));
    float TRAIL_DIFFUSION = calculate_setting(TRAIL_DIFFUSION_SETTING,entity_space_pos,0);
    TRAIL_DIFFUSION = clamp(TRAIL_DIFFUSION,0.001,1.0);//keeps jitter from exceeding the valid domain
    if(TRAIL_DIFFUSION>0){
        TRAIL_DIFFUSION= TRAIL_DIFFUSION*TRAIL_DIFFUSION;//better scaling for slider
        TRAIL_DIFFUSION = 4/(pow(5,(TRAIL_DIFFUSION))-1);//better scaling for slider
        can_color = getBlur(texcoord, can_tex,TRAIL_DIFFUSION);
    }
    else{
        can_color = texture(can_tex,texcoord);
    }

    // Use entity space position for position-based sweeps
    float trail_persistence = calculate_setting(TRAIL_PERSISTENCE_SETTING, entity_space_pos, 0.0);
    trail_persistence = clamp(trail_persistence,0.0,0.999);
    //ONE pow compensates the whole clock, because the decay and the deposit
    //are two halves of one moving average: what the trail keeps is P and what
    //it takes in is 1-P. Raising P to the step's length makes two steps at
    //half time exactly one step at full time, so a slower sim is the same
    //creature rather than one drawing thicker.
    if (TIME_SCALE != 1.0) {
        trail_persistence = pow(trail_persistence, TIME_SCALE);
    }
    can_out = can_color * trail_persistence + (1 - trail_persistence) * brush_color;

    // Draw trail mode: add velocity based on mouse drag
    if (draw_mode && canvas_draw_active && draw_power > 0.0) {
        float distance_to_mouse;

        // Calculate raw mouse velocity for tiling correction
        vec2 raw_mouse_vel = (mouse - previous_mouse);

        if (tiling_mode) {
            // In tiling mode, check 9-cell neighborhood (3x3) for wrapped distance
            // This allows trail drawing across wrapped edges/corners
            float min_distance = 999.0;
            vec2 min_draw_vector = vec2(999);
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    vec2 wrapped_mouse = mouse + vec2(dx, dy);
                    float dist = length(aspect_correct_uv(texcoord - wrapped_mouse));
                    min_distance = min(min_distance, dist);
                    vec2 vel = (wrapped_mouse-previous_mouse);
                    min_draw_vector = length(vel)<length(min_draw_vector)?vel:min_draw_vector;
                }
            }
            distance_to_mouse = min_distance;
            raw_mouse_vel = min_draw_vector;
        } else {
            // Normal mode: direct distance calculation
            distance_to_mouse = length(aspect_correct_uv(texcoord - mouse));
        }

        // Apply brush mode to get final draw vector
        vec2 draw_vector = calculate_draw_vector(
            brush_mode, raw_mouse_vel, fixed_direction_heading,
            texcoord, mouse
        );
        draw_vector *= draw_power/5;

        // Apply Gaussian kernel and add to velocity channels (RG)
        float kernel_weight = draw_kernel(distance_to_mouse, draw_size);
        can_out.xy += draw_vector * kernel_weight/draw_size*(1-trail_persistence);
    }

    // Right-click eraser: hard circle erase within draw_size radius
    if (erase_mode && canvas_draw_active) {
        float erase_distance;
        if (tiling_mode) {
            float min_dist = 999.0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    vec2 wrapped = mouse + vec2(dx, dy);
                    min_dist = min(min_dist, length(aspect_correct_uv(texcoord - wrapped)));
                }
            }
            erase_distance = min_dist;
        } else {
            erase_distance = length(aspect_correct_uv(texcoord - mouse));
        }
        if (erase_distance < draw_size*2) {//Match the reticle size from frame_assembly.frag
            can_out = vec4(0.0, 0.0, 0.0, 1.0);
        }
    }

    // Fill mode: apply brush to entire canvas for one frame (no kernel, no persistence factor)
    if (fill_mode && canvas_draw_active && draw_power > 0.0) {
        vec2 fill_vector;
        if (fill_direction_type == 0) {
            fill_vector = vec2(sin(fixed_direction_heading), cos(fixed_direction_heading));
        } else if (fill_direction_type == 3) {
            // Fixed Direction - Negative (heading + PI)
            float neg_heading = fixed_direction_heading + 3.1415;
            fill_vector = vec2(sin(neg_heading), cos(neg_heading));
        } else if (fill_direction_type == 1) {
            vec2 to_center = vec2(0.5) - texcoord;
            vec2 corrected = aspect_correct_uv(to_center);
            float len = length(corrected);
            fill_vector = len > 0.0 ? corrected / len : vec2(0.0);
        } else {
            vec2 from_center = texcoord - vec2(0.5);
            vec2 corrected = aspect_correct_uv(from_center);
            float len = length(corrected);
            fill_vector = len > 0.0 ? corrected / len : vec2(0.0);
        }
        fill_vector *= draw_power / 5.0;
        can_out.xy += .25*fill_vector;
    }

    // Self-healing NaN scrub. entity_update.glsl stops particles seeding NaN,
    // but getBlur() above is a 5-tap kernel reading the PREVIOUS canvas, so any
    // NaN already stored keeps poisoning its four neighbours every frame and
    // the hole grows forever. Zeroing it here lets existing damage heal instead
    // of requiring the canvas to be cleared by hand.
    //
    // Magnitude bound rather than isnan(): false for NaN and false for Inf,
    // and a driver assuming finite math cannot fold it away.
    //
    // The bound is 1e30, not the 1e6 used for particle state. Particle
    // positions and velocities have a physical scale of order 1, so 1e6 there
    // means "already broken". Canvas values do NOT: they accumulate, and a
    // normal run was measured at max 12930 after three generations. A 1e6
    // bound here would eventually start deleting the brightest real trails,
    // which is data loss dressed up as a safety check. 1e30 is unreachable by
    // accumulation but still below infinity.
    if(!all(lessThan(abs(can_out), vec4(1e30)))) can_out = vec4(0.0, 0.0, 0.0, 1.0);
}
