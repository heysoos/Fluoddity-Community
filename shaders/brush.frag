#version 430

in vec2 uv;
in vec4 pos_vel;
in vec4 view_col;
in vec2 frag_world;
flat in vec2 tile_lo;
flat in vec2 tile_hi;
out vec4 brush_out;

uniform int frame_count;
uniform int TILE_MODE;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}


// The deposit's weight is the other half of the trail's moving average, and it
// has to be the number canvas.frag decayed this texel by. Both are copied from
// there verbatim rather than shared through a uniform: trail_persistence is
// POSITION-dependent - sweeps and jitter - so one value for the whole canvas
// is wrong wherever a sweep is on. Guarded by test_trail_deposit.py.
#define COHORTS 64//TODO THIS IS A HACK, OFTEN WRONG. Cohort sweeps a little broken

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
//Must match canvas.frag's, or the decay and the deposit keep different clocks.
uniform float TIME_SCALE = 1.0;

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

float gaussian(vec2 pos, float sigma) {
    float sigma2 = sigma * sigma;
    float norm = 1.0 / (2.0 * 3.14159265359 * sigma2);
    float exponent = -(dot(pos, pos)) / (2.0 * sigma2);
    return norm * exp(exponent);
}

void main() {
    // The canvas pass has already zeroed this texel on frame 0; there is
    // nothing to add on top of it.
    if (frame_count == 0) { discard; }

    // Boxed: a particle sprite has ~1px extent, so without this clip a particle
    // near a seam deposits trail into the neighbouring box, which that box's
    // sensors then read. Keeps boxes genuinely independent.
    if (TILE_MODE != 0) {
        if (any(lessThan(frag_world, tile_lo)) || any(greaterThan(frag_world, tile_hi))) {
            discard;
        }
    }

    float kernel_func = gaussian(uv - .5, .163);
    if (length(uv - .5) > .5 || view_col.w == 0) { discard; }
    vec2 vel = pos_vel.zw;

    // frag_world IS canvas.frag's entity_space_pos: brush.vert scales the
    // vertex by (1/sqrt(ca), sqrt(ca)) into clip space, which that formula
    // undoes exactly.
    float p = calculate_setting(TRAIL_PERSISTENCE_SETTING, frag_world, 0.0);
    p = clamp(p, 0.0, 0.999);
    if (TIME_SCALE != 1.0) { p = pow(p, TIME_SCALE); }

    // Blending is ONE, ONE now. The old SRC_ALPHA factor applied the kernel a
    // second time, so squaring it here is what keeps the falloff identical.
    brush_out = vec4(vel * (1.0 - p) * kernel_func * kernel_func, 0.0, 0.0);
}
