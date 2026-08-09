#version 450
layout(local_size_x = 64) in;

//SAME STRUCT USED IN BRUSH.VERT AND CAM_BRUSH.VERT
struct Entity {
    vec2 pos;
    vec2 vel;
    float size;
    float cohort;      // Normalized cohort value (0-1) for parameter sweep calculations
    float padding[2];  // Align to 16-byte boundary for vec4
    vec4 color;
};  // Total: 48 bytes (12 floats)
// The brain buffer, its uniforms and brain_at() live in shaders/brains/_header.glsl,
// which is prepended ahead of every brain_*.glsl.
layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};
// Per-particle brains, written only when WRITE_RULES is set (click-to-adopt).
// Sized to BRAIN_LEN floats per particle by the host, NOT MAX_BRAIN_FLOATS:
// at the max stride this would be ~1 KB per particle, about 600 MB.
layout(std430, binding = 2) buffer BrainReadbackBuffer {
    float particle_brains[];
};
// SYNCHRONIZED: This struct must match canvas.frag
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
uniform float WORLD_SIZE;
uniform int frame_count;
uniform sampler2D canvas; //trails canvas
uniform sampler2D field_texture; // Force/Strafe field (.xy=force, .zw=strafe)
uniform bool advanced_drawing_resources_initialized; // True when field_texture has valid data
uniform float force_field_strength; // Multiplier for force field effects
uniform float strafe_field_strength; // Multiplier for strafe field effects
uniform vec2 canvas_resolution;
uniform PhysicsSetting DRAG_SETTING; 
uniform PhysicsSetting STRAFE_POWER_SETTING;
uniform PhysicsSetting SENSOR_ANGLE_SETTING;
uniform PhysicsSetting GLOBAL_FORCE_MULT_SETTING;
uniform PhysicsSetting SENSOR_DISTANCE_SETTING;
uniform PhysicsSetting AXIAL_FORCE_SETTING;
uniform PhysicsSetting LATERAL_FORCE_SETTING;
uniform PhysicsSetting SENSOR_GAIN_SETTING;
uniform PhysicsSetting MUTATION_SCALE_SETTING;
uniform PhysicsSetting HAZARD_RATE_SETTING;
uniform float HUE_SENSITIVITY;
uniform bool COLOR_BY_COHORT;
uniform bool DISABLE_SYMMETRY;
uniform int ABSOLUTE_ORIENTATION; // 0=Off, 1=Y axis, 2=Radial
uniform float ORIENTATION_MIX; // Blend factor for orientation calculations
uniform int BOUNDARY_CONDITIONS_MODE; //0-1-2 == BOUNCE-RESET-WRAP
uniform int RESET_MODE; //0-1-2 == GRID-RANDOM-RING
uniform int COHORTS; //each cohort gets its own rule and starting location
uniform float RULE_SEED;
uniform bool WRITE_RULES; // Set true for one frame when rule buffer readback is needed

// Tournament mode: partition the canvas into a TOURNAMENT_GRID x TOURNAMENT_GRID grid
uniform int TOURNAMENT_MODE;   // 0 = off, 1 = on
uniform int TOURNAMENT_GRID;   // grid side length (4 => 16 tiles)
// Reshuffles initial conditions between generations so a genome is not scored
// on one lucky starting layout. 0.0 = the original deterministic reset.
uniform float RESET_SEED;
// 1 = each tile's physics comes from its own configs[] entry, so the optimizer
// can search physics as well as the brain.
uniform int TOURNAMENT_PHYSICS;

// Multi-load control uniforms (small, stay as uniforms)
uniform int MULTILOAD_COUNT; // Number of loaded configs (0 = normal mode)
uniform float MULTI_LOAD_CURRENT_PROGRESS; // Current position in config ring (0-1)
uniform float MULTI_LOAD_SIMULTANEOUS_CONFIGS; // How many configs to span
uniform int MULTI_LOAD_ASSIGNMENT_MODE; // 0 = Cohorts, 1 = Random
uniform bool MULTI_LOAD_PER_CONFIG_INITIAL_CONDITIONS; // If true, use per-config reset modes
uniform bool MULTI_LOAD_PER_CONFIG_COHORTS; // If true, use per-config cohort counts
uniform bool MULTI_LOAD_PER_CONFIG_HAZARD_RATE; // If true, use per-config hazard rates

// Multi-load config data (large arrays, packed into SSBO)
struct MultiLoadConfig {
    // Physics parameters as PhysicsSetting structs (10 params * 6 floats = 60 floats)
    PhysicsSetting axial_force;
    PhysicsSetting lateral_force;
    PhysicsSetting sensor_gain;
    PhysicsSetting mutation_scale;
    PhysicsSetting drag;
    PhysicsSetting strafe_power;
    PhysicsSetting sensor_angle;
    PhysicsSetting global_force_mult;
    PhysicsSetting sensor_distance;
    PhysicsSetting hazard_rate;

    // Simulation settings (6 ints)
    int disable_symmetry;      // bool as int for alignment
    int absolute_orientation;  // 0=Off, 1=Y axis, 2=Radial
    int boundary_conditions;
    int reset_mode;
    int cohorts;
    int color_by_cohort;       // bool as int for alignment

    // Appearance and orientation mix (3 floats)
    float hue_sensitivity;
    float orientation_mix;
    float rule_seed;
};

layout(std430, binding = 3) buffer MultiLoadConfigBuffer {
    MultiLoadConfig configs[64];
};


////////////////////////////CONSTANTS
#define PI 3.1415926
// The real allocated/dispatched particle count, which is
// 600000 * WORLD_SIZE * PARTICLE_DENSITY. It must NOT be re-derived from
// WORLD_SIZE here: ACTIVE_COUNT is the denominator for the index-derived slices
// below (cohorts, tournament tiles), so a denominator larger than the buffer
// would compress every slice into the bottom of the range - at density 0.5 the
// top half of the tournament grid would render empty.
#define ACTIVE_COUNT float(ENTITY_COUNT)
#define SQRT_WORLD_SIZE (sqrt(WORLD_SIZE))
#define SQRT_WORLD_SIZE (sqrt(WORLD_SIZE))
// Multi-load helper: Calculate which config index this particle should use
int tournament_home_tile(uint index);   // defined below

int get_particle_config_index() {
    // Auto tournament with physics in the search space: every tile reads its
    // own PhysicsSetting block, reusing the multi-load config SSBO. Checked
    // before MULTILOAD_COUNT because tournament mode does not load configs.
    if (TOURNAMENT_MODE == 1 && TOURNAMENT_PHYSICS == 1) {
        return tournament_home_tile(gl_GlobalInvocationID.x);
    }
    if (MULTILOAD_COUNT == 0) return -1; // Not in multi-load mode

    // Calculate normalized index (0 to 1) for this particle
    float normalized_index = float(gl_GlobalInvocationID.x) / float(ACTIVE_COUNT);

    // For "Random" assignment mode, hash the normalized_index for stable pseudo-random assignment
    if (MULTI_LOAD_ASSIGNMENT_MODE == 1) {
        normalized_index = hash(vec2(normalized_index, 0.0));
    }

    // Calculate config index using circular ring formula
    // If SIMULTANEOUS_CONFIGS == 2, span across 2 full indices as normalized_index sweeps 0 to 1
    float offset = MULTI_LOAD_SIMULTANEOUS_CONFIGS / float(MULTILOAD_COUNT) * normalized_index;
    float ring_position = fract(MULTI_LOAD_CURRENT_PROGRESS + offset);
    int config_index = int(floor(float(MULTILOAD_COUNT) * ring_position));

    // Clamp to valid range
    return clamp(config_index, 0, MULTILOAD_COUNT - 1);
}

// Tournament: a particle's stable "home tile" is derived from its buffer index,
// so a particle never migrates between tiles even if it drifts spatially.
int tournament_home_tile(uint index){
    int n = TOURNAMENT_GRID * TOURNAMENT_GRID;
    int tile = int(floor(float(index) / float(ACTIVE_COUNT) * float(n)));
    return clamp(tile, 0, n - 1);
}
// Entity-space bounding box [lo, hi] of a tile index.
void tournament_tile_box(int tile, out vec2 lo, out vec2 hi){
    float ca = canvas_resolution.x / canvas_resolution.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    int tx = tile % TOURNAMENT_GRID;
    int ty = tile / TOURNAMENT_GRID;
    vec2 cell = (2.0 * half_extent) / float(TOURNAMENT_GRID);
    lo = -half_extent + vec2(float(tx), float(ty)) * cell;
    hi = lo + cell;
}
                            //Entities with index > ACTIVE_COUNT aren't rendered or updated
int get_particle_cohorts() {
    int idx = get_particle_config_index();
    // Use per-config value only if multi-load is active AND per-config checkbox is enabled
    if (idx >= 0 && MULTI_LOAD_PER_CONFIG_COHORTS) {
        return configs[idx].cohorts;
    }
    return COHORTS;
}
//Calculate the actual setting value for this particle. When sweeps are
//active, physics settings can depend on entity position and cohort
// SYNCHRONIZED: This function must match canvas.frag and sim.py::calculate_setting
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag, sim.py
float calculate_setting(PhysicsSetting setting, vec2 pos, float cohort){
    //if no sweep modes are active and no jitter, just return slider value
    if(setting.y_sweep == 0.0 && setting.cohort_sweep == 0.0 && setting.x_sweep == 0.0 && setting.jitter == 0.0)
        {return setting.slider_value;}
    //otherwise calculate parameter sweeps
    pos = (pos+1)/2.;//convert to 0..1 for use as a mix coefficient
    cohort = floor(cohort) / float(get_particle_cohorts()); //convert to 0..1 for mixing

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
    // hash() returns 0..1, so (hash(...)*2.-1.) returns -1..1
    if(setting.jitter != 0.0) {
        float random = hash(vec2(float(frame_count)+result, pos.x + pos.y * 1000.0)) * 2.0 - 1.0;
        result += setting.jitter * result * random;
    }

    return result;
}



// Helper functions to get config values (return array value if multi-load, else single uniform)

PhysicsSetting get_particle_axial_force() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].axial_force : AXIAL_FORCE_SETTING;
}

PhysicsSetting get_particle_lateral_force() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].lateral_force : LATERAL_FORCE_SETTING;
}

PhysicsSetting get_particle_sensor_gain() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].sensor_gain : SENSOR_GAIN_SETTING;
}

PhysicsSetting get_particle_mutation_scale() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].mutation_scale : MUTATION_SCALE_SETTING;
}

PhysicsSetting get_particle_drag() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].drag : DRAG_SETTING;
}

PhysicsSetting get_particle_strafe_power() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].strafe_power : STRAFE_POWER_SETTING;
}

PhysicsSetting get_particle_sensor_angle() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].sensor_angle : SENSOR_ANGLE_SETTING;
}

PhysicsSetting get_particle_global_force_mult() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].global_force_mult : GLOBAL_FORCE_MULT_SETTING;
}

PhysicsSetting get_particle_sensor_distance() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].sensor_distance : SENSOR_DISTANCE_SETTING;
}

bool get_particle_disable_symmetry() {
    int idx = get_particle_config_index();
    return idx >= 0 ? bool(configs[idx].disable_symmetry) : DISABLE_SYMMETRY;
}

int get_particle_absolute_orientation() {
    int idx = get_particle_config_index();
    return idx >= 0 ? int(configs[idx].absolute_orientation) : ABSOLUTE_ORIENTATION;
}
PhysicsSetting get_particle_hazard_rate() {
    int idx = get_particle_config_index();
    return idx >= 0 &&MULTI_LOAD_PER_CONFIG_HAZARD_RATE? (configs[idx].hazard_rate) : HAZARD_RATE_SETTING;
}

//HARDCODED TO BE GLOBAL FOR NOW
int get_particle_boundary_conditions() {
    //int idx = get_particle_config_index();
    //return idx >= 0 ? configs[idx].boundary_conditions : BOUNDARY_CONDITIONS_MODE;
    return BOUNDARY_CONDITIONS_MODE;
}

int get_particle_reset_mode() {
    int idx = get_particle_config_index();
    // Use per-config value only if multi-load is active AND per-config checkbox is enabled
    if (idx >= 0 && MULTI_LOAD_PER_CONFIG_INITIAL_CONDITIONS) {
        return configs[idx].reset_mode;
    }
    return RESET_MODE;
}


//HARDCODED TO BE GLOBAL FOR NOW
float get_particle_hue_sensitivity() {
    //int idx = get_particle_config_index();
    //return idx >= 0 ? configs[idx].hue_sensitivity : HUE_SENSITIVITY;
    return HUE_SENSITIVITY;
}

//HARDCODED TO BE GLOBAL FOR NOW
bool get_particle_color_by_cohort() {
    //int idx = get_particle_config_index();
    //return idx >= 0 ? bool(configs[idx].color_by_cohort) : COLOR_BY_COHORT;
    return COLOR_BY_COHORT;
}

float get_particle_rule_seed() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].rule_seed : RULE_SEED;
}

// Which slot of brain_params this particle reads. Multi-load configs own
// slots by config index; tournament mode by tile; otherwise slot 0.
uint get_particle_brain_base() {
    int idx = get_particle_config_index();
    uint slot = 0u;
    if (idx >= 0) {
        slot = uint(idx);
    } else if (TOURNAMENT_MODE == 1) {
        slot = uint(tournament_home_tile(gl_GlobalInvocationID.x));
    }
    return slot * uint(MAX_BRAIN_FLOATS);
}


////////////////////////////////////
//FOURIER NOISE IS IMPORTED INTO THIS SHADER
//FROM fourier4_4.glsl
////////////////////////////////////

//rotate p around origin by angle a
void pR(inout vec2 p, float a) {
	p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}


//convert p (entity space) to texture coords and retrieve canvas
vec4 get_can(vec2 p){
    vec2 res=textureSize(canvas,0);
    float ca = res.x / res.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv = p / (2.0 * half_extent) + 0.5;
    if(get_particle_boundary_conditions() == 2) uv = fract(uv);
    return texture(canvas, uv);
}
vec4 get_field(vec2 p){
    if(!advanced_drawing_resources_initialized)return vec4(0);
    vec2 res=textureSize(field_texture,0);
    float ca = res.x / res.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv = p / (2.0 * half_extent) + 0.5;
    if(get_particle_boundary_conditions() == 2) uv = fract(uv);
    return texture(field_texture, uv);
}

vec2 safenorm(vec2 p){
    return length(p)==0?vec2(0):normalize(p);
}

float get_cohort(uint index) {
    return float(get_particle_cohorts()) * float(index) / float(ACTIVE_COUNT);
}

//Return all entities to their initialization state
void reset(uint index){

    float size=index<ACTIVE_COUNT?.0015/SQRT_WORLD_SIZE: 0;
    float cohort_val = get_cohort(index);
    float aspect = sqrt(canvas_resolution.x/canvas_resolution.y);

    vec4 color=vec4(0,0,1,.045);
    //set pos and vel to random values on a small disk
    float cohort_scale = 0.019;//Size of each disk
    // RESET_SEED shifts every draw below. It is 0.0 outside auto-tournament
    // mode, where adding it is a no-op, so ordinary resets are bit-identical.
    vec2 pos=cohort_scale*vec2(hash(vec2(cohort_val+RESET_SEED)),hash(vec2(cohort_val+index+2.142+RESET_SEED)));
    vec2 vel=.00005*(vec2(hash(vec2(cohort_val+RESET_SEED,index)),hash(vec2(cohort_val,pos.y)))*2-1);

    //RESET_MODE: 0=Grid, 1=Random, 2=Ring
    int reset_mode = get_particle_reset_mode();
    int cohorts = get_particle_cohorts();
    if(reset_mode == 0) {
        //GRID: position different cohorts at different places in a grid
        float spots=float(cohorts);
        float spot_rows=ceil(aspect*sqrt(spots));
        vec2 gridcell=vec2(int(cohort_val)%int(spot_rows),(int(cohort_val))/int(spot_rows));
        //pR(pos,floor(cohort_val)*3.1415*2*spots);
        //this aspect transform is good enough, but not perfect
        pos+=1.8*((gridcell)/spot_rows)*vec2(aspect);
        pos+= 1.8*(1/2.*(1./vec2(spot_rows,spots/spot_rows)-1))*vec2(aspect,1/aspect);
    }
    else if(reset_mode == 1) {
        //RANDOM: scatter cohorts randomly across the canvas, homogenous start
        pos= vec2(hash(vec2(cohort_val, 1.0)), hash(vec2(cohort_val, 2.0))) * 2.0 - 1.0;
        pos.x*=aspect;
        pos.y/=aspect;
    }
    else if(reset_mode == 2) {
        //RING: arrange cohorts in a ring pattern
        float angle = cohort_val / float(cohorts) * 2.0 * PI;
        float radius = 0.5;
        pos += vec2(cos(angle), sin(angle)) * radius;
        //pos += 0.02 * vec2(hash(vec2(cohort_val)), hash(vec2(cohort_val + 1.0))); // Small jitter
    }

    
    //Tournament: place the particle uniformly inside its home tile (with a small margin).
    if(TOURNAMENT_MODE == 1){
        int htile = tournament_home_tile(index);
        vec2 lo, hi; tournament_tile_box(htile, lo, hi);
        vec2 margin = (hi - lo) * 0.04;
        lo += margin; hi -= margin;
        vec2 r = vec2(hash(vec2(cohort_val + RESET_SEED, float(index)+0.1)),
                      hash(vec2(float(index)+0.2, cohort_val + RESET_SEED)));
        pos = mix(lo, hi, r);
    }

    //store to persistent entity buffer
    entities[index]=Entity(pos,vel,size,cohort_val/float(cohorts),float[2](0,0),color);
}

// Per-particle brain mutation lives in shaders/brains/_header.glsl as
// brain_at(); main() sets g_brain_mut and g_brain_cohort before evaluating.


//Used to enforce left-right symmetry in the local coordinates vec2(forward, left)
vec2 y_reflect(vec2 p){
    return p*vec2(1,-1);
}
vec2 x_reflect(vec2 p){
    return p*vec2(-1,1);
}
//reflect across the boundary [-1,1] to keep particle positions from leaving the canvas
float edgeflect(float x){
    return sign(x)*(1-abs(1-abs(x)));
}

//Somewhat arbitrary generator of functions with 4 float inputs and 4 float outputs,
//varying the brain should smoothly change the behavior of the function.
//eval_brain() is the modality dispatch, in shaders/brains/_dispatch.glsl.
vec4 black_box(vec2 L,vec2 R,uint base){
    return eval_brain(base, vec4(L,R));
}



//This function determines entity output by plugging sensor values into a noise function called black_box()
//The calculation is performed twice, once in mirrored coordinates, and the two values are averaged.
//This keeps entities from displaying clockwise/counterclockwise bias.
//PARAMETERS:
//--L and R: velocity field measurements from left sensor and right sensor.
//--axis: forward vector that defines our orientation.
//--base: offset of this particle's brain in brain_params.
//--pos: entity position (for parameter sweeps)
//--cohort: entity cohort (for parameter sweeps)
//RETURNS:
//--force: A "push" vector that will be added to entity.vel
//--strafe: A "hop" vector that will be added to entity.pos and have no effect on velocity
//--color: vec2 to be used as parameters in a coloring function
void calculate_entity_behavior( vec2 L,vec2 R, vec2 axis, uint base, vec2 pos, float cohort, out vec2 force, out vec2 strafe, out vec2 color){

    //build a local coordinate frame where "axis" is forward.
    vec2 forward=safenorm(axis);
    vec2 left=vec2(forward.y,-forward.x);

    //Convert L and R to local coordinates.
    //Ie. decompose each into an axial component and a lateral component
    L=vec2(dot(L,forward),dot(L,left));
    R=vec2(dot(R,forward),dot(R,left));

    //calculate black box noise values
    vec4 baseterm= black_box(L,R,base);
    vec4 mirrorterm=black_box(y_reflect(R),y_reflect(L),base);
    if(DISABLE_SYMMETRY){mirrorterm = vec4(0);}//disable symmetry by zeroing the mirror term

    //Combine base and mirror terms
    force = baseterm.xy+y_reflect(mirrorterm.xy);
    strafe = baseterm.zw + y_reflect(mirrorterm.zw);

    //Convert force and strafe back to world coordinates
    force=forward*force.x*calculate_setting(get_particle_axial_force(),pos,cohort)+left*force.y*calculate_setting(get_particle_lateral_force(),pos,cohort);
    strafe = forward*strafe.x*calculate_setting(get_particle_axial_force(),pos,cohort) + left * strafe.y * calculate_setting(get_particle_lateral_force(),pos,cohort);

    color = baseterm.xy+(mirrorterm.xy); //Just an arbitrary function of blackbox output. Reuses force terms.
    return;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ENTITY_COUNT) return;

    // Inactive entities get zeroed out. Position offscreen so they don't accidentally get clicked on
    if (index >= ACTIVE_COUNT) {
        entities[index] = Entity(vec2(10000), vec2(0), 0.0, 0.0, float[2](0,0), vec4(0));
        return;
    }
    Entity e=entities[index];
    float cohort = get_cohort(index);

    uint brain_base = get_particle_brain_base();

    //Each cohort gets a random mutation, applied on read by brain_at().
    g_brain_mut = calculate_setting(get_particle_mutation_scale(),e.pos,cohort);
    g_brain_cohort = get_particle_rule_seed()+floor(cohort);

    //Same two-coefficient probe as the old all-zero Rule check: index 0 is the
    //first frequency component, index 40 the first amplitude of centre 5.
    bool blank = brain_params[brain_base]==0.0 && brain_params[brain_base+40u]==0.0;
    if(blank){
        g_brain_fallback = true;
        g_brain_seed = (TOURNAMENT_MODE == 1)
            ? get_particle_rule_seed()+float(tournament_home_tile(index))
            : get_particle_rule_seed()+floor(cohort);
    }

    // Only write brains when explicitly requested (expensive - BRAIN_LEN floats
    // per particle). Click-to-adopt reads this back for one entity, and needs
    // the MUTATED values, which is why it goes through brain_at().
    if(WRITE_RULES) {
        uint out_base = index * uint(BRAIN_LEN);
        for(int i = 0; i < BRAIN_LEN; i++) {
            particle_brains[out_base + uint(i)] = brain_at(brain_base, i);
        }
    }


    //frame_count == 0 signals a simulation reset
    if (frame_count==0||calculate_setting(get_particle_hazard_rate(),e.pos,cohort)>hash(vec2(float(index)/float(ACTIVE_COUNT),frame_count))){reset(index);return;}



    //Calculate position offsets for the two sensors.
    float sample_dist = 1./SQRT_WORLD_SIZE*.005 * calculate_setting(get_particle_sensor_distance(),e.pos,cohort);
    
    //variable sample distance?
    //sample_dist *= (get_can(e.pos).z*10);
    //GOOD 1./dot(normalize(e.vel),normalize(get_can(e.pos).xy));
    //length(e.vel)/.05;//length(get_can(e.pos).xy)/.01;
    
    int ORIENTATION_MODE =get_particle_absolute_orientation();
    float mix_amt = min(1,ORIENTATION_MODE)*ORIENTATION_MIX;
    vec2 orientation = safenorm(e.vel);//vector facing the same direction as velocity, with length==samplen
    if(ORIENTATION_MODE==1){orientation = mix(orientation,vec2(0,1),mix_amt);}
    else if(ORIENTATION_MODE==2){orientation = mix(orientation,-normalize(e.pos),mix_amt);}
    vec2 left_sensor_offset = orientation*sample_dist;
    vec2 right_sensor_offset = orientation*sample_dist;
    pR(left_sensor_offset,calculate_setting(get_particle_sensor_angle(),e.pos,cohort)*PI);//rotate them opposite directions
    pR(right_sensor_offset,-calculate_setting(get_particle_sensor_angle(),e.pos,cohort)*PI);

    //read the trails from canvas (tournament: keep sample points inside the home tile)
    vec2 lsample = e.pos + left_sensor_offset;
    vec2 rsample = e.pos + right_sensor_offset;
    if(TOURNAMENT_MODE == 1){
        vec2 tlo, thi; tournament_tile_box(tournament_home_tile(index), tlo, thi);
        // Inset by one texel: get_can() samples bilinearly, so clamping exactly to
        // the seam would still blend in texels belonging to the neighbouring tile.
        float ca_s = canvas_resolution.x / canvas_resolution.y;
        vec2 texel = 2.0 * vec2(sqrt(ca_s), 1.0/sqrt(ca_s)) / canvas_resolution;
        tlo += texel; thi -= texel;
        lsample = clamp(lsample, tlo, thi);
        rsample = clamp(rsample, tlo, thi);
    }
    vec4 ltap = get_can(lsample);
    vec4 rtap = get_can(rsample);

    
    //rescale sensor values
    float sensor_scaling = SQRT_WORLD_SIZE*38.855*calculate_setting(get_particle_sensor_gain(),e.pos,cohort);
    ltap *= sensor_scaling;
    rtap *= sensor_scaling;

    //compute entity action
    vec2 strafe =vec2(0);
    vec2 force = vec2(0);
    vec2 col_params = vec2(0);
    calculate_entity_behavior(ltap.xy,rtap.xy,orientation,brain_base,e.pos,cohort,force,strafe,col_params);

    //rescale output forces
    force *= 1./SQRT_WORLD_SIZE*calculate_setting(get_particle_global_force_mult(),e.pos,cohort)/400.;
    strafe *= 1./SQRT_WORLD_SIZE*calculate_setting(get_particle_global_force_mult(),e.pos,cohort)/20.;


    //e.color is interpreted as vec4(hue,saturation,brightness,alpha)
    //We just set brightness to 1 and modulate hue and saturation
    e.color.x = get_particle_hue_sensitivity()*col_params.x;//hue can be anything
    //Hardcoding saturation for now. 
    //low saturation arises naturally due to a mix of hues from different particles. 
    //Use col_params.y for something else?
    //e.color.y = sin(col_params.y)/2.+.5;//saturation must be 0..1
    e.color.y = .8;

    if(get_particle_color_by_cohort()) {e.color.x = hash(vec2(floor(cohort)));} //just assign a random hue to each cohort
    e.color.z=1;//brightness 1.
    e.color.w=0.045; //low alpha

    //Accelerate: Apply drag and add force to e.vel,
    e.vel = e.vel*calculate_setting(get_particle_drag(),e.pos,cohort) + force;
    //Move: add e.vel and strafe to e.pos
    e.pos += e.vel;
    e.pos += strafe*calculate_setting(get_particle_strafe_power(),e.pos,cohort);

    //ADVANCED DRAWING force / strafe
    vec4 draw_sample =get_field(e.pos);
    e.vel += .01*force_field_strength*draw_sample.xy;
    e.pos += .01*strafe_field_strength*draw_sample.zw;

    //BOUNDARY_CONDITIONS_MODE:  0-1-2 == BOUNCE-RESET-WRAP
    float ca = canvas_resolution.x / canvas_resolution.y;
    float x_edge = sqrt(ca);
    float y_edge = 1.0 / sqrt(ca);
    int boundary_mode = get_particle_boundary_conditions();
    if(boundary_mode==0){
        //reflect particles off canvas boundaries
        if (e.pos.x < -x_edge || e.pos.x > x_edge){
            e.vel.x=-e.vel.x;
            e.pos.x=edgeflect(e.pos.x/x_edge)*x_edge;
        }
        if (e.pos.y < -y_edge || e.pos.y > y_edge){
            e.vel.y=-e.vel.y;
            e.pos.y=edgeflect(e.pos.y/y_edge)*y_edge;
        }
    }
    else if(boundary_mode==1){
        //reset to initial conditions
        if(e.pos.x<-x_edge||e.pos.x>x_edge||e.pos.y<-y_edge||e.pos.y>y_edge){
            reset(index);
            return;//reset expects to be the last thing we do. It handles entity buffer storage
        }
    }
    else if(boundary_mode==2){
        //wrap: X wraps [-x_edge,x_edge], Y wraps [-y_edge, y_edge]
        e.pos.x = x_edge * 2.0 * (fract(e.pos.x / (x_edge * 2.0) - 0.5) - 0.5);
        e.pos.y = y_edge * 2.0 * (fract(e.pos.y / (y_edge * 2.0) - 0.5) - 0.5);
    }

    //Tournament: override world boundaries with per-tile bounce so tiles stay isolated.
    if(TOURNAMENT_MODE == 1){
        vec2 tlo, thi; tournament_tile_box(tournament_home_tile(index), tlo, thi);
        if(e.pos.x < tlo.x){ e.pos.x = tlo.x; e.vel.x = abs(e.vel.x); }
        if(e.pos.x > thi.x){ e.pos.x = thi.x; e.vel.x = -abs(e.vel.x); }
        if(e.pos.y < tlo.y){ e.pos.y = tlo.y; e.vel.y = abs(e.vel.y); }
        if(e.pos.y > thi.y){ e.pos.y = thi.y; e.vel.y = -abs(e.vel.y); }
    }

    // Non-finite guard. NaN fails <, > and == alike, so a NaN position walks
    // straight through the tile bounce above, splats its brush quad at an
    // arbitrary place on the canvas, and seeds the trail with NaN. getBlur()
    // is a 5-tap kernel, so one NaN texel turns its neighbours NaN every
    // frame: the hole grows and never heals, which is what the black boxes
    // are. Searching physics makes this reachable - extreme parameter draws
    // overflow the velocity integration - but a hand-cranked slider can do it
    // too, so the guard is unconditional.
    //
    // Written as a magnitude bound rather than isnan()/isinf(): the bound is
    // false for NaN, false for Inf, and also catches finite-but-absurd values
    // that would overflow on the next step. Drivers compiling with fast-math
    // assumptions may fold isnan() away; they cannot fold this.
    if(!(all(lessThan(abs(e.pos), vec2(1e6))) &&
         all(lessThan(abs(e.vel), vec2(1e6))) &&
         abs(e.size) < 1e6 &&
         all(lessThan(abs(e.color), vec4(1e6))))){
        reset(index);
        return;
    }

    //Commit new entity state to buffers
    entities[index]=e;


}
