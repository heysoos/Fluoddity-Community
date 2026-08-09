#version 430 
        
uniform vec2 canvas_resolution;
//SYNC WITH ENTITY_UPDATE.GLSL AND CAM_BRUSH.VERT
struct Entity {
    vec2 pos;
    vec2 vel;
    float size;
    float cohort;      // Normalized cohort value (0-1) for parameter sweep calculations
    float padding[2];  // Align to 16-byte boundary for vec4
    vec4 color;
};  // Total: 48 bytes (12 floats)
layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};

// Tournament tiling: deposits must not spill into a neighbouring tile.
uniform int TOURNAMENT_MODE;      // 0 = off, 1 = on
uniform int TOURNAMENT_GRID;      // grid side length
uniform float TOURNAMENT_ACTIVE;  // active particle count (matches ACTIVE_COUNT)

// First texel of tile k along one axis. SYNCHRONIZED with entity_update.glsl
// and canvas.frag - see the note there for why this is integer arithmetic.
int tile_lo_texel(int k, int g, int res){
    if(k <= 0) return 0;
    if(k >= g) return res;
    int b = 2 * g;
    return (2 * k * res - g + b - 1) / b;        // ceil division, exact
}

out vec2 uv;
out vec4 pos_vel;
out vec4 view_col;
out vec2 frag_world;         // entity-space position of this fragment
flat out vec2 tile_lo;       // home tile bounds (entity space)
flat out vec2 tile_hi;
void main() {
    int instance_id = gl_InstanceID;
    int vertex_id = gl_VertexID;
    
    // Read entity position
    vec2 entity_pos = entities[instance_id].pos;
    vec2 entity_vel = entities[instance_id].vel;
    // Generate quad vertices based on vertex_id (0-3)
    // Create small square centered at entity position
    float size = entities[instance_id].size;
    vec2 offsets[4] = vec2[](
        vec2(-size, -size),  // bottom-left
        vec2( size, -size),  // bottom-right
        vec2( size,  size),  // top-right
        vec2(-size,  size)   // top-left
    );
    vec2 uv_coords[4] = vec2[](
        vec2(0,0),
        vec2(1,0),
        vec2(1,1),
        vec2(0,1)
    );
    vec2 particle_uv=uv_coords[vertex_id];
    vec2 vertex_pos = entity_pos + offsets[vertex_id];
    
    // Entity space to clip space: entity bounds [-x_edge,x_edge]x[-y_edge,y_edge] -> [-1,1]^2
    float ca = canvas_resolution.x / canvas_resolution.y;
    gl_Position = vec4(vertex_pos, 0.0, 1.0) * vec4(1.0/sqrt(ca), sqrt(ca), 1, 1);

    uv = particle_uv;
    pos_vel=vec4(entity_pos,entity_vel);
    view_col=entities[instance_id].color;

    // Home tile box, matching tournament_home_tile()/tournament_tile_box() in
    // entity_update.glsl. The fragment stage clips deposits to this box.
    frag_world = vertex_pos;
    tile_lo = vec2(-1e9);
    tile_hi = vec2(1e9);
    if (TOURNAMENT_MODE == 1) {
        int n = TOURNAMENT_GRID * TOURNAMENT_GRID;
        int tile = clamp(int(floor(float(instance_id) / TOURNAMENT_ACTIVE * float(n))), 0, n - 1);
        vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
        // Seams on texel edges, in integer arithmetic, matching
        // tile_lo_texel() in entity_update.glsl and canvas.frag. An even
        // division puts the seam inside a texel whenever the canvas does not
        // divide by the grid, and at the default world size it is 647 texels
        // across against a grid of up to 8.
        ivec2 res = ivec2(canvas_resolution);
        ivec2 k = ivec2(tile % TOURNAMENT_GRID, tile / TOURNAMENT_GRID);
        vec2 lo_uv = vec2(tile_lo_texel(k.x, TOURNAMENT_GRID, res.x),
                          tile_lo_texel(k.y, TOURNAMENT_GRID, res.y)) / canvas_resolution;
        vec2 hi_uv = vec2(tile_lo_texel(k.x + 1, TOURNAMENT_GRID, res.x),
                          tile_lo_texel(k.y + 1, TOURNAMENT_GRID, res.y)) / canvas_resolution;
        tile_lo = (2.0 * lo_uv - 1.0) * half_extent;
        tile_hi = (2.0 * hi_uv - 1.0) * half_extent;
    }
}