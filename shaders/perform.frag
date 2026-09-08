#version 330 core
// Corner pinning: sample the frame through the inverse homography, so the
// picture lands on an arbitrary quadrilateral and a skewed projector comes
// out square on the wall.
in vec2 disp_uv;
out vec4 fragColor;

uniform sampler2D view_tex;
uniform mat3 inv_h;          // display -> source. Column-major; see the writer.
uniform vec2 fb_size;        // pixels, for round markers on any aspect
uniform int show_guides;     // 0 unless calibration mode is on
uniform int held_corner;     // -1 when the pointer holds none
uniform vec2 corners[4];     // TL, TR, BR, BL in display space

const float GRID_DIV = 8.0;
const vec3 GRID_RGB = vec3(0.15, 1.0, 0.45);
const vec3 EDGE_RGB = vec3(1.0, 1.0, 1.0);
const vec3 MARK_RGB = vec3(0.25, 0.75, 1.0);
const vec3 HELD_RGB = vec3(1.0, 0.85, 0.15);

// A line one pixel wide wherever `v` crosses a whole number, whatever the
// warp does to the spacing. fwidth is what keeps it a constant width on
// screen rather than a constant width in source space.
float gridline(vec2 v) {
    vec2 d = abs(fract(v + 0.5) - 0.5) / max(fwidth(v), vec2(1e-6));
    return 1.0 - min(min(d.x, d.y), 1.0);
}

void main() {
    vec3 p = inv_h * vec3(disp_uv, 1.0);

    // The sign of w is the vanishing line. Beyond it the map folds back and
    // would draw a mirrored ghost of the picture; inverse_homography fixes
    // the sign so that the quad itself is positive.
    bool covered = p.z > 0.0;
    vec2 src = covered ? p.xy / p.z : vec2(0.0);
    covered = covered
           && all(greaterThanEqual(src, vec2(0.0)))
           && all(lessThanEqual(src, vec2(1.0)));

    vec3 col = covered ? texture(view_tex, src).rgb : vec3(0.0);

    if (show_guides == 1) {
        if (covered) {
            // Drawn in SOURCE space and warped with the picture: a line that
            // looks straight against the wall IS the alignment.
            col = mix(col, GRID_RGB, gridline(src * GRID_DIV) * 0.7);
            vec2 e = min(src, 1.0 - src) / max(fwidth(src), vec2(1e-6));
            col = mix(col, EDGE_RGB, 1.0 - min(min(e.x, e.y), 1.0));
        }
        // Markers sit in DISPLAY space so a corner dragged off the picture
        // can still be found. Circles on any aspect, hence fb_size.
        for (int i = 0; i < 4; ++i) {
            float d = length((disp_uv - corners[i]) * fb_size);
            float ring = (1.0 - smoothstep(7.0, 9.0, d))
                       * smoothstep(3.0, 5.0, d);
            col = mix(col, i == held_corner ? HELD_RGB : MARK_RGB, ring);
        }
    }

    fragColor = vec4(col, 1.0);
}
