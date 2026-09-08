#version 330 core
// The projector's display pass. One full-framebuffer quad: the warp is a
// property of the fragment, not of these vertices, so nothing here moves.
in vec2 in_position;
out vec2 disp_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    // Normalised display space, v = 0 at the BOTTOM. GL coordinates.
    disp_uv = in_position * 0.5 + 0.5;
}
