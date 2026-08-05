"""Blit a rectangle of the assembled view into the square capture framebuffer.

The assembled view is window-shaped, but CLIP needs the tournament grid as an
exact square. Rather than reverse-engineering the camera projection, the source
rectangle is derived from the camera's own tex_to_screen(), so the crop stays
correct at any window size, zoom or pan.
"""
from __future__ import annotations

VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
uniform vec2 src_lo;   // source rect in 0..1 texture space
uniform vec2 src_hi;
void main() {
    vec2 t = in_pos * 0.5 + 0.5;          // 0..1 across the target quad
    uv = mix(src_lo, src_hi, t);
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG = """
#version 330
in vec2 uv;
out vec4 frag;
uniform sampler2D src;
void main() {
    // Outside the grid means the camera is showing empty space; black is the
    // honest answer and check_capture() will flag it if it dominates.
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        frag = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    frag = vec4(texture(src, uv).rgb, 1.0);
}
"""


class CaptureBlit:
    def __init__(self, ctx):
        import numpy as np

        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype="f4")
        self.vbo = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])

    def draw(self, src_tex, src_lo, src_hi) -> None:
        src_tex.use(location=0)
        self.prog["src"] = 0
        self.prog["src_lo"] = tuple(float(v) for v in src_lo)
        self.prog["src_hi"] = tuple(float(v) for v in src_hi)
        self.vao.render()

    def release(self) -> None:
        for obj in (self.vao, self.vbo, self.prog):
            try:
                obj.release()
            except Exception:
                pass
