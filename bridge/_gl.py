"""Minimal GL constants and a VBO-less fullscreen blit.

Neither Fluoddity nor Fluoddity-Core depends on PyOpenGL, so the handful of
GL enums Spout needs are spelled out here rather than imported.
"""
import moderngl

# From <GL/gl.h>. Spout's sendTexture/receiveTexture take a texture target.
GL_TEXTURE_2D = 0x0DE1


# Fullscreen quad from gl_VertexID alone — no VBO, no attributes.
# Rendered as TRIANGLE_FAN with 4 vertices:
#   0 -> (-1,-1)   1 -> (1,-1)   2 -> (1,1)   3 -> (-1,1)
_BLIT_VERT = """
#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID == 1 || gl_VertexID == 2) ? 1.0 : -1.0,
                  (gl_VertexID >= 2) ? 1.0 : -1.0);
    uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_BLIT_FRAG = """
#version 330
uniform sampler2D src;
uniform bool force_opaque;
in vec2 uv;
out vec4 frag;
void main() {
    vec4 c = clamp(texture(src, uv), 0.0, 1.0);
    frag = force_opaque ? vec4(c.rgb, 1.0) : c;
}
"""


class Blitter:
    """Draws a source texture into whatever framebuffer is bound.

    Used to convert a float32 render target into the 8-bit texture Spout
    shares, and to resample between differing resolutions.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_BLIT_VERT, fragment_shader=_BLIT_FRAG)
        self.vao = ctx.vertex_array(self.program, [])

    def blit(self, texture: moderngl.Texture, framebuffer: moderngl.Framebuffer,
             force_opaque: bool = True, linear: bool = True) -> None:
        prev_filter = texture.filter
        if linear:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        framebuffer.use()  # also sets the viewport to the framebuffer size
        self.ctx.disable(moderngl.BLEND)
        texture.use(location=0)
        self.program["src"].value = 0
        self.program["force_opaque"].value = force_opaque
        self.vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        texture.filter = prev_filter

    def release(self) -> None:
        self.vao.release()
        self.program.release()
