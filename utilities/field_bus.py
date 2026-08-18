"""The field injection bus: owns the destination texture and composites into it.

The bus is stateless per frame. Every destination is cleared and rebuilt from
the stack, so a disabled layer contributes nothing without anything having to
undo it. A source that needs memory owns its own buffer.
"""
from __future__ import annotations

import moderngl

from state.field_stack import MAPPINGS
from utilities.gl_helpers import read_shader, tryset

BLEND_STATE = {
    "replace":  None,
    "add":      (moderngl.ONE, moderngl.ONE, moderngl.FUNC_ADD),
    "multiply": (moderngl.DST_COLOR, moderngl.ZERO, moderngl.FUNC_ADD),
    "max":      (moderngl.ONE, moderngl.ONE, moderngl.MAX),
}


class FieldBus:
    """Owns the force/strafe destination texture and the composite pass."""

    # force is .xy, strafe is .zw of one RGBA32F texture, so entity_update's
    # get_field() is unchanged by this feature.
    DEST_CHANNELS = {
        "force":  (True, True, False, False),
        "strafe": (False, False, True, True),
    }
    SCALAR_DESTINATIONS = ()

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._res = (0, 0)
        self._tex = None
        self._fbo = None
        self._composite = None
        self._composite_vao = None
        self._scratch = None
        self._scratch_fbo = None

    # -- public ---------------------------------------------------------

    @property
    def field_texture(self):
        return self._tex

    @property
    def resolution(self) -> tuple[int, int]:
        return self._res

    def ensure(self, canvas_width: int, canvas_height: int, scale: float) -> None:
        """Create or resize the destination texture. Never zero-sized."""
        w = max(1, int(canvas_width * scale))
        h = max(1, int(canvas_height * scale))
        if self._tex is not None and self._res == (w, h):
            return
        self._release_target()
        self._res = (w, h)
        tex = self.ctx.texture((w, h), 4, dtype="f4")
        # LINEAR because the bus runs below canvas resolution by default and
        # nearest sampling is visibly blocky there.
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._tex = tex
        self._fbo = self.ctx.framebuffer(color_attachments=[tex])
        self._fbo.clear()
        self._ensure_composite()

    def clear(self) -> None:
        if self._fbo is None:
            return
        self._fbo.color_mask = (True, True, True, True)
        self._fbo.clear()

    def composite_one(self, src_tex, layer) -> None:
        """Map `src_tex` through `layer` and blend it into the destination."""
        if self._fbo is None or src_tex is None:
            return
        self._ensure_composite()

        if layer.blur > 0.0:
            src_tex.build_mipmaps()
        src_tex.use(location=0)

        prog = self._composite
        tryset(prog, "src", 0)
        tryset(prog, "mapping", MAPPINGS.index(layer.mapping))
        tryset(prog, "strength", float(layer.strength))
        tryset(prog, "sign_mul", float(layer.sign))
        tryset(prog, "blur_lod", float(layer.blur))
        tryset(prog, "texel", (1.0 / self._res[0], 1.0 / self._res[1]))
        tryset(prog, "scalar_out", layer.destination in self.SCALAR_DESTINATIONS)

        # The mask is set BEFORE use(): moderngl applies a framebuffer's stored
        # state when it is bound, so a mask set afterwards misses this pass.
        self._fbo.color_mask = self.DEST_CHANNELS[layer.destination]
        self._fbo.use()

        state = BLEND_STATE[layer.blend]
        if state is None:
            self.ctx.disable(moderngl.BLEND)
        else:
            src_factor, dst_factor, equation = state
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = src_factor, dst_factor
            self.ctx.blend_equation = equation

        self._composite_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self._fbo.color_mask = (True, True, True, True)

    @property
    def scratch_texture(self):
        """The shared target procedural sources render into.

        Safe to share because a layer's composite consumes it immediately
        after that layer's own evaluation.
        """
        return self._scratch

    def render_into_scratch(self, vao) -> moderngl.Texture:
        """Render a source's fullscreen quad into scratch and return it."""
        self._ensure_scratch()
        self._scratch_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)
        return self._scratch

    def cleanup(self) -> None:
        self._release_target()
        if self._composite_vao is not None:
            self._composite_vao.release()
            self._composite_vao = None
        if self._composite is not None:
            self._composite.release()
            self._composite = None

    # -- internal -------------------------------------------------------

    def _ensure_composite(self) -> None:
        if self._composite is not None:
            return
        self._composite = self.ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=read_shader("shaders/field/composite.frag"),
        )
        self._composite_vao = self.ctx.vertex_array(self._composite, [])

    def _ensure_scratch(self) -> None:
        if self._scratch is not None:
            return
        w, h = self._res
        tex = self.ctx.texture((w, h), 4, dtype="f4")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._scratch = tex
        self._scratch_fbo = self.ctx.framebuffer(color_attachments=[tex])

    def _release_target(self) -> None:
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        if self._tex is not None:
            self._tex.release()
            self._tex = None
        if self._scratch_fbo is not None:
            self._scratch_fbo.release()
            self._scratch_fbo = None
        if self._scratch is not None:
            self._scratch.release()
            self._scratch = None
        self._res = (0, 0)
