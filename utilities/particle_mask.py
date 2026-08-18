"""Per-region activity mask for the particle simulation.

One job: turn an incoming texture (and/or a built-in vignette) into a small
field saying how much activity each part of the canvas should get. It knows
nothing about the simulation -- `entity_update.glsl` samples the result and
decides what to do with it.

GPU resources are created lazily on first use and released when the mask is
switched off, so the default configuration costs nothing at all.

Sizing. The target is the canvas divided by :data:`DOWNSCALE`, which keeps its
aspect ratio identical to the canvas. That is what lets `entity_update` reuse
`get_field()`'s canvas-space-to-UV maths verbatim instead of carrying a second
mapping that could drift out of step with it.
"""

import moderngl

from utilities.gl_helpers import read_shader, tryset

# The mask is a low-frequency field; quarter resolution is plenty and keeps
# the blur taps and the per-frame pass cheap next to a 240k-particle sim.
DOWNSCALE = 4
MIN_SIZE = 16


class ParticleMask:
    """Owns the mask texture and the pass that fills it.

    Args:
        ctx: the ModernGL context. Resources are created on the GL thread on
            first :meth:`update`.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._resources = None
        self._size = (0, 0)

    @property
    def texture(self):
        """The mask texture, or None if the mask has never been generated."""
        if self._resources is None:
            return None
        return self._resources["tex"]

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def update(self, canvas_width: int, canvas_height: int, params,
               external_texture=None):
        """Regenerate the mask. Call once per rendered frame.

        Args:
            canvas_width, canvas_height: the simulation canvas dimensions. The
                mask target is derived from these so their aspects match.
            params: a :class:`state.render_params.MaskParams`.
            external_texture: the incoming Spout texture, or None. Used
                regardless of whether ``spout.frag`` is the selected field
                override -- the field decides where particles go, the mask
                decides how much they do there, and the two are independent.

        Returns:
            The mask texture, or None when the mask is inactive.
        """
        if not params.active:
            # Nothing would read it. Free the resources rather than keeping a
            # stale texture and a per-frame pass alive.
            self.cleanup()
            return None

        width = max(canvas_width // DOWNSCALE, MIN_SIZE)
        height = max(canvas_height // DOWNSCALE, MIN_SIZE)
        self._ensure_resources(width, height)
        r = self._resources

        previous_fbo = self.ctx.fbo
        r["fbo"].use()

        if external_texture is not None:
            # Unit 6 belongs to the mask in entity_update; using it here too
            # keeps the two ends of this feature on one unit. The render target
            # is the mask FBO, so there is no read/write conflict.
            external_texture.use(location=6)
        tryset(r["program"], "external_tex", 6)
        tryset(r["program"], "mask_connected", external_texture is not None)
        tryset(r["program"], "mask_resolution", (width, height))
        tryset(r["program"], "mask_floor", params.floor)
        tryset(r["program"], "mask_gamma", params.gamma)
        tryset(r["program"], "mask_blur", params.blur)
        tryset(r["program"], "mask_vignette", params.vignette)
        tryset(r["program"], "mask_vignette_softness", params.vignette_softness)
        tryset(r["program"], "mask_source", int(params.source))

        self.ctx.disable(moderngl.BLEND)
        r["vao"].render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        previous_fbo.use()
        return r["tex"]

    def cleanup(self):
        """Release all GPU resources. Safe to call when there are none."""
        if self._resources is None:
            return
        r = self._resources
        r["fbo"].release()
        r["tex"].release()
        r["vao"].release()
        r["program"].release()
        self._resources = None
        self._size = (0, 0)

    # ------------------------------------------------------------------

    def _ensure_resources(self, width, height):
        if self._resources is not None and self._size == (width, height):
            return
        self.cleanup()
        self._size = (width, height)

        ctx = self.ctx
        program = ctx.program(
            # canvas.vert generates a fullscreen quad from gl_VertexID, so no
            # VBO is needed -- same trick advanced_drawing.py uses.
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=read_shader("shaders/particle_mask.frag"),
        )
        vao = ctx.vertex_array(program, [])

        # f2 rather than f4: the mask is a smooth 0..1 field plus a small
        # gradient, so half precision is ample and halves the sampling cost.
        tex = ctx.texture((width, height), 4, dtype="f2")
        # Linear so per-particle sampling interpolates instead of showing the
        # quarter-res grid; clamped so the gradient does not wrap at the edges.
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.clear(1.0, 1.0, 0.0, 0.0)  # full activity until the first pass runs

        self._resources = dict(program=program, vao=vao, tex=tex, fbo=fbo)
