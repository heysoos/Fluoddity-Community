"""Datamosh post-process: particle motion displaces the incoming texture.

One job: take the incoming Spout frame and drag its pixels along the flow the
simulation is already producing, instead of drawing particles over it. It knows
nothing about the simulation beyond the two velocity textures it is handed.

Two passes. The first condenses those two velocity maps into one field in
canvas-uv units (`shaders/datamosh_flow.frag`) and builds a mip chain over it;
the second does the displacement (`shaders/datamosh.frag`). The chain is what
makes the controls behave: its top level is the frame's mean displacement
magnitude, which the response curve is normalized against, and its lower levels
are a free blur of the field, which is what stroke size is.

GPU resources are created lazily and released the moment the mode is switched
off, so the default configuration costs nothing -- not a pass, not a texture.

Sizing. The ping-pong pair matches the *source* resolution, so the feed comes
through pixel-for-pixel and goes back out to vvvv at the resolution it arrived
at. RGBA16F rather than RGBA8: the buffer is resampled every frame and the
feedback path would quantize itself to mush in 8 bits.
"""

import math

import moderngl

from utilities.gl_helpers import read_shader, tryset

MIN_SIZE = 16

# The flow field is a smooth, low-frequency thing next to a 240k-particle sim,
# and halving it also shortens the mip chain it carries.
FLOW_DOWNSCALE = 2
FLOW_MIN_SIZE = 32


class Datamosh:
    """Owns the feedback buffers and the passes that fill them.

    Args:
        ctx: the ModernGL context. Resources are created on the GL thread on
            first :meth:`process`.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._resources = None
        self._flow = None
        self._size = (0, 0)
        self._flow_size = (0, 0)
        self._read_index = 0
        self._needs_reseed = True

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def reseed(self):
        """Refill the buffer from the live source on the next pass.

        The way back from a fully melted image: with mosh_refresh at 0 nothing
        resets on its own, which is the point of that setting and also the
        reason this exists.
        """
        self._needs_reseed = True

    def process(self, particle_texture, external_texture,
                flow_canvas, flow_brush, canvas_size, params):
        """Run one datamosh step and return the moshed frame.

        Call once per *displayed* frame. Running it per motion-blur
        accumulation sample would advect the image several times per frame and
        make the smear rate depend on the blur settings.

        Args:
            particle_texture: the assembled particle frame. Used as the source
                in particle mode, as the fallback when no sender is connected,
                and as the crisp overlay when ``mosh_ink`` is up.
            external_texture: the incoming Spout texture, or None.
            flow_canvas: the persistent trail map (RG velocity).
            flow_brush: this frame's velocity splat (RG velocity).
            canvas_size: ``(width, height)`` of those two, for the aspect fit.
            params: a :class:`state.render_params.MoshParams`.

        Returns:
            The moshed texture, or None when the mode is inactive -- in which
            case the caller keeps whatever it already had.
        """
        if not params.active:
            self.cleanup()
            return None

        source = external_texture if external_texture is not None else particle_texture
        if source is None:
            return None

        width, height = source.size
        width = max(int(width), MIN_SIZE)
        height = max(int(height), MIN_SIZE)

        previous_fbo = self.ctx.fbo
        self._ensure_resources(width, height)
        cw, ch = canvas_size
        aspect = max(cw, 1) / max(ch, 1)
        self._run_flow_pass(flow_canvas, flow_brush, cw, ch, aspect, params)
        r = self._resources

        read_tex = r["textures"][self._read_index]
        write_index = 1 - self._read_index
        r["fbos"][write_index].use()
        self.ctx.viewport = (0, 0, width, height)

        read_tex.use(location=0)
        tryset(r["program"], "prev_tex", 0)
        if external_texture is not None:
            external_texture.use(location=1)
        tryset(r["program"], "source_tex", 1)
        if particle_texture is not None:
            particle_texture.use(location=2)
        tryset(r["program"], "particle_tex", 2)
        self._flow["tex"].use(location=3)
        tryset(r["program"], "flow_tex", 3)

        tryset(r["program"], "flow_cover", self._cover(aspect, width / height))
        tryset(r["program"], "flow_max_lod", self._flow["max_lod"])
        tryset(r["program"], "source_resolution", (float(width), float(height)))
        tryset(r["program"], "source_connected", external_texture is not None)
        tryset(r["program"], "mosh_source", int(params.source))
        tryset(r["program"], "mosh_amount", params.amount)
        tryset(r["program"], "mosh_contrast", params.contrast)
        tryset(r["program"], "mosh_scale", params.scale)
        tryset(r["program"], "mosh_swirl", params.swirl)
        # A reseed is just one frame at full refresh: the live source replaces
        # the buffer outright, and the next frame carries on from there.
        tryset(r["program"], "mosh_refresh",
               1.0 if self._needs_reseed else params.refresh)
        tryset(r["program"], "mosh_block", params.block)
        tryset(r["program"], "mosh_chroma", params.chroma)
        tryset(r["program"], "mosh_ink", params.ink)
        self._needs_reseed = False

        self.ctx.disable(moderngl.BLEND)
        r["vao"].render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        self._read_index = write_index
        self._restore(previous_fbo)
        return r["textures"][write_index]

    def cleanup(self):
        """Release all GPU resources. Safe to call when there are none."""
        self._release_flow()
        if self._resources is None:
            return
        self._release_buffers()
        # Whatever comes back after this has to start from the live frame, not
        # from a buffer that no longer exists.
        self._needs_reseed = True

    # ------------------------------------------------------------------

    def _run_flow_pass(self, flow_canvas, flow_brush, canvas_w, canvas_h,
                       aspect, params):
        """Build the mipped flow field the displacement pass reads."""
        width = max(int(canvas_w) // FLOW_DOWNSCALE, FLOW_MIN_SIZE)
        height = max(int(canvas_h) // FLOW_DOWNSCALE, FLOW_MIN_SIZE)
        self._ensure_flow(width, height)
        f = self._flow

        f["fbo"].use()
        self.ctx.viewport = (0, 0, width, height)

        flow_canvas.use(location=4)
        tryset(f["program"], "flow_can_tex", 4)
        flow_brush.use(location=5)
        tryset(f["program"], "flow_brush_tex", 5)
        # Entity space spans +-sqrt(ca) in x and +-1/sqrt(ca) in y; brush.vert
        # is where that mapping is defined, and the shader undoes it to turn a
        # velocity into a uv offset.
        x_edge = math.sqrt(aspect)
        tryset(f["program"], "canvas_edges", (x_edge, 1.0 / x_edge))
        tryset(f["program"], "mosh_flow_mix", params.flow_mix)

        self.ctx.disable(moderngl.BLEND)
        f["vao"].render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        # The chain is the point of this pass: its top level is the frame's
        # mean magnitude -- the response curve's reference -- and the levels
        # below it are the stroke-size blur.
        f["tex"].build_mipmaps()

    def _restore(self, fbo):
        """Rebind whatever was bound on entry, or the screen if it is gone.

        A resize releases this object's own framebuffers, and the one bound on
        entry can be one of them -- restoring a released framebuffer raises.
        Falling back to the screen keeps a resize from taking the app down.
        """
        try:
            if fbo is not None:
                fbo.use()
                return
        except Exception:
            pass
        if self.ctx.screen is not None:
            self.ctx.screen.use()

    @staticmethod
    def _cover(flow_aspect: float, output_aspect: float) -> tuple[float, float]:
        """Scale factors that fit the flow field over the output, cropping.

        Cover rather than stretch so displacement angles survive: a 1:1 canvas
        stretched onto a 16:9 feed would send diagonals off at the wrong angle,
        which is precisely what the eye picks up in a directional effect.
        """
        if flow_aspect > output_aspect:
            return (output_aspect / flow_aspect, 1.0)
        return (1.0, flow_aspect / output_aspect)

    def _release_flow(self):
        if self._flow is None:
            return
        f = self._flow
        f["fbo"].release()
        f["tex"].release()
        f["vao"].release()
        f["program"].release()
        self._flow = None
        self._flow_size = (0, 0)

    def _release_buffers(self):
        r = self._resources
        for fbo in r["fbos"]:
            fbo.release()
        for tex in r["textures"]:
            tex.release()
        r["vao"].release()
        r["program"].release()
        self._resources = None
        self._size = (0, 0)

    def _ensure_flow(self, width, height):
        if self._flow is not None and self._flow_size == (width, height):
            return
        self._release_flow()
        self._flow_size = (width, height)

        ctx = self.ctx
        program = ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=read_shader("shaders/datamosh_flow.frag"),
        )
        vao = ctx.vertex_array(program, [])
        tex = ctx.texture((width, height), 4, dtype="f2")
        # Trilinear: stroke size is a fractional mip level, so it has to
        # interpolate between levels or the control would step.
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.build_mipmaps()
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.clear(0.0, 0.0, 0.0, 1.0)

        self._flow = dict(program=program, vao=vao, tex=tex, fbo=fbo,
                          max_lod=float(int(math.log2(max(width, height)))))

    def _ensure_resources(self, width, height):
        if self._resources is not None and self._size == (width, height):
            return
        if self._resources is not None:
            self._release_buffers()
        self._size = (width, height)

        ctx = self.ctx
        program = ctx.program(
            # canvas.vert builds the fullscreen quad from gl_VertexID, so there
            # is no VBO -- same trick particle_mask.py uses.
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=read_shader("shaders/datamosh.frag"),
        )
        vao = ctx.vertex_array(program, [])

        textures = []
        fbos = []
        for _ in range(2):
            tex = ctx.texture((width, height), 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            # Clamped, not repeating: the smear runs off the edge of the frame
            # rather than dragging the opposite edge back in. The canvas wraps,
            # but the incoming video does not.
            tex.repeat_x = False
            tex.repeat_y = False
            fbo = ctx.framebuffer(color_attachments=[tex])
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            textures.append(tex)
            fbos.append(fbo)

        self._read_index = 0
        self._needs_reseed = True
        self._resources = dict(program=program, vao=vao,
                               textures=textures, fbos=fbos)
