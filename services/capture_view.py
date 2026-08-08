"""Render the tournament grid from the CANVAS, not from the window view.

The capture used to crop the on-screen image, because in cam_brush_mode the
particles are only ever composited over the trails in a window-shaped target -
sim.view_tex holds trails alone, so the finished picture existed nowhere else.
That forced the capture to reverse-engineer where the canvas sat inside the
window, and every crop bug so far has been that rect disagreeing with the
texture it was cropping: first a frame of camera lag, then a top-down/bottom-up
mirror that put 20% of each tile's neighbour into its crop and blacked out the
whole bottom row.

cam_brush.vert takes window_size as a uniform, so the same particle pass can be
run into a square target with an identity camera. The canvas then fills that
target exactly. There is no crop rect at all, which is the point - the class of
bug is gone rather than fixed again.

Two further consequences, both improvements:

  - the capture no longer depends on where the user is looking. Pan, zoom and
    window size cannot change what the optimizer scores.
  - it is rendered at exactly grid*224, so CLIP gets its native resolution with
    no resampling, and a run is reproducible across window sizes. Previously
    the sampling density of the capture was whatever the window happened to be.
"""
from __future__ import annotations

import moderngl

from utilities.gl_helpers import tryset


class CaptureView:
    """Owns the offscreen canvas render used for scoring. GL-dependent."""

    def __init__(self, ctx, sim, camera):
        self.ctx = ctx
        self.sim = sim
        self.camera = camera
        self._side = 0
        self._tex = None
        self._fbo = None
        self._assembler = None
        self._bloom = None

    @property
    def side(self) -> int:
        return self._side

    def resize(self, side: int) -> None:
        side = max(1, int(side))
        if side == self._side and self._fbo is not None:
            return
        self._release_target()
        self._side = side
        # f4, matching cam_brush_target: frame assembly expects pre-gamma HDR
        # values and clamps them itself.
        self._tex = self.ctx.texture((side, side), 4, dtype="f4")
        self._fbo = self.ctx.framebuffer(color_attachments=[self._tex])
        # Its own assembler: assemble_frame rebuilds its resources whenever the
        # input size changes, so sharing the camera's would thrash it between
        # window-sized and capture-sized every frame.
        from utilities.frame_assembler import FrameAssembler

        self._assembler = FrameAssembler(self.ctx, self._tex)

    def render(self, ui_state, assemble_kwargs, side):
        """-> the finished square texture, or None if it could not be built."""
        self.resize(side)
        src = self._render_particles(ui_state)
        if src is None:
            return None

        out = self._assembler.assemble_frame(
            src, total_samples=1, current_sample_index=0,
            **self._capture_kwargs(assemble_kwargs))
        if out is None:
            return None

        prefs = ui_state.preferences
        if prefs.bloom_enabled and not ui_state.sim.watercolor_mode:
            if self._bloom is None:
                from utilities.bloom import BloomProcessor

                self._bloom = BloomProcessor(self.ctx)
            out = self._bloom.process(
                out, prefs.bloom_threshold, prefs.bloom_intensity,
                prefs.bloom_radius, tonemap_softness=prefs.tonemap_softness)
        return out

    def _render_particles(self, ui_state):
        """The cam_brush pass with an IDENTITY camera.

        cam_pos 0 and cam_zoom 1 against a square window_size make the vertex
        shader's letterbox scale exactly (1, 1) for a square canvas, so entity
        space maps onto the target one-to-one.
        """
        if not self.camera.cam_brush_mode:
            # The other view modes are the canvas texture itself; there is no
            # separate particle pass to redo.
            return self.sim.view_tex

        prog = self.camera.cam_brush_program
        self._fbo.use()
        self.ctx.viewport = (0, 0, self._side, self._side)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        prog["cam_pos"].value = (0.0, 0.0)
        prog["cam_zoom"].value = 1.0
        prog["canvas_resolution"].value = self.sim.view_tex.size
        prog["window_size"].value = (float(self._side), float(self._side))
        tryset(prog, "WATERCOLOR_MODE", ui_state.sim.watercolor_mode)
        # Tiling repeats the canvas across the VIEW; the capture is exactly one
        # canvas, so there is nothing to repeat.
        tryset(prog, "tiling_mode_enabled", False)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.camera.cam_brush_vao.render(
            mode=moderngl.TRIANGLE_FAN, instances=self.sim.entity_count,
            vertices=4)
        self.ctx.disable(moderngl.BLEND)
        return self._tex

    @staticmethod
    def _capture_kwargs(assemble_kwargs) -> dict:
        """The shared assembly kwargs, with every camera-dependent entry
        neutralised. Anything left pointing at the live camera would put the
        user's viewpoint back into the score."""
        kw = dict(assemble_kwargs)
        kw.update(
            screen_aspect=1.0,
            camera_position=(0.0, 0.0),
            camera_zoom=1.0,
            tiling_mode=False,
            view_min=(0.0, 0.0),
            view_max=(0.0, 0.0),
            tiling_scale=(1.0, 1.0),
            # UI overlays are not part of the creature.
            sweep_mode=False,
            sweep_reticle_visible=False,
            trail_draw_radius=0.0,
            draw_target_overlay_opacity=0.0,
        )
        return kw

    def _release_target(self) -> None:
        for obj in (self._fbo, self._tex):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._fbo = None
        self._tex = None
        if self._assembler is not None:
            try:
                self._assembler.cleanup_resources()
            except Exception:
                pass
            self._assembler = None

    def release(self) -> None:
        self._release_target()
        self._side = 0
        if self._bloom is not None:
            try:
                self._bloom.cleanup()
            except Exception:
                pass
            self._bloom = None
