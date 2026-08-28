"""Render the projector's own frame, at a FIXED viewpoint.

Perform mode shows the whole canvas filling the display, whatever the laptop's
camera is doing - so it cannot mirror the laptop's picture and has to redo the
particle pass with an identity camera, exactly as services/capture_view.py does
for the tournament capture.

`cam_brush.vert` takes `window_size` as a uniform, so the same particle pass
runs into a target of any shape: at cam_pos 0 and cam_zoom 1 the vertex
shader's own letterbox fits the canvas into that shape with no distortion and
no crop rect.

The cost is one more particle pass per motion-blur sample - the same work the
laptop's view already does, and the reason the projector's framing is fixed
rather than a second camera anyone can steer.
"""
from __future__ import annotations

import moderngl

from camera import DisplayFrame, tiling_scale, tiling_view_bounds
from utilities.gl_helpers import tryset


class PerformView:
    """Owns the offscreen render the perform window displays. GL-dependent."""

    def __init__(self, ctx, sim, camera):
        self.ctx = ctx
        self.sim = sim
        self.camera = camera
        self._size = (0, 0)
        self._tex = None
        self._fbo = None
        self._assembler = None
        self._bloom = None
        self.frame = None          # the DisplayFrame the perform window draws

    @property
    def size(self) -> tuple:
        return self._size

    def resize(self, size) -> None:
        w, h = max(1, int(size[0])), max(1, int(size[1]))
        if (w, h) == self._size and self._fbo is not None:
            return
        self._release_target()
        self._size = (w, h)
        # f4, matching cam_brush_target: frame assembly expects pre-gamma HDR
        # values and clamps them itself.
        self._tex = self.ctx.texture((w, h), 4, dtype="f4")
        self._fbo = self.ctx.framebuffer(color_attachments=[self._tex])
        # Its own assembler: assemble_frame rebuilds its resources whenever the
        # input size changes, so sharing the camera's would thrash it between
        # window-sized and projector-sized every sample.
        from utilities.frame_assembler import FrameAssembler

        self._assembler = FrameAssembler(self.ctx, self._tex)
        self.frame = None

    def render(self, ui_state, assemble_kwargs, size,
               total_samples: int, sample_index: int) -> None:
        """Accumulate one sample. Publishes `self.frame` on the final one.

        Driven from the SAME loop as the laptop's view and with the same
        sample counts, so the projector gets the identical motion blur. A
        projector rendered once per frame would be the screen with the worse
        picture, which is backwards.
        """
        self.resize(size)
        src = self._render_particles(ui_state)
        assembled = self._assembler.assemble_frame(
            src, total_samples=total_samples, current_sample_index=sample_index,
            **self._fixed_kwargs(assemble_kwargs, ui_state))
        if assembled is None:
            return

        prefs = ui_state.preferences
        if prefs.bloom_enabled and not ui_state.sim.watercolor_mode:
            assembled = self._bloom_of(assembled, prefs)

        # An identity DisplayFrame: this texture is already at the projector's
        # shape and framing, so the perform window's fit is a no-op and its
        # display shader has nothing left to do but blit.
        self.frame = DisplayFrame(
            texture=assembled, cam_pos=(0.0, 0.0), cam_zoom=1.0,
            tex_size=assembled.size, window_size=self._size)

    def _render_particles(self, ui_state):
        """The cam_brush pass with an IDENTITY camera at the projector's shape."""
        if not self.camera.cam_brush_mode:
            # The other view modes are the canvas texture itself; there is no
            # separate particle pass to redo, and no camera baked into it.
            return self.sim.view_tex

        tiling = self._tiling(ui_state)
        prog = self.camera.cam_brush_program
        w, h = self._size
        self._fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        prog["cam_pos"].value = (0.0, 0.0)
        prog["cam_zoom"].value = 1.0
        prog["canvas_resolution"].value = self.sim.view_tex.size
        prog["window_size"].value = (float(w), float(h))
        tryset(prog, "WATERCOLOR_MODE", ui_state.sim.watercolor_mode)
        tryset(prog, "tiling_mode_enabled", tiling)
        if tiling:
            # Tiling repeats the canvas across the VIEW, so the bounds are a
            # property of the viewpoint - and this viewpoint is not the
            # laptop's. Recomputed for the fixed camera at the projector's
            # shape; the laptop's bounds would tile the wrong rectangle.
            view_min, view_max = tiling_view_bounds(
                self._size, self.sim.view_tex.size, (0.0, 0.0), 1.0)
            tryset(prog, "view_min", tuple(view_min))
            tryset(prog, "view_max", tuple(view_max))

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.camera.cam_brush_vao.render(
            mode=moderngl.TRIANGLE_FAN, instances=self.sim.entity_count,
            vertices=4)
        self.ctx.disable(moderngl.BLEND)
        return self._tex

    @staticmethod
    def _tiling(ui_state) -> bool:
        from state import view_modes

        return ui_state.sim.current_view_option == view_modes.CAMERA_TILED

    def _fixed_kwargs(self, assemble_kwargs, ui_state) -> dict:
        """The shared assembly kwargs with every camera-dependent entry
        replaced by the fixed viewpoint's.

        Anything left pointing at the live camera puts the laptop's zoom back
        into the projected picture, which is the whole complaint.
        """
        w, h = self._size
        tiling = self._tiling(ui_state)
        view_min = view_max = (0.0, 0.0)
        if tiling:
            lo, hi = tiling_view_bounds(self._size, self.sim.view_tex.size,
                                        (0.0, 0.0), 1.0)
            view_min, view_max = tuple(lo), tuple(hi)
        kw = dict(assemble_kwargs)
        kw.update(
            screen_aspect=w / max(h, 1),
            camera_position=(0.0, 0.0),
            camera_zoom=1.0,
            tiling_mode=tiling,
            view_min=view_min,
            view_max=view_max,
            tiling_scale=tiling_scale(self._size, self.sim.view_tex.size),
            # The overlays are hidden on both screens while performing, but
            # they are pointer positions in the LAPTOP's window and would land
            # somewhere arbitrary here even so.
            sweep_mode=False,
            sweep_reticle_visible=False,
            trail_draw_radius=0.0,
            draw_target_overlay_opacity=0.0,
        )
        return kw

    def _bloom_of(self, tex, prefs):
        if self._bloom is None:
            from utilities.bloom import BloomProcessor

            self._bloom = BloomProcessor(self.ctx)
        return self._bloom.process(
            tex, prefs.bloom_threshold, prefs.bloom_intensity,
            prefs.bloom_radius, tonemap_softness=prefs.tonemap_softness)

    def _release_target(self) -> None:
        for obj in (self._fbo, self._tex):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._fbo = self._tex = None
        if self._assembler is not None:
            try:
                self._assembler.cleanup()
            except Exception:
                pass
            self._assembler = None

    def cleanup(self) -> None:
        self._release_target()
        self._size = (0, 0)
        self.frame = None
        if self._bloom is not None:
            try:
                self._bloom.cleanup()
            except Exception:
                pass
            self._bloom = None
