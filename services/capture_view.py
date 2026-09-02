"""Render the tournament grid from the CANVAS, not from the window view.

cam_brush.vert takes window_size as a uniform, so the same particle pass can
run into a square target with an identity camera, filling it exactly with no
crop rect. Pan, zoom and window size therefore cannot change what the
optimizer scores, and it renders at exactly grid*224 so CLIP gets its native
resolution with no resampling.
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
        self._tile_tex = None
        self._tile_fbo = None

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

    def render(self, ui_state, assemble_kwargs, side, grayscale=False):
        """-> the assembled square texture, WITHOUT bloom, or None.

        Bloom is deliberately not applied here: it has to run per tile, after
        the grid is split, or a bright creature glows into its neighbours'
        pictures. See draw_grid().
        """
        self.resize(side)
        src = self._render_particles(ui_state, grayscale)
        if src is None:
            return None
        return self._assembler.assemble_frame(
            src, total_samples=1, current_sample_index=0,
            **self._capture_kwargs(assemble_kwargs, grayscale))

    def draw_grid(self, fbo, src, grid, blit, ui_state, tile_px):
        """Fill the bound capture framebuffer, blooming each tile ALONE.

        Blooming the whole grid at once leaks neighbouring tiles' glow into
        every crop, and the optimizer would score that as belonging to the
        creature. Blooming a tile in isolation makes that impossible by
        construction: the pass cannot see anything outside the tile. See
        CLAUDE.md.
        """
        from services.tile_geometry import tile_uv_box

        prefs = ui_state.preferences
        bloom = prefs.bloom_enabled and not ui_state.sim.watercolor_mode
        res = self.sim.view_tex.size
        if bloom:
            self._ensure_tile_target(tile_px)

        for tile in range(grid * grid):
            # Tile 0 is bottom-left, matching tournament_home_tile(), and both
            # the source UVs and the destination viewport are bottom-up here.
            # crop_bounds() applies the one flip, on readback.
            tx, ty = tile % grid, tile // grid
            # The tile's TEXEL box, not tx/grid: a tile owns a whole number of
            # texels (see services/tile_geometry), so at 647/8 they are seven
            # of 81 and one of 80. Cropping at the even split would put a
            # sliver of the neighbour into the picture the optimizer scores.
            lo, hi = tile_uv_box(tx, ty, grid, res)

            if not bloom:
                fbo.use()
                self.ctx.viewport = (tx * tile_px, ty * tile_px,
                                     tile_px, tile_px)
                blit.draw(src, lo, hi)
                continue

            self._tile_fbo.use()
            self.ctx.viewport = (0, 0, tile_px, tile_px)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            blit.draw(src, lo, hi)

            bloomed = self._bloom_of(
                self._tile_tex, prefs, ui_state.sim.watercolor_mode)

            fbo.use()
            self.ctx.viewport = (tx * tile_px, ty * tile_px, tile_px, tile_px)
            blit.draw(bloomed, (0.0, 0.0), (1.0, 1.0))

    def _bloom_of(self, tex, prefs, watercolor):
        if self._bloom is None:
            from utilities.bloom import BloomProcessor

            self._bloom = BloomProcessor(self.ctx)
        return self._bloom.process(
            tex, prefs.bloom_threshold, prefs.bloom_intensity,
            prefs.bloom_radius, tonemap_softness=prefs.tonemap_softness)

    def _ensure_tile_target(self, tile_px):
        if self._tile_tex is not None and self._tile_tex.size == (tile_px, tile_px):
            return
        self._release_tile_target()
        self._tile_tex = self.ctx.texture((tile_px, tile_px), 4, dtype="f4")
        self._tile_fbo = self.ctx.framebuffer(color_attachments=[self._tile_tex])

    def _render_particles(self, ui_state, grayscale=False):
        """The cam_brush pass with an IDENTITY camera.

        cam_pos 0 and cam_zoom 1 against a square window_size make the vertex
        shader's letterbox scale exactly (1, 1) for a square canvas, so entity
        space maps onto the target one-to-one.

        `grayscale` zeroes every particle's saturation for THIS pass only: the
        program is shared with the laptop's own view, so the uniform is put
        back before returning.
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
        tryset(prog, "GRAYSCALE", bool(grayscale))

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.blend_equation = moderngl.FUNC_ADD
        try:
            self.camera.cam_brush_vao.render(
                mode=moderngl.TRIANGLE_FAN, instances=self.sim.entity_count,
                vertices=4)
        finally:
            self.ctx.disable(moderngl.BLEND)
            if grayscale:
                tryset(prog, "GRAYSCALE", False)
        return self._tex

    @staticmethod
    def _capture_kwargs(assemble_kwargs, grayscale=False) -> dict:
        """The shared assembly kwargs, with every camera-dependent entry
        neutralised. Anything left pointing at the live camera would put the
        user's viewpoint back into the score.

        `grayscale` has to ride here as well as in the particle pass: the
        Canvas view and the trail overlay are coloured by the ASSEMBLER."""
        kw = dict(assemble_kwargs)
        kw.update(
            grayscale=bool(grayscale),
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

    def _release_tile_target(self) -> None:
        for obj in (self._tile_fbo, self._tile_tex):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._tile_fbo = None
        self._tile_tex = None

    def release(self) -> None:
        self._release_target()
        self._release_tile_target()
        self._side = 0
        if self._bloom is not None:
            try:
                self._bloom.cleanup()
            except Exception:
                pass
            self._bloom = None
