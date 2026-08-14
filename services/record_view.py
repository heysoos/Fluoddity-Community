"""Crops the assembled texture to the world, into a fixed-size target.

The target size is frozen when recording starts, because the encoder cannot
take a dimension change mid-stream; the source rect is re-derived every frame,
so a zoom rescales into that target rather than resizing the file.
"""
from __future__ import annotations

from services.capture_blit import CaptureBlit
from services.record_crop import world_crop


class RecordView:
    """Owns the blit target for one recording. Release it when the take ends."""

    def __init__(self, ctx, size) -> None:
        self.ctx = ctx
        self.size = (int(size[0]), int(size[1]))
        self.texture = ctx.texture(self.size, 4, dtype="f4")
        self.fbo = ctx.framebuffer(color_attachments=[self.texture])
        self.blit = CaptureBlit(ctx)

    def crop(self, src_tex, view_rect):
        """Blit the world part of `src_tex` into the target and return it.

        `view_rect` is the camera's canvas rect for `src_tex` - a GL texture
        coordinate, which is also what CaptureBlit samples with, so the two
        agree without a flip.
        """
        lo, hi = world_crop(view_rect)
        self.fbo.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.blit.draw(src_tex, lo, hi)
        return self.texture

    def release(self) -> None:
        for obj in (self.blit, self.fbo, self.texture):
            try:
                obj.release()
            except Exception:
                pass
