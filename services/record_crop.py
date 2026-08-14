"""Where a recording's frame sits inside the assembled texture.

The camera stores the canvas rect alongside every assembled texture; the
recorder trims to it so empty space around the world never reaches the file.
Both functions are pure so the geometry can be checked without a GL context.
"""
from __future__ import annotations

# ffmpeg's H.264 encoder rejects odd dimensions, and the recorder's own padding
# would put a black edge back on the side that was just trimmed.
_EVEN = 2


def world_crop(view_rect):
    """The part of the texture that holds world, as ((lo_u, lo_v), (hi_u, hi_v)).

    `view_rect` is Camera.canvas_view_rect's output for the texture being
    recorded. Intersecting it with the texture covers both directions: zoomed
    out the canvas is smaller than the view and the rect tightens onto it;
    zoomed in the canvas overflows and the rect becomes the whole texture, so
    the crop is an identity and the framing is untouched.
    """
    (lo_u, lo_v), (hi_u, hi_v) = view_rect
    lo = (min(max(lo_u, 0.0), 1.0), min(max(lo_v, 0.0), 1.0))
    hi = (min(max(hi_u, 0.0), 1.0), min(max(hi_v, 0.0), 1.0))
    # A canvas panned entirely off screen clamps both edges to the same side;
    # ordering them keeps the rect degenerate instead of sampling backwards.
    return ((min(lo[0], hi[0]), min(lo[1], hi[1])),
            (max(lo[0], hi[0]), max(lo[1], hi[1])))


def crop_output_size(rect, tex_size):
    """The encodable pixel size `rect` covers of a `tex_size` texture."""
    (lo_u, lo_v), (hi_u, hi_v) = rect
    width, height = tex_size
    return (_encodable((hi_u - lo_u) * width),
            _encodable((hi_v - lo_v) * height))


def record_sizes(rect, tex_size, supersample_k: int):
    """(blit_target_size, video_size) for cropping to `rect`.

    The video size is derived in OUTPUT space rather than by cropping first and
    dividing after: an even crop divided by the kernel can land on an odd
    number, which the recorder pads - putting a black edge back on the side the
    crop just removed. The target is then a whole number of kernels, because
    save_frame_gpu divides it by exactly that.
    """
    k = max(1, int(supersample_k))
    width, height = tex_size
    video = crop_output_size(rect, (width // k, height // k))
    return (video[0] * k, video[1] * k), video


def _encodable(pixels: float) -> int:
    return max(_EVEN, int(round(pixels)) // _EVEN * _EVEN)
