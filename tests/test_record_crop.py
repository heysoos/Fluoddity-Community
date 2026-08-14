"""The rect a recording is cropped to, and the size that rect encodes at.

The camera stores where the canvas sits inside each assembled texture; the
recorder trims to it so empty space never reaches the file. Every rect here is
taken from a PANNED camera - see the orientation tests at the bottom.
"""
import pytest

from services.record_crop import crop_output_size, record_sizes, world_crop


# --- the rect: intersect what the camera saw with what the texture holds -----

def test_a_canvas_smaller_than_the_view_is_trimmed_to():
    """Zoomed out, the canvas occupies part of the frame and the rest is the
    black bar the crop exists to remove."""
    lo, hi = world_crop(((0.25, 0.1), (0.75, 0.9)))
    assert lo == pytest.approx((0.25, 0.1))
    assert hi == pytest.approx((0.75, 0.9))


def test_a_canvas_overflowing_the_view_crops_nothing():
    """Zoomed in, the view is entirely inside the canvas, so every pixel of the
    texture is world and the blit must be an identity."""
    lo, hi = world_crop(((-0.4, -1.2), (1.6, 2.3)))
    assert lo == pytest.approx((0.0, 0.0))
    assert hi == pytest.approx((1.0, 1.0))


def test_one_axis_can_be_trimmed_while_the_other_is_full():
    """The usual case: a square canvas in a wide window has bars left and right
    and none top or bottom."""
    lo, hi = world_crop(((0.2, -0.3), (0.8, 1.1)))
    assert lo == pytest.approx((0.2, 0.0))
    assert hi == pytest.approx((0.8, 1.0))


def test_a_canvas_entirely_off_screen_does_not_invert():
    """Panned right off the edge there is no world left to frame; the rect must
    stay degenerate rather than crossing over and sampling backwards."""
    lo, hi = world_crop(((1.4, 0.2), (2.1, 0.6)))
    assert lo[0] <= hi[0] and lo[1] <= hi[1]


def test_the_rect_is_never_wider_than_the_texture():
    lo, hi = world_crop(((-5.0, -5.0), (5.0, 5.0)))
    assert (lo, hi) == ((0.0, 0.0), (1.0, 1.0))


# --- the size: what that rect encodes at ------------------------------------

def test_the_output_covers_the_cropped_pixels():
    size = crop_output_size(((0.25, 0.0), (0.75, 1.0)), (1920, 1080))
    assert size == (960, 1080)


def test_an_odd_pixel_count_rounds_because_h264_demands_even():
    """H.264 rejects odd dimensions, and the recorder's own padding would put
    back a black edge on the side we just trimmed."""
    w, h = crop_output_size(((0.0, 0.0), (0.5005, 1.0)), (1001, 999))
    assert w % 2 == 0 and h % 2 == 0


def test_an_identity_rect_keeps_the_whole_texture():
    assert crop_output_size(((0.0, 0.0), (1.0, 1.0)), (800, 600)) == (800, 600)


def test_a_degenerate_rect_still_yields_an_encodable_size():
    """A canvas panned off screen must not ask ffmpeg for a zero-sized video."""
    w, h = crop_output_size(((0.5, 0.5), (0.5, 0.5)), (1920, 1080))
    assert w >= 2 and h >= 2 and w % 2 == 0 and h % 2 == 0


# --- the two sizes: the blit target, and what ffmpeg is told ----------------

def test_without_supersampling_the_target_is_the_video():
    fbo, video = record_sizes(((0.25, 0.0), (0.75, 1.0)), (1920, 1080), 1)
    assert fbo == video == (960, 1080)


def test_the_video_stays_even_after_supersampling():
    """The video size is derived in OUTPUT space, not by cropping first and
    dividing after - dividing an even crop by the kernel can land on an odd
    number, and the recorder would pad the black edge back on."""
    _fbo, video = record_sizes(((0.0, 0.0), (0.5005, 1.0)), (2002, 1998), 2)
    assert video[0] % 2 == 0 and video[1] % 2 == 0


def test_the_blit_target_is_a_whole_number_of_kernels():
    """save_frame_gpu divides the target by the kernel, so a target that is not
    a multiple of it cannot produce the video size promised to ffmpeg."""
    fbo, video = record_sizes(((0.1, 0.2), (0.9, 0.7)), (1920, 1080), 3)
    assert fbo == (video[0] * 3, video[1] * 3)


def test_the_target_never_asks_for_more_pixels_than_the_texture_holds():
    """The crop is a subset of the texture, so upscaling into the target would
    invent detail that was never rasterised."""
    fbo, _video = record_sizes(((0.0, 0.0), (1.0, 1.0)), (1920, 1080), 1)
    assert fbo[0] <= 1920 and fbo[1] <= 1080


def test_a_degenerate_rect_still_yields_an_encodable_video():
    fbo, video = record_sizes(((0.5, 0.5), (0.5, 0.5)), (1920, 1080), 2)
    assert video == (2, 2) and fbo == (4, 4)


# --- orientation: a centred camera hides flips, so pan before trusting this --

def test_the_crop_follows_a_vertical_pan_rather_than_mirroring_it():
    """canvas_view_rect returns a GL coordinate (v=0 at the bottom) while
    tex_to_screen is top-down; the two cancel exactly when the canvas is
    centred. A rect low in the frame must stay low."""
    lo, hi = world_crop(((0.2, 0.0), (0.8, 0.4)))
    assert hi[1] == pytest.approx(0.4), "a low rect must not flip to the top"
    assert lo[1] == pytest.approx(0.0)
