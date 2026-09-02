"""The capture must not depend on where the user is looking.

It used to crop the on-screen image, so pan, zoom and window size all leaked
into what the optimizer scored - and a top-down/bottom-up mix-up in that crop
put 20% of each tile's neighbour into its picture and blacked out the whole
bottom row of the grid. CaptureView re-renders the canvas instead, with an
identity camera, so there is no crop and nothing to get wrong.

No GL: the point is which uniforms are chosen, and that is pure logic.
"""
import pytest

from services.capture_view import CaptureView


class _Uniform:
    def __init__(self):
        self.value = None


class _Prog:
    # tryset() skips uniforms the program does not declare, so the fake has to
    # declare the same set cam_brush.vert does or the test would pass by
    # silently skipping the assignment it is checking.
    DECLARED = ("cam_pos", "cam_zoom", "canvas_resolution", "window_size",
                "WATERCOLOR_MODE", "tiling_mode_enabled", "view_min", "view_max",
                "GRAYSCALE")

    def __init__(self):
        self._u = {name: _Uniform() for name in self.DECLARED}

    def __getitem__(self, k):
        return self._u.setdefault(k, _Uniform())

    def __setitem__(self, k, v):
        self._u.setdefault(k, _Uniform()).value = v

    def __contains__(self, k):
        return k in self._u

    def get(self, k):
        u = self._u.get(k)
        return u.value if u else None


class _Vao:
    def __init__(self):
        self.renders = 0

    def render(self, **kw):
        self.renders += 1


class _Ctx:
    def __init__(self):
        self.viewport = None
        self.blend_func = None
        self.blend_equation = None

    def clear(self, *a):
        pass

    def enable(self, *a):
        pass

    def disable(self, *a):
        pass


class _Tex:
    def __init__(self, size):
        self.size = size

    def release(self):
        pass


class _Fbo:
    def use(self):
        pass

    def release(self):
        pass


class _Sim:
    def __init__(self, canvas=(647, 647)):
        self.view_tex = _Tex(canvas)
        self.entity_count = 1000


class _Cam:
    def __init__(self, position=(0.3, -0.2), zoom=2.5, brush=True):
        self.position = position
        self.zoom = zoom
        self.cam_brush_mode = brush
        self.cam_brush_program = _Prog()
        self.cam_brush_vao = _Vao()


class _UI:
    def __init__(self):
        self.sim = type("S", (), {"watercolor_mode": False})()
        self.preferences = type("P", (), {
            "bloom_enabled": False, "bloom_threshold": 0.11,
            "bloom_intensity": 0.23, "bloom_radius": 1.0,
            "tonemap_softness": 2.5,
        })()


def view_at(camera, side=896):
    """A CaptureView with its GL target pre-faked, so render can be exercised
    without a context."""
    cv = CaptureView(_Ctx(), _Sim(), camera)
    cv._side = side
    cv._tex = _Tex((side, side))
    cv._fbo = _Fbo()
    return cv


# ---- the particle pass --------------------------------------------------

@pytest.mark.parametrize("position,zoom", [
    ((0.0, 0.0), 1.0), ((0.3, -0.2), 2.5), ((-0.9, 0.75), 0.4),
])
def test_the_particle_pass_ignores_the_camera(position, zoom):
    """Whatever the user is looking at, the capture renders the whole canvas
    from the origin at zoom 1."""
    cam = _Cam(position=position, zoom=zoom)
    cv = view_at(cam)
    cv._render_particles(_UI())
    prog = cam.cam_brush_program
    assert prog.get("cam_pos") == (0.0, 0.0)
    assert prog.get("cam_zoom") == 1.0


def test_the_target_is_square_so_a_square_canvas_fills_it_exactly():
    """cam_brush.vert letterboxes by tex_aspect vs window_aspect. Square
    against square is scale (1, 1) - the canvas maps one-to-one, no border."""
    cam = _Cam()
    cv = view_at(cam, side=896)
    cv._render_particles(_UI())
    prog = cam.cam_brush_program
    assert prog.get("window_size") == (896.0, 896.0)
    assert prog.get("canvas_resolution") == (647, 647)


def test_tiling_is_off_because_the_capture_is_exactly_one_canvas():
    cam = _Cam()
    cv = view_at(cam)
    cv._render_particles(_UI())
    assert cam.cam_brush_program.get("tiling_mode_enabled") is False


def test_the_particles_are_actually_drawn():
    cam = _Cam()
    cv = view_at(cam)
    assert cv._render_particles(_UI()) is cv._tex
    assert cam.cam_brush_vao.renders == 1


def test_a_non_brush_view_mode_uses_the_canvas_texture_itself():
    """Those view modes have no separate particle pass to redo."""
    cam = _Cam(brush=False)
    cv = view_at(cam)
    assert cv._render_particles(_UI()) is cv.sim.view_tex
    assert cam.cam_brush_vao.renders == 0


# ---- the assembly kwargs ------------------------------------------------

def test_every_camera_dependent_kwarg_is_neutralised():
    """Anything left pointing at the live camera puts the user's viewpoint
    back into the score by the back door."""
    live = dict(
        screen_aspect=1.777, camera_position=(0.3, -0.2), camera_zoom=2.5,
        tiling_mode=True, view_min=(-2.0, -2.0), view_max=(2.0, 2.0),
        tiling_scale=(3.0, 3.0), brightness=4.0, exposure=0.5,
    )
    kw = CaptureView._capture_kwargs(live)
    assert kw["screen_aspect"] == 1.0
    assert kw["camera_position"] == (0.0, 0.0)
    assert kw["camera_zoom"] == 1.0
    assert kw["tiling_mode"] is False
    assert kw["tiling_scale"] == (1.0, 1.0)


def test_the_look_of_the_creature_is_preserved():
    """Brightness, exposure and tonemap ARE the creature's appearance and must
    survive - CLIP should see what the user would see."""
    kw = CaptureView._capture_kwargs(
        dict(brightness=4.0, exposure=0.5, tonemap_softness=2.5, ink_weight=0.8))
    assert kw["brightness"] == 4.0
    assert kw["exposure"] == 0.5
    assert kw["tonemap_softness"] == 2.5
    assert kw["ink_weight"] == 0.8


def test_ui_overlays_are_excluded():
    """A sweep reticle or a draw-target overlay is not part of the creature."""
    kw = CaptureView._capture_kwargs(
        dict(sweep_mode=True, sweep_reticle_visible=True,
             trail_draw_radius=0.2, draw_target_overlay_opacity=0.7))
    assert kw["sweep_reticle_visible"] is False
    assert kw["sweep_mode"] is False
    assert kw["trail_draw_radius"] == 0.0
    assert kw["draw_target_overlay_opacity"] == 0.0


def test_the_caller_s_dict_is_not_mutated():
    """It is the live per-frame dict the renderer is still using."""
    live = dict(camera_zoom=2.5, tiling_mode=True)
    CaptureView._capture_kwargs(live)
    assert live == {"camera_zoom": 2.5, "tiling_mode": True}


# ---- per-tile bloom -----------------------------------------------------

class _Blit:
    def __init__(self):
        self.calls = []

    def draw(self, src, lo, hi):
        self.calls.append((src, tuple(lo), tuple(hi)))


class _Bloom:
    def __init__(self):
        self.inputs = []

    def process(self, tex, *a, **kw):
        self.inputs.append(tex)
        return _Tex(tex.size)


class _GridCtx(_Ctx):
    """Adds the allocation the tile target needs."""

    def texture(self, size, comps, dtype=None):
        return _Tex(size)

    def framebuffer(self, color_attachments=None):
        return _Fbo()


def grid_view(bloom_on=True):
    cv = CaptureView(_GridCtx(), _Sim(), _Cam())
    cv._bloom = _Bloom()
    ui = _UI()
    ui.preferences.bloom_enabled = bloom_on
    return cv, ui


def test_each_tile_is_bloomed_on_its_own():
    """The whole point: the bloom pass cannot see outside the tile, so a bright
    creature cannot glow into its neighbour's picture."""
    cv, ui = grid_view()
    cv.draw_grid(_Fbo(), _Tex((448, 448)), 2, _Blit(), ui, 224)
    assert len(cv._bloom.inputs) == 4, "one bloom per tile"
    assert all(t.size == (224, 224) for t in cv._bloom.inputs), \
        "bloomed at tile size, not grid size"


def test_the_source_rect_of_each_tile_is_its_own_quarter():
    cv, ui = grid_view()
    blit, grid_tex = _Blit(), _Tex((448, 448))
    cv.draw_grid(_Fbo(), grid_tex, 2, blit, ui, 224)
    # An extract reads the GRID texture; the write-back reads the bloomed tile.
    # Filtering on the rect would drop tile 3, whose rect really is (0,0)-(1,1).
    extracts = [c for c in blit.calls if c[0] is grid_tex]
    assert len(extracts) == 4
    # Not tx/grid: a tile owns a whole number of TEXELS, and the canvas is 647
    # across at the default world size, so the seam is at 323/647 rather than
    # at 0.5. Cropping on the even split puts a sliver of the neighbouring
    # tile into the picture the optimizer scores. See services/tile_geometry.
    from services.tile_geometry import tile_uv_box

    rects = sorted((c[1], c[2]) for c in extracts)
    assert rects == sorted([tile_uv_box(tx, ty, 2, (647, 647))
                            for ty in range(2) for tx in range(2)])
    assert all(r[0] != (0.5, 0.5) and r[1] != (0.5, 0.5) for r in rects)


def test_tile_zero_is_bottom_left_to_match_the_shader():
    """tournament_home_tile() numbers tile 0 bottom-left; getting this backwards
    would score every genome against a different tile's picture."""
    cv, ui = grid_view()
    blit, grid_tex = _Blit(), _Tex((448, 448))
    cv.draw_grid(_Fbo(), grid_tex, 2, blit, ui, 224)
    first_extract = next(c for c in blit.calls if c[0] is grid_tex)
    assert first_extract[1] == (0.0, 0.0)


def test_with_bloom_off_the_tiles_still_come_from_their_own_texels():
    """Bloom is the only per-tile PASS that can be skipped; the crop cannot.

    This used to blit the whole grid in one go and let the even split of the
    destination do the tiling, which is only correct when the canvas divides by
    the grid. It is 647 texels across at the default world size.
    """
    cv, ui = grid_view(bloom_on=False)
    blit = _Blit()
    cv.draw_grid(_Fbo(), _Tex((448, 448)), 2, blit, ui, 224)
    from services.tile_geometry import tile_uv_box

    assert sorted((c[1], c[2]) for c in blit.calls) == sorted(
        [tile_uv_box(tx, ty, 2, (647, 647))
         for ty in range(2) for tx in range(2)])
    assert cv._bloom.inputs == []


def test_watercolor_mode_skips_bloom_as_it_does_on_screen():
    cv, ui = grid_view()
    ui.sim.watercolor_mode = True
    blit = _Blit()
    cv.draw_grid(_Fbo(), _Tex((448, 448)), 2, blit, ui, 224)
    assert cv._bloom.inputs == []


def test_the_capture_can_be_asked_for_density_alone():
    """Greyscale scoring forces saturation to zero in the PARTICLE PASS, so
    what the encoder sees is particle density with no colour term at all - a
    luma mix of the colour frame would still vary with hue."""
    cam = _Cam()
    prog = cam.cam_brush_program
    seen = []

    class _SpyVao:
        def render(self, **kw):
            seen.append(prog.get("GRAYSCALE"))

    cam.cam_brush_vao = _SpyVao()
    cv = view_at(cam)
    cv._render_particles(_UI())
    cv._render_particles(_UI(), grayscale=True)
    assert seen == [False, True]
    # The program is shared with the laptop's own view, so the pass puts the
    # uniform back before returning.
    assert prog.get("GRAYSCALE") is False


def test_greyscale_reaches_the_assembler_too():
    # The Canvas view and the trail overlay are coloured in the assembler,
    # not the particle pass, so the flag has to ride in the kwargs as well.
    kw = CaptureView._capture_kwargs({"view_mode": 0}, grayscale=True)
    assert kw["grayscale"] is True
    assert CaptureView._capture_kwargs({"view_mode": 0})["grayscale"] is False
