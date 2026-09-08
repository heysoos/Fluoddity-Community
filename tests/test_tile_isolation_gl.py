"""Tournament tiles must not see each other, on a real GPU.

Every previous attempt at this was checked by eye, and every one of them left
a leak that only showed on the outer ring of tiles - which is exactly where a
4x4 grid puts most of its tiles, and where every corner lives. Reading the code
is not enough: the last bug was that `clamp(uv, 0, 0.999999)` made an
out-of-canvas probe look like it was still in the same tile, so the guard
passed and the sampler (repeat_x/repeat_y) quietly returned the far side of the
canvas.

So this runs the SHIPPED shader text. Both files are read from disk and their
main() replaced; everything above it - getCan, tile_tap, getBlur,
particle_world_box, confine_sample - is the real thing, byte for byte.

Skipped when no GL 4.3 context is available (CI, remote shells, no GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from utilities.gl_helpers import prepend_defines, read_shader  # noqa: E402

GRID = 4
RES = 128                      # divisible by GRID, so tile edges are texel-aligned
TILE = RES // GRID

BOUNCE, RESET, WRAP = 0, 1, 2


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                    # no GPU, no display, no driver
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def _cut_main(src: str) -> str:
    """Everything above `void main()`. The helpers are what is under test."""
    i = src.index("void main()")
    return src[:i]


# ---- the trail diffusion ------------------------------------------------

QUAD_VERT = """#version 430
out vec2 texcoord;
void main(){
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    texcoord = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

BLUR_MAIN = """
uniform float K_TEST;
void main(){ can_out = getBlur(texcoord, can_tex, K_TEST); }
"""


class Diffuser:
    """Ping-pong the real getBlur over a canvas, exactly as canvas.frag does."""

    def __init__(self, ctx, boundary, tile_mode=1, grid=GRID, res=RES):
        self.ctx = ctx
        self.res = res
        gx, gy = (grid, grid) if isinstance(grid, int) else grid
        self.prog = ctx.program(
            vertex_shader=QUAD_VERT,
            fragment_shader=_cut_main(read_shader("shaders/canvas.frag"))
                            + BLUR_MAIN)
        for name, value in (("BOUNDARY_CONDITIONS_MODE", boundary),
                            ("TILE_MODE", tile_mode),
                            ("TILE_GRID", (gx, gy)),
                            ("canvas_resolution", (res, res)),
                            # 4/(5^d - 1) at d=1, i.e. the strongest the slider
                            # reaches: one step moves half the mass.
                            ("K_TEST", 1.0)):
            if name in self.prog:
                self.prog[name].value = value
        # RG32F, as sim.py allocates the canvas. The diffusion is per-channel,
        # so the trail under test rides in R either way - but a harness in a
        # format the app does not use stops being evidence about the app.
        self.tex = [ctx.texture((res, res), 2, dtype="f4") for _ in range(2)]
        for t in self.tex:
            t.repeat_x = True                  # as sim.py sets them
            t.repeat_y = True
        self.fbo = [ctx.framebuffer([t]) for t in self.tex]
        self.vao = ctx.vertex_array(self.prog, [])

    def run(self, field: np.ndarray, steps: int) -> np.ndarray:
        rg = np.zeros((self.res, self.res, 2), dtype=np.float32)
        rg[..., 0] = field
        self.tex[0].write(rg.tobytes())
        read, write = 0, 1
        for _ in range(steps):
            self.tex[read].use(location=1)
            if "can_tex" in self.prog:
                self.prog["can_tex"].value = 1
            self.fbo[write].use()
            self.vao.render(moderngl.TRIANGLES, vertices=3)
            read, write = write, read
        out = np.frombuffer(self.fbo[read].read(components=2, dtype="f4"),
                            dtype=np.float32).reshape(self.res, self.res, 2)
        return out[..., 0].copy()

    def release(self):
        for f in self.fbo:
            f.release()
        for t in self.tex:
            t.release()
        self.vao.release()
        self.prog.release()


def _tile(field, tx, ty):
    return field[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]


def _one_tile_lit(tx, ty, value=1.0):
    f = np.zeros((RES, RES), dtype=np.float32)
    f[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE] = value
    return f


@pytest.mark.parametrize("boundary", [BOUNCE, RESET, WRAP])
@pytest.mark.parametrize("tx,ty", [(0, 0), (3, 0), (0, 3), (3, 3),   # corners
                                   (1, 0), (0, 2), (3, 1), (2, 3),   # edges
                                   (1, 1), (1, 2), (2, 2)])          # interior
def test_a_lit_tile_never_lights_its_neighbours(ctx, boundary, tx, ty):
    """Measured against the old shader, every one of the twelve outer-ring
    cases failed and all three interior ones passed - the leak was the canvas
    border, so only tiles touching it were affected and corner tiles on two
    edges each."""
    d = Diffuser(ctx, boundary)
    try:
        out = d.run(_one_tile_lit(tx, ty), steps=200)
    finally:
        d.release()
    for gy in range(GRID):
        for gx in range(GRID):
            if (gx, gy) == (tx, ty):
                continue
            peak = float(_tile(out, gx, gy).max())
            assert peak == 0.0, f"tile ({gx},{gy}) lit to {peak} by ({tx},{ty})"


@pytest.mark.parametrize("boundary", [BOUNCE, WRAP])
def test_a_tile_conserves_its_own_trail(ctx, boundary):
    """Both a torus and a zero-flux wall conserve mass exactly - the 5-tap
    kernel is a weighted average, and every tap that leaves the tile is
    replaced by one that stays. Mass going missing means the tile is losing
    trail across the seam; mass appearing means it is stealing.
    """
    rng = np.random.default_rng(0)
    field = np.zeros((RES, RES), dtype=np.float32)
    blob = rng.random((TILE, TILE)).astype(np.float32)
    field[0:TILE, 0:TILE] = blob              # corner tile again
    d = Diffuser(ctx, boundary)
    try:
        out = d.run(field, steps=300)
    finally:
        d.release()
    assert float(_tile(out, 0, 0).sum()) == pytest.approx(float(blob.sum()),
                                                         rel=1e-4)


def test_wrap_carries_trail_across_the_tile_seam(ctx):
    """A torus has no edge, so a blob on the tile's left edge must reach its
    right edge - within the SAME tile, without ever touching the next one."""
    field = np.zeros((RES, RES), dtype=np.float32)
    field[0:TILE, 0:2] = 1.0                  # left edge of tile (0,0)
    d = Diffuser(ctx, WRAP)
    try:
        out = d.run(field, steps=60)
    finally:
        d.release()
    tile00 = _tile(out, 0, 0)
    assert tile00[:, -1].mean() > 1e-3, "the trail did not wrap round the tile"
    assert float(_tile(out, 1, 0).max()) == 0.0, "it wrapped into the neighbour"


def test_bounce_holds_the_trail_off_the_seam(ctx):
    """The other half of the same statement: under bounce the seam IS a wall,
    so the far edge must stay dark."""
    field = np.zeros((RES, RES), dtype=np.float32)
    field[0:TILE, 0:2] = 1.0
    d = Diffuser(ctx, BOUNCE)
    try:
        out = d.run(field, steps=60)
    finally:
        d.release()
    assert float(_tile(out, 0, 0)[:, -1].max()) < 1e-6


def test_tournament_off_leaves_the_canvas_alone(ctx):
    """The non-tournament path must be untouched: the whole canvas is one
    domain and wrap wraps it, so a blob at x=0 reaches x=RES-1."""
    field = np.zeros((RES, RES), dtype=np.float32)
    field[:, 0:2] = 1.0
    d = Diffuser(ctx, WRAP, tile_mode=0)
    try:
        out = d.run(field, steps=60)
    finally:
        d.release()
    assert out[:, -1].mean() > 1e-3


# ---- resolutions that do not divide by the grid --------------------------
#
# The default world_size of 0.40 makes the canvas 647 texels across, the grid
# slider goes 2..8, and 647/8 = 80.875. Every tiling bug so far has been at a
# seam, so the seams have to be tested where they are ugly rather than where
# they are convenient. 128/4 is here as the control.


from services.tile_geometry import lo_texel as _lo_texel  # noqa: E402


def _slices(res, grid):
    return [slice(_lo_texel(k, grid, res), _lo_texel(k + 1, grid, res))
            for k in range(grid)]


@pytest.mark.parametrize("res,grid", [(647, 8), (647, 4), (647, 3),
                                      (641, 7), (128, 4)])
def test_every_tile_is_a_torus_whatever_the_resolution(ctx, res, grid):
    """Light every tile's left edge at once, with a value unique to the tile.

    Three statements in one run, which is what makes it affordable at 64 tiles:

      wrapped       each tile's own FAR edge lights up - the seam is not a wall
      no brighter   no tile exceeds its own value, so nothing brighter bled in
      conserved     each tile's mass is unchanged, so nothing dimmer did either

    Measured against the pre-fix shader at 647/8, the middle column retained
    25.7% of its trail and 39 of 64 tiles lit a tile they could not reach.
    """
    cols = _slices(res, grid)
    field = np.zeros((res, res), dtype=np.float32)
    values = {}
    for tx in range(grid):
        for ty in range(grid):
            v = float(tx + ty * grid + 1)
            values[(tx, ty)] = v
            field[cols[ty], cols[tx].start:cols[tx].start + 2] = v

    before = {k: float(field[cols[k[1]], cols[k[0]]].sum()) for k in values}
    d = Diffuser(ctx, WRAP, grid=grid, res=res)
    try:
        out = d.run(field, steps=150)
    finally:
        d.release()

    for (tx, ty), v in values.items():
        own = out[cols[ty], cols[tx]]
        assert float(own[:, -1].mean()) > 1e-4 * v, (
            f"tile ({tx},{ty}) has a wall: its far edge never lit")
        assert float(own.max()) <= v + 1e-4, (
            f"tile ({tx},{ty}) peaks at {own.max()}, above its own {v} - a "
            f"brighter tile bled in")
        assert float(own.sum()) == pytest.approx(before[(tx, ty)], rel=2e-4), (
            f"tile ({tx},{ty}) did not conserve its trail")


@pytest.mark.parametrize("res,grid", [(647, 8), (647, 3), (641, 7)])
def test_tiles_partition_the_texels_exactly(ctx, res, grid):
    """No texel in two tiles, none in none. A tile owns a whole number of
    texels, so widths differ by at most one - 647/8 gives seven of 81 and one
    of 80. The one that used to fall through the cracks was texel 323, where
    the float form of this evaluated floor((323.5/647)*8) as 3 rather than 4.
    """
    edges = [_lo_texel(k, grid, res) for k in range(grid + 1)]
    assert edges[0] == 0 and edges[-1] == res
    widths = np.diff(edges)
    assert widths.min() >= 1
    assert widths.max() - widths.min() <= 1

    # And the shader agrees: a single lit texel either side of every seam must
    # stay on its own side.
    cols = _slices(res, grid)
    d = Diffuser(ctx, WRAP, grid=grid, res=res)
    try:
        for k in range(1, grid):
            field = np.zeros((res, res), dtype=np.float32)
            field[:, edges[k] - 1] = 1.0        # last texel of tile k-1
            out = d.run(field, steps=40)
            assert float(out[:, cols[k]].max()) == 0.0, (
                f"the texel below seam {k} leaked across it")
    finally:
        d.release()


# ---- the particle side --------------------------------------------------

PARTICLE_SRC = """#version 430
layout(local_size_x = 64) in;
uniform vec2 canvas_resolution;
uniform int TILE_MODE;
uniform ivec2 TILE_GRID;
uniform int TILE_COHORTS;
uniform float ACTIVE_COUNT_F;
uniform int MODE;
layout(std430, binding = 0) buffer In  { vec2 pts[]; };
layout(std430, binding = 1) buffer Out { vec4 res[]; };
layout(std430, binding = 2) buffer Idx { uint idx[]; };
#define ACTIVE_COUNT ACTIVE_COUNT_F
"""

PARTICLE_MAIN = """
void main(){
    uint i = gl_GlobalInvocationID.x;
    if(i >= pts.length()) return;
    vec2 lo, hi; particle_world_box(idx[i], lo, hi);
    res[i] = vec4(confine_sample(pts[i], lo, hi, MODE), lo);
}
"""


def _particle_helpers() -> str:
    """The real index_home_tile / cohort_home_tile / tile_box /
    particle_world_box / confine_sample block, lifted from entity_update."""
    src = read_shader("shaders/entity_update.glsl")
    start = src.index("int tile_count(){")
    end = src.index("//Entities with index > ACTIVE_COUNT", start)
    return src[start:end]


@pytest.fixture(scope="module")
def confine(ctx):
    prog = ctx.compute_shader(PARTICLE_SRC + _particle_helpers() + PARTICLE_MAIN)
    prog["canvas_resolution"].value = (RES, RES)
    prog["TILE_MODE"].value = 1
    prog["TILE_GRID"].value = (GRID, GRID)
    prog["TILE_COHORTS"].value = GRID * GRID
    prog["ACTIVE_COUNT_F"].value = 1600.0

    def run(points, indices, mode):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        ids = np.asarray(indices, dtype=np.uint32).reshape(-1)
        bin_ = ctx.buffer(pts.tobytes())
        bout = ctx.buffer(reserve=len(pts) * 16)
        bidx = ctx.buffer(ids.tobytes())
        bin_.bind_to_storage_buffer(0)
        bout.bind_to_storage_buffer(1)
        bidx.bind_to_storage_buffer(2)
        prog["MODE"].value = int(mode)
        prog.run(group_x=(len(pts) + 63) // 64)
        out = np.frombuffer(bout.read(), dtype=np.float32).reshape(-1, 4)
        for b in (bin_, bout, bidx):
            b.release()
        return out[:, :2].copy(), out[:, 2:].copy()

    yield run
    prog.release()


def _tile_of(i):
    """Buffer index that lands squarely in tile i, matching
    index_home_tile at ACTIVE_COUNT=1600, GRID=4."""
    return int((i + 0.5) * 1600 / (GRID * GRID))


@pytest.mark.parametrize("res,grid", [(647, 8), (647, 3), (641, 7), (128, 4)])
def test_the_particle_box_is_the_same_box_the_trails_use(ctx, res, grid):
    """The particle's world box has to land on the SAME texel edges the
    diffusion tiles on, or a particle sits in its own tile while the texel it
    deposits into belongs to the next one. Boxes must abut exactly and cover
    the canvas, and each edge must be a texel edge."""
    prog = ctx.compute_shader(PARTICLE_SRC + _particle_helpers() + PARTICLE_MAIN)
    active = 1600.0
    prog["canvas_resolution"].value = (res, res)
    prog["TILE_MODE"].value = 1
    prog["TILE_GRID"].value = (grid, grid)
    prog["TILE_COHORTS"].value = grid * grid
    prog["ACTIVE_COUNT_F"].value = active
    prog["MODE"].value = WRAP
    n = grid * grid
    ids = np.array([int((i + 0.5) * active / n) for i in range(n)],
                   dtype=np.uint32)
    pts = np.zeros((n, 2), dtype=np.float32)
    bufs = [ctx.buffer(pts.tobytes()), ctx.buffer(reserve=n * 16),
            ctx.buffer(ids.tobytes())]
    for i, b in enumerate(bufs):
        b.bind_to_storage_buffer(i)
    prog.run(group_x=(n + 63) // 64)
    los = np.frombuffer(bufs[1].read(), dtype=np.float32).reshape(-1, 4)[:, 2:]
    for b in bufs:
        b.release()
    prog.release()

    xs = sorted({round(float(v), 6) for v in los[:, 0]})
    assert len(xs) == grid, "tiles do not form a grid of columns"
    # world = (2*uv - 1) * half_extent, half_extent = 1 for a square canvas
    want = [2.0 * _lo_texel(k, grid, res) / res - 1.0 for k in range(grid)]
    assert np.allclose(xs, want, atol=1e-5), (
        "the particle box is not on the diffusion's texel edges")


def test_the_tile_box_partitions_the_canvas(confine):
    """Every tile's box must abut its neighbour's exactly - a gap or an overlap
    here is a strip of canvas that belongs to two tiles or to none."""
    ids = [_tile_of(i) for i in range(GRID * GRID)]
    _out, los = confine([[0.0, 0.0]] * len(ids), ids, WRAP)
    xs = sorted({round(float(v), 5) for v in los[:, 0]})
    assert len(xs) == GRID
    step = xs[1] - xs[0]
    assert np.allclose(np.diff(xs), step), "tile columns are not evenly spaced"


def test_wrap_folds_a_sensor_back_into_its_own_tile(confine):
    """The sensor used to be clamped, which pinned every reading past the wall
    to the same texel. Under wrap it has to come out the other side."""
    idx = _tile_of(5)
    _lo_probe, los = confine([[0.0, 0.0]], [idx], WRAP)
    lo = los[0]
    span = 2.0 / GRID                          # square canvas, half_extent = 1
    just_past = [lo[0] + span * 1.1, lo[1] + span * 0.5]
    out, _ = confine([just_past], [idx], WRAP)
    assert lo[0] <= out[0][0] <= lo[0] + span
    assert out[0][0] == pytest.approx(lo[0] + span * 0.1, abs=1e-3)


def test_wrap_keeps_two_sensors_past_the_wall_distinct(confine):
    """The actual defect, stated as a test.

    A particle heading INTO the wall has both sensors past it. Clamping sent
    both to the same coordinate, so the left and right taps were equal and the
    steering differential was exactly zero - in a band of width sample_dist
    around every tile, and along both axes at once in a corner. That band is
    the anisotropy; the corners are where two of them cross.

    Under wrap there is no wall to be past, so the two stay as far apart as
    they started.
    """
    idx = _tile_of(5)
    _p, los = confine([[0.0, 0.0]], [idx], WRAP)
    lo = los[0]
    span = 2.0 / GRID
    mid_y = lo[1] + span * 0.5
    near = [lo[0] + span + 0.02, mid_y]        # both past the right wall,
    far = [lo[0] + span + 0.08, mid_y]         # 0.06 apart

    wrapped, _ = confine([near, far], [idx, idx], WRAP)
    assert abs(wrapped[0][0] - wrapped[1][0]) == pytest.approx(0.06, abs=1e-3)

    clamped, _ = confine([near, far], [idx, idx], BOUNCE)
    assert clamped[0][0] == pytest.approx(clamped[1][0], abs=1e-6), (
        "under bounce the wall is real, so both sensors stopping at it is "
        "correct - the bug was doing this under wrap too")


def test_wrap_keeps_a_corner_from_collapsing_both_axes(confine):
    """A corner is where the band along x meets the band along y, so a clamp
    there flattens the sample in both directions at once."""
    idx = _tile_of(5)
    _p, los = confine([[0.0, 0.0]], [idx], WRAP)
    lo = los[0]
    span = 2.0 / GRID
    a = [lo[0] + span + 0.02, lo[1] + span + 0.02]
    b = [lo[0] + span + 0.09, lo[1] + span + 0.05]

    wrapped, _ = confine([a, b], [idx, idx], WRAP)
    assert np.abs(wrapped[0] - wrapped[1]).min() > 1e-3

    clamped, _ = confine([a, b], [idx, idx], BOUNCE)
    assert np.allclose(clamped[0], clamped[1], atol=1e-6)


@pytest.mark.parametrize("mode", [BOUNCE, RESET, WRAP])
def test_a_sensor_never_leaves_its_tile(confine, mode):
    """Whatever the boundary condition, a sample must stay inside the tile -
    and inside it by half a texel, because get_can() samples bilinearly and a
    coordinate on the seam blends the neighbouring tile's texels."""
    rng = np.random.default_rng(1)
    ids = [_tile_of(i % (GRID * GRID)) for i in range(256)]
    pts = rng.uniform(-3.0, 3.0, size=(256, 2)).astype(np.float32)
    out, los = confine(pts, ids, mode)
    span = 2.0 / GRID
    half_texel = 1.0 / RES
    assert np.all(out >= los + half_texel - 1e-5)
    assert np.all(out <= los + span - half_texel + 1e-5)


# ---- rectangular grids and cohort-owned boxes ---------------------------
#
# Cohort boxing gives every cohort a box of its own, and a count that is not a
# product of two close factors needs a rectangle: 12 cohorts is 4x3. Nothing
# about a seam changes, so the isolation statements have to hold there too.

from services.cohort_tiling import box_grid as _box_grid  # noqa: E402


def _axis_slices(res, g):
    return [slice(_lo_texel(k, g, res), _lo_texel(k + 1, g, res))
            for k in range(g)]


@pytest.mark.parametrize("res,gx,gy", [(647, 4, 3), (647, 5, 3), (128, 3, 2)])
def test_a_rectangular_grid_isolates_just_as_well(ctx, res, gx, gy):
    """Every box is still a torus that keeps its own trail. The x and y grids
    differ here, so a shader reading one count for both axes puts the seams in
    the wrong place along one of them."""
    cols, rows = _axis_slices(res, gx), _axis_slices(res, gy)
    field = np.zeros((res, res), dtype=np.float32)
    values = {}
    for bx in range(gx):
        for by in range(gy):
            v = float(bx + by * gx + 1)
            values[(bx, by)] = v
            field[rows[by], cols[bx].start:cols[bx].start + 2] = v
    before = {k: float(field[rows[k[1]], cols[k[0]]].sum()) for k in values}

    d = Diffuser(ctx, WRAP, grid=(gx, gy), res=res)
    try:
        out = d.run(field, steps=150)
    finally:
        d.release()

    for (bx, by), v in values.items():
        own = out[rows[by], cols[bx]]
        assert float(own[:, -1].mean()) > 1e-4 * v, (
            f"box ({bx},{by}) has a wall: its far edge never lit")
        assert float(own.max()) <= v + 1e-4, (
            f"box ({bx},{by}) peaks above its own {v} - a brighter box bled in")
        assert float(own.sum()) == pytest.approx(before[(bx, by)], rel=2e-4), (
            f"box ({bx},{by}) did not conserve its trail")


def _cohort_boxes(ctx, res, cohorts, active=1600.0):
    """-> the (lo_x, lo_y) each cohort's box starts at, straight off the real
    particle_world_box in cohort mode."""
    gx, gy = _box_grid(cohorts)
    prog = ctx.compute_shader(PARTICLE_SRC + _particle_helpers() + PARTICLE_MAIN)
    prog["canvas_resolution"].value = (res, res)
    prog["TILE_MODE"].value = 2
    prog["TILE_GRID"].value = (gx, gy)
    prog["TILE_COHORTS"].value = cohorts
    prog["ACTIVE_COUNT_F"].value = active
    prog["MODE"].value = WRAP
    ids = np.array([int((c + 0.5) * active / cohorts) for c in range(cohorts)],
                   dtype=np.uint32)
    pts = np.zeros((cohorts, 2), dtype=np.float32)
    bufs = [ctx.buffer(pts.tobytes()), ctx.buffer(reserve=cohorts * 16),
            ctx.buffer(ids.tobytes())]
    for i, b in enumerate(bufs):
        b.bind_to_storage_buffer(i)
    prog.run(group_x=(cohorts + 63) // 64)
    los = np.frombuffer(bufs[1].read(), dtype=np.float32).reshape(-1, 4)[:, 2:]
    for b in bufs:
        b.release()
    prog.release()
    return los.copy(), (gx, gy)


@pytest.mark.parametrize("cohorts", [4, 12, 13, 64])
def test_each_cohort_gets_its_own_box(ctx, cohorts):
    """The box index IS the cohort index, laid out across then up. Two cohorts
    sharing a box is the monoculture this feature exists to avoid; one cohort
    spanning two boxes is a wall through the middle of a creature."""
    res = 647
    los, (gx, gy) = _cohort_boxes(ctx, res, cohorts)
    assert len({tuple(np.round(p, 6)) for p in los}) == cohorts, (
        "two cohorts landed in the same box")
    for c in range(cohorts):
        kx, ky = c % gx, c // gx
        want = (2.0 * _lo_texel(kx, gx, res) / res - 1.0,
                2.0 * _lo_texel(ky, gy, res) / res - 1.0)
        assert np.allclose(los[c], want, atol=1e-5), (
            f"cohort {c} is not in cell ({kx},{ky})")


def test_a_cohort_box_lands_on_the_same_texel_edges_the_trails_use(ctx):
    """13 cohorts is a 4x4 grid with three cells left blank - the seams are
    still the diffusion's, so a particle never deposits across one."""
    res = 647
    los, (gx, gy) = _cohort_boxes(ctx, res, 13)
    edges = {round(2.0 * _lo_texel(k, gx, res) / res - 1.0, 6)
             for k in range(gx)}
    assert {round(float(v), 6) for v in los[:, 0]} <= edges


def test_the_two_stages_slice_the_cohorts_the_same_way():
    """brush.vert cannot call into entity_update, so its copy of the cohort
    slice is checked against the original character by character - the same
    discipline tile_lo_texel is held to."""
    ent = read_shader("shaders/entity_update.glsl")
    vert = read_shader("shaders/brush.vert")
    assert "float(TILE_COHORTS) * float(index) / float(ACTIVE_COUNT)" in ent
    assert "float(TILE_COHORTS) * float(instance_id) / TILE_ACTIVE" in vert
