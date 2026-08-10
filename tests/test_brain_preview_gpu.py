"""The Inspector must draw what the particles compute.

It is a second consumer of the brain GLSL, so the risk is that it drifts into
being a second IMPLEMENTATION. These check the drawn field against the brain
functions evaluated directly, and that the per-unit tiles sum to the whole.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

MODALITIES = ["fourier", "gabor", "lenia", "mlp"]


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


@pytest.fixture(scope="module")
def preview(ctx):
    from services.brain_preview import BrainPreview

    return BrainPreview(ctx, tile=64)


def upload(ctx, layout, seed=5):
    from services.brains import REGISTRY
    from utilities.gl_helpers import pack_brains

    p = np.asarray(REGISTRY[layout.modality].random(
        np.random.default_rng(seed), layout), dtype=np.float32).reshape(-1)
    buf = ctx.buffer(pack_brains([p], layout))
    return buf, p


def atlas(preview, tex):
    raw = np.frombuffer(tex.read(), dtype=np.uint8)
    return raw.reshape(tex.height, tex.width, 4)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_atlas_has_a_tile_per_unit_plus_the_total(ctx, preview, name):
    from services.brains import REGISTRY

    layout = REGISTRY[name].layout_from_settings({})
    buf, _ = upload(ctx, layout)
    tex = preview.render(layout, buf)
    n = preview.unit_count(layout)
    assert preview.grid ** 2 >= n + 1
    assert tex.width == tex.height == preview.grid * preview.tile
    buf.release()


@pytest.mark.parametrize("name", MODALITIES)
def test_the_atlas_draws_structure(ctx, preview, name):
    """The whole-brain tile must never be flat, and most unit tiles must not be.

    NOT every unit tile: a Gabor filter is localised over the full 4D input, so
    one whose centre sits far out in the two axes a slice holds at zero is
    legitimately invisible THERE. Measured over 40 random brains, 4.6% of
    filters are silent in the (L.axial, R.axial) slice (one brain reached 25%),
    while 0.0% are silent over the whole |x|<=2 box - so this is the slice
    hiding them, not a dead brain.
    """
    from services.brains import REGISTRY

    layout = REGISTRY[name].layout_from_settings({})
    buf, _ = upload(ctx, layout)
    tex = preview.render(layout, buf, channel=0, gain=1.0)
    img = atlas(preview, tex)
    n = preview.unit_count(layout)

    def tile_std(slot):
        gx, gy = slot % preview.grid, slot // preview.grid
        t = img[gy * preview.tile:(gy + 1) * preview.tile,
                gx * preview.tile:(gx + 1) * preview.tile, :3]
        return float(t.std())

    assert tile_std(0) > 1.0, f"{name}: the whole-brain tile is flat"
    flat = [s for s in range(1, n + 1) if tile_std(s) < 1.0]
    assert len(flat) <= 0.4 * n, (
        f"{name}: {len(flat)} of {n} unit tiles are flat - too many to be "
        f"slice geometry"
    )
    buf.release()


@pytest.mark.parametrize("name", MODALITIES)
def test_the_units_sum_to_the_whole_brain(ctx, name):
    """eval_brain_unit is only trustworthy if summing it reproduces eval_brain.
    Checked in float, off-screen, rather than through the 8-bit atlas."""
    from pathlib import Path

    from services.brains import REGISTRY
    from utilities.gl_helpers import pack_brains, read_shader, shader_prepend

    ctx_ = ctx
    layout = REGISTRY[name].layout_from_settings({})
    p = np.asarray(REGISTRY[name].random(np.random.default_rng(2), layout),
                   dtype=np.float32).reshape(-1)

    head = """#version 430
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer InBuf  { vec4 xs[]; };
layout(std430, binding = 1) buffer OutBuf { vec4 ys[]; };
layout(std430, binding = 3) buffer SumBuf { vec4 zs[]; };
"""
    main = """
void main(){
    uint i = gl_GlobalInvocationID.x;
    if(i >= xs.length()) return;
    ys[i] = eval_brain(0u, xs[i]);
    vec4 acc = vec4(0.0);
    for(int u = 0; u < BRAIN_SHAPE.x; u++) acc += eval_brain_unit(0u, u, xs[i]);
    zs[i] = acc;
}
"""
    sh = Path(__file__).resolve().parent.parent / "shaders"
    src = head
    for f in ("fourier4_4.glsl", "brains/_header.glsl", "brains/fourier.glsl",
              "brains/gabor.glsl", "brains/lenia.glsl", "brains/mlp.glsl",
              "brains/_dispatch.glsl"):
        src += (sh / f).read_text()
    prog = ctx_.compute_shader(src + main)

    N = 256
    xs = (np.random.default_rng(0).standard_normal((N, 4)) * 0.6).astype(np.float32)
    b_in = ctx_.buffer(xs.tobytes())
    b_out = ctx_.buffer(reserve=N * 16)
    b_sum = ctx_.buffer(reserve=N * 16)
    b_par = ctx_.buffer(pack_brains([p], layout))
    b_scratch = ctx_.buffer(reserve=4096)
    b_in.bind_to_storage_buffer(0)
    b_out.bind_to_storage_buffer(1)
    b_sum.bind_to_storage_buffer(3)
    b_par.bind_to_storage_buffer(4)
    b_scratch.bind_to_storage_buffer(2)
    shape = (tuple(layout.shape) + (0, 0, 0, 0))[:4]
    for k, v in (("BRAIN_MODALITY", REGISTRY[name].modality_id),
                 ("BRAIN_LEN", layout.length),
                 ("BRAIN_SHAPE", tuple(int(s) for s in shape))):
        if k in prog:
            prog[k].value = v
    prog.run(group_x=(N + 31) // 32)
    ctx_.finish()
    whole = np.frombuffer(b_out.read(), dtype=np.float32).reshape(N, 4)
    summed = np.frombuffer(b_sum.read(), dtype=np.float32).reshape(N, 4)

    if name == "mlp":
        # The output bias belongs to no hidden unit, so it is the difference.
        h = layout.shape[0]
        summed = summed + p[9 * h:]
    scale = max(float(np.abs(whole).mean()), 1e-6)
    err = float(np.abs(whole - summed).mean()) / scale
    assert err < 1e-4, f"{name}: units sum to {err:.3%} away from the whole"


@pytest.mark.parametrize("name", MODALITIES)
def test_the_random_projection_draws_structure(ctx, preview, name):
    """Held to the same standard as the fixed slices. A random plane cuts
    through the units at an angle, so if anything it should show MORE than an
    axis-aligned one, never less."""
    from services.brains import REGISTRY

    layout = REGISTRY[name].layout_from_settings({})
    buf, _ = upload(ctx, layout)
    img = atlas(preview, preview.render(layout, buf, axes=None, seed=3))
    assert float(img[..., :3].std()) > 1.0, f"{name}: the random plane is flat"
    buf.release()


def test_the_same_seed_renders_the_same_picture(ctx, preview):
    """End to end, through the shader. The basis is rebuilt every frame, so a
    seed that did not fully determine it would strobe on screen while every
    CPU-side test still passed."""
    from services.brains import default_layout

    layout = default_layout()
    buf, _ = upload(ctx, layout)
    a = atlas(preview, preview.render(layout, buf, axes=None, seed=7)).copy()
    b = atlas(preview, preview.render(layout, buf, axes=None, seed=7))
    assert np.array_equal(a, b)
    buf.release()


def test_reseeding_renders_a_different_picture(ctx, preview):
    from services.brains import default_layout

    layout = default_layout()
    buf, _ = upload(ctx, layout)
    a = atlas(preview, preview.render(layout, buf, axes=None, seed=7)).copy()
    b = atlas(preview, preview.render(layout, buf, axes=None, seed=8))
    assert not np.array_equal(a, b), "Reseed did not reach the shader"
    buf.release()


def test_the_seed_does_not_touch_an_axis_aligned_slice(ctx, preview):
    """The four fixed slices are fixed planes. If the seed leaked into them the
    Reseed button would silently change views it has no business changing."""
    from services.brains import default_layout

    layout = default_layout()
    buf, _ = upload(ctx, layout)
    a = atlas(preview, preview.render(layout, buf, axes=(0, 2), seed=1)).copy()
    b = atlas(preview, preview.render(layout, buf, axes=(0, 2), seed=99))
    assert np.array_equal(a, b)
    buf.release()


def test_a_random_plane_is_not_one_of_the_fixed_ones(ctx, preview):
    """Otherwise the new option is just a fifth copy of an existing view."""
    from services.brain_preview import AXES
    from services.brains import default_layout

    layout = default_layout()
    buf, _ = upload(ctx, layout)
    rnd = atlas(preview, preview.render(layout, buf, axes=None, seed=5)).copy()
    for label, axes in AXES:
        if axes is None:
            continue
        fixed = atlas(preview, preview.render(layout, buf, axes=axes))
        assert not np.array_equal(rnd, fixed), f"random == {label}"
    buf.release()


def test_a_blank_fourier_brain_draws_the_generated_rule(ctx, preview):
    """The reported bug, as a test.

    Fourier's "no rule loaded" IS an all-zero buffer - the shader answers it by
    generating a per-cohort rule. Rendering that buffer directly gives black
    tiles while the particles move perfectly well, which is what switching from
    MLP back to Fourier, and pressing Z, both looked like.
    """
    from services.brains import default_layout, is_fallback
    from utilities.gl_helpers import pack_brains

    layout = default_layout()
    blank = np.zeros(layout.length, dtype=np.float32)
    assert is_fallback(blank, layout), "the probe disagrees with the shader"
    buf = ctx.buffer(pack_brains([blank], layout))

    def dead_fraction(img, slot=0):
        """Share of a tile at exactly zero response.

        diverging(0) is pure black, so this is the direct measure of "the tile
        shows nothing". Whole-atlas std cannot say it - the background clear and
        the axis overlay carry variance of their own.
        """
        gx, gy = slot % preview.grid, slot // preview.grid
        t = img[gy * preview.tile:(gy + 1) * preview.tile,
                gx * preview.tile:(gx + 1) * preview.tile, :3]
        return float((t.max(axis=2) == 0).mean())

    dark = atlas(preview, preview.render(layout, buf, fallback=False))
    lit = atlas(preview, preview.render(layout, buf, fallback=True,
                                        rule_seed=0.25))
    assert dead_fraction(dark) > 0.95, "the blank buffer was not blank"
    assert dead_fraction(lit) < 0.5, "the generated rule drew nothing"
    buf.release()


def test_every_generated_unit_tile_is_drawn(ctx, preview):
    """Not just the whole-brain tile: eval_brain_unit needs its own fallback
    path, or the total is right and all ten unit tiles stay black."""
    from services.brain_preview import FALLBACK_UNITS
    from services.brains import default_layout
    from utilities.gl_helpers import pack_brains

    layout = default_layout()
    buf = ctx.buffer(pack_brains([np.zeros(layout.length, np.float32)], layout))
    img = atlas(preview, preview.render(layout, buf, fallback=True,
                                        rule_seed=0.5))
    assert preview.unit_count(layout, True) == FALLBACK_UNITS
    flat = []
    for slot in range(1, FALLBACK_UNITS + 1):
        gx, gy = slot % preview.grid, slot // preview.grid
        t = img[gy * preview.tile:(gy + 1) * preview.tile,
                gx * preview.tile:(gx + 1) * preview.tile, :3]
        if float(t.std()) < 1.0:
            flat.append(slot - 1)
    assert not flat, f"generated units {flat} drew nothing"
    buf.release()


def test_a_new_rule_seed_changes_the_generated_rule(ctx, preview):
    """Pressing the reset key picks a new rule_seed, which is the ONLY thing
    that changes when no brain is loaded - so if the seed does not reach the
    preview, the Inspector looks frozen while the particles change."""
    from services.brains import default_layout
    from utilities.gl_helpers import pack_brains

    layout = default_layout()
    buf = ctx.buffer(pack_brains([np.zeros(layout.length, np.float32)], layout))
    a = atlas(preview, preview.render(layout, buf, fallback=True,
                                      rule_seed=0.1)).copy()
    b = atlas(preview, preview.render(layout, buf, fallback=True,
                                      rule_seed=0.9))
    assert not np.array_equal(a, b), "rule_seed does not reach the preview"
    buf.release()


@pytest.mark.parametrize("name", MODALITIES)
def test_the_output_projection_draws_structure(ctx, preview, name):
    from services.brain_preview import CHANNEL_RANDOM
    from services.brains import REGISTRY

    layout = REGISTRY[name].layout_from_settings({})
    buf, _ = upload(ctx, layout)
    img = atlas(preview, preview.render(layout, buf, channel=CHANNEL_RANDOM,
                                        axes=None, seed=2))
    assert float(img[..., :3].std()) > 1.0, f"{name}: output projection is flat"
    buf.release()


def test_the_output_projection_is_not_one_of_the_channels(ctx, preview):
    from services.brain_preview import CHANNEL_RANDOM
    from services.brains import default_layout

    layout = default_layout()
    buf, _ = upload(ctx, layout)
    rnd = atlas(preview, preview.render(layout, buf, channel=CHANNEL_RANDOM,
                                        seed=4)).copy()
    for ch in range(CHANNEL_RANDOM):
        one = atlas(preview, preview.render(layout, buf, channel=ch))
        assert not np.array_equal(rnd, one), f"output projection == channel {ch}"
    buf.release()


def test_the_tile_uv_is_flipped_for_imgui(preview, ctx):
    """The framebuffer's origin is bottom-left and ImGui's is top-left; handing
    it the raw range draws every tile upside down."""
    from services.brains import default_layout

    buf, _ = upload(ctx, default_layout())
    preview.render(default_layout(), buf)
    (u0, v0), (u1, v1) = preview.uv_for(0)
    assert v0 > v1, "v is not flipped, tiles will render upside down"
    assert u0 < u1
    buf.release()
