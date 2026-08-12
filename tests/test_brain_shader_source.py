"""No GPU in CI, so the GLSL contract is asserted on source text."""
from pathlib import Path

import numpy as np

from services.brains import MAX_BRAIN_FLOATS, default_layout
from utilities.gl_helpers import pack_brains

ROOT = Path(__file__).resolve().parent.parent


def read(p):
    return (ROOT / p).read_text()


def test_the_header_mirrors_the_python_slot_constants():
    """Three constants live in both services/brains/__init__.py and
    _header.glsl, and they index the same buffer from opposite sides. A drift
    is silent: the host writes one slot and the GPU reads another."""
    from services.brains import COHORT_BRAIN_SLOT0, MAX_COHORT_BRAINS

    src = read("shaders/brains/_header.glsl")
    for name, value in (("MAX_BRAIN_FLOATS", MAX_BRAIN_FLOATS),
                        ("COHORT_BRAIN_SLOT0", COHORT_BRAIN_SLOT0),
                        ("MAX_COHORT_BRAINS", MAX_COHORT_BRAINS)):
        assert f"#define {name} {value}\n" in src, (
            f"{name} is {value} in Python and something else in the shader"
        )


def test_mlp_glsl_mirrors_its_python_limits():
    """MAX_MLP_WIDTH sizes the deep path's locals and MAX_MLP_DEPTH sizes the
    BRAIN_LAYERS uniform, and both bound what layout_from_settings will build.
    A shader narrower than Python indexes past the end of an array."""
    from services.brains.mlp import MAX_DEEP_WIDTH, MAX_DEPTH

    src = read("shaders/brains/mlp.glsl")
    assert f"#define MAX_MLP_WIDTH {MAX_DEEP_WIDTH}\n" in src
    assert f"#define MAX_MLP_DEPTH {MAX_DEPTH}\n" in src


def test_the_depth_one_path_is_still_there():
    """Every genome, archive and config on disk was written by it. It is kept
    verbatim behind a uniform branch, so depth 1 cannot regress in what it
    COMPUTES - and MAX_MLP_WIDTH is what keeps it from regressing in what it
    costs."""
    src = read("shaders/brains/mlp.glsl")
    assert src.count("if (BRAIN_DEPTH <= 1)") == 2, (
        "brain_mlp and mlp_unit must each keep the old path"
    )
    assert "5 * h + j" in src and "9 * h" in src, "the old offsets are gone"


def test_header_declares_the_flat_brain_buffer():
    src = read("shaders/brains/_header.glsl")
    assert "buffer BrainBuffer" in src
    assert "float brain_params[]" in src
    assert "uniform int   BRAIN_MODALITY;" in src
    assert "uniform int   BRAIN_LEN;" in src
    assert "uniform ivec4 BRAIN_SHAPE;" in src


def test_dispatch_branches_on_every_modality():
    src = read("shaders/brains/_dispatch.glsl")
    for i, fn in enumerate(("brain_fourier", "brain_gabor", "brain_lenia")):
        assert f"BRAIN_MODALITY == {i}" in src
        assert fn in src
    assert "brain_mlp" in src


def test_entity_update_delegates_to_the_shared_dispatch():
    """black_box must not branch itself - adding a fifth modality has to touch
    only its own two files plus _dispatch."""
    src = read("shaders/entity_update.glsl")
    i = src.index("vec4 black_box")
    body = src[i:i + 200]
    assert "eval_brain" in body
    assert "BRAIN_MODALITY" not in body


def test_shader_prepend_order_puts_declarations_before_use():
    """shader_prepend inserts after #version, so the LAST prepend lands FIRST.
    Getting this backwards is a compile error that only shows on the GPU, which
    CI does not have."""
    src = read("sim.py")
    i = src.index("read_shader('shaders/entity_update.glsl')")
    block = src[i:i + 1600]
    order = [
        block.index("_dispatch.glsl"),
        block.index("f'shaders/brains/{_brain}.glsl'"),
        block.index("_header.glsl"),
        block.index("fourier4_4.glsl"),
    ]
    assert order == sorted(order), (
        "prepends must run dispatch -> brains -> header -> fourier4_4 so the "
        "compiled order is fourier4_4, header, brains, dispatch"
    )


def test_fourier_glsl_has_the_contract_signature():
    src = read("shaders/brains/fourier.glsl")
    assert "vec4 brain_fourier(uint base, vec4 x)" in src


def test_fourier_glsl_is_pure():
    """It may read only brain_params, base, x and BRAIN_SHAPE. Touching entity
    state would break the Brain Inspector, which calls it from a fragment pass."""
    src = read("shaders/brains/fourier.glsl")
    for forbidden in ("entities[", "get_can(", "canvas", "e.pos", "e.vel"):
        assert forbidden not in src, f"brain function is not pure: {forbidden}"


def test_the_adopted_brain_buffer_is_one_brain():
    """It holds the ONE particle that was read back, so it must scale with
    neither the particle count nor MAX_BRAIN_FLOATS.

    Asserts on the reserve= expression rather than the whole function, because
    the comment above it legitimately names both to explain why neither is used.
    """
    src = read("sim.py")
    i = src.index("def realloc_brain_buffers")
    body = src[i:i + 1200]
    j = body.index("reserve=")
    alloc = body[j:body.index(")", j)]
    assert "layout.length" in alloc, f"the stride is wrong: {alloc!r}"
    assert "MAX_BRAIN_FLOATS" not in alloc, (
        f"the adopted brain must not use the max stride: {alloc!r}"
    )
    assert "entity_count" not in alloc, (
        f"the adopted brain must not be a row per particle: {alloc!r}"
    )


def test_the_writeback_is_gated_and_scoped_to_one_particle():
    """Two independent guards, and the second is not an optimisation.

    WRITE_RULES keeps the writeback off except on the frame click-to-adopt asks
    for it. WRITE_RULES_INDEX narrows it to the ONE particle that is read back,
    which is the whole of that buffer. Without it, all 600k particles re-derive
    their mutation to produce bytes nobody looks at - and there is nowhere to
    put them.

    Asserted on presence, not on surrounding syntax - anchoring these on exact
    shader text has broken on four separate edits that changed nothing real.
    """
    src = read("shaders/entity_update.glsl")
    assert "uniform int WRITE_RULES_INDEX" in src
    assert "uint(WRITE_RULES_INDEX) == index" in src, (
        "the writeback is not scoped to one particle"
    )
    assert "WRITE_RULES_INDEX < 0" not in src, (
        "the write-all branch is back, and there is one brain to write into"
    )
    assert "brain_write(brain_base, 0u)" in src, "the write is not at offset 0"

    host = read("sim.py")
    assert "'WRITE_RULES_INDEX'" in host, "the host never sets the scope"
    assert "_pending_entity_id" in host


def test_there_is_no_blank_brain_state_left():
    """A brain is always in the buffer, so nothing downstream has to ask which
    of two it is looking at.

    The GPU used to detect an all-zero slot and generate a Fourier rule instead.
    That gave Fourier a second, hidden way to have a brain that the other three
    modalities had no equivalent for - and with MUTATION_SCALE defaulting to 0.0
    and num_cohorts to 64, the difference was 64 independent rules against 64
    copies of one. Cohort brains come from the host now, for every modality.
    """
    entity = _code("shaders/entity_update.glsl")
    for gone in ("g_brain_fallback", "bool blank ="):
        assert gone not in entity, f"{gone} survived the removal"
    assert "BRAIN_PER_COHORT" in entity, "cohort brains are never selected"

    dispatch = _code("shaders/brains/_dispatch.glsl")
    for gone in ("fallback_centers", "fourier_write_fallback",
                 "g_brain_fallback"):
        assert gone not in dispatch, f"{gone} survived the removal"


def test_the_host_generates_a_brain_per_cohort():
    """The replacement, on the host side, and it must apply to every modality -
    the point of moving it off the GPU."""
    host = read("sim.py")
    assert "_write_cohort_brains" in host
    assert "'BRAIN_PER_COHORT'" in host, "the uniform is never pushed"

    brains = read("services/brains/__init__.py")
    assert "def generated_brains" in brains
    # Via the registry, so it cannot be Fourier-only ever again.
    i = brains.index("def generated_brains")
    assert "get(layout.modality)" in brains[i:i + 1200]


def _code(path: str) -> str:
    """Shader source with // comments stripped.

    Scanning the raw text conflates a CALL with a mention: a comment explaining
    why the preview must not call brain_fourier() failed the check below.
    """
    import re

    return re.sub(r"//[^\n]*", "", read(path))


def test_the_inspector_does_not_reimplement_the_brain():
    """It calls eval_brain/eval_brain_unit, which are the same functions the
    compute shader dispatches. A second implementation would drift, and the
    picture would stop being evidence about what the particles do."""
    src = _code("shaders/brain_preview.frag")
    assert "eval_brain_unit(" in src and "eval_brain(" in src
    for fn in ("brain_fourier", "brain_gabor", "brain_lenia", "brain_mlp"):
        assert f"{fn}(" not in src, (
            f"the preview reaches past the dispatch to {fn}")


def test_the_inspector_needs_no_special_case_to_show_the_running_brain():
    """Slot 0 always holds a real brain - the loaded rule, or cohort 0's
    generated one - so the preview just draws it.

    It used to have to re-derive the shader's "is this buffer blank" verdict,
    and because it did not, it drew the blank buffer: black tiles while the
    particles ran. Deleting the state deletes the class of bug.
    """
    src = _code("shaders/brain_preview.frag")
    for gone in ("g_brain_fallback", "PREVIEW_FALLBACK", "PREVIEW_SEED"):
        assert gone not in src, f"{gone} survived the removal"


def test_the_inspector_can_isolate_one_unit():
    src = read("shaders/brain_preview.frag")
    assert "PREVIEW_UNIT" in src


def test_every_modality_exposes_a_per_unit_function():
    """Needed because a unit cannot be isolated by pointing `base` at it:
    Fourier's phase offset is derived from the centre INDEX, so centre 5 seen at
    index 0 is a different function from the one the particles run."""
    dispatch = read("shaders/brains/_dispatch.glsl")
    for fn in ("fourier_unit", "gabor_unit", "lenia_unit", "mlp_unit"):
        assert fn in dispatch, f"{fn} is not dispatched"
        assert fn in read(f"shaders/brains/{fn.split('_')[0]}.glsl")


def test_the_brain_window_passes_textures_the_way_this_imgui_wants():
    """imgui_bundle needs ImTextureRef/ImVec2, not a raw int and a tuple. A
    mismatch only shows when the window is opened, which no test does."""
    src = read("ui/brain_window.py")
    i = src.index("imgui.image(")
    call = src[i:i + 200]
    assert "ImTextureRef" in call and "ImVec2" in call


def test_pack_brains_pads_each_brain_to_the_stride():
    layout = default_layout()
    a = np.arange(layout.length, dtype=np.float32)
    b = np.zeros(layout.length, dtype=np.float32)
    blob = pack_brains([a, b], layout)
    assert len(blob) == 2 * MAX_BRAIN_FLOATS * 4
    got = np.frombuffer(blob, dtype=np.float32)
    assert np.array_equal(got[:layout.length], a)
    assert np.all(got[layout.length:MAX_BRAIN_FLOATS] == 0)
    assert np.array_equal(got[MAX_BRAIN_FLOATS:MAX_BRAIN_FLOATS + layout.length], b)
