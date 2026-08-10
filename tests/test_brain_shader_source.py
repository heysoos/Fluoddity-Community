"""No GPU in CI, so the GLSL contract is asserted on source text."""
from pathlib import Path

import numpy as np

from services.brains import MAX_BRAIN_FLOATS, default_layout
from utilities.gl_helpers import pack_brains

ROOT = Path(__file__).resolve().parent.parent


def read(p):
    return (ROOT / p).read_text()


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


def test_per_particle_buffer_is_sized_by_brain_len_not_max():
    """MAX_BRAIN_FLOATS per particle would be ~600 MB at 600k particles. The
    per-particle buffer must use the ACTIVE length.

    Asserts on the reserve= expression rather than the whole function, because
    the comment above it legitimately names MAX_BRAIN_FLOATS to explain why it
    is not used.
    """
    src = read("sim.py")
    i = src.index("def realloc_brain_buffers")
    body = src[i:i + 1200]
    j = body.index("reserve=")
    alloc = body[j:body.index(")", j)]
    assert "layout.length" in alloc, f"per-particle stride is wrong: {alloc!r}"
    assert "MAX_BRAIN_FLOATS" not in alloc, (
        f"per-particle buffer must not use the max stride: {alloc!r}"
    )


def test_the_writeback_is_gated_and_scoped_to_one_particle():
    """Two independent guards, and the second is not an optimisation.

    WRITE_RULES keeps the writeback off except on the frame click-to-adopt asks
    for it. WRITE_RULES_INDEX narrows it to the ONE particle that is read back -
    readback_rule() takes a single entity's slice and nothing else reads the
    buffer. Without it, all 600k particles re-derive their mutation to produce
    bytes nobody looks at: measured 13 ms a click, a hitch you can feel.

    Asserted on presence, not on surrounding syntax - anchoring these on exact
    shader text has broken on four separate edits that changed nothing real.
    """
    src = read("shaders/entity_update.glsl")
    assert "uniform int WRITE_RULES_INDEX" in src
    assert "WRITE_RULES_INDEX < 0" in src, "the writeback is not scoped"
    assert "brain_write(" in src

    host = read("sim.py")
    assert "'WRITE_RULES_INDEX'" in host, "the host never sets the scope"
    assert "_pending_entity_id" in host


def test_the_writeback_emits_the_fallback_rule_not_the_blank_buffer():
    """When the fallback is active the particle runs a GENERATED rule while
    brain_params still holds the blank buffer that triggered it. Writing the
    buffer made click-to-adopt copy zeros, which re-blanked slot 0 and flipped
    every cohort onto its own random rule.

    The writeback must use the SAME helper the evaluation does, mutation
    included, or the adopted rule is not the one the particle was running.
    Behaviour is covered on the GPU (tests/test_brain_readback_gpu.py); this is
    the CI-side guard, since CI has no GPU.
    """
    dispatch = read("shaders/brains/_dispatch.glsl")
    assert "fourier_noise(fallback_centers()" in dispatch, (
        "the evaluation does not run the generated fallback rule"
    )
    i = dispatch.index("void brain_write")
    body = dispatch[i:]
    assert "g_brain_fallback" in body and "fourier_write_fallback" in body, (
        "the writeback does not emit the generated rule when the fallback is on"
    )


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


def test_the_inspector_reproduces_the_generated_rule():
    """The sim decides at RUNTIME whether to read the buffer or generate a rule.
    The preview binds the same buffer, so unless it is told, it renders the
    blank buffer that TRIGGERED the fallback - black tiles while the particles
    move on the generated rule, which is exactly what was reported.
    """
    src = _code("shaders/brain_preview.frag")
    assert "g_brain_fallback" in src and "g_brain_seed" in src, (
        "the preview never sets the fallback state, so it cannot show it")

    dispatch = _code("shaders/brains/_dispatch.glsl")
    i = dispatch.index("vec4 eval_brain_unit")
    body = dispatch[i:dispatch.index("float brain_param_at", i)]
    assert "g_brain_fallback" in body, (
        "per-unit tiles ignore the fallback, so they stay black")
    assert "fallback_centers()" in body


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
