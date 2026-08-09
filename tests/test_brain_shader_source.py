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


def test_particle_brains_are_written_only_when_requested():
    """Writing BRAIN_LEN floats per particle is expensive; it happens for one
    frame when click-to-adopt asks for it."""
    src = read("shaders/entity_update.glsl")
    i = src.index("if(WRITE_RULES)")
    body = src[i:i + 1600]
    assert "particle_brains[" in body
    assert "BRAIN_LEN" in body


def test_the_writeback_emits_the_fallback_rule_not_the_blank_buffer():
    """When the fallback is active the particle runs a GENERATED rule while
    brain_params still holds the blank buffer that triggered it. Writing the
    buffer made click-to-adopt copy zeros, which re-blanked slot 0 and flipped
    every cohort onto its own random rule. Guarded at source as well as on the
    GPU (tests/test_brain_readback_gpu.py) because CI has no GPU."""
    src = read("shaders/entity_update.glsl")
    i = src.index("if(WRITE_RULES)")
    body = src[i:i + 1600]
    j = body.index("g_brain_fallback")
    assert "generate_random_centers" in body[j:], (
        "the fallback branch of the writeback does not emit the generated rule"
    )


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
