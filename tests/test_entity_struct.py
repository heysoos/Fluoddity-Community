"""The entity struct is 32 bytes, and what it no longer stores is derivable.

Two of the twelve floats were redundant. `cohort` is a pure function of the
particle index - `get_cohort()` computes it in the shader and nothing ever read
the stored copy - and `color` was a vec4 whose brightness and alpha were the
same constants for every particle on every frame. Only the hue varied, and the
saturation only ever takes two values.

The buffer is read and written in full on every physics step, so this is a
third off the hottest traffic in the engine.

Saturation IS still stored, and that is not an oversight: reset() deliberately
writes a DESATURATED particle where main() writes 0.8, and HAZARD_RATE respawns
particles continuously - so hardcoding 0.8 would change what a hazard-heavy
preset looks like. It costs nothing, because a vec2 member forces the struct to
an 8-byte alignment and the slot would be padding either way.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ("entity_update.glsl", "brush.vert", "cam_brush.vert")


def struct_of(name: str) -> str:
    src = (ROOT / "shaders" / name).read_text(encoding="utf-8")
    m = re.search(r"struct Entity \{(.*?)\};", src, re.S)
    assert m, f"{name} declares no Entity struct"
    # Comments are per-file prose; the FIELDS are what has to agree.
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return " ".join(body.split())


def test_the_struct_is_thirty_two_bytes():
    from sim import SIZE_OF_ENTITY_STRUCT

    assert SIZE_OF_ENTITY_STRUCT == 32


def test_all_three_shaders_declare_the_same_fields():
    """entity_update writes the buffer; both vertex shaders read it. A field
    order that disagrees reads another field's bytes and never fails."""
    shapes = {name: struct_of(name) for name in SHADERS}
    assert len(set(shapes.values())) == 1, shapes


def test_python_and_the_shader_agree_on_the_width():
    from sim import SIZE_OF_ENTITY_STRUCT

    floats = struct_of("entity_update.glsl")
    counted = (2 + 2                                   # pos, vel
               + len(re.findall(r"float (\w+);", floats))
               + sum(int(n) for n in re.findall(r"float \w+\[(\d+)\];", floats)))
    assert counted == SIZE_OF_ENTITY_STRUCT // 4


def test_nothing_reads_a_stored_cohort_any_more():
    for name in SHADERS:
        src = (ROOT / "shaders" / name).read_text(encoding="utf-8")
        src = re.sub(r"//[^\n]*", "", src)
        # Not a bare ".cohort" search: .cohort_sweep and .cohorts are other
        # things entirely and both are still live.
        assert not re.search(r"\.cohort(?!\w)", src), (
            f"{name} still reads the removed field")


def test_the_vertex_shaders_rebuild_the_constants():
    """Brightness and alpha were identical for every particle every frame."""
    for name in ("brush.vert", "cam_brush.vert"):
        src = (ROOT / "shaders" / name).read_text(encoding="utf-8")
        assert "0.045" in src, f"{name} must supply the deposit alpha itself"


@pytest.mark.gpu
def test_a_reset_particle_is_still_desaturated():
    """This is the whole reason saturation survived the slim-down.

    reset() writes 0 where the update writes 0.8, and HAZARD_RATE respawns
    particles every frame - so a hardcoded 0.8 in the vertex shaders would
    change what a hazard-heavy preset looks like.
    """
    import numpy as np
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"no GL context: {exc}")

    from sim import SIZE_OF_ENTITY_STRUCT, Sim
    from state import SimState

    SAT = 6
    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    sim.apply_state(SimState())
    sim.reset()

    def column(i):
        raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
        return raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)[:sim.entity_count, i]

    sim.entity_update(ctx)                         # frame_count 0 resets every one
    ctx.finish()
    assert (column(SAT) == 0.0).all(), "reset must leave the particle desaturated"

    sim.frame_count = 25                           # past the unconditional reset
    sim.entity_update(ctx)
    ctx.finish()
    assert np.allclose(column(SAT), 0.8)

    ctx.release()


def test_nobody_hardcodes_the_stride():
    """A stale stride does not fail - it reads other fields AS positions.

    Four files carried a literal 12 through the slim-down. Two of them are GPU
    tests that then passed in isolation and failed in a full run, because what
    they were comparing was whatever the misaligned columns happened to hold.
    """
    offenders = []
    for path in list(ROOT.glob("tests/*.py")) + list(ROOT.glob("tools/*.py")):
        src = path.read_text(encoding="utf-8")
        for read in re.finditer(r"entities\.read\(\)", src):
            # Only the reshape belonging to THIS read; other buffers in the
            # same file have their own widths and are none of our business.
            window = src[read.end():read.end() + 200]
            m = re.search(r"reshape\(\s*-1\s*,\s*(\d+)\s*\)", window)
            if m:
                offenders.append(f"{path.name}: reshape(-1, {m.group(1)})")
    assert not offenders, (
        "use SIZE_OF_ENTITY_STRUCT // 4: " + ", ".join(offenders))
