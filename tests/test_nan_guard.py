"""A non-finite particle must never reach the canvas.

Symptom that prompted this: black boxes appearing on the tournament render one
by one, growing, and not aligned to the tile grid.

Chain of causation:
  1. an extreme physics draw overflows the velocity integration -> Inf -> NaN
  2. the tournament tile clamp uses `<` and `>`, and NaN fails every
     comparison, so the particle is NOT confined to its tile
  3. its brush quad splats at an arbitrary canvas position, writing NaN
  4. canvas.frag's getBlur() is a 5-tap kernel, so every frame each NaN texel
     poisons its four neighbours - the hole grows and never heals

Step 2 is why the boxes are not tile-aligned, and step 4 is why they grow.
"""
import numpy as np
import pytest

from sim import SIZE_OF_ENTITY_STRUCT

STRIDE = SIZE_OF_ENTITY_STRUCT // 4      # 12 floats: pos2 vel2 size1 pad3 col4


def _shader_src():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent
            / "shaders" / "entity_update.glsl").read_text()


def test_guard_runs_before_the_commit():
    """A guard after `entities[index]=e` would let the bad state through for a
    frame, which is all it takes to poison the canvas."""
    src = _shader_src()
    guard = src.index("Non-finite guard")
    commit = src.index("entities[index]=e;", guard)
    assert guard < commit


def test_guard_does_not_rely_on_isnan():
    """Drivers compiling with fast-math assumptions may fold isnan() away."""
    import re

    src = _shader_src()
    tail = src[src.index("Non-finite guard"):]
    tail = tail[:tail.index("entities[index]=e;")]
    code = re.sub(r"//.*", "", tail)          # the rationale mentions isnan
    assert "lessThan(abs(" in code
    assert "isnan" not in code and "isinf" not in code


@pytest.mark.gpu
@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf, 1e30])
def test_a_poisoned_particle_is_respawned_not_committed(poison):
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    from sim import Sim
    from state import SimState

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    st = SimState()
    st.HAZARD_RATE = 0.0          # else the hazard reset masks the guard
    sim.apply_state(st)
    sim.apply_tournament(True, grid=4, mutation=0.0)
    sim.reset()
    sim.entity_update(ctx)
    ctx.finish()

    # entity_update() does not advance frame_count, and frame_count == 0 makes
    # the shader reset EVERY particle unconditionally - which would wipe the
    # poison and make this test vacuous. Move past the reset frame first.
    sim.frame_count = 25

    buf = np.frombuffer(sim.entities.read(), dtype=np.float32).copy()
    buf = buf.reshape(-1, STRIDE)
    buf[7, 0:2] = poison        # position
    buf[9, 2:4] = poison        # velocity
    buf[11, 4] = poison         # size
    sim.entities.write(buf.tobytes())

    sim.entity_update(ctx)
    ctx.finish()

    out = np.frombuffer(sim.entities.read(), dtype=np.float32).reshape(-1, STRIDE)
    assert np.isfinite(out).all(), "a non-finite particle survived the update"
    assert (np.abs(out) < 1e6).all()

    ctx.release()


@pytest.mark.gpu
def test_poison_does_not_escape_its_tile():
    """The guard respawns into the tile, so isolation survives. Without it the
    NaN particle draws outside its tile - the reason the black boxes did not
    line up with the grid."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    from services.cohort_tiling import tile_of
    from sim import Sim
    from state import SimState

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    st = SimState()
    st.HAZARD_RATE = 0.0
    sim.apply_state(st)
    sim.apply_tournament(True, grid=4, mutation=0.0)
    sim.reset()
    sim.entity_update(ctx)
    ctx.finish()
    sim.frame_count = 25          # past the unconditional reset frame

    n = sim.entity_count
    victim = n // 3
    buf = np.frombuffer(sim.entities.read(), dtype=np.float32).copy().reshape(-1, STRIDE)
    buf[victim, 0:2] = np.nan
    sim.entities.write(buf.tobytes())
    sim.entity_update(ctx)
    ctx.finish()

    pos = np.frombuffer(sim.entities.read(), dtype=np.float32).reshape(-1, STRIDE)[victim, 0:2]
    t = tile_of(victim, n, 4)
    tx, ty = t % 4, t // 4
    assert -1.0 - 1e-3 <= pos[0] <= 1.0 + 1e-3
    lo_x, hi_x = -1.0 + tx * 0.5, -1.0 + (tx + 1) * 0.5
    lo_y, hi_y = -1.0 + ty * 0.5, -1.0 + (ty + 1) * 0.5
    assert lo_x - 1e-3 <= pos[0] <= hi_x + 1e-3, "respawned outside its tile"
    assert lo_y - 1e-3 <= pos[1] <= hi_y + 1e-3, "respawned outside its tile"

    ctx.release()


@pytest.mark.gpu
def test_existing_canvas_damage_heals_itself():
    """The particle guard cannot undo NaN already stored in the canvas: getBlur
    reads the PREVIOUS frame, so a stored NaN keeps poisoning its neighbours
    forever. canvas.frag scrubs it so the holes close on their own instead of
    needing a manual clear."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    from sim import Sim
    from state import SimState

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    st = SimState()
    st.HAZARD_RATE = 0.0
    sim.apply_state(st)
    sim.reset()
    sim.update(ctx)
    ctx.finish()

    # Poison a patch of the canvas the way an escaped particle would have.
    tex = sim.can_textures[sim.can_read_index]
    w, h = tex.size
    poisoned = np.full((16, 16, tex.components), np.nan, dtype=np.float32)
    tex.write(poisoned.tobytes(), viewport=(w // 3, h // 3, 16, 16))
    ctx.finish()

    for _ in range(4):
        sim.update(ctx)
    ctx.finish()

    for t in sim.can_textures:
        data = np.frombuffer(t.read(), dtype=np.float32)
        assert np.isfinite(data).all(), "NaN survived in the canvas"

    ctx.release()
