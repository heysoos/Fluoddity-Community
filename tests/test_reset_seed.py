"""Initial conditions must reshuffle between generations.

AutoTournamentService.gen_seed was computed and written to the JSONL but never
applied to anything, so every generation replayed the identical starting layout.
A genome was therefore scored on one fixed arrangement and could be rewarded for
suiting it rather than for being good.

RESET_SEED carries gen_seed into reset(). It is 0.0 everywhere else, and adding
0.0 inside the hash arguments is a no-op, so ordinary resets are unchanged.
"""
import numpy as np
import pytest

from sim import Sim, SIZE_OF_ENTITY_STRUCT


def test_reset_seed_defaults_to_zero():
    """Normal play must keep the original deterministic reset."""
    import inspect

    src = inspect.getsource(Sim.__init__)
    assert "self.reset_seed = 0.0" in src


def test_shader_threads_the_seed_into_every_draw():
    """Guards against seeding only one of the hashes and leaving the layout
    effectively fixed."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "shaders" / "entity_update.glsl").read_text()
    assert "uniform float RESET_SEED;" in src
    body = src[src.index("void reset(uint index)"):]
    body = body[:body.index("\n}")]
    assert body.count("RESET_SEED") >= 4, (
        "seed must reach the base position, the velocity and the tournament "
        "scatter, or generations will still share a layout")


@pytest.mark.gpu
def test_different_seeds_give_different_layouts_and_same_seed_repeats():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    sim.apply_tournament(True, grid=4, mutation=0.0)

    def layout(seed):
        sim.reset_seed = seed
        sim.reset()
        sim.entity_update(ctx)
        ctx.finish()
        raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
        return raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)[:, 0:2].copy()

    a = layout(1000.0)
    b = layout(1001.0)
    a_again = layout(1000.0)

    assert np.allclose(a, a_again), "same seed must reproduce exactly"
    moved = np.linalg.norm(a - b, axis=1)
    assert (moved > 1e-6).mean() > 0.9, "a new seed must move nearly every particle"

    ctx.release()


@pytest.mark.gpu
def test_seed_zero_keeps_particles_inside_their_tile():
    """Reshuffling must not break tile isolation."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    from services.cohort_tiling import tile_of

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    sim.apply_tournament(True, grid=4, mutation=0.0)
    sim.reset_seed = 7777.0
    sim.reset()
    sim.entity_update(ctx)
    ctx.finish()

    raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
    pos = raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)[:, 0:2]
    n = sim.entity_count
    step = max(1, n // 2000)
    bad = 0
    for i in range(0, n, step):
        t = tile_of(i, n, 4)
        tx, ty = t % 4, t // 4
        lo_x, hi_x = -1.0 + tx * 0.5, -1.0 + (tx + 1) * 0.5
        lo_y, hi_y = -1.0 + ty * 0.5, -1.0 + (ty + 1) * 0.5
        x, y = pos[i]
        if not (lo_x - 1e-3 <= x <= hi_x + 1e-3 and lo_y - 1e-3 <= y <= hi_y + 1e-3):
            bad += 1
    assert bad == 0, f"{bad} particles spawned outside their home tile"

    ctx.release()
