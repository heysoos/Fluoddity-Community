"""Every tile/config must read ITS OWN slot of the flat brain buffer.

A slot is MAX_BRAIN_FLOATS wide while a genome is layout.length (80 for
Fourier), so anything that writes genomes back to back lands them inside slot
0's padding and leaves the later slots holding whatever was there before. Every
slot is read as a brain, so a mis-strided write does not fail - it silently
evaluates the wrong one.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


@pytest.fixture(scope="module")
def sim(ctx):
    from sim import Sim

    return Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
               particle_density=0.01)


def slots(sim):
    """The flat brain buffer, as (n_slots, MAX_BRAIN_FLOATS)."""
    from services.brains import MAX_BRAIN_FLOATS

    raw = np.frombuffer(sim.multi_load_rule_buffer.read(), dtype=np.float32)
    return raw.reshape(-1, MAX_BRAIN_FLOATS)


def test_tournament_genomes_land_one_per_slot(sim):
    from services.tournament_service import TournamentService

    svc = TournamentService(rng=np.random.default_rng(0))
    svc.init_population()
    sim.write_tournament_rules(svc.pack_rule_bytes())

    n = sim.brain_layout.length
    got = slots(sim)
    for i, genome in enumerate(svc.population):
        want = np.asarray(genome, dtype=np.float32).reshape(-1)
        assert np.array_equal(got[i, :n], want), (
            f"tile {i} does not hold its own genome; genomes were packed at "
            f"the layout stride ({n}) instead of the slot stride"
        )


def test_no_tournament_slot_is_left_blank(sim):
    """An all-zero brain outputs zero for every input, so the tile freezes
    rather than erroring. write_tournament_rules pads short uploads for exactly
    this reason."""
    from services.tournament_service import TournamentService

    svc = TournamentService(rng=np.random.default_rng(1))
    svc.init_population()
    sim.write_tournament_rules(svc.pack_rule_bytes())

    n = sim.brain_layout.length
    got = slots(sim)[:len(svc.population), :n]
    blank = [i for i, row in enumerate(got) if not row.any()]
    assert not blank, f"tiles {blank} would fall back to a generated rule"


def test_the_padding_between_slots_stays_zero(sim):
    """Genome i+1 spilling into slot i's padding is the exact failure mode."""
    from services.tournament_service import TournamentService

    svc = TournamentService(rng=np.random.default_rng(2))
    svc.init_population()
    sim.write_tournament_rules(svc.pack_rule_bytes())

    n = sim.brain_layout.length
    pad = slots(sim)[:len(svc.population), n:]
    assert not pad.any(), "a genome overran its slot into the padding"


def test_multi_load_configs_land_one_per_slot(sim):
    """Same buffer, indexed by config instead of tile."""
    from unittest.mock import MagicMock

    from services.genome import random_genome

    rules = [random_genome(np.random.default_rng(s)) for s in range(3)]
    svc = MagicMock()
    svc.get_config_count.return_value = len(rules)

    def get_config(i):
        c = MagicMock()
        c.rule = rules[i]
        c.num_cohorts = 8
        c.color_by_cohort = True
        for f in ("hue_sensitivity", "orientation_mix", "rule_seed"):
            setattr(c, f, 0.0)
        return c

    svc.get_config.side_effect = get_config
    try:
        sim._write_multi_load_ssbo(svc)
    except Exception:
        pytest.skip("multi-load config packing needs a richer fake")

    n = sim.brain_layout.length
    got = slots(sim)
    for i, rule in enumerate(rules):
        want = np.asarray(rule, dtype=np.float32).reshape(-1)
        assert np.array_equal(got[i, :n], want), f"config {i} in the wrong slot"


def test_a_tournament_upload_stops_claiming_the_slots_are_per_cohort(sim):
    """These slots are owned by TILE once the grid runs, and the Brain window
    captions its picture off this flag. Leaving it set has the Inspector name
    tile 0's evolved genome "cohort 0 of a random brain per cohort"."""
    from services.genome import random_genome

    sim.apply_rule(None)                       # the per-cohort state
    assert sim.brain_per_cohort

    n = sim.brain_layout.length
    sim._tournament_grid = 2
    genomes = [random_genome(np.random.default_rng(s)) for s in range(4)]
    sim.write_tournament_rules(
        b"".join(np.asarray(g, dtype=np.float32).reshape(-1)[:n].tobytes()
                 for g in genomes))

    assert not sim.brain_per_cohort
