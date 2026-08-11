import numpy as np
from services.tournament_service import TournamentService
from services.genome import GENOME_SHAPE


def make_service(seed=0):
    svc = TournamentService(rng=np.random.default_rng(seed))
    svc.init_population()
    return svc


def test_init_population_makes_16_distinct_genomes():
    svc = make_service()
    assert len(svc.population) == svc.tiles == 16
    for g in svc.population:
        assert g.shape == GENOME_SHAPE and g.dtype == np.float32
    # Not all identical
    assert not np.array_equal(svc.population[0], svc.population[1])
    assert svc.initialized is True


def test_toggle_select():
    svc = make_service()
    svc.toggle_select(3)
    assert svc.selected == {3}
    svc.toggle_select(3)
    assert svc.selected == set()


def test_next_generation_keeps_selected_genomes_pinned():
    svc = make_service()
    svc.inject_randoms = 0
    svc.toggle_select(5)
    svc.toggle_select(10)
    keep5 = svc.population[5].copy()
    keep10 = svc.population[10].copy()
    svc.next_generation()
    assert len(svc.population) == 16
    assert np.array_equal(svc.population[5], keep5)   # survivor stays in its tile
    assert np.array_equal(svc.population[10], keep10)


def test_next_generation_injects_requested_random_count():
    svc = make_service()
    svc.inject_randoms = 3
    svc.toggle_select(0)                     # 1 survivor => 15 empty tiles
    before = [g.copy() for g in svc.population]
    svc.next_generation()
    changed = sum(0 if np.array_equal(svc.population[i], before[i]) else 1 for i in range(16))
    assert changed >= svc.inject_randoms
    assert np.array_equal(svc.population[0], before[0])


def test_next_generation_with_no_selection_mutates_all():
    svc = make_service()
    svc.inject_randoms = 0
    before = [g.copy() for g in svc.population]
    svc.next_generation()
    # With nothing selected, every tile should change (whole population mutated)
    assert all(not np.array_equal(svc.population[i], before[i]) for i in range(16))


def test_next_generation_clears_selection():
    svc = make_service()
    svc.toggle_select(2)
    svc.toggle_select(9)
    svc.next_generation()
    assert svc.selected == set()


def test_reset_clears_selection():
    svc = make_service()
    svc.toggle_select(4)
    svc.reset()
    assert svc.selected == set()


def test_undo_restores_selection_that_bred_the_generation():
    svc = make_service()
    svc.toggle_select(6)
    svc.next_generation()
    assert svc.selected == set()
    svc.undo()
    assert svc.selected == {6}


def test_undo_restores_previous_population():
    svc = make_service()
    snapshot = [g.copy() for g in svc.population]
    svc.next_generation()
    svc.undo()
    for i in range(16):
        assert np.array_equal(svc.population[i], snapshot[i])


def test_pack_rule_bytes_layout():
    svc = make_service()
    data = svc.pack_rule_bytes()
    assert len(data) == 16 * 320   # 16 genomes * 80 floats * 4 bytes
    arr = np.frombuffer(data, dtype=np.float32).reshape(16, 10, 8)
    assert np.array_equal(arr[7], svc.population[7])


def test_dirty_flag_lifecycle():
    svc = make_service()
    assert svc.is_dirty() is True     # init marks dirty
    svc.clear_dirty()
    assert svc.is_dirty() is False
    svc.toggle_select(1)
    svc.next_generation()
    assert svc.is_dirty() is True


def test_grid_is_configurable():
    svc = TournamentService(grid=6)
    svc.init_population()
    assert svc.tiles == 36
    assert len(svc.population) == 36


def test_set_grid_resizes_and_clears_selection():
    svc = TournamentService(grid=4)
    svc.init_population()
    svc.toggle_select(3)
    svc.set_grid(2)
    assert svc.tiles == 4
    assert len(svc.population) == 4
    assert svc.selected == set()
    assert svc.is_dirty()


def test_set_grid_to_same_value_is_a_no_op():
    svc = TournamentService(grid=4)
    svc.init_population()
    before = [g.copy() for g in svc.population]
    svc.set_grid(4)
    assert all((a == b).all() for a, b in zip(before, svc.population))


# ---- the active modality ------------------------------------------------
#
# init_population, set_grid and reset all called services.genome.random_genome,
# which is hardcoded to Fourier's (10, 8). Under any other modality the manual
# tournament bred 80-float genomes and handed them to a GPU expecting 168, 120
# or 148 - and sim.write_tournament_rules re-slices at layout.length, so 16
# genomes arrived as 7 straddling ones plus 9 generated fillers.

import pytest  # noqa: E402

from services.brains import REGISTRY  # noqa: E402

MODALITIES = sorted(REGISTRY)


def a_service(name, grid=2, seed=0):
    lay = REGISTRY[name].layout_from_settings({})
    svc = TournamentService(grid=grid, rng=np.random.default_rng(seed),
                            layout=lay)
    svc.init_population()
    return svc, lay


@pytest.mark.parametrize("name", MODALITIES)
def test_a_fresh_population_matches_the_active_layout(name):
    svc, lay = a_service(name)
    assert all(np.asarray(g).size == lay.length for g in svc.population)
    assert all(np.asarray(g).dtype == np.float32 for g in svc.population)


@pytest.mark.parametrize("name", MODALITIES)
def test_breeding_stays_in_the_active_layout(name):
    svc, lay = a_service(name)
    svc.crossover_enabled = True
    svc.toggle_select(0)
    svc.toggle_select(1)
    svc.next_generation()
    assert all(np.asarray(g).size == lay.length for g in svc.population)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_pack_is_exactly_one_genome_per_tile(name):
    svc, lay = a_service(name, grid=4)
    flat = np.frombuffer(svc.pack_rule_bytes(), dtype=np.float32)
    assert flat.size == svc.tiles * lay.length


@pytest.mark.parametrize("name", MODALITIES)
def test_mutation_moves_the_genome_without_leaving_the_layout(name):
    svc, lay = a_service(name)
    svc.mutation_strength = 0.5
    svc.inject_randoms = 0
    svc.toggle_select(0)
    parent = np.asarray(svc.population[0]).reshape(-1).copy()
    svc.next_generation()
    kids = [np.asarray(g).reshape(-1) for i, g in enumerate(svc.population)
            if i != 0]
    assert all(k.size == lay.length for k in kids)
    assert any(not np.allclose(k, parent) for k in kids)


def test_a_layout_switch_repopulates_at_the_new_width():
    svc, _ = a_service("fourier")
    gab = REGISTRY["gabor"].layout_from_settings({})
    svc.set_layout(gab)
    assert all(np.asarray(g).size == gab.length for g in svc.population)
    assert svc.selected == set()


def test_a_scales_only_layout_change_leaves_the_population_alone():
    """Scales change what a genome MEANS, not how wide it is. Rerolling here
    would throw away the tiles the user is in the middle of selecting."""
    m = REGISTRY["fourier"]
    svc = TournamentService(grid=2, layout=m.layout_from_settings({}))
    svc.init_population()
    before = [np.asarray(g).copy() for g in svc.population]
    svc.set_layout(m.layout_from_settings({"freq_scale": 1.5}))
    assert all(np.array_equal(a, b) for a, b in zip(before, svc.population))


def test_the_default_layout_is_still_fourier_shaped():
    """Everything above this section constructs TournamentService without a
    layout and expects (10, 8); that presentation must survive."""
    svc = TournamentService(grid=2)
    svc.init_population()
    assert all(np.asarray(g).shape == GENOME_SHAPE for g in svc.population)
