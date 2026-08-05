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
