import numpy as np
from services.tournament_service import TournamentService
from services.genome import GENOME_SHAPE


def make_service(seed=0):
    svc = TournamentService(rng=np.random.default_rng(seed))
    svc.init_population()
    return svc


def test_init_population_makes_16_distinct_genomes():
    svc = make_service()
    assert len(svc.population) == svc.TILES == 16
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
