import numpy as np
import pytest

from services.auto_tournament_service import (
    Action,
    AutoTournamentService,
    Phase,
    snapshot_steps,
)
from services.tournament_service import TournamentService


class FakeScorer:
    """Scores each tile by its mean brightness, so tests control fitness."""

    def __init__(self):
        self.prompt = ""
        self.calls = 0

    def set_prompt(self, text, distractors=None):
        self.prompt = text

    def score(self, images):
        self.calls += 1
        return images.reshape(len(images), -1).mean(axis=1).astype(np.float32) / 255.0


def make(grid=2, **kw):
    ts = TournamentService(grid=grid)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=FakeScorer(), logger=None)
    svc.configure(steps_per_gen=100, snapshots_per_gen=2, sim_steps_per_frame=10, **kw)
    return svc, ts


def crops_for(svc, value=128):
    return np.full((svc.tournament.tiles, 224, 224, 3), value, dtype=np.uint8)


def run_one_generation(svc, value=128):
    """Drive update() until one full generation completes."""
    seen = []
    for _ in range(500):
        a = svc.update()
        seen.append(a)
        if a is Action.CAPTURE:
            svc.submit_frames(crops_for(svc, value))
        elif a is Action.SCORE:
            svc.score_and_tell()
            return seen
    raise AssertionError("generation never completed")


def test_snapshot_steps_are_evenly_spaced_and_end_on_the_last_step():
    assert snapshot_steps(300, 4) == [75, 150, 225, 300]
    assert snapshot_steps(300, 1) == [300]
    assert snapshot_steps(100, 3) == [33, 67, 100]


def test_starts_idle_and_does_nothing():
    svc, _ = make()
    assert svc.phase is Phase.IDLE
    assert svc.update() is Action.NONE


def test_first_action_after_start_is_write_rules():
    svc, _ = make()
    svc.start("coral")
    assert svc.update() is Action.WRITE_RULES
    assert svc.phase is Phase.ROLLOUT


def test_full_generation_sequence():
    svc, _ = make()
    svc.start("coral")
    seen = run_one_generation(svc)
    assert seen[0] is Action.WRITE_RULES
    assert seen.count(Action.CAPTURE) == 2
    assert seen[-1] is Action.SCORE
    assert svc.generation == 1
    assert svc.fitness is not None and len(svc.fitness) == 4


def test_captures_land_on_the_snapshot_steps():
    svc, _ = make()
    svc.configure(steps_per_gen=100, snapshots_per_gen=4, sim_steps_per_frame=5)
    svc.start("coral")
    at = []
    for _ in range(500):
        a = svc.update()
        if a is Action.CAPTURE:
            at.append(svc.step_in_gen)
            svc.submit_frames(crops_for(svc))
        elif a is Action.SCORE:
            svc.score_and_tell()
            break
    assert at == [25, 50, 75, 100]


@pytest.mark.parametrize("steps,snaps,per_frame", [(300, 4, 10), (50, 1, 1), (137, 3, 7)])
def test_snapshot_count_holds_for_odd_configurations(steps, snaps, per_frame):
    svc, _ = make()
    svc.configure(steps_per_gen=steps, snapshots_per_gen=snaps,
                  sim_steps_per_frame=per_frame)
    svc.start("coral")
    n = 0
    for _ in range(5000):
        a = svc.update()
        if a is Action.CAPTURE:
            n += 1
            svc.submit_frames(crops_for(svc))
        elif a is Action.SCORE:
            svc.score_and_tell()
            break
    assert n == snaps


def test_seed_is_shared_within_a_generation_and_changes_between():
    svc, _ = make()
    svc.start("coral")
    s0 = svc.gen_seed
    run_one_generation(svc)
    assert svc.gen_seed == s0 + 1
    run_one_generation(svc)
    assert svc.gen_seed == s0 + 2


def test_elite_injection_ranks_selected_tiles_first():
    svc, ts = make(grid=2)
    svc.start("coral")
    for _ in range(500):
        a = svc.update()
        if a is Action.CAPTURE:
            imgs = crops_for(svc, 10)
            imgs[2] = 250        # tile 2 is genuinely brightest
            svc.submit_frames(imgs)
        elif a is Action.SCORE:
            ts.toggle_select(0)  # but the human picks tile 0
            svc.score_and_tell()
            break
    assert svc.fitness[0] == svc.fitness.max()
    assert ts.selected == set(), "selection is cleared after tell"


def test_nan_fitness_is_replaced_with_the_generation_minimum():
    svc, _ = make()
    svc.scorer.score = lambda imgs: np.array([np.nan, 0.2, 0.5, 0.1], dtype=np.float32)
    svc.start("coral")
    run_one_generation(svc)
    assert np.all(np.isfinite(svc.fitness))
    assert svc.fitness[0] == pytest.approx(0.1)


def test_abort_restarts_the_generation():
    svc, _ = make()
    svc.start("coral")
    for _ in range(4):
        svc.update()
    g = svc.generation
    svc.abort_generation()
    assert svc.step_in_gen == 0
    assert svc.update() is Action.WRITE_RULES
    assert svc.generation == g


def test_pause_and_resume_preserve_optimizer_state():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    before = svc.optimizer.state_dict()["popsize"]
    svc.pause()
    assert svc.update() is Action.NONE
    svc.start("coral")
    assert svc.optimizer.state_dict()["popsize"] == before
    assert svc.generation == 1


def test_changing_the_prompt_does_not_restart_the_search():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    svc.set_prompt("a city map")
    assert svc.generation == 1
    assert svc.scorer.prompt == "a city map"


def test_reset_clears_the_optimizer():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    svc.reset()
    assert svc.generation == 0
    assert svc.phase is Phase.IDLE
    assert svc.optimizer is None


def test_reset_clears_the_logged_history(tmp_path):
    """The plot reads logger.history(); a stale curve after Reset makes the new
    search look like it started where the old one left off."""
    from services.run_logger import RunLogger

    svc, _ = make()
    svc.logger = RunLogger(root=tmp_path)
    svc.start("coral")
    run_one_generation(svc)
    assert svc.logger.history()["fit_best"] != []

    svc.reset()
    assert svc.logger.history()["fit_best"] == []
    svc.logger.close()


def test_sigma_is_logged_every_generation(tmp_path):
    """The sigma trace is plotted alongside fitness, so it must be recorded."""
    from services.run_logger import RunLogger

    svc, _ = make()
    svc.logger = RunLogger(root=tmp_path)
    svc.start("coral")
    run_one_generation(svc)
    run_one_generation(svc)
    sig = svc.logger.history()["sigma"]
    assert len(sig) == 2
    assert all(s > 0 for s in sig)
    svc.logger.close()


def test_population_is_written_into_the_tournament_service():
    svc, ts = make(grid=2)
    svc.start("coral")
    svc.update()
    assert len(ts.population) == 4
    assert all(g.shape == (10, 8) for g in ts.population)
    assert ts.is_dirty()


def test_fitness_is_averaged_over_snapshots():
    svc, _ = make()
    svc.configure(snapshots_per_gen=2)
    svc.start("coral")
    vals = [0, 200]
    i = 0
    for _ in range(500):
        a = svc.update()
        if a is Action.CAPTURE:
            svc.submit_frames(crops_for(svc, vals[i]))
            i += 1
        elif a is Action.SCORE:
            svc.score_and_tell()
            break
    assert svc.fitness[0] == pytest.approx(100 / 255.0, rel=1e-3)


def test_checkpoint_state_roundtrip():
    svc, _ = make()
    svc.start("coral")
    run_one_generation(svc)
    st = svc.checkpoint_state()
    assert st["generation"] == 1
    assert st["prompt"] == "coral"
    assert st["settings"]["grid"] == 2
    assert st["genome_spec_signature"] == "brain:80"

    svc2, _ = make()
    svc2.restore(st)
    assert svc2.generation == 1
    assert svc2.prompt == "coral"
    assert svc2.phase is Phase.PAUSED


def test_grid_change_without_reset_rebuilds_instead_of_crashing():
    """cmaes fixes popsize at construction and asserts on it in tell(). A grid
    change by any route that did not reset the optimizer must not crash."""
    svc, ts = make(grid=2)
    svc.start("coral")
    run_one_generation(svc)
    assert svc.optimizer.popsize == 4

    ts.set_grid(4)               # bypasses the CommandHandler reset path
    run_one_generation(svc)      # must not raise
    assert svc.optimizer.popsize == 16
    assert len(svc.fitness) == 16
