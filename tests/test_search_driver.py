import numpy as np
import pytest

from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.prompt_driver import PromptDriver
from services.tournament_service import TournamentService


class FakeScorer:
    """Scores each tile by mean brightness, matching test_auto_tournament_service."""

    def __init__(self):
        self.prompt = ""

    def set_prompt(self, text, distractors=None):
        self.prompt = text

    def score(self, images):
        return images.reshape(len(images), -1).mean(axis=1).astype(np.float32) / 255.0


def make(grid=2):
    ts = TournamentService(grid=grid)
    ts.init_population()
    return PromptDriver(ts, FakeScorer()), ts


def crops(n, value=128):
    return np.full((n, 224, 224, 3), value, dtype=np.uint8)


def test_ask_returns_the_requested_shape_and_dtype():
    d, _ = make()
    z = d.ask(4)
    assert z.shape == (4, BRAIN_SPEC.dim)
    assert z.dtype == np.float32


def test_optimizer_is_built_lazily_on_first_ask():
    d, _ = make()
    assert d.optimizer is None
    d.ask(4)
    assert d.optimizer is not None
    assert d.optimizer.popsize == 4


def test_ask_rebuilds_when_the_population_size_changes():
    """cmaes fixes popsize at construction and asserts on it in tell()."""
    d, _ = make()
    d.ask(4)
    d.ask(16)
    assert d.optimizer.popsize == 16


def test_tell_averages_the_score_over_snapshots():
    d, _ = make()
    z = d.ask(4)
    fit = d.tell(z, [crops(4, 0), crops(4, 200)])
    assert fit[0] == pytest.approx(100 / 255.0, rel=1e-3)


def test_tell_replaces_non_finite_scores_with_the_generation_minimum():
    d, _ = make()
    d.scorer.score = lambda imgs: np.array([np.nan, 0.2, 0.5, 0.1], dtype=np.float32)
    z = d.ask(4)
    fit = d.tell(z, [crops(4)])
    assert np.all(np.isfinite(fit))
    assert fit[0] == pytest.approx(0.1)
    assert d.status()["nan_replaced"] == 1


def test_tell_with_no_snapshots_returns_zeros():
    d, _ = make()
    z = d.ask(4)
    assert np.array_equal(d.tell(z, []), np.zeros(4, np.float32))


def test_elite_injection_ranks_selected_tiles_first_and_clears_the_selection():
    d, ts = make()
    z = d.ask(4)
    imgs = crops(4, 10)
    imgs[2] = 250                 # tile 2 is genuinely brightest
    ts.toggle_select(0)           # but the human picks tile 0
    fit = d.tell(z, [imgs])
    assert fit[0] == fit.max()
    assert ts.selected == set()
    assert d.status()["elites_injected"] == 1


def test_set_prompt_forwards_to_the_scorer():
    d, _ = make()
    d.set_prompt("coral")
    assert d.prompt == "coral"
    assert d.scorer.prompt == "coral"


def test_set_x0_builds_an_optimizer_centred_on_that_genome():
    d, _ = make()
    z0 = np.full(BRAIN_SPEC.dim, 0.25, dtype=np.float32)
    d.set_x0(z0)
    assert d.optimizer is not None
    # The GRAND mean, not the per-dimension mean: 4 samples at sigma 0.5 give
    # each of the 80 dimensions a standard error of 0.25, so the worst single
    # dimension wanders far enough to make a per-dimension bound flaky. Pooling
    # all 320 values gives a standard error of 0.028, so this is a ~5 sigma
    # bound on the thing actually being asserted - that x0 moved the centre.
    assert float(d.ask(4).mean()) == pytest.approx(0.25, abs=0.15)


def test_set_spec_discards_the_optimizer():
    """The search dimension changed; a covariance for the old one is meaningless."""
    d, _ = make()
    d.ask(4)
    d.set_spec(BRAIN_PHYSICS_SPEC)
    assert d.optimizer is None
    assert d.ask(4).shape == (4, BRAIN_PHYSICS_SPEC.dim)


def test_set_spec_to_the_same_spec_keeps_the_optimizer():
    d, _ = make()
    d.ask(4)
    before = d.optimizer
    d.set_spec(BRAIN_SPEC)
    assert d.optimizer is before


def test_reset_discards_the_optimizer_and_the_loaded_start_point():
    d, _ = make()
    d.set_x0(np.full(BRAIN_SPEC.dim, 2.0, dtype=np.float32))
    d.reset()
    assert d.optimizer is None
    assert np.allclose(d.ask(64).mean(axis=0), 0.0, atol=0.4)


def test_sigma_falls_back_to_sigma0_before_the_first_ask():
    d, _ = make()
    d.sigma0 = 0.3
    assert d.sigma == pytest.approx(0.3)


def test_checkpoint_roundtrip_restores_the_optimizer_and_prompt():
    d, _ = make()
    d.set_prompt("coral")
    z = d.ask(4)
    d.tell(z, [crops(4)])
    st = d.checkpoint_state()
    assert st["optimizer_name"] == "CMA-ES"
    assert st["prompt"] == "coral"

    d2, _ = make()
    d2.restore(st)
    assert d2.prompt == "coral"
    assert d2.optimizer is not None
    assert np.allclose(d2.optimizer.ask(4), d.optimizer.ask(4), atol=1e-5)


def test_status_reports_what_the_log_and_ui_need():
    d, _ = make()
    d.set_prompt("coral")
    d.tell(d.ask(4), [crops(4)])
    st = d.status()
    assert set(st) >= {"prompt", "algorithm", "sigma", "nan_replaced", "elites_injected"}
    assert st["sigma"] > 0


def test_an_equal_spec_does_not_restart_the_search():
    """main._refresh_driver_specs builds a fresh spec_for(layout) each call and
    passes reset=False on a scales-only change, because "throwing away the
    optimizer's covariance would cost the run for nothing". Comparing spec
    objects by identity defeated that: nudging a decode-scale slider silently
    restarted CMA-ES mid-run, with nothing on screen to say so."""
    from services.brains import REGISTRY
    from services.genome_spec import spec_for

    m = REGISTRY["fourier"]
    d, _ = make()
    d.set_spec(spec_for(m.layout_from_settings({"freq_scale": 3.0})))
    d.ask(4)
    before = d.optimizer
    assert before is not None

    scaled = spec_for(m.layout_from_settings({"freq_scale": 1.5}))
    d.set_spec(scaled)
    assert d.optimizer is before, "a scale tweak restarted the search"
    # The NEW object must still be adopted - decode() reads its scales.
    assert d.spec is scaled


def test_a_real_space_change_still_discards_the_optimizer():
    d, _ = make()
    d.ask(4)
    assert d.optimizer is not None
    d.set_spec(BRAIN_PHYSICS_SPEC)
    assert d.optimizer is None
