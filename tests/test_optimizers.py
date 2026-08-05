import numpy as np
import pytest

from services.optimizers import ALGORITHMS, make_optimizer

DIM = 80
POP = 16
NAMES = list(ALGORITHMS)


def sphere(z):
    """Fitness, higher is better. Optimum 0.0 at the origin."""
    return -float(np.sum(z * z))


def rastrigin(z):
    return -float(10 * len(z) + np.sum(z * z - 10 * np.cos(2 * np.pi * z)))


def run(name, fn, gens, seed=0, sigma0=0.5):
    opt = make_optimizer(name, DIM, POP, sigma0, seed)
    best = -np.inf
    for _ in range(gens):
        z = opt.ask(POP)
        f = np.array([fn(zi) for zi in z], dtype=np.float32)
        opt.tell(z, f)
        best = max(best, float(f.max()))
    return best, opt


@pytest.mark.parametrize("name", NAMES)
def test_ask_returns_correct_shape_and_dtype(name):
    opt = make_optimizer(name, DIM, POP, 0.5, 0)
    z = opt.ask(POP)
    assert z.shape == (POP, DIM)
    assert z.dtype == np.float32


@pytest.mark.parametrize("name", NAMES)
def test_tell_treats_higher_fitness_as_better(name):
    """The sign convention lives in one place: tell() takes fitness where
    higher wins, and implementations negate internally for cmaes."""
    opt = make_optimizer(name, DIM, POP, 0.5, 0)
    z = opt.ask(POP)
    f = np.arange(POP, dtype=np.float32)  # the LAST sample is the best
    opt.tell(z, f)
    zb, fb = opt.best()
    assert fb == pytest.approx(POP - 1)
    assert np.allclose(zb, z[-1])


@pytest.mark.parametrize("name", ["CMA-ES", "Sep-CMA-ES", "GA"])
def test_learning_optimizers_improve_on_sphere(name):
    """Random Search is excluded deliberately: it does not learn, and in 80-D
    its best-of-640 equals its best-of-16. That is exactly why it is the
    control for whether the fitness signal is real."""
    start, _ = run(name, sphere, 1)
    end, _ = run(name, sphere, 40)
    assert end > start


@pytest.mark.parametrize("name", ["CMA-ES", "Sep-CMA-ES"])
def test_cma_converges_on_sphere(name):
    best, _ = run(name, sphere, 150)
    assert best > -5.0, f"{name} failed to converge on 80-D sphere: {best}"


def test_cma_beats_random_search_on_rastrigin():
    cma, _ = run("CMA-ES", rastrigin, 120)
    rnd, _ = run("Random Search", rastrigin, 120)
    assert cma > rnd


@pytest.mark.parametrize("name", NAMES)
def test_best_returns_the_best_seen(name):
    _, opt = run(name, sphere, 20)
    z, f = opt.best()
    assert z.shape == (DIM,)
    assert f == pytest.approx(sphere(z), rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("name", NAMES)
def test_state_dict_roundtrip_reproduces_the_next_ask(name):
    """Explicit named arrays, not a pickle - so a checkpoint is not tied to the
    installed cmaes version. This test is what pins the capture."""
    _, opt = run(name, sphere, 12, seed=3)
    d = opt.state_dict()
    expected = opt.ask(POP)

    fresh = make_optimizer(name, DIM, POP, 0.5, 999)
    fresh.load_state_dict(d)
    assert np.allclose(fresh.ask(POP), expected, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_state_dict_roundtrip_also_restores_best(name):
    _, opt = run(name, sphere, 12, seed=3)
    fresh = make_optimizer(name, DIM, POP, 0.5, 999)
    fresh.load_state_dict(opt.state_dict())
    z0, f0 = opt.best()
    z1, f1 = fresh.best()
    assert np.allclose(z0, z1)
    assert f0 == pytest.approx(f1)


@pytest.mark.parametrize("name", NAMES)
def test_state_dict_has_no_pickled_objects(name):
    _, opt = run(name, sphere, 3)
    for k, v in opt.state_dict().items():
        assert isinstance(v, (int, float, str, bool, np.ndarray, list, tuple)), (
            f"{name}.state_dict()[{k!r}] is {type(v)}; must be npz-serializable"
        )


def test_x0_seeds_the_initial_mean():
    x0 = np.full(DIM, 0.4, dtype=np.float32)
    opt = make_optimizer("CMA-ES", DIM, POP, 0.05, 0, x0=x0)
    z = opt.ask(POP)
    assert np.allclose(z.mean(axis=0), x0, atol=0.1)


@pytest.mark.parametrize("name", NAMES)
def test_sigma_is_exposed(name):
    opt = make_optimizer(name, DIM, POP, 0.5, 0)
    assert opt.sigma > 0


def test_cma_sigma_shrinks_as_it_converges():
    """The sigma readout warns when mutation strength approaches sigma; that is
    only meaningful if sigma actually falls during a run."""
    opt = make_optimizer("CMA-ES", DIM, POP, 0.5, 0)
    start = opt.sigma
    for _ in range(120):
        z = opt.ask(POP)
        opt.tell(z, np.array([sphere(zi) for zi in z], dtype=np.float32))
    assert opt.sigma < start
