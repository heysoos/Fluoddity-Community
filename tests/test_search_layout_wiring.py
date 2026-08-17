"""The search space follows the SELECTED brain, not Fourier.

AutoTournamentService._resolve_spec() chose between two module-level constants
that are permanently Fourier, and ran again on every _begin_generation() - so
the layout-aware spec main.py pushes at the driver was overwritten once per
generation. Measured before the fix: with Gabor, Lenia or MLP active the driver
searched 80 floats instead of 168, 120 or 148, and the tournament population
came back as (10, 8) Fourier arrays.

Downstream that is not a clean failure. sim.write_tournament_rules re-slices the
upload at layout.length, so a 16-tile Gabor generation handed it 1280 floats to
cut into 168-float brains: 7 genomes straddling Fourier boundaries, then 9
generated ones to fill the grid. The optimizer scored that and learned from it,
and the entries landed in the gabor-n12 archive stamped spec="brain:80".
"""
import numpy as np
import pytest

from services.auto_tournament_service import AutoTournamentService
from services.brains import REGISTRY
from services.genome_spec import physics_spec_for, spec_for
from services.imgep_driver import ImgepDriver
from services.physics_genome import PHYSICS_DIM
from services.tournament_service import TournamentService

MODALITIES = ["fourier", "gabor", "lenia", "mlp"]


def layout(name, **settings):
    return REGISTRY[name].layout_from_settings(settings)


def a_service(lay, grid=2, **kw):
    ts = TournamentService(grid=grid, layout=lay)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=None, spec=spec_for(lay))
    svc.configure(**kw)
    return svc, ts


@pytest.mark.parametrize("name", MODALITIES)
def test_the_search_space_follows_the_selected_modality(name):
    lay = layout(name)
    svc, _ = a_service(lay)
    svc.start("anything")
    assert svc.spec.dim == lay.length
    assert svc.driver.spec.dim == lay.length


@pytest.mark.parametrize("name", MODALITIES)
def test_a_generation_produces_brains_of_the_active_layout(name):
    lay = layout(name)
    svc, ts = a_service(lay)
    svc.start("anything")
    assert all(np.asarray(g).size == lay.length for g in ts.population)


@pytest.mark.parametrize("name", MODALITIES)
def test_physics_search_extends_the_active_layout(name):
    lay = layout(name)
    svc, _ = a_service(lay, physics_enabled=True)
    svc.start("anything")
    assert svc.spec.dim == lay.length + PHYSICS_DIM


@pytest.mark.parametrize("name", MODALITIES)
def test_the_checkpoint_records_the_layout_it_searched(name):
    """A checkpoint restored under a different modality must be rejected. Widths
    alone cannot do it - Fourier at 21 centres and Gabor at 12 filters are both
    168 floats - so the layout signature has to be in there too."""
    lay = layout(name)
    svc, _ = a_service(lay)
    svc.start("anything")
    assert lay.signature() in svc.checkpoint_state()["brain_layout_signature"]


def test_resolving_the_spec_every_generation_keeps_the_optimizer():
    """_resolve_spec runs on every _begin_generation(). Rebuilding the spec
    object each time would turn this into an optimizer reset per generation -
    strictly worse than the Fourier-only behaviour it replaces."""
    svc, _ = a_service(layout("gabor"))
    svc.start("anything")
    before = svc.driver.optimizer
    assert before is not None
    svc._begin_generation()
    assert svc.driver.optimizer is before


def test_toggling_physics_still_rebuilds_the_optimizer():
    """The one case that MUST reset: the dimension genuinely changed."""
    svc, _ = a_service(layout("gabor"))
    svc.start("anything")
    before = svc.driver.optimizer
    svc.configure(physics_enabled=True)
    svc._begin_generation()
    assert svc.driver.optimizer is not before
    assert svc.driver.spec.dim == layout("gabor").length + PHYSICS_DIM


# ---- built AFTER the brain was switched -------------------------------
# main._ensure_auto_service and _ensure_archive_service construct these lazily,
# the first time Auto or Explore is opened - which can be long after the brain
# was changed. Both wiring calls in _apply_brain_layout are guarded by
# `is not None`, so a switch made before that first open is never delivered, and
# _resolve_spec re-derives from the service's OWN layout every generation, so it
# cannot self-correct either.
#
# The tests above all pass `spec=` explicitly, which is exactly the argument
# production does not pass; that is how this survived them.


@pytest.mark.parametrize("name", MODALITIES)
def test_a_service_built_with_no_spec_inherits_the_tournaments_layout(name):
    lay = layout(name)
    ts = TournamentService(grid=2, layout=lay)
    svc = AutoTournamentService(ts, scorer=None, logger=None)
    assert svc.spec.dim == lay.length
    assert svc.spec.layout == lay


@pytest.mark.parametrize("name", MODALITIES)
def test_a_service_built_with_no_spec_does_not_corrupt_the_population(name):
    """The crash, end to end. An 80-float genome in a 168-float population is
    not caught until something tries to breed it, and then it is a ValueError
    inside orchestrate_frame - which takes the app down."""
    lay = layout(name)
    ts = TournamentService(grid=2, layout=lay)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=None, logger=None)
    svc.start("anything")
    assert all(np.asarray(g).size == lay.length for g in ts.population)
    ts.next_generation()          # ValueError before the fix


@pytest.mark.parametrize("name", MODALITIES)
def test_the_imgep_driver_built_with_no_spec_inherits_it_too(name):
    lay = layout(name)
    ts = TournamentService(grid=2, layout=lay)
    assert ImgepDriver(ts, None, None).spec.dim == lay.length


def test_a_driver_with_no_tournament_still_builds():
    """tests/test_archive_state.py introspects ImgepDriver(None, None, []), and
    the archive window can outlive its tournament. Falling back beats raising."""
    assert ImgepDriver(None, None, []).spec.dim > 0


def test_a_layout_switch_mid_run_moves_the_search_with_it():
    svc, ts = a_service(layout("fourier"))
    svc.start("anything")
    gab = layout("gabor")
    ts.set_layout(gab)
    svc.set_layout(gab)
    svc._begin_generation()
    assert svc.spec.dim == gab.length
    assert all(np.asarray(g).size == gab.length for g in ts.population)


# ---- a rollout in flight when the space moves --------------------------
# The test above calls _begin_generation() by hand, which is exactly what
# production does NOT do: abort_generation() restarts the rollout without
# re-asking, so the OLD population reached tell() under the NEW spec.

def _fourier_and_mlp():
    from services.brains import layout_from_signature

    return (layout_from_signature("fourier-n10"),
            layout_from_signature("mlp-n16.8.8-a0.0.0"))


def test_a_layout_switch_drops_a_population_the_new_spec_cannot_read():
    """Crashed a real run: physics search on under Fourier is an 88-wide z, the
    brain was switched to a 324-wide MLP, and the rollout already in flight was
    scored against the new spec."""
    f, m = _fourier_and_mlp()
    svc, ts = a_service(f, physics_enabled=True)
    svc.start("anything")
    assert svc.current_z.shape[1] == physics_spec_for(f).dim

    ts.set_layout(m)
    svc.set_layout(m)

    z = svc.current_z
    assert z is None or z.shape[1] == svc.spec.dim, (
        f"population is {z.shape[1]} wide but the spec in force is "
        f"{svc.spec.dim}; it cannot be decoded")


def test_the_next_update_asks_for_a_population_in_the_new_space():
    """Dropping it is only half: the phase machine must re-ask rather than
    score nothing."""
    f, m = _fourier_and_mlp()
    svc, ts = a_service(f, physics_enabled=True)
    svc.start("anything")
    ts.set_layout(m)
    svc.set_layout(m)
    for _ in range(4):
        svc.update()
        if svc.current_z is not None:
            break
    assert svc.current_z is not None, "never re-asked after the space moved"
    assert svc.current_z.shape[1] == svc.spec.dim


def test_the_spec_refuses_a_vector_from_another_space():
    """decode() SLICES, so a short z truncates silently and the failure lands
    inside whichever modality is running, naming neither space."""
    _f, m = _fourier_and_mlp()
    spec = physics_spec_for(m)
    with pytest.raises(ValueError, match="expects 332"):
        spec.decode(np.zeros(88, dtype=np.float32))
