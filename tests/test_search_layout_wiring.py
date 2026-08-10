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


def test_a_layout_switch_mid_run_moves_the_search_with_it():
    svc, ts = a_service(layout("fourier"))
    svc.start("anything")
    gab = layout("gabor")
    ts.set_layout(gab)
    svc.set_layout(gab)
    svc._begin_generation()
    assert svc.spec.dim == gab.length
    assert all(np.asarray(g).size == gab.length for g in ts.population)
