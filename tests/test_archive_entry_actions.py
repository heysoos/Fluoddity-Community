"""Export, seed and delete for a single archive entry.

These run in CommandHandler, whose other paths all need a GL context, a camera
and a sim - so they are exercised here directly against the three attributes
they actually touch.
"""
import json

import numpy as np
import pytest

from command_handler import CommandHandler
from services.archive import Archive, Candidate
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.physics_genome import PHYSICS_PARAMS
from state import SimState
from state.ui_state import UIState


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def archive_with(n=3, spec="brain:80", physics=None):
    arc = Archive(store=None, dim=4, liveness_min=0.0, capacity=50)
    rng = np.random.default_rng(0)
    for i in range(n):
        e = np.zeros(4, dtype=np.float32)
        e[i % 4] = 1.0
        arc.consider(
            Candidate(brain=(rng.normal(size=(10, 8)) * 0.2).astype(np.float32),
                      physics=(np.zeros(8, np.float32) if physics is None
                               else np.asarray(physics, np.float32)),
                      embedding=_unit(e[None])[0], liveness=0.5, spec=spec),
            novelty=1.0)
    return arc


def handler(tmp_path, archive, auto_service=None):
    ch = CommandHandler(None, None, None, None, None, None, None, None, tmp_path)
    ch.archive = archive
    ch.auto_service = auto_service
    return ch


class _Driver:
    """Stands in for PromptDriver, which has set_x0."""

    def __init__(self):
        self.x0 = None

    def set_x0(self, z):
        self.x0 = np.asarray(z, dtype=np.float32)


class _Service:
    def __init__(self, driver):
        self.driver = driver

    def set_x0(self, z):
        self.driver.set_x0(z)


class _NoX0Driver:
    """Stands in for ImgepDriver, which deliberately has no set_x0."""


# ---- export -------------------------------------------------------------

def test_export_writes_a_config_under_the_name_the_user_chose(tmp_path):
    """The name comes from the save dialog, not from the entry id. Auto-naming
    was the reported problem: you could not tell what a save was called."""
    arc = archive_with()
    ch = handler(tmp_path, arc)
    ch._export_archive_entry(UIState(), arc.entries[1].id, "my creature")
    assert (tmp_path / "my creature.json").is_file()


def test_export_tells_the_user_where_the_file_went(tmp_path):
    """The file lands in the configs folder, which is not on screen, so a
    console print left the button looking like it had done nothing."""
    arc = archive_with()
    ui = UIState()
    handler(tmp_path, arc)._export_archive_entry(ui, arc.entries[1].id, "keeper")
    assert "keeper.json" in ui.archive.notice


def test_the_exported_config_round_trips_to_the_same_brain(tmp_path):
    """The point of exporting: it must open in the normal single-simulation
    view as the creature that was archived."""
    from services.genome_io import import_genome

    arc = archive_with()
    ch = handler(tmp_path, arc)
    ch._export_archive_entry(UIState(), arc.entries[0].id, "chosen")
    path = tmp_path / "chosen.json"

    z, n_clamped, meta = import_genome(path)
    brain = BRAIN_SPEC.decode(np.asarray(z, dtype=np.float32))["brain"]
    assert np.allclose(brain, arc.brains[0], atol=1e-3)
    assert n_clamped == 0, "an archived brain is already inside the squash range"
    assert meta.get("archive_id") == arc.entries[0].id


def test_export_records_the_provenance_metadata(tmp_path):
    arc = archive_with()
    arc.entries[0].goal = "coral reef"
    arc.entries[0].source = "expedition"
    ch = handler(tmp_path, arc)
    ch._export_archive_entry(UIState(), arc.entries[0].id, "chosen")
    blob = json.loads((tmp_path / "chosen.json").read_text())
    meta = json.dumps(blob)
    assert "coral reef" in meta and "expedition" in meta


def test_exporting_a_physics_entry_applies_its_absolute_values(tmp_path):
    """The archive stores decoded physics, so applying it needs no origin."""
    values = np.arange(1, 9, dtype=np.float32)
    arc = archive_with(spec="brain:80,physics:8", physics=values)
    ch = handler(tmp_path, arc)
    ui_state = UIState()
    ch._export_archive_entry(ui_state, arc.entries[0].id, "chosen")
    for j, (name, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS):
        assert getattr(ui_state.sim, name) == pytest.approx(float(values[j]))


def test_a_brain_only_entry_leaves_physics_alone(tmp_path):
    arc = archive_with(spec="brain:80")
    ch = handler(tmp_path, arc)
    ui_state = UIState()
    before = {n: getattr(ui_state.sim, n) for n, _g, _lo, _hi in PHYSICS_PARAMS}
    ch._export_archive_entry(ui_state, arc.entries[0].id, "chosen")
    for name, was in before.items():
        assert getattr(ui_state.sim, name) == was


def test_exporting_an_unknown_id_writes_nothing(tmp_path):
    ch = handler(tmp_path, archive_with())
    ui = UIState()
    ch._export_archive_entry(ui, 999, "chosen")
    assert list(tmp_path.glob("*.json")) == []
    assert "no longer in the archive" in ui.archive.warning


# ---- seed ---------------------------------------------------------------

def _ast(entry_id):
    from state.archive_state import ArchiveState

    a = ArchiveState()
    a.seed_entry_id = int(entry_id)
    return a


def test_seeding_sets_x0_on_a_driver_that_has_one(tmp_path):
    arc = archive_with()
    drv = _Driver()
    ch = handler(tmp_path, arc, auto_service=_Service(drv))
    ast = _ast(arc.entries[2].id)
    ch._seed_from_archive(ast)
    assert drv.x0 is not None and drv.x0.shape == (BRAIN_SPEC.dim,)
    assert str(arc.entries[2].id) in ast.notice, "the user must be told it took"


def test_seeding_is_refused_for_a_driver_with_no_x0(tmp_path):
    """IMGEP draws parents from the archive by novelty, so seeding has no
    meaning there - and saying so on the CONSOLE is not saying so at all."""
    arc = archive_with()
    ch = handler(tmp_path, arc, auto_service=_Service(_NoX0Driver()))
    ast = _ast(arc.entries[0].id)
    ch._seed_from_archive(ast)
    assert "Auto (CLIP)" in ast.warning
    assert not ast.notice


def test_seeding_with_no_service_is_harmless(tmp_path):
    arc = archive_with()
    ast = _ast(arc.entries[0].id)
    handler(tmp_path, arc)._seed_from_archive(ast)
    assert not ast.notice and not ast.warning


def test_seeding_an_unknown_id_does_nothing(tmp_path):
    drv = _Driver()
    ch = handler(tmp_path, archive_with(), auto_service=_Service(drv))
    ast = _ast(999)
    ch._seed_from_archive(ast)
    assert drv.x0 is None
    assert not ast.notice
    assert drv.x0 is None


# ---- delete -------------------------------------------------------------

def test_delete_removes_the_entry(tmp_path):
    arc = archive_with(n=3)
    ch = handler(tmp_path, arc)
    target = arc.entries[1].id
    ast = UIState().archive
    ast.delete_entry_id = target
    ch._delete_archive_entry(ast)
    assert len(arc) == 2
    assert all(e.id != target for e in arc.entries)


def test_delete_keeps_the_vectors_and_entries_aligned(tmp_path):
    """_remove swaps the last entry into the hole, so the parallel arrays and
    the entry list must move together or every later novelty is wrong."""
    arc = archive_with(n=4)
    ch = handler(tmp_path, arc)
    keep = {e.id: arc.embeddings[i].copy() for i, e in enumerate(arc.entries)}
    ast = UIState().archive
    ast.delete_entry_id = arc.entries[0].id
    del keep[ast.delete_entry_id]
    ch._delete_archive_entry(ast)
    for i, e in enumerate(arc.entries):
        assert np.allclose(arc.embeddings[i], keep[e.id])


def test_delete_clears_the_selection_it_just_removed(tmp_path):
    arc = archive_with()
    ch = handler(tmp_path, arc)
    ast = UIState().archive
    ast.delete_entry_id = arc.entries[0].id
    ast.selected_entry_id = ast.delete_entry_id
    ch._delete_archive_entry(ast)
    assert ast.selected_entry_id == -1


def test_delete_leaves_a_different_selection_alone(tmp_path):
    arc = archive_with(n=3)
    ch = handler(tmp_path, arc)
    ast = UIState().archive
    ast.delete_entry_id = arc.entries[0].id
    ast.selected_entry_id = arc.entries[2].id
    ch._delete_archive_entry(ast)
    assert ast.selected_entry_id == arc.entries[-1].id or ast.selected_entry_id >= 0


def test_deleting_an_unknown_id_does_nothing(tmp_path):
    arc = archive_with(n=3)
    ch = handler(tmp_path, arc)
    ast = UIState().archive
    ast.delete_entry_id = 999
    ch._delete_archive_entry(ast)
    assert len(arc) == 3
