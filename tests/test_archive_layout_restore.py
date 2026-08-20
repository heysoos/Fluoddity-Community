"""Reopening an archive under the brain it was being searched under.

The app remembers which archive you were in - archive_name is in preferences -
and used to forget which brain you were working on in it, so an archive whose
entries are all one modality reopened under another with no native rows at all.
"""
from __future__ import annotations

from services.brains import BrainLayout, default_layout
from state.archive_state import ArchiveState
from state.brain_state import BrainState
from state.preferences_state import PreferencesState

FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)


class _Bag:
    pass


class _Sim:
    def __init__(self):
        self.brain_layout = FOURIER


class _App:
    def __init__(self):
        self.sim = _Sim()
        self.applied = []

    def _apply_brain_layout(self, layout, ui_state):
        self.sim.brain_layout = layout
        self.applied.append(layout)
        return True

    def _restore_archive_layout(self, ui_state):
        from main import App

        return App._restore_archive_layout(self, ui_state)


def _ui(signature):
    u = _Bag()
    u.archive = ArchiveState()
    u.archive.layout_signature = signature
    u.brain = BrainState()
    u.preferences = PreferencesState()
    return u


def test_an_archive_recorded_under_another_brain_switches_to_it():
    app, ui = _App(), _ui("gabor-n12")
    assert app._restore_archive_layout(ui) is True
    assert app.sim.brain_layout.signature() == "gabor-n12"


def test_the_brain_window_moves_too():
    """_handle_brain_layout applies ui_state.brain every frame, so a restore
    that moves only the sim is undone by the very next frame."""
    app, ui = _App(), _ui("gabor-n12")
    app._restore_archive_layout(ui)
    assert ui.brain.modality == "gabor"
    assert ui.brain.settings.get("filters") == 12


def test_an_archive_already_on_that_brain_does_nothing():
    app, ui = _App(), _ui("fourier-n10")
    assert app._restore_archive_layout(ui) is False
    assert app.applied == []


def test_an_archive_with_no_recorded_layout_is_left_alone():
    """Every archive written before this one. No migration runs."""
    app, ui = _App(), _ui("")
    assert app._restore_archive_layout(ui) is False
    assert app.applied == []


def test_a_signature_this_build_cannot_rebuild_keeps_the_current_brain():
    """Guessing a plausible layout of the wrong width is the one outcome worse
    than refusing."""
    app, ui = _App(), _ui("mlp-n99999999-aZZZ")
    assert app._restore_archive_layout(ui) is False
    assert app.sim.brain_layout is FOURIER
    assert "mlp-n99999999-aZZZ" in ui.archive.warning


def test_the_restore_is_not_wired_into_the_layout_change_itself():
    """_apply_brain_layout writes the settings that _restore reads. Calling
    one from the other would put the outgoing layout straight back."""
    import inspect

    from main import App

    src = inspect.getsource(App._apply_brain_layout)
    assert "_restore_archive_layout" not in src
