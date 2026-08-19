"""Delete and Relayout must work while merely BROWSING an archive.

Both were dispatched inside _handle_explore, which bails as soon as the
Explore driver is detached - and _clear_explore_flags then wipes the one-shot,
so the button did nothing and said nothing. _handle_archive_preview was moved
out for exactly this reason; these two were missed.

Driven through the DISPATCH, in the order process_commands runs it. The
methods themselves were always fine, which is why the existing tests - which
call _delete_archive_entry directly - passed throughout.
"""
import numpy as np

from command_handler import CommandHandler
from services.archive import Archive, Candidate
from state.ui_state import UIState


def _archive(n=6):
    arc = Archive(store=None, dim=8, liveness_min=0.0, capacity=50,
                  min_separation=0.0)
    for i in range(n):
        e = np.zeros(8, dtype=np.float32)
        e[i] = 1.0

        arc.consider(
            Candidate(brain=np.zeros(80, np.float32),
                      physics=np.zeros(8, np.float32),
                      embedding=e, liveness=0.5, spec="brain:80"),
            novelty=1.0)
    return arc


class _Refit:
    def __init__(self):
        self.asked = 0

    def request_refit(self):
        self.asked += 1


def _browsing(tmp_path, arc, refit=None):
    """A handler in the state the archive browser runs in on its own: an
    archive open, and no Explore driver behind it."""
    ch = CommandHandler(None, None, None, None, None, None, None, None,
                        tmp_path)
    ch.archive = arc
    ch.map_layout_service = refit
    ch.auto_service = None
    ch.imgep_driver = None
    return ch


def _dispatch(ch, ui):
    """The REAL entry point, so the dispatch order is the app's and not a
    copy of it that can drift."""
    ch.process_commands(ui, False)


def test_delete_works_while_only_browsing(tmp_path):
    arc = _archive(6)
    ch = _browsing(tmp_path, arc)
    ui = UIState()
    ui.archive.delete_entry_id = 2
    gone = arc.entries[2].id

    _dispatch(ch, ui)

    assert len(arc) == 5, "the Delete button did nothing and said nothing"
    assert all(e.id != gone for e in arc.entries)
    assert ui.archive.delete_entry_id == -1


def test_relayout_works_while_only_browsing(tmp_path):
    arc = _archive(6)
    refit = _Refit()
    ch = _browsing(tmp_path, arc, refit)
    ui = UIState()
    ui.archive.refit_projection_requested = True

    _dispatch(ch, ui)

    assert refit.asked == 1
    assert ui.archive.refit_projection_requested is False


def test_seeding_a_run_still_needs_the_search(tmp_path, monkeypatch):
    """Not every browser one-shot is archive-only: seeding sets the
    optimizer's mean, so with no driver there is nothing to seed - and
    reaching it anyway would run it against a driver that is None."""
    arc = _archive(6)
    ch = _browsing(tmp_path, arc)
    called = []
    monkeypatch.setattr(CommandHandler, "_seed_from_archive",
                        lambda self, ast: called.append(ast.seed_entry_id))
    ui = UIState()
    ui.archive.seed_entry_id = 1

    _dispatch(ch, ui)

    assert called == [], "seeding must stay behind the Explore gate"
    assert ui.archive.seed_entry_id == -1
    assert len(arc) == 6
