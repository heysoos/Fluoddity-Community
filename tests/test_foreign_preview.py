"""Hovering an entry that belongs to another brain.

The preview BORROWS the entry's layout: the sim's buffer is resized and the
modality uniform follows it, while the archive, the optimizer and the Brain
window keep the user's own brain. That is the whole difference between a hover
and a switch - one costs a buffer resize, the other rebuilds the archive and
rescores every entry in it.

The same-brain preview lives in tests/test_archive_preview.py; the archive that
holds several brains at once is tests/test_mixed_archive.py.
"""
from __future__ import annotations

import numpy as np

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore
from services.brains import BrainLayout, default_layout

DIM = 8
FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)


def _cand(vec, brain_len):
    e = np.asarray(vec, dtype=np.float32)
    e = e / max(float(np.linalg.norm(e)), 1e-8)
    return Candidate(embedding=e, brain=np.arange(brain_len, dtype=np.float32),
                     physics=np.zeros(8, np.float32), liveness=0.5,
                     spec="brain", goal="", run_id="r", gen=0, tile=0,
                     viable=True)


def _axis(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def _fill(root, layout, n, start_axis):
    arc = Archive(store=ArchiveStore(root, layout), layout=layout, dim=DIM)
    for j in range(n):
        arc.consider(_cand(_axis(start_axis + j), layout.length),
                     novelty=1.0, force=True)
    arc.maybe_flush(force=True)
    arc.store.close()


def _mixed(tmp_path, gabor=GABOR):
    """An archive holding both brains, opened under Fourier."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, 0)
    _fill(root, gabor, 2, 4)
    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()
    foreign = next(i for i in range(len(arc.entries)) if not arc.is_native(i))
    native = next(i for i in range(len(arc.entries)) if arc.is_native(i))
    return arc, foreign, native


class _Sim:
    def __init__(self, layout):
        self.brain_layout = layout
        self.applied = []
        self.reallocs = []

    def realloc_brain_buffers(self, layout):
        self.brain_layout = layout
        self.reallocs.append(layout)

    def apply_rule(self, rule):
        # Recorded WITH the layout that was live. Applying a genome under the
        # wrong one is what this mechanism exists to prevent, and the rule on
        # its own cannot show it.
        self.applied.append((rule, self.brain_layout))


class _RuleManager:
    def __init__(self, base):
        self.stack = [(base, 0.0)]

    def push_rule(self, rule, seed):
        self.stack.append((np.array(rule), seed))

    def pop_rule(self):
        if len(self.stack) > 1:
            self.stack.pop()
            return self.stack[-1]
        self.stack = []
        return None, None


def _handler(arc, sim):
    from command_handler import CommandHandler

    h = object.__new__(CommandHandler)
    h.archive = arc
    h.archive_store = None
    h.sim = sim
    h.rule_manager = _RuleManager(np.zeros(sim.brain_layout.length, np.float32))
    h.param_lock_service = None
    h._run_config_cache = {}
    h._run_physics_written = ""
    h._pending_brain_rule = None
    h._archive_preview_id = -1
    h._archive_preview_pushed = False
    h._archive_preview_physics = None
    h._archive_preview_arc = None
    h._borrow = None
    h.apply_brain_layout = None
    # _handle_brain_layout keeps the window's readouts live off these.
    h.auto_service = None
    h.imgep_driver = None
    return h


def _ui():
    from state.archive_state import ArchiveState
    from state.auto_tournament_state import AutoTournamentState
    from state.brain_state import BrainState
    from state.sim_state import SimState
    from state.tournament_state import TournamentState

    class _UI:
        pass

    u = _UI()
    u.sim = SimState()
    u.archive = ArchiveState()
    u.auto_tournament = AutoTournamentState()
    u.tournament = TournamentState()
    u.brain = BrainState()
    u.archive.live_preview = True
    return u


def test_hovering_a_foreign_entry_runs_it_under_its_own_layout(tmp_path):
    arc, foreign, _ = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()

    ui.archive.preview_entry_id = foreign
    h._handle_archive_preview(ui)

    rule, live = sim.applied[-1]
    assert live == GABOR, "the genome must reach the GPU under its own brain"
    assert rule.size == GABOR.length
    assert np.array_equal(rule, arc.brain_at(foreign))


def test_the_archive_and_the_search_keep_the_user_s_own_brain(tmp_path):
    """What makes a hover cheap enough to be a hover."""
    arc, foreign, _ = _mixed(tmp_path)
    h, ui = _handler(arc, _Sim(FOURIER)), _ui()
    switches = []
    h.apply_brain_layout = lambda lay, _u: switches.append(lay)

    ui.archive.preview_entry_id = foreign
    h._handle_archive_preview(ui)

    assert switches == []
    assert arc.signature == FOURIER.signature()


def test_un_hovering_gives_the_layout_back_before_the_rule_under_it(tmp_path):
    """Order, not just eventual state: what comes off the rule stack is the
    user's own brain, and apply_rule measures a rule against whatever layout is
    live - so a restore in the wrong order is silently refused on width."""
    arc, foreign, _ = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()

    ui.archive.preview_entry_id = foreign
    h._handle_archive_preview(ui)
    ui.archive.preview_entry_id = -1
    h._handle_archive_preview(ui)

    assert sim.brain_layout == FOURIER
    rule, live = sim.applied[-1]
    assert live == FOURIER and rule.size == FOURIER.length


def test_moving_between_two_foreign_entries_does_not_strand_a_layout(tmp_path):
    arc, foreign, _ = _mixed(tmp_path)
    other = next(i for i in range(len(arc.entries))
                 if not arc.is_native(i) and i != foreign)
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()

    for row in (foreign, other, foreign):
        ui.archive.preview_entry_id = row
        h._handle_archive_preview(ui)
    ui.archive.preview_entry_id = -1
    h._handle_archive_preview(ui)

    assert sim.brain_layout == FOURIER
    assert h._borrow is None


def test_a_borrow_suppresses_the_per_frame_layout_apply(tmp_path):
    """_handle_brain_layout runs every frame off the Brain window, which still
    reads the user's brain. Left alone it would undo the hover - with a full
    archive teardown - once a frame."""
    arc, foreign, _ = _mixed(tmp_path)
    h, ui = _handler(arc, _Sim(FOURIER)), _ui()
    applied = []
    h.apply_brain_layout = lambda lay, _u: applied.append(lay)

    ui.archive.preview_entry_id = foreign
    h._handle_archive_preview(ui)
    h._handle_brain_layout(ui)
    assert applied == []

    ui.archive.preview_entry_id = -1
    h._handle_archive_preview(ui)
    h._handle_brain_layout(ui)
    assert applied, "the suppression lasts exactly as long as the borrow"


def test_committing_a_foreign_entry_switches_from_the_user_s_own_layout(tmp_path):
    """A click makes the borrow permanent. The switch has to see the layout it
    is switching FROM, or it early-returns and nothing is rebuilt."""
    arc, foreign, _ = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()
    seen = []
    h.apply_brain_layout = lambda lay, _u: seen.append((lay, sim.brain_layout))

    ui.archive.preview_entry_id = foreign
    h._handle_archive_preview(ui)              # hovering
    ui.archive.load_entry_id = foreign
    h._handle_archive_preview(ui)              # clicked

    assert len(seen) == 1
    switched_to, switched_from = seen[0]
    assert switched_to == GABOR
    assert switched_from == FOURIER
    assert h._borrow is None, "the borrow ends at the commit"


def test_committing_without_hovering_first_still_switches(tmp_path):
    """Live preview off, or a click that outran the hover."""
    arc, foreign, _ = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()
    ui.archive.live_preview = False
    seen = []
    h.apply_brain_layout = lambda lay, _u: seen.append(lay)

    ui.archive.load_entry_id = foreign
    h._handle_archive_preview(ui)

    assert seen == [GABOR]
    assert ui.brain.modality == "gabor"


def test_a_native_entry_is_previewed_at_its_own_width_not_the_pool_s(tmp_path):
    """The genome column is padded to the WIDEST layout the archive holds, and
    apply_rule refuses a rule of the wrong width - so the pooled column breaks
    the ordinary same-brain preview too, once one foreign entry is wider."""
    arc, _foreign, native = _mixed(tmp_path)
    assert arc.brains.shape[1] == GABOR.length, "the pool is padded"
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()

    ui.archive.preview_entry_id = native
    h._handle_archive_preview(ui)

    rule, live = sim.applied[-1]
    assert live == FOURIER, "a native entry borrows nothing"
    assert rule.size == FOURIER.length
    assert sim.reallocs == []


def test_a_signature_this_build_cannot_rebuild_is_not_previewed(tmp_path):
    """Silent, deliberately: this runs off the pointer position, and the click
    is where the refusal is said out loud."""
    arc, foreign, _ = _mixed(tmp_path)
    arc.entries[foreign].layout = "quantum-n4"
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()

    ui.archive.preview_entry_id = foreign
    h._handle_archive_preview(ui)

    assert sim.applied == [] and sim.reallocs == []


def test_a_tournament_takes_the_borrowed_layout_back(tmp_path):
    """The canvas becomes a grid of simulations, so there is nothing for one
    rule to be previewed in - and the grid breeds genomes of the user's layout."""
    arc, foreign, _ = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h, ui = _handler(arc, sim), _ui()

    ui.archive.preview_entry_id = foreign
    h._handle_archive_preview(ui)
    ui.tournament.enabled = True
    h._handle_archive_preview(ui)

    assert sim.brain_layout == FOURIER
    assert h._borrow is None


# ---- the preview gate and the adopt gate are not the same gate ----------

def test_a_manual_tournament_grid_is_not_written_into_by_a_hover(tmp_path):
    """Slot 0 under a grid is TILE 0, so a hover would change one square of a
    running grid - the defect _grid_owner() fixes for the Z and G keys. The
    Manual tab sets neither sub-mode, so a tab-selection gate misses it."""
    arc, _foreign, native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.tournament.enabled = True          # the window is open: a grid is up
    depth = len(h.rule_manager.stack)

    ui.archive.preview_entry_id = native
    h._handle_archive_preview(ui)

    assert len(h.rule_manager.stack) == depth, "a grid must not be previewed into"


def test_clicking_a_foreign_entry_adopts_its_brain_with_the_grid_up(tmp_path):
    """Adopting a LAYOUT is not a single-sim operation: the Brain window's
    modality combo already switches layout while a tournament runs."""
    arc, foreign, _native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.tournament.enabled = True
    ui.archive.enabled = True             # the Explore tab is selected
    seen = []
    h.apply_brain_layout = lambda lay, _u: seen.append(lay)

    ui.archive.load_entry_id = foreign
    h._handle_archive_preview(ui)

    assert [l.signature() for l in seen] == [GABOR.signature()]
    assert ui.brain.modality == "gabor"


def test_clicking_a_native_entry_under_a_grid_does_not_push_a_rule(tmp_path):
    """Nothing to adopt and nowhere to run it: the grid owns slot 0."""
    arc, _foreign, native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.tournament.enabled = True
    depth = len(h.rule_manager.stack)

    ui.archive.load_entry_id = native
    h._handle_archive_preview(ui)

    assert len(h.rule_manager.stack) == depth


def test_with_no_grid_the_explore_tab_no_longer_blocks_anything(tmp_path):
    """A tab selection is not a grid. With the tournament window shut the
    browser works whichever tab was last on screen."""
    arc, _foreign, native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.archive.enabled = True             # tab selected, window shut
    depth = len(h.rule_manager.stack)

    ui.archive.preview_entry_id = native
    h._handle_archive_preview(ui)

    assert len(h.rule_manager.stack) > depth
