"""Live preview: run an archive entry in the single sim from the browser.

The rule is pushed onto the same stack File > Load's preview uses, so the
invariant that matters is that it always comes back off: a preview that leaks
a push leaves the user's own creature buried under an archive entry with no
way to reach it.
"""
import numpy as np
import pytest

from state.archive_state import ArchiveState
from state.auto_tournament_state import AutoTournamentState
from state.sim_state import SimState


class _Entry:
    def __init__(self, i, spec="brain:80"):
        self.id = i
        self.spec = spec


class _FakeArchive:
    def __init__(self, n=4, spec="brain:80"):
        rng = np.random.default_rng(0)
        self.entries = [_Entry(i, spec) for i in range(n)]
        self.brains = rng.normal(0, 0.5, (n, 10, 8)).astype(np.float32)
        # PHYSICS_PARAMS is 8 long; values are absolute, not offsets.
        self.physics = np.full((n, 8), 0.25, dtype=np.float32)


class _RuleManager:
    """services.rule_manager.RuleManager's contract, verbatim.

    The top of the stack IS the current rule, so pop removes it and returns the
    new top - and (None, None) when there was nothing underneath. Getting that
    last case wrong in the fake is how a None reaching sim.apply_rule hides.
    """

    BASE_SEED = 7

    def __init__(self, base=None):
        self.rule_history = []
        if base is not None:
            self.rule_history.append((base, self.BASE_SEED))

    def push_rule(self, rule, seed):
        self.rule_history.append((np.array(rule), seed))

    def pop_rule(self):
        if len(self.rule_history) > 1:
            self.rule_history.pop()
            return self.rule_history[-1]
        if len(self.rule_history) == 1:
            self.rule_history = []
            return (None, None)
        return (None, None)

    @property
    def depth(self):
        """Pushes ON TOP of the rule that was already current."""
        return max(0, len(self.rule_history) - 1)

    @property
    def base(self):
        return self.rule_history[0]


class _Sim:
    def __init__(self):
        self.applied = []

    def apply_rule(self, rule):
        # The RAW argument. np.array(None) is a 0-d object array, which reads
        # as "not None" and would hide exactly the bug below.
        self.applied.append(rule)


class _UI:
    def __init__(self):
        self.sim = SimState()
        self.archive = ArchiveState()
        self.auto_tournament = AutoTournamentState()


def _handler(archive=None):
    from command_handler import CommandHandler

    h = object.__new__(CommandHandler)
    h.archive = archive if archive is not None else _FakeArchive()
    h.sim = _Sim()
    h.rule_manager = _RuleManager(np.zeros((10, 8), dtype=np.float32))
    h.param_lock_service = None
    h._archive_preview_id = -1
    h._archive_preview_pushed = False
    h._archive_preview_physics = None
    return h


def hover(h, ui, entry_id):
    ui.archive.preview_entry_id = entry_id
    h._handle_archive_preview(ui)


# ---- the toggle gates everything ------------------------------------------

def test_hovering_does_nothing_while_the_toggle_is_off():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = False
    hover(h, ui, 2)
    assert h.rule_manager.depth == 0
    assert h.sim.applied == []


def test_hovering_runs_the_entry_when_the_toggle_is_on():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)
    assert h.rule_manager.depth == 1
    assert np.allclose(h.sim.applied[-1], h.archive.brains[2])


def test_turning_the_toggle_off_puts_back_what_was_running():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 1)
    ui.archive.live_preview = False
    h._handle_archive_preview(ui)
    assert h.rule_manager.depth == 0
    assert np.allclose(h.sim.applied[-1], h.rule_manager.base[0])


# ---- the stack must not grow ----------------------------------------------

def test_moving_between_entries_swaps_rather_than_stacks():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    for entry_id in (0, 1, 2, 3, 0, 3):
        hover(h, ui, entry_id)
        assert h.rule_manager.depth == 1, "each preview must replace the last"
    assert np.allclose(h.sim.applied[-1], h.archive.brains[3])


def test_leaving_the_browser_restores_the_original_rule():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)
    hover(h, ui, -1)          # the pointer is over nothing
    assert h.rule_manager.depth == 0
    assert np.allclose(h.sim.applied[-1], h.rule_manager.base[0])


def test_hovering_the_same_entry_repeatedly_is_a_no_op():
    """This runs every frame; re-applying at 60 Hz would restart the rule
    continuously."""
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 1)
    n = len(h.sim.applied)
    for _ in range(5):
        hover(h, ui, 1)
    assert len(h.sim.applied) == n
    assert h.rule_manager.depth == 1


def test_the_rule_seed_comes_back_with_the_rule():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    ui.sim.rule_seed = 99
    hover(h, ui, 0)
    hover(h, ui, -1)
    assert ui.sim.rule_seed == _RuleManager.BASE_SEED


# ---- physics ---------------------------------------------------------------

def _physics_names():
    from services.physics_genome import PHYSICS_PARAMS

    return [name for name, _g, _lo, _hi in PHYSICS_PARAMS]


def test_an_entry_searched_with_physics_brings_its_physics():
    """A genome found under a searched slider is not the same creature under
    whatever the sliders happen to be."""
    h = _handler(_FakeArchive(spec="brain:80+physics:8"))
    ui = _UI()
    ui.archive.live_preview = True
    hover(h, ui, 1)
    for name in _physics_names():
        assert getattr(ui.sim, name) == pytest.approx(0.25)


def test_the_physics_sliders_come_back_on_leaving():
    h = _handler(_FakeArchive(spec="brain:80+physics:8"))
    ui = _UI()
    ui.archive.live_preview = True
    before = {n: getattr(ui.sim, n) for n in _physics_names()}
    hover(h, ui, 1)
    hover(h, ui, -1)
    assert {n: getattr(ui.sim, n) for n in _physics_names()} == before


def test_a_brain_only_entry_leaves_the_sliders_alone():
    h, ui = _handler(), _UI()          # default spec has no physics
    ui.archive.live_preview = True
    before = {n: getattr(ui.sim, n) for n in _physics_names()}
    hover(h, ui, 1)
    assert {n: getattr(ui.sim, n) for n in _physics_names()} == before


def test_swapping_between_physics_entries_restores_the_original_not_the_last():
    """The snapshot is of what the user had, and must survive being previewed
    over - otherwise leaving restores the previous ENTRY's physics."""
    arc = _FakeArchive(spec="brain:80+physics:8")
    arc.physics[2] = 0.75
    h, ui = _handler(arc), _UI()
    ui.archive.live_preview = True
    before = {n: getattr(ui.sim, n) for n in _physics_names()}
    hover(h, ui, 1)
    hover(h, ui, 2)
    hover(h, ui, -1)
    assert {n: getattr(ui.sim, n) for n in _physics_names()} == before


# ---- committing ------------------------------------------------------------

def test_clicking_keeps_the_entry_after_the_pointer_leaves():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)
    ui.archive.load_entry_id = 2
    h._handle_archive_preview(ui)
    hover(h, ui, -1)
    assert np.allclose(h.sim.applied[-1], h.archive.brains[2])
    assert ui.archive.load_entry_id == -1


def test_committing_clears_the_restore_point():
    """Committing drops the undo, so a later preview restores the entry the
    user chose rather than what was running before it."""
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)
    ui.archive.load_entry_id = 2
    h._handle_archive_preview(ui)
    assert h._archive_preview_id == -1
    assert h._archive_preview_physics is None


def test_clicking_without_hovering_first_still_loads():
    """The map's click path can fire on an entry that was never the hover
    target, e.g. when live preview was toggled on mid-drag."""
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    ui.archive.load_entry_id = 3
    h._handle_archive_preview(ui)
    assert np.allclose(h.sim.applied[-1], h.archive.brains[3])


def test_a_commit_reports_itself():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    ui.archive.load_entry_id = 0
    h._handle_archive_preview(ui)
    assert "#0" in ui.archive.notice


# ---- the mode guard --------------------------------------------------------

@pytest.mark.parametrize("mode", ["archive", "auto_tournament"])
def test_tournament_mode_refuses_to_preview(mode):
    """The canvas is a grid of simulations there; one rule swapped into it
    would mean nothing."""
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    getattr(ui, mode).enabled = True
    hover(h, ui, 2)
    assert h.rule_manager.depth == 0
    assert h.sim.applied == []


def test_entering_tournament_mode_ends_a_running_preview():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)
    ui.archive.enabled = True
    h._handle_archive_preview(ui)
    assert h.rule_manager.depth == 0
    assert np.allclose(h.sim.applied[-1], h.rule_manager.base[0])


def test_a_pending_click_is_dropped_rather_than_queued_in_tournament_mode():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    ui.archive.enabled = True
    ui.archive.load_entry_id = 2
    h._handle_archive_preview(ui)
    assert ui.archive.load_entry_id == -1
    assert h.sim.applied == []


# ---- degenerate input ------------------------------------------------------

def test_an_entry_that_was_deleted_mid_hover_does_not_crash():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 999)
    assert h.rule_manager.depth == 0


def test_no_archive_is_survivable():
    h, ui = _handler(), _UI()
    h.archive = None
    ui.archive.live_preview = True
    hover(h, ui, 1)
    assert h.sim.applied == []


def test_the_rule_lock_blocks_the_push_but_not_the_bookkeeping():
    """With the rule locked nothing may reach the GPU, and leaving must not
    then pop a rule that was never pushed."""
    h, ui = _handler(), _UI()
    h.param_lock_service = type(
        "PLS", (), {"should_block_rule_push": staticmethod(lambda: True)})()
    ui.archive.live_preview = True
    hover(h, ui, 1)
    assert h.rule_manager.depth == 0 and h.sim.applied == []
    hover(h, ui, -1)
    assert h.rule_manager.depth == 0 and h.sim.applied == []


def test_leaving_with_nothing_underneath_does_not_apply_a_none_rule():
    """RuleManager.pop_rule returns (None, None) when the popped rule was the
    only one on the stack. There is then nothing to go back to, and passing
    that None straight to the GPU is not the answer."""
    h, ui = _handler(), _UI()
    h.rule_manager = _RuleManager()          # no rule was ever current
    ui.archive.live_preview = True
    hover(h, ui, 1)
    hover(h, ui, -1)
    assert all(r is not None for r in h.sim.applied)
