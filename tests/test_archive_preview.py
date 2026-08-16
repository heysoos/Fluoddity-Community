"""Live preview: run an archive entry in the single sim from the browser.

The rule is pushed onto the same stack File > Load's preview uses, so the
invariant that matters is that it always comes back off: a preview that leaks
a push leaves the user's own creature buried under an archive entry with no
way to reach it.
"""
import numpy as np
import pytest

from services.brains import default_layout
from state.archive_state import ArchiveState
from state.auto_tournament_state import AutoTournamentState
from state.sim_state import SimState


class _Entry:
    def __init__(self, i, spec="brain:80", run_id="run-a", layout=""):
        self.id = i
        self.spec = spec
        self.run_id = run_id
        self.layout = layout
        self.thumb = f"{i:06d}.jpg"


class _FakeArchive:
    # `seed` because two archives must hold DIFFERENT creatures: with one
    # fixed seed, "switched to another archive" is indistinguishable from
    # "did nothing", and every assertion about it passes vacuously.
    def __init__(self, n=4, spec="brain:80", seed=0):
        rng = np.random.default_rng(seed)
        self.entries = [_Entry(i, spec) for i in range(n)]
        self.brains = rng.normal(0, 0.5, (n, 10, 8)).astype(np.float32)
        # PHYSICS_PARAMS is 8 long; values are absolute, not offsets.
        self.physics = np.full((n, 8), 0.25, dtype=np.float32)
        # One archive holds every brain, so the preview asks each row which one
        # it belongs to before deciding it can run it. All native here; the
        # foreign case is tests/test_mixed_archive.py.
        self.layout = default_layout()
        self.signature = self.layout.signature()

    def layout_at(self, i):
        return self.entries[i].layout or self.signature

    def is_native(self, i):
        return self.layout_at(i) == self.signature

    def brain_at(self, i):
        return np.asarray(self.brains[i], dtype=np.float32).reshape(-1)

    def thumb_key(self, i):
        t = self.entries[i].thumb
        return f"{self.layout_at(i)}/{t}" if t else ""


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


class _FakeStore:
    """Only what the preview asks of a store: one config per run id."""

    def __init__(self, configs=None):
        self.configs = dict(configs or {})
        self.saved = {}

    def load_run_config(self, run_id):
        return self.configs.get(run_id)

    def save_run_config(self, run_id, config_json):
        self.saved.setdefault(run_id, config_json)
        return True


def _handler(archive=None, store=None):
    from command_handler import CommandHandler

    h = object.__new__(CommandHandler)
    h.archive = archive if archive is not None else _FakeArchive()
    h.archive_store = store
    h._run_config_cache = {}
    h._run_physics_written = ""
    h.sim = _Sim()
    h.rule_manager = _RuleManager(np.zeros((10, 8), dtype=np.float32))
    h.param_lock_service = None
    h._archive_preview_id = -1
    h._archive_preview_pushed = False
    h._archive_preview_physics = None
    h._archive_preview_arc = None
    h._borrow = None
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
    assert np.allclose(h.sim.applied[-1], h.archive.brain_at(2))


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
    assert np.allclose(h.sim.applied[-1], h.archive.brain_at(3))


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
    assert np.allclose(h.sim.applied[-1], h.archive.brain_at(2))
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
    assert np.allclose(h.sim.applied[-1], h.archive.brain_at(3))


def test_a_commit_reports_itself():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    ui.archive.load_entry_id = 0
    h._handle_archive_preview(ui)
    assert "#0" in ui.archive.notice


def test_a_commit_says_so_when_the_entry_stored_no_physics():
    """An archive written before run configs existed, or one whose run had
    physics search off, recorded nothing but the brain - so a click moves no
    sliders. Said plainly, that is a limit of what was stored; unsaid, it
    reads as a control that only half works."""
    h, ui = _handler(), _UI()               # default spec is brain-only
    h._run_config_cache = {"run-a": None}   # and no run config either
    ui.archive.live_preview = True
    ui.archive.load_entry_id = 0
    h._handle_archive_preview(ui)
    assert "brain only" in ui.archive.notice


def test_a_commit_stays_quiet_when_the_entry_did_store_physics():
    h, ui = _handler(_FakeArchive(spec="brain:80+physics:8")), _UI()
    ui.archive.live_preview = True
    ui.archive.load_entry_id = 0
    h._handle_archive_preview(ui)
    assert "brain only" not in ui.archive.notice


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


# ---- switching archives under a live preview ------------------------------

def test_switching_archives_ends_the_preview():
    """Entry ids restart at 0 in every archive, so the id being previewed
    names a different creature in the new one - or none at all."""
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)
    assert h.rule_manager.depth == 1

    # A switch rebuilds the Archive object, and the pointer is over the combo
    # rather than the gallery on that frame.
    h.archive = _FakeArchive(seed=1)
    hover(h, ui, -1)
    assert h.rule_manager.depth == 0
    assert np.allclose(h.sim.applied[-1], h.rule_manager.base[0])


def test_after_a_switch_hovering_previews_the_new_archive():
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)

    fresh = _FakeArchive(seed=1)
    h.archive = fresh
    hover(h, ui, 2)
    assert h.rule_manager.depth == 1
    assert np.allclose(h.sim.applied[-1], fresh.brain_at(2))


def test_a_switch_restores_the_physics_of_the_archive_that_set_them():
    h = _handler(_FakeArchive(spec="brain:80+physics:8"))
    ui = _UI()
    ui.archive.live_preview = True
    before = {n: getattr(ui.sim, n) for n in _physics_names()}
    hover(h, ui, 1)

    h.archive = _FakeArchive(spec="brain:80+physics:8", seed=1)
    hover(h, ui, -1)
    assert {n: getattr(ui.sim, n) for n in _physics_names()} == before


def test_a_switch_while_still_hovering_shows_the_new_archives_entry():
    """Not a bug: id 2 of the new archive is what hovering row 2 now means.
    What must not happen is the OLD archive's brain staying on the GPU."""
    h, ui = _handler(), _UI()
    ui.archive.live_preview = True
    hover(h, ui, 2)
    old = h.archive

    fresh = _FakeArchive(seed=1)
    h.archive = fresh
    h._handle_archive_preview(ui)          # preview_entry_id is still 2
    assert h.rule_manager.depth == 1, "still exactly one push"
    assert np.allclose(h.sim.applied[-1], fresh.brain_at(2))
    assert not np.allclose(fresh.brain_at(2), old.brain_at(2))


# ---- the run's own physics -------------------------------------------------
#
# An entry stores its brain and, with physics search on, the deltas the
# optimizer moved. The config those deltas are RELATIVE to lives in a per-run
# file, because with search off it is the entry's physics entirely.

def _run_config(**overrides):
    from services.config_saver import ConfigSaver

    state = SimState()
    for name, value in overrides.items():
        setattr(state, name, value)
    return ConfigSaver().create_config(state, None).to_json()


def test_a_brain_only_entry_brings_the_physics_of_its_run():
    """The whole bug: with physics search off nothing about an entry's physics
    was recorded, so it replayed under whatever the sliders happened to say."""
    store = _FakeStore({"run-a": _run_config(SENSOR_GAIN=3.5, TRAIL_PERSISTENCE=0.5)})
    h, ui = _handler(store=store), _UI()
    ui.sim.SENSOR_GAIN = 0.1
    ui.archive.live_preview = True

    hover(h, ui, 1)
    assert ui.sim.SENSOR_GAIN == pytest.approx(3.5)
    # Not a physics slider, and exactly why the whole config is recorded.
    assert ui.sim.TRAIL_PERSISTENCE == pytest.approx(0.5)


def test_the_searched_deltas_win_over_the_runs_config():
    """Two layers: the run config is the base, the entry's own vector overrides
    it for the parameters the optimizer actually moved."""
    store = _FakeStore({"run-a": _run_config(SENSOR_GAIN=3.5, TRAIL_PERSISTENCE=0.5)})
    h, ui = _handler(_FakeArchive(spec="brain:80+physics:8"), store=store), _UI()
    ui.archive.live_preview = True

    hover(h, ui, 1)
    assert ui.sim.SENSOR_GAIN == pytest.approx(0.25)   # searched, from _phys
    assert ui.sim.TRAIL_PERSISTENCE == pytest.approx(0.5)  # not searched, from the run


def test_leaving_puts_back_everything_the_run_config_touched():
    store = _FakeStore({"run-a": _run_config(SENSOR_GAIN=3.5, TRAIL_PERSISTENCE=0.5)})
    h, ui = _handler(store=store), _UI()
    ui.sim.SENSOR_GAIN, ui.sim.TRAIL_PERSISTENCE = 0.1, 0.9
    ui.archive.live_preview = True

    hover(h, ui, 1)
    hover(h, ui, -1)
    assert ui.sim.SENSOR_GAIN == pytest.approx(0.1)
    assert ui.sim.TRAIL_PERSISTENCE == pytest.approx(0.9)


def test_moving_between_entries_restores_the_user_not_the_previous_entry():
    """The snapshot is taken once, before the FIRST push."""
    store = _FakeStore({"run-a": _run_config(SENSOR_GAIN=3.5),
                        "run-b": _run_config(SENSOR_GAIN=1.5)})
    arc = _FakeArchive()
    arc.entries[2].run_id = "run-b"
    h, ui = _handler(arc, store=store), _UI()
    ui.sim.SENSOR_GAIN = 0.1
    ui.archive.live_preview = True

    hover(h, ui, 1)
    hover(h, ui, 2)
    assert ui.sim.SENSOR_GAIN == pytest.approx(1.5)
    hover(h, ui, -1)
    assert ui.sim.SENSOR_GAIN == pytest.approx(0.1)


def test_a_run_recorded_before_this_existed_leaves_the_sliders_alone():
    """Every archive on disk predates run configs. A missing file must mean
    'do nothing', which is exactly what those entries did before."""
    h, ui = _handler(store=_FakeStore()), _UI()
    ui.sim.SENSOR_GAIN = 0.1
    ui.archive.live_preview = True

    hover(h, ui, 1)
    assert ui.sim.SENSOR_GAIN == pytest.approx(0.1)
    assert h._archive_preview_physics is None, "nothing to restore, nothing snapshotted"


def test_the_run_config_is_read_once_per_run():
    """The browser hovers on every frame."""
    store = _FakeStore({"run-a": _run_config(SENSOR_GAIN=3.5)})
    reads = []
    inner = store.load_run_config
    store.load_run_config = lambda rid: (reads.append(rid), inner(rid))[1]

    h, ui = _handler(store=store), _UI()
    ui.archive.live_preview = True
    for entry_id in (1, 2, 3, 1, 2):
        hover(h, ui, entry_id)
    assert reads == ["run-a"]
