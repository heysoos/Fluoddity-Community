"""Restoring a snapshot: the fields, the locks, and the tournament split."""
import numpy as np
import pytest

from command_handler import CommandHandler
from services import undo_history as uh
from state.ui_state import UIState


class FakeSim:
    def __init__(self):
        self.brain_layout = None
        self.applied = []

    def apply_rule(self, rule):
        self.applied.append(None if rule is None else np.array(rule, copy=True))


class FakeRuleManager:
    def __init__(self):
        self.pushed = []

    def push_rule(self, rule, seed):
        self.pushed.append((np.array(rule, copy=True), seed))


def handler(**over):
    kwargs = dict(sim=FakeSim(), camera=None, ui=None,
                  rule_manager=FakeRuleManager(), entity_picker=None,
                  video_service=None, config_saver=None,
                  multi_load_service=None, user_configs_dir=None)
    kwargs.update(over)
    return CommandHandler(**kwargs)


def _perturb(value):
    """A different value of the same kind. bool before int - bool IS an int."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, dict):
        return {**value, "__probe__": 1.0}
    if isinstance(value, list):
        return list(value) + [1.0]
    return value


def test_every_undoable_field_round_trips():
    """Catches a field that is captured but never applied - the same defect
    as one that is never captured, wearing a different hat."""
    st = UIState()
    snap = uh.capture(st, None, None)
    for name, module, _panel in uh.CONTAINERS:
        target = getattr(st, name)
        for f in module.UNDOABLE_FIELDS:
            setattr(target, f, _perturb(getattr(target, f)))

    handler().apply_undo_snapshot(snap, st)

    for name, module, _panel in uh.CONTAINERS:
        target = getattr(st, name)
        for f in module.UNDOABLE_FIELDS:
            assert getattr(target, f) == snap.fields[name][f], f


def test_restoring_does_not_touch_excluded_fields():
    st = UIState()
    snap = uh.capture(st, None, None)
    st.sim.going = False
    st.preferences.show_tutorial_window = False

    handler().apply_undo_snapshot(snap, st)

    assert st.sim.going is False
    assert st.preferences.show_tutorial_window is False


def test_a_missing_field_is_skipped_rather_than_defaulted():
    """A snapshot taken before a field existed must stay applicable."""
    st = UIState()
    snap = uh.capture(st, None, None)
    del snap.fields["sim"]["DRAG"]
    st.sim.DRAG = 0.123

    handler().apply_undo_snapshot(snap, st)

    assert st.sim.DRAG == pytest.approx(0.123)


def test_the_rule_reaches_the_gpu_when_nobody_owns_the_grid():
    st = UIState()
    rule = np.arange(80, dtype=np.float32)
    snap = uh.capture(st, rule, None)
    sim = FakeSim()

    skipped = handler(sim=sim).apply_undo_snapshot(snap, st)

    assert skipped is False
    assert len(sim.applied) == 1
    assert np.array_equal(sim.applied[0], rule)


def test_the_layout_goes_back_before_the_rule():
    """apply_rule measures a rule against the LIVE layout and silently refuses
    a mismatch, so restoring the rule first would drop it without an error.

    The Snapshot is built directly rather than through capture(), which would
    call settings_of() on a layout this test does not have.
    """
    class FakeLayout:
        def __init__(self, sig):
            self._sig = sig

        def signature(self):
            return self._sig

    st = UIState()
    rule = np.arange(80, dtype=np.float32)      # fourier-n10 is 80 floats
    snap = uh.Snapshot(fields=uh.capture(st, None, None).fields, rule=rule,
                       brain_signature="fourier-n10", brain_settings={})
    sim = FakeSim()
    sim.brain_layout = FakeLayout("mlp-n16-a0")
    h = handler(sim=sim)

    order = []

    def fake_apply_layout(layout, ui_state):
        order.append("layout")
        sim.brain_layout = layout
        return True

    plain_apply = sim.apply_rule

    def recording_apply(r):
        order.append("rule")
        plain_apply(r)

    h.apply_brain_layout = fake_apply_layout
    sim.apply_rule = recording_apply

    h.apply_undo_snapshot(snap, st)

    assert order == ["layout", "rule"]
    assert sim.brain_layout.signature() == "fourier-n10"


def test_a_matching_layout_is_not_switched():
    """A switch tears down and rebuilds the archive - never do it for nothing."""
    class FakeLayout:
        def signature(self):
            return "fourier-n10"

    st = UIState()
    snap = uh.Snapshot(fields=uh.capture(st, None, None).fields,
                       rule=np.arange(80, dtype=np.float32),
                       brain_signature="fourier-n10", brain_settings={})
    sim = FakeSim()
    sim.brain_layout = FakeLayout()
    h = handler(sim=sim)
    switched = []
    h.apply_brain_layout = lambda layout, ui_state: switched.append(layout)

    h.apply_undo_snapshot(snap, st)

    assert switched == []
    assert len(sim.applied) == 1


def test_the_brain_half_is_skipped_while_a_tournament_owns_the_grid():
    st = UIState()
    st.tournament.enabled = True
    rule = np.arange(80, dtype=np.float32)
    snap = uh.capture(st, rule, None)
    st.sim.AXIAL_FORCE = 0.9
    sim = FakeSim()
    h = handler(sim=sim)
    h.tournament_service = object()

    skipped = h.apply_undo_snapshot(snap, st)

    assert skipped is True
    assert sim.applied == []
    assert st.sim.AXIAL_FORCE == pytest.approx(snap.fields["sim"]["AXIAL_FORCE"])
