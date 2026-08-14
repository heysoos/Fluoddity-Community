"""Capturing, comparing and naming a snapshot."""
import numpy as np

from services import undo_history as uh
from state.ui_state import UIState


def test_capture_deep_copies_mutable_fields():
    st = UIState()
    snap = uh.capture(st, None, None)
    st.sim.x_sweeps["DRAG"] = 1.0
    assert snap.fields["sim"]["x_sweeps"]["DRAG"] == 0.0


def test_same_is_true_for_an_unchanged_state():
    st = UIState()
    assert uh.same(uh.capture(st, None, None), uh.capture(st, None, None))


def test_same_is_false_after_a_scalar_change():
    st = UIState()
    before = uh.capture(st, None, None)
    st.sim.AXIAL_FORCE += 0.01
    assert not uh.same(before, uh.capture(st, None, None))


def test_same_is_false_after_a_preference_change():
    st = UIState()
    before = uh.capture(st, None, None)
    st.preferences.brightness += 1.0
    assert not uh.same(before, uh.capture(st, None, None))


def test_excluded_fields_do_not_register_as_a_change():
    st = UIState()
    before = uh.capture(st, None, None)
    st.sim.going = not st.sim.going
    st.preferences.show_tutorial_window = not st.preferences.show_tutorial_window
    assert uh.same(before, uh.capture(st, None, None))


def test_same_is_false_after_a_rule_change():
    st = UIState()
    rule = np.zeros(80, dtype=np.float32)
    before = uh.capture(st, rule, None)
    changed = rule.copy()
    changed[3] = 1.0
    assert not uh.same(before, uh.capture(st, changed, None))


def test_capture_copies_the_rule():
    st = UIState()
    rule = np.zeros(80, dtype=np.float32)
    snap = uh.capture(st, rule, None)
    rule[0] = 9.0
    assert snap.rule[0] == 0.0


def test_a_label_difference_is_not_a_state_difference():
    st = UIState()
    a = uh.capture(st, None, None)
    b = uh.capture(st, None, None)
    object.__setattr__(b, "label", "something")
    assert uh.same(a, b)


def test_describe_names_a_single_physics_field_by_its_ui_label():
    st = UIState()
    before = uh.capture(st, None, None)
    st.sim.SENSOR_GAIN += 0.1
    assert uh.describe(before, uh.capture(st, None, None)) == "Sensor Gain"


def test_describe_names_the_container_when_several_fields_move():
    st = UIState()
    before = uh.capture(st, None, None)
    st.sim.SENSOR_GAIN += 0.1
    st.sim.DRAG += 0.1
    assert uh.describe(before, uh.capture(st, None, None)) == "Physics"


def test_describe_names_the_rule():
    st = UIState()
    rule = np.zeros(80, dtype=np.float32)
    before = uh.capture(st, rule, None)
    changed = rule.copy()
    changed[0] = 1.0
    assert uh.describe(before, uh.capture(st, changed, None)) == "Rule"


def test_describe_falls_back_to_a_titled_field_name():
    st = UIState()
    before = uh.capture(st, None, None)
    st.preferences.brightness += 1.0
    assert uh.describe(before, uh.capture(st, None, None)) == "Brightness"
