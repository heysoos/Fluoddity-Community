"""The cycle walks the set list on a timer and a key press restarts it."""
import pytest

from command_handler import CommandHandler
from services.slot_cycle import SlotCycle
from state.set_list import SetList, Slot
from state.ui_state import UIState
from tests.test_exhibition_keys import _handler, _write

SL = SetList(slots=[Slot("1", "Core", "a"), Slot("2"), Slot("3", "Core", "b"),
                    Slot("4", rig="only-rig")])


def test_the_clock_starts_on_the_first_look_and_steps_in_order():
    c = SlotCycle()
    assert c.due(SL, 100.0, 60) is None
    assert c.due(SL, 159.0, 60) is None
    assert c.due(SL, 160.0, 60).filename == "a"
    assert c.due(SL, 200.0, 60) is None
    assert c.due(SL, 220.0, 60).filename == "b"
    assert c.due(SL, 280.0, 60).rig == "only-rig"
    assert c.due(SL, 340.0, 60).filename == "a", "wraps"


def test_a_load_from_elsewhere_restarts_the_clock_from_that_slot():
    c = SlotCycle()
    c.due(SL, 0.0, 60)
    c.note_loaded(SL.slots[2], 50.0)
    assert c.due(SL, 100.0, 60) is None
    assert c.due(SL, 110.0, 60).rig == "only-rig"


def test_an_empty_set_list_never_fires():
    c = SlotCycle()
    assert c.due(SetList.default(), 0.0, 1) is None
    assert c.due(SetList.default(), 1000.0, 1) is None


def test_the_dispatch_loads_the_next_slot_when_due(tmp_path):
    ch = _handler(tmp_path)
    _write(tmp_path, "a", 1.0)
    _write(tmp_path, "b", 2.0)
    ui = UIState()
    ui.perform.set_list = SetList(slots=[Slot("1", "Core", "a"), Slot("2", "Core", "b")])
    ui.perform.cycle_enabled = True
    ui.preferences.cycle_seconds = 10.0

    ch._handle_slot_load(ui, now=0.0)
    assert ch.ui.loaded == []
    ch._handle_slot_load(ui, now=5.0)
    assert ch.ui.loaded == []
    ch._handle_slot_load(ui, now=10.0)
    assert ch.ui.loaded == ["a"]
    assert ui.sim.SENSOR_GAIN == pytest.approx(1.0)
    ch._handle_slot_load(ui, now=20.0)
    assert ch.ui.loaded == ["a", "b"]


def test_a_key_press_restarts_the_timer_and_the_order(tmp_path):
    ch = _handler(tmp_path)
    _write(tmp_path, "a", 1.0)
    _write(tmp_path, "b", 2.0)
    ui = UIState()
    ui.perform.set_list = SetList(slots=[Slot("1", "Core", "a"), Slot("2", "Core", "b")])
    ui.perform.cycle_enabled = True
    ui.preferences.cycle_seconds = 10.0

    ch._handle_slot_load(ui, now=0.0)
    ui.perform.slot_load = ("b", "Core", "")
    ch._handle_slot_load(ui, now=8.0)
    assert ch.ui.loaded == ["b"]
    ch._handle_slot_load(ui, now=12.0)
    assert ch.ui.loaded == ["b"], "the visitor's choice stays up a full interval"
    ch._handle_slot_load(ui, now=18.0)
    assert ch.ui.loaded == ["b", "a"]


def test_switching_the_cycle_off_and_on_starts_fresh(tmp_path):
    ch = _handler(tmp_path)
    _write(tmp_path, "a", 1.0)
    ui = UIState()
    ui.perform.set_list = SetList(slots=[Slot("1", "Core", "a")])
    ui.perform.cycle_enabled = True
    ui.preferences.cycle_seconds = 10.0
    ch._handle_slot_load(ui, now=0.0)
    ui.perform.cycle_enabled = False
    ch._handle_slot_load(ui, now=50.0)
    ui.perform.cycle_enabled = True
    ch._handle_slot_load(ui, now=51.0)
    assert ch.ui.loaded == [], "a stale clock would fire at once"
    ch._handle_slot_load(ui, now=61.0)
    assert ch.ui.loaded == ["a"]
