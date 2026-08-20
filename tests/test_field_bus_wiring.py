"""The bus is wired to the sim, and the override path is gone.

The old feature was gated on "is the Drawing Controls window open", which is
why closing the window silently stopped it. The bus must have no such gate.
"""
import inspect

import simulation_runner
from state import preferences_state
from state.ui_state import UIState
from utilities import advanced_drawing


def test_the_override_shader_path_is_gone():
    processor = advanced_drawing.AdvancedDrawingProcessor
    for name in ("process_override", "_ensure_override_resources",
                 "_cleanup_override", "get_available_override_shaders",
                 "resolve_override_shader_path"):
        assert not hasattr(processor, name), f"{name} should have been removed"


def test_the_override_preferences_are_gone():
    fields = preferences_state.PreferencesState.__dataclass_fields__
    assert "shader_driven_field" not in fields
    assert "field_override_shader" not in fields


def test_the_new_preferences_are_classified():
    fields = preferences_state.PreferencesState.__dataclass_fields__
    assert "field_bus_scale" in fields
    assert "show_field_stack" in fields
    classified = (set(preferences_state.UNDOABLE_FIELDS)
                  | set(preferences_state.NOT_UNDOABLE))
    assert {"field_bus_scale", "show_field_stack"} <= classified


def test_opening_the_window_is_not_an_undo_step():
    assert "show_field_stack" in preferences_state.NOT_UNDOABLE


def test_the_bus_resolution_is_undoable():
    assert "field_bus_scale" in preferences_state.UNDOABLE_FIELDS


def test_ui_state_carries_a_field_stack():
    assert hasattr(UIState(), "field_stack")
    assert UIState().field_stack.layers == []


def test_the_runner_takes_a_field_bus():
    params = inspect.signature(simulation_runner.SimulationRunner.__init__).parameters
    assert "field_bus" in params


def test_the_bus_rebuild_is_not_gated_on_a_window():
    frame = inspect.getsource(simulation_runner.SimulationRunner.run_simulation_frame)
    call = next(l for l in frame.splitlines() if "_rebuild_field_bus" in l)
    assert call.strip().startswith("self._rebuild_field_bus"), (
        "the rebuild must be called unconditionally, not inside a branch")

    body = inspect.getsource(simulation_runner.SimulationRunner._rebuild_field_bus)
    assert "field_bus.rebuild" in body
    assert "advanced_drawing_enabled" not in body, (
        "the bus must not be gated on whether a window is open")


def test_the_bus_is_off_under_a_tournament():
    body = inspect.getsource(simulation_runner.SimulationRunner._rebuild_field_bus)
    assert "tournament.enabled" in body, (
        "a shared field across isolated tiles would be scored in place of the genome")
