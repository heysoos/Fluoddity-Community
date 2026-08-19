"""A preset must not silently delete an injection setup.

Every preset in the library was saved before this feature existed, so it
carries no stack - and installing "no stack" over a setup that took a long
time to build is deleting user work on an unrelated action.
"""
from __future__ import annotations

from types import SimpleNamespace

from services.field_handler import FieldHandler
from state.field_stack import (FieldLayer, FieldStack, stack_from_dict,
                               stack_to_dict)


class _Sim:
    def get_canvas_dimensions(self):
        return 64, 64


class _Cache:
    def __init__(self, data=None):
        self._data = data

    def get(self, *_a):
        return self._data


def _handler(field_data=None):
    handler = FieldHandler.__new__(FieldHandler)
    handler.sim = _Sim()
    handler.cache = _Cache(field_data)
    handler.field_bus = None
    handler.param_lock_service = None
    return handler


def _ui_state(stack):
    return SimpleNamespace(field_stack=stack, sim=SimpleNamespace(),
                           preferences=SimpleNamespace())


def _config(stack_dict=None):
    return SimpleNamespace(field_stack=stack_dict or {},
                           force_field_strength=None,
                           strafe_field_strength=None)


def _mine():
    return FieldStack(layers=[FieldLayer(source="webcam", mapping="curl")])


def test_a_preset_with_no_stack_leaves_mine_alone():
    """This is every preset in the library."""
    stack = _mine()
    mine = stack.layers[0]
    _handler().apply_for_config(_config(), "x.json", _ui_state(stack))
    assert stack.layers == [mine], "the preset deleted the stack"


def test_a_preset_with_a_stack_replaces_mine():
    stack = _mine()
    theirs = stack_to_dict(FieldStack(layers=[FieldLayer(source="noise")]))
    _handler().apply_for_config(_config(theirs), "x.json", _ui_state(stack))
    assert [l.source for l in stack.layers] == ["noise"]


def test_the_lock_keeps_mine_even_against_a_preset_that_has_one():
    stack = _mine()
    stack.locked = True
    mine = stack.layers[0]
    theirs = stack_to_dict(FieldStack(layers=[FieldLayer(source="noise")]))
    _handler().apply_for_config(_config(theirs), "x.json", _ui_state(stack))
    assert stack.layers == [mine], "the lock did not hold"


def test_the_lock_survives_a_save_and_load():
    stack = FieldStack(layers=[FieldLayer()], locked=True)
    assert stack_from_dict(stack_to_dict(stack)).locked is True


def test_loading_a_preset_carries_its_global_switch():
    stack = _mine()
    theirs = stack_to_dict(FieldStack(layers=[FieldLayer()], enabled=False))
    _handler().apply_for_config(_config(theirs), "x.json", _ui_state(stack))
    assert stack.enabled is False
