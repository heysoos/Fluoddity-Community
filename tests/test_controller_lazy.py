"""The joystick scan must not run unless something reads the controller.

GLFW initialises its Win32 joystick backend lazily, on the first call to any
joystick function, and that initialisation runs IDirectInput8::EnumDevices over
every HID node on the machine. A device that does not answer the probe costs a
ten-second timeout each, so the scan is not bounded by anything Fluoddity can
see. It is only ever worth paying when the controller camera is actually being
read.
"""
import inspect

import pytest

import controller_input
from controller_input import ControllerCam, process_controller_input

JOYSTICK_API = ("joystick_present", "get_joystick_axes", "get_joystick_buttons",
                "get_joystick_name", "set_joystick_callback")


@pytest.fixture
def no_joystick_calls(monkeypatch):
    """Make every GLFW joystick entry point fail loudly."""
    calls = []

    def boom(name):
        def fn(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"glfw.{name} was called")
        return fn

    for name in JOYSTICK_API:
        monkeypatch.setattr(controller_input.glfw, name, boom(name))
    return calls


def test_an_inactive_frame_never_touches_the_joystick_api(no_joystick_calls):
    state = {'joystick_id': None, 'prev_buttons': []}
    process_controller_input(ControllerCam(), state, 0.016, active=False)
    assert no_joystick_calls == []
    assert state['joystick_id'] is None


def test_an_active_frame_does_scan(monkeypatch):
    seen = []
    monkeypatch.setattr(controller_input.glfw, "joystick_present",
                        lambda jid: seen.append(jid) or False)
    state = {'joystick_id': None, 'prev_buttons': []}
    process_controller_input(ControllerCam(), state, 0.016, active=True)
    assert seen, "an active frame must look for a controller"


def test_startup_does_not_scan():
    """App.__init__ must leave the id unknown rather than go looking for one.

    The scan used to sit here, which is what put a DirectInput enumeration in
    front of the window appearing.
    """
    import main

    src = inspect.getsource(main.App.__init__)
    assert "find_joystick" not in src
