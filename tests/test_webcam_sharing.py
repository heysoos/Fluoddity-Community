"""Two webcam layers must not fight over one device.

DirectShow refuses the second open of a camera, and two layers churning a
device between them - each opening what the other just closed, every frame -
is what a run with two webcam layers did not survive.
"""
from __future__ import annotations

from services.field_sources import _WebcamSource
from state.field_stack import FieldLayer


class _Bus:
    """Just the claim, which is all a webcam source asks of the bus."""

    def __init__(self):
        self._claims = {}

    def claim_device(self, name, source):
        return self._claims.setdefault(name, source)

    def new_frame(self):
        self._claims.clear()


def _source():
    s = _WebcamSource.__new__(_WebcamSource)
    s.error = None
    s._reader = None
    s._tex = None
    s._opened = None
    s._serial = -1
    s.opened_with = []
    return s


def test_the_first_layer_keeps_the_device():
    bus = _Bus()
    first, second = object(), object()
    assert bus.claim_device("Cam", first) is first
    assert bus.claim_device("Cam", second) is first


def test_the_second_layer_says_so_instead_of_opening_it():
    bus = _Bus()
    bus.claim_device("Cam", object())
    loser = _source()
    layer = FieldLayer(source="webcam", params={"_device": "Cam"})
    assert loser.evaluate(bus, layer, None) is None
    assert "already used" in (loser.error or "")
    assert loser._reader is None, "the loser opened the device anyway"


def test_two_different_cameras_both_get_one():
    bus = _Bus()
    a, b = object(), object()
    assert bus.claim_device("Cam A", a) is a
    assert bus.claim_device("Cam B", b) is b


def test_a_claim_does_not_outlive_the_frame():
    """Cleared per rebuild, so a layer that has gone stops holding a device."""
    bus = _Bus()
    gone = object()
    bus.claim_device("Cam", gone)
    bus.new_frame()
    fresh = object()
    assert bus.claim_device("Cam", fresh) is fresh


def test_an_unselected_camera_claims_nothing():
    bus = _Bus()
    source = _source()
    layer = FieldLayer(source="webcam")
    assert source.evaluate(bus, layer, None) is None
    assert source.error == "no camera selected"
    assert not bus._claims, "an empty device name was claimed"
