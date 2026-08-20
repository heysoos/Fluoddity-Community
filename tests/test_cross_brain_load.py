"""Loading a config saved under another brain must arrive whole.

The config carries the physics, the modality AND the creature. Its own
`brain_layout` is what triggers the switch, so the load and the switch happen
in that order inside one frame - and the rule is pushed while the OLD brain is
still live, where apply_rule refuses it on width. Without the handoff the
switch then writes a generated brain over it, and the user gets the right
sliders around a creature nobody chose.
"""
from __future__ import annotations

import numpy as np
import pytest

from command_handler import CommandHandler
from services.brains import default_layout, get
from ui.brain_window import layout_for


def _handler():
    h = object.__new__(CommandHandler)
    h._pending_brain_rule = None
    return h


class _Config:
    def __init__(self, rule, signature, settings=None):
        self.rule = rule
        self.brain_layout = signature
        self.brain_settings = dict(settings or {})


class _Brain:
    modality = "fourier"
    settings: dict = {}


class _UI:
    def __init__(self):
        self.brain = _Brain()


def _rule_for(layout, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(get(layout.modality).random(rng, layout),
                      dtype=np.float32).reshape(-1)


@pytest.mark.parametrize("modality", ["gabor", "lenia", "mlp"])
def test_the_creature_reaches_the_layout_the_config_asked_for(modality):
    """The defect itself, at the seam that carries it."""
    layout = layout_for(modality, {})
    rule = _rule_for(layout)
    h = _handler()
    h._restore_brain_settings(_Config(rule, layout.signature()), _UI())

    got = h.take_pending_brain_rule(layout.signature())
    assert got is not None, "the switch was handed nothing to apply"
    assert np.allclose(got, rule)


def test_a_rule_is_never_applied_twice():
    """Consumed on read, so an unrelated later switch cannot pick it up."""
    layout = layout_for("gabor", {})
    h = _handler()
    h._restore_brain_settings(_Config(_rule_for(layout), layout.signature()),
                              _UI())

    assert h.take_pending_brain_rule(layout.signature()) is not None
    assert h.take_pending_brain_rule(layout.signature()) is None


def test_a_rule_for_another_brain_is_refused():
    """What makes this safe to apply unconditionally after a switch."""
    gabor = layout_for("gabor", {})
    h = _handler()
    h._restore_brain_settings(_Config(_rule_for(gabor), gabor.signature()),
                              _UI())

    assert h.take_pending_brain_rule(default_layout().signature()) is None


def test_the_match_is_the_signature_and_never_the_width():
    """Two layouts of one width are not one brain. Fourier at 10 centres is 80
    floats; so is a Lenia layout, over completely different meanings - and a
    width match would decode one as the other and call it a load."""
    fourier = default_layout()
    twin = next((layout_for(m, s)
                 for m, s in (("lenia", {"bumps": b}) for b in range(1, 40))
                 if layout_for("lenia", {"bumps": s["bumps"]}).length
                 == fourier.length), None)
    if twin is None:
        pytest.skip("no Lenia layout is exactly Fourier's width")

    h = _handler()
    h._restore_brain_settings(_Config(_rule_for(twin), twin.signature()), _UI())
    assert twin.length == fourier.length
    assert h.take_pending_brain_rule(fourier.signature()) is None


def test_a_config_naming_no_brain_is_read_as_the_fourier_it_is():
    """Pre-modality files are Fourier by history, and this used to stop at
    saying so: the absence was read as "stay put", so hovering one while
    another modality was live ran an 80-float genome under a wider layout,
    where apply_rule refuses it without a word. The GENOME is the evidence -
    all 138 unsigned presets in the library are exactly fourier-n10 - so it is
    handed over under that signature like any other."""
    h = _handler()
    h._restore_brain_settings(_Config(np.zeros(80, np.float32), ""), _UI())
    got = h.take_pending_brain_rule(default_layout().signature())
    assert got is not None, "an unsigned Fourier preset handed over nothing"
    assert len(got) == default_layout().length


def test_an_unsigned_config_at_an_unknown_width_is_left_alone():
    """The other half, and the reason the width is checked rather than
    assumed. An unsigned file is not always Fourier - a tile saved by an
    earlier build of this branch can be unsigned at another width - and there
    the file genuinely does not say which brain it wants."""
    h = _handler()
    wide = layout_for("mlp", {})
    h._restore_brain_settings(
        _Config(np.zeros(wide.length, np.float32), ""), _UI())
    assert h._pending_brain_rule is None
    assert CommandHandler._config_signature(
        _Config(np.zeros(wide.length, np.float32), "")) == ""


class _Layout:
    def __init__(self, sig):
        self._sig = sig

    def signature(self):
        return self._sig


class _Sim:
    def __init__(self, sig):
        self.brain_layout = _Layout(sig)


def test_a_load_that_succeeds_is_not_reported_as_ignored():
    """The width guard would print 'ignoring a 168-float rule' a few statements
    before the switch applies that very rule."""
    gabor = layout_for("gabor", {})
    h = _handler()
    h.apply_brain_layout = lambda *a: None
    h.sim = _Sim(default_layout().signature())
    h._restore_brain_settings(_Config(_rule_for(gabor), gabor.signature()),
                              _UI())

    assert h._switch_will_apply(np.zeros(4, np.float32))


def test_a_same_brain_load_still_goes_straight_to_the_gpu():
    """No switch is coming, so nothing else would ever apply it."""
    fourier = default_layout()
    h = _handler()
    h.apply_brain_layout = lambda *a: None
    h.sim = _Sim(fourier.signature())
    h._restore_brain_settings(_Config(_rule_for(fourier), fourier.signature()),
                              _UI())

    assert not h._switch_will_apply(np.zeros(4, np.float32))


def test_nothing_pending_means_nothing_deferred():
    h = _handler()
    h.apply_brain_layout = lambda *a: None
    h.sim = _Sim(default_layout().signature())
    assert not h._switch_will_apply(np.zeros(4, np.float32))
