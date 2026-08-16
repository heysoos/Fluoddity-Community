"""Every holder of the resident encoder is named where it is replaced.

Swapping encoder builds one scorer and hands it round. `AutoTournamentService`
does not hold one of its own - its `scorer` is a PROPERTY forwarding to
`self.driver` - so assigning through the service reaches whichever driver is
installed and no other. Auto's driver and Explore's are swapped in and out by
mode, so one of them is always the one that misses.
"""
import pytest

import main
from main import App


class _Scorer:
    def __init__(self, key):
        self.model = type("M", (), {"key": key})()


class _Driver:
    def __init__(self, scorer=None):
        self.scorer = scorer


class _Service:
    """The real forwarding property: a service owns no scorer of its own."""

    def __init__(self, driver):
        self.driver = driver

    @property
    def scorer(self):
        return getattr(self.driver, "scorer", None)

    @scorer.setter
    def scorer(self, value):
        self.driver.scorer = value


class _App:
    _ensure_scorer = App._ensure_scorer

    def __init__(self, driving):
        old = _Scorer("clip-b32")
        self.vision_scorer = old
        self.prompt_driver = _Driver(old)
        self.imgep_driver = _Driver(old)
        self.auto_service = _Service(
            self.prompt_driver if driving == "auto" else self.imgep_driver)
        self.ui = type("UI", (), {"auto_unavailable": ""})()


@pytest.fixture
def built(monkeypatch):
    """Swap the ONNX session out; this test is about who gets the object."""
    made = _Scorer("siglip2-b16")
    import services.vision_scorer as vs

    monkeypatch.setattr(vs, "VisionScorer", lambda key: made)
    return made


@pytest.mark.parametrize("driving", ("auto", "explore"))
def test_both_drivers_get_the_new_encoder_whichever_is_installed(driving, built):
    app = _App(driving)
    assert app._ensure_scorer("siglip2-b16") is True
    assert app.vision_scorer is built
    assert app.prompt_driver.scorer is built, "Auto kept the replaced encoder"
    assert app.imgep_driver.scorer is built, "Explore kept the replaced encoder"


def test_a_failed_build_leaves_every_holder_on_the_old_one(monkeypatch):
    """A half-swapped app scores in two spaces at once, which is worse than
    scoring in the outgoing one."""
    import services.vision_scorer as vs

    def boom(_key):
        raise RuntimeError("no weights")

    monkeypatch.setattr(vs, "VisionScorer", boom)
    app = _App("auto")
    old = app.vision_scorer

    assert app._ensure_scorer("clip-l14") is False
    assert app.vision_scorer is old
    assert app.prompt_driver.scorer is old
    assert app.imgep_driver.scorer is old


def test_an_unchanged_key_rebuilds_nothing(monkeypatch):
    import services.vision_scorer as vs

    monkeypatch.setattr(
        vs, "VisionScorer",
        lambda key: pytest.fail("rebuilt an encoder that was already resident"))
    app = _App("auto")
    assert app._ensure_scorer("clip-b32") is True
