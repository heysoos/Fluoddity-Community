"""One resident encoder, replaced rather than stacked.

Keeping every encoder loaded would cost about a gigabyte of weights for a
switch that happens once per archive.
"""
import services.vision_scorer as vs
from main import App


class FakeScorer:
    built = 0

    def __init__(self, key, *a, **kw):
        FakeScorer.built += 1
        self.model = type("M", (), {"key": key})()


class FakeApp:
    def __init__(self, scorer=None, auto_service=None, imgep_driver=None):
        self.vision_scorer = scorer
        self.auto_service = auto_service
        self.imgep_driver = imgep_driver
        self.ui = type("U", (), {"auto_unavailable": ""})()


def _ensure(app, key, monkeypatch, cls=FakeScorer):
    monkeypatch.setattr(vs, "VisionScorer", cls)
    return App._ensure_scorer(app, key)


def test_the_same_key_reuses_the_resident_scorer(monkeypatch):
    FakeScorer.built = 0
    app = FakeApp(FakeScorer("clip-b32"))
    FakeScorer.built = 0
    assert _ensure(app, "clip-b32", monkeypatch) is True
    assert FakeScorer.built == 0, "a matching key must not rebuild"


def test_a_different_key_replaces_it(monkeypatch):
    app = FakeApp(FakeScorer("clip-b32"))
    FakeScorer.built = 0
    assert _ensure(app, "siglip2-b16", monkeypatch) is True
    assert app.vision_scorer.model.key == "siglip2-b16"
    assert FakeScorer.built == 1


def test_the_first_scorer_is_built_on_demand(monkeypatch):
    app = FakeApp()
    FakeScorer.built = 0
    assert _ensure(app, "clip-l14", monkeypatch) is True
    assert app.vision_scorer.model.key == "clip-l14"


def test_a_failure_leaves_the_previous_scorer_in_place(monkeypatch):
    """A half-built swap would leave the app scoring with nothing."""
    def boom(key, *a, **kw):
        raise RuntimeError("no such file")

    kept = FakeScorer("clip-b32")
    app = FakeApp(kept)
    assert _ensure(app, "clip-l14", monkeypatch, cls=boom) is False
    assert app.vision_scorer is kept
    assert "no such file" in app.ui.auto_unavailable


def test_a_failure_with_nothing_resident_leaves_none(monkeypatch):
    def boom(key, *a, **kw):
        raise RuntimeError("missing weights")

    app = FakeApp()
    assert _ensure(app, "clip-l14", monkeypatch, cls=boom) is False
    assert app.vision_scorer is None


def test_a_swap_repoints_the_running_service(monkeypatch):
    """AutoTournamentService holds its own reference; leaving it stale would
    keep scoring with the previous encoder."""
    service = type("S", (), {"scorer": None})()
    app = FakeApp(FakeScorer("clip-b32"), auto_service=service)
    _ensure(app, "clip-b16", monkeypatch)
    assert service.scorer is app.vision_scorer


def test_a_swap_repoints_the_explore_driver_too(monkeypatch):
    """ImgepDriver takes its scorer at construction and is NOT rebuilt on an
    archive switch, so a stale reference would score the new archive with the
    old archive's encoder."""
    driver = type("D", (), {"scorer": None})()
    app = FakeApp(FakeScorer("clip-b32"), imgep_driver=driver)
    _ensure(app, "siglip2-b16", monkeypatch)
    assert driver.scorer is app.vision_scorer
    assert driver.scorer.model.key == "siglip2-b16"


def test_a_successful_reuse_clears_no_previous_error(monkeypatch):
    app = FakeApp(FakeScorer("clip-b32"))
    app.ui.auto_unavailable = "stale message"
    _ensure(app, "clip-b32", monkeypatch)
    assert app.vision_scorer.model.key == "clip-b32"
