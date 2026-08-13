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

    # _follow_auto_encoder delegates the swap itself, so the fake gets the
    # real one bound to it rather than a second implementation.
    def _ensure_scorer(self, key):
        return App._ensure_scorer(self, key)


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


# -- Auto's picker, per frame ------------------------------------------------
# The service is built once, on the tab's enable edge, and returns early ever
# after - so the encoder was read exactly once and the combo changed nothing.

class FakeService:
    def __init__(self, driver=None):
        self.scorer = None
        self.driver = driver or object()
        self.prompts = []
        self.aborts = 0

    def set_prompt(self, text):
        self.prompts.append(text)

    def abort_generation(self):
        self.aborts += 1


def _ui_state(key="clip-b32", prompt="", enabled=True):
    auto = type("A", (), {"model_key": key, "prompt": prompt,
                          "enabled": enabled})()
    return type("S", (), {"auto_tournament": auto})()


def _follow(app, state, monkeypatch, present=True, cls=FakeScorer):
    import tools.fetch_models as fm

    monkeypatch.setattr(vs, "VisionScorer", cls)
    monkeypatch.setattr(fm, "is_present", lambda key: present)
    return App._follow_auto_encoder(app, state)


def test_moving_the_combo_swaps_the_resident_encoder(monkeypatch):
    svc = FakeService()
    app = FakeApp(FakeScorer("clip-b32"), auto_service=svc)
    _follow(app, _ui_state("siglip2-b16"), monkeypatch)
    assert app.vision_scorer.model.key == "siglip2-b16"
    assert svc.scorer is app.vision_scorer


def test_leaving_the_combo_alone_rebuilds_nothing(monkeypatch):
    app = FakeApp(FakeScorer("clip-b32"), auto_service=FakeService())
    FakeScorer.built = 0
    _follow(app, _ui_state("clip-b32"), monkeypatch)
    assert FakeScorer.built == 0


def test_the_goal_is_re_embedded_in_the_new_space(monkeypatch):
    """The prompt embedding belongs to the outgoing encoder, and at 512 against
    768 it is not even a shape error until the first tile arrives."""
    svc = FakeService()
    app = FakeApp(FakeScorer("clip-b32"), auto_service=svc)
    _follow(app, _ui_state("clip-l14", prompt="coral reef"), monkeypatch)
    assert svc.prompts == ["coral reef"]
    assert svc.aborts == 1, "a half-scored generation ranks nothing"


def test_a_missing_encoder_says_so_and_keeps_the_resident_one(monkeypatch):
    kept = FakeScorer("clip-b32")
    app = FakeApp(kept, auto_service=FakeService())
    _follow(app, _ui_state("clip-l14"), monkeypatch, present=False)
    assert app.vision_scorer is kept
    assert app.ui.auto_unavailable == "model_missing"


def test_it_stands_down_while_explore_owns_the_driver(monkeypatch):
    """Explore's encoder belongs to its archive; Auto's combo must not reach
    across and swap it out from under a running search."""
    driver = type("D", (), {"scorer": None})()
    svc = FakeService(driver=driver)
    app = FakeApp(FakeScorer("clip-b32"), auto_service=svc,
                  imgep_driver=driver)
    _follow(app, _ui_state("siglip2-b16"), monkeypatch)
    assert app.vision_scorer.model.key == "clip-b32"


def test_a_closed_auto_tab_swaps_nothing(monkeypatch):
    app = FakeApp(FakeScorer("clip-b32"), auto_service=FakeService())
    _follow(app, _ui_state("clip-l14", enabled=False), monkeypatch)
    assert app.vision_scorer.model.key == "clip-b32"


def test_it_does_nothing_before_the_service_exists(monkeypatch):
    """The enable edge builds the service and reads the combo itself; there is
    nothing to repoint until then."""
    app = FakeApp()
    _follow(app, _ui_state("clip-l14"), monkeypatch)
    assert app.vision_scorer is None
