"""The Auto tab shows what the ENCODER sees, not what the user sees.

The capture is a separate render with its own view, crop, greyscale and
overlay rules, so the only honest way to check any of them is to look at the
captured tile itself. Tile 0 is published to a texture after every capture.
"""
import numpy as np


class _Tex:
    def __init__(self, size):
        self.size = size
        self.writes = []
        self.released = 0

    def write(self, data):
        self.writes.append(bytes(data))

    def release(self):
        self.released += 1


class _Ctx:
    def __init__(self):
        self.made = []
        self.screen = type("S", (), {"use": lambda self: None})()
        self.viewport = None

    def texture(self, size, components, data=None):
        t = _Tex(size)
        self.made.append((size, components))
        return t


def _app(crops, monkeypatch):
    import main as m

    monkeypatch.setattr(m.glfw, "get_framebuffer_size", lambda w: (640, 480))
    app = type("A", (), {})()
    app.ctx = _Ctx()
    app.window = object()
    app.ui = type("U", (), {"capture_preview_tex": None})()
    app.sim_runner = type("R", (), {"last_assemble_kwargs": {}})()
    app.tournament_service = type("T", (), {"grid": 2})()
    app.auto_service = type("S", (), {"grayscale": False})()
    app.capture_view = type("CV", (), {
        "render": lambda self, ui, kw, side, grayscale=False: object(),
        "draw_grid": lambda self, *a: None})()
    app.tile_capture = type("TC", (), {
        "resize": lambda self, grid: None,
        "capture": lambda self, fn: crops})()
    app.capture_blit = object()
    app.capture_preview = None
    # The real publisher, bound to the fake: _capture_tiles is driven unbound.
    from main import App

    app._publish_capture_preview = (
        lambda crops: App._publish_capture_preview(app, crops))
    return app


def _crops(n=4, value=7):
    return np.full((n, 224, 224, 3), value, dtype=np.uint8)


def test_tile_zero_is_published_after_a_capture(monkeypatch):
    from main import App

    crops = _crops()
    app = _app(crops, monkeypatch)
    out = App._capture_tiles(app, object())
    assert out is crops
    tex = app.ui.capture_preview_tex
    assert tex is not None and tex.size == (224, 224)
    assert tex.writes == [crops[0].tobytes()]
    assert app.ctx.made == [((224, 224), 3)]


def test_the_preview_texture_is_reused_between_captures(monkeypatch):
    from main import App

    app = _app(_crops(value=1), monkeypatch)
    App._capture_tiles(app, object())
    first = app.ui.capture_preview_tex
    app.tile_capture.capture = lambda fn: _crops(value=2)
    App._capture_tiles(app, object())
    assert app.ui.capture_preview_tex is first
    assert len(first.writes) == 2
    assert len(app.ctx.made) == 1, "one upload target, written twice"


def test_a_failed_capture_publishes_nothing(monkeypatch):
    from main import App

    app = _app(None, monkeypatch)
    app.capture_view.render = lambda ui, kw, side, grayscale=False: None
    assert App._capture_tiles(app, object()) is None
    assert app.ui.capture_preview_tex is None
