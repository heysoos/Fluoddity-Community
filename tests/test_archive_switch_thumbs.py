"""A brain switch stays in the same directory, so it must not look like a reload.

Clicking an archive entry that belongs to another brain switches layout, and a
layout change IS an archive switch - to a SIBLING directory under the same
archive name. Everything keyed by the directory therefore survives it: the
thumbnails (the key carries the signature) and the map's layout (the ids and
their positions are unchanged). Dropping either re-decodes the whole map and
leaves the browser blank for a frame, which reads as a reload and clamps the
window's scroll back to the top.
"""
import main as main_mod
from main import App
from services.thumb_cache import ThumbCache


class _State:
    class archive:
        running = False


class _Releaser:
    """The real release path, with nothing behind it but the caches."""

    _release_archive = App._release_archive

    def __init__(self):
        self.auto_service = None
        self.imgep_driver = None
        self.archive = None
        self.goal_list = None
        self.archive_store = None
        self.thumb_cache = ThumbCache(lambda k: f"t:{k}", capacity=8)
        self.atlas_cache = ThumbCache(lambda k: f"a:{k}", capacity=8)
        self.thumb_cache.get("x")
        self.atlas_cache.get("x")


def test_a_brain_switch_keeps_the_thumbnails():
    app = _Releaser()
    app._release_archive(_State(), keep_thumbs=True)
    assert len(app.thumb_cache) == 1
    assert len(app.atlas_cache) == 1


def test_an_archive_switch_still_drops_them():
    """Entry ids restart in every archive and the key derives from the id, so
    a kept texture would show the previous archive's picture."""
    app = _Releaser()
    app._release_archive(_State())
    assert app.thumb_cache is None
    assert app.atlas_cache is None


def test_the_brain_switch_path_asks_to_keep_them():
    """The one caller that stays in the same directory. Read off the source,
    because building a real App needs a GL context."""
    import inspect
    src = inspect.getsource(App._apply_brain_layout)
    assert "_release_archive(ui_state, keep_thumbs=True)" in src
    assert inspect.getsource(App._switch_archive).count("keep_thumbs") == 0


def test_the_rebuild_reuses_a_cache_that_survived():
    """The other half: keeping them through the release buys nothing if the
    rebuild hands back new ones. Read off the source for the same reason."""
    import inspect
    src = inspect.getsource(App._build_archive_set)
    for name in ("thumb_cache", "atlas_cache"):
        assert f"self.{name} is not None" in src, name
        assert f"self.{name}.set_loader(" in src, name
