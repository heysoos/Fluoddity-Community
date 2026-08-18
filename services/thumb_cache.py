"""LRU of GL textures for archive thumbnails.

An archive of 20,000 entries cannot hold 20,000 live textures. The gallery only
ever shows a screenful, so a small LRU with explicit release() is enough - and
explicit release matters because ModernGL's gc_mode='auto' would otherwise free
them at unpredictable moments during a frame.

The loader is injected so the eviction policy is testable without a GL context.
"""
from __future__ import annotations

from collections import OrderedDict

# The most textures the cache will hold however small the gallery's tiles get.
# 160px RGB, so this is roughly 80 MB.
MAX_CAPACITY = 1024


def gl_loader(ctx, stores, max_px: int | None = None):
    """The real loader: read a thumbnail JPEG into an RGB texture.

    `stores` maps a brain layout signature to the store that owns those
    entries' thumbnails, and a key is `"<signature>/<filename>"`. A bare
    filename cannot identify a thumbnail: an id is unique inside one layout's
    directory and nowhere else, so every brain in an archive has a 000000.jpg
    and one cache keyed on the name alone would hand out the wrong picture.

    `max_px` decodes at reduced scale through JPEG's own DCT scaling. It is
    what lets the MAP hold a picture for every cell: a thumbnail drawn at 32px
    does not need the stored 160, and the smaller texture is what the cache is
    sized in. Draft only ever reduces by 1/2, 1/4 or 1/8, so the result is the
    smallest such size still at or above `max_px`.
    """

    def load(key: str):
        try:
            from PIL import Image

            sig, _, name = str(key).rpartition("/")
            store = stores.get(sig) if hasattr(stores, "get") else None
            if store is None:
                return None
            path = store.thumb_path(name)
            with Image.open(path) as img:
                if max_px:
                    img.draft("RGB", (int(max_px), int(max_px)))
                rgb = img.convert("RGB")
                return ctx.texture(rgb.size, 3, rgb.tobytes())
        except Exception:
            # A missing or unreadable thumbnail costs a placeholder, never a
            # crash mid-frame.
            return None

    return load


class ThumbCache:
    def __init__(self, loader, capacity: int = 256,
                 max_capacity: int = MAX_CAPACITY):
        self._load = loader
        self.capacity = int(capacity)
        # reserve() may raise the capacity but never take it below this.
        self._floor = int(capacity)
        # Per instance, because the ceiling is a MEMORY budget and the map's
        # textures are decoded small: the same number of them costs a fraction
        # of the gallery's 160px ones.
        self.max_capacity = max(int(capacity), int(max_capacity))
        self._items: OrderedDict = OrderedDict()

    def reserve(self, n: int) -> None:
        """Hold room for `n` thumbnails in one frame.

        A frame that touches more than `capacity` evicts every texture and
        re-decodes the whole visible set on the next one. Call this with the
        count about to be drawn, before drawing any of it.
        """
        self.capacity = max(self._floor, min(int(n), self.max_capacity))

    def __len__(self) -> int:
        return len(self._items)

    def get(self, name: str):
        if not name:
            return None
        if name in self._items:
            self._items.move_to_end(name)
            return self._items[name]
        try:
            tex = self._load(name)
        except Exception:
            # gl_loader swallows its own errors, but this is the last line: an
            # exception raised here lands mid-ImGui-frame, which corrupts the
            # whole frame rather than one thumbnail.
            return None
        if tex is None:
            return None            # not cached, so a transient failure retries
        self._items[name] = tex
        while len(self._items) > self.capacity:
            _, old = self._items.popitem(last=False)
            _release(old)
        return tex

    def peek(self, name: str):
        """The texture if it is already resident, without loading one.

        Deliberately does NOT count as a use: a peek is a look, so sweeping the
        map over an archive cannot evict what the gallery is showing.
        """
        return self._items.get(name) if name else None

    def invalidate(self, name: str) -> None:
        tex = self._items.pop(name, None)
        _release(tex)

    def release(self) -> None:
        for tex in self._items.values():
            _release(tex)
        self._items.clear()


def _release(tex) -> None:
    if tex is not None:
        try:
            tex.release()
        except Exception:
            pass
