"""LRU of GL textures for archive thumbnails.

An archive of 20,000 entries cannot hold 20,000 live textures. The gallery only
ever shows a screenful, so a small LRU with explicit release() is enough - and
explicit release matters because ModernGL's gc_mode='auto' would otherwise free
them at unpredictable moments during a frame.

The loader is injected so the eviction policy is testable without a GL context.
"""
from __future__ import annotations

from collections import OrderedDict


def gl_loader(ctx, store):
    """The real loader: read a thumbnail JPEG into an RGB texture."""

    def load(name: str):
        try:
            from PIL import Image

            path = store.thumb_path(name)
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                return ctx.texture(rgb.size, 3, rgb.tobytes())
        except Exception:
            # A missing or unreadable thumbnail costs a placeholder, never a
            # crash mid-frame.
            return None

    return load


class ThumbCache:
    def __init__(self, loader, capacity: int = 256):
        self._load = loader
        self.capacity = int(capacity)
        self._items: OrderedDict = OrderedDict()

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
