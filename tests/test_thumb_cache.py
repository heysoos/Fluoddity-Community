import numpy as np

from services.thumb_cache import MAX_CAPACITY, ThumbCache


class _FakeTex:
    def __init__(self, name):
        self.name = name
        self.released = False

    def release(self):
        self.released = True


def cache(capacity=3):
    made = []

    def loader(name):
        t = _FakeTex(name)
        made.append(t)
        return t

    return ThumbCache(loader=loader, capacity=capacity), made


def test_get_loads_once_and_then_hits():
    c, made = cache()
    a = c.get("000001.jpg")
    b = c.get("000001.jpg")
    assert a is b
    assert len(made) == 1


def test_capacity_evicts_the_least_recently_used():
    c, _ = cache(capacity=2)
    first = c.get("a")
    c.get("b")
    c.get("a")            # 'a' is now the most recent
    c.get("c")            # evicts 'b'
    assert first.released is False
    assert len(c) == 2


def test_evicted_textures_are_released():
    c, made = cache(capacity=1)
    c.get("a")
    c.get("b")
    assert made[0].released is True


def test_a_failed_load_is_not_cached_and_is_retried():
    """A missing thumbnail must not become a permanent hole in the gallery -
    it may simply not have been written yet when the entry was admitted."""
    calls = []

    def loader(name):
        calls.append(name)
        return None

    c = ThumbCache(loader=loader, capacity=3)
    assert c.get("missing.jpg") is None
    assert c.get("missing.jpg") is None
    assert len(calls) == 2
    assert len(c) == 0


def test_invalidate_releases_and_forgets():
    c, made = cache()
    c.get("a")
    c.invalidate("a")
    assert made[0].released is True
    assert len(c) == 0


def test_release_frees_everything():
    c, made = cache()
    c.get("a")
    c.get("b")
    c.release()
    assert all(t.released for t in made)
    assert len(c) == 0


def test_an_empty_name_is_a_miss():
    c, made = cache()
    assert c.get("") is None
    assert made == []


def test_a_loader_that_raises_does_not_escape_into_the_frame():
    """gl_loader swallows its own errors, but the cache is the last line: an
    exception here lands mid-ImGui-frame and corrupts the whole UI, not just
    one thumbnail."""
    def loader(name):
        raise OSError("disk gone")

    c = ThumbCache(loader=loader, capacity=3)
    assert c.get("boom.jpg") is None
    assert len(c) == 0


def test_invalidating_something_absent_is_harmless():
    c, _ = cache()
    c.invalidate("never-loaded.jpg")
    assert len(c) == 0


# ---- reserve --------------------------------------------------------------
#
# The gallery's tile size is a slider, so how many thumbnails one frame draws
# is not knowable here. A frame that touches more than `capacity` evicts every
# texture and re-decodes the whole visible set on the next frame, forever.


def test_reserve_raises_capacity_to_fit_the_frame():
    c, _ = cache(capacity=3)
    c.reserve(40)
    assert c.capacity == 40


def test_reserve_never_drops_below_the_capacity_it_was_built_with():
    """The floor is the construction capacity: a small frame must not shrink
    the cache, or scrolling back up re-decodes what was just on screen."""
    c, _ = cache(capacity=256)
    c.reserve(4)
    assert c.capacity == 256


def test_reserve_clamps_at_the_ceiling():
    c, _ = cache(capacity=3)
    c.reserve(10_000_000)
    assert c.capacity == MAX_CAPACITY


def test_nothing_is_evicted_after_reserving_room_for_the_frame():
    c, made = cache(capacity=2)
    c.reserve(6)
    for name in "abcdef":
        c.get(name)
    assert len(c) == 6
    assert not any(t.released for t in made)


# ---- peek, and the per-frame decode budget ------------------------------
#
# Zooming the map changes which entry wins each cell, so the representative set
# churns and every changed cell is a MISS. A JPEG decode is ~1 ms, so a frame
# that fetches every changed cell stalls. peek() is what lets a caller draw
# what it already has and fetch only a few new ones per frame.

def test_peek_returns_a_resident_texture():
    c, made = cache()
    got = c.get("a")
    assert c.peek("a") is got
    assert len(made) == 1


def test_peek_never_loads():
    c, made = cache()
    assert c.peek("a") is None
    assert made == [], "peek decoded a thumbnail"


def test_peek_does_not_disturb_the_eviction_order():
    """A peek is a look, not a use: counting it as a use would let whatever the
    map happens to sweep over evict what the gallery is showing."""
    c, _ = cache(capacity=2)
    first = c.get("a")
    c.get("b")
    c.peek("a")           # if this counted as a use, 'b' would go next
    c.get("c")
    assert c.peek("a") is None
    assert first.released is True
    assert c.peek("b") is not None


def test_an_empty_name_peeks_to_nothing():
    c, _ = cache()
    assert c.peek("") is None


def test_swapping_the_loader_keeps_what_is_already_decoded():
    """A brain switch rebuilds the Archive over the SAME directory, so the
    keys and the files behind them are unchanged - only the object the loader
    resolves through is new. Re-decoding the whole map for that is what made a
    cross-brain click look like a reload."""
    first = ThumbCache(lambda k: f"old:{k}", capacity=8)
    assert first.get("a") == "old:a"
    first.set_loader(lambda k: f"new:{k}")
    assert first.get("a") == "old:a", "already decoded, so untouched"
    assert first.get("b") == "new:b", "anything new goes through the new one"
