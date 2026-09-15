"""Walk a set list on a timer: the next filled slot every `interval` seconds.

A load from anywhere else (a key press) restarts the clock from that slot, so
a visitor's choice stays up for the whole interval.
"""
from state.set_list import SetList, Slot


class SlotCycle:
    def __init__(self):
        self._since = None      # when the current slot went up, None = never
        self._current = None    # (filename, category) on screen, if known

    def note_loaded(self, slot: Slot, now: float) -> None:
        """A slot went up by some other road; time the next step from it."""
        self._current = (slot.filename, slot.category)
        self._since = now

    def due(self, set_list: SetList, now: float, interval: float) -> Slot | None:
        """The slot to load now, or None. Starts the clock on the first call."""
        filled = [s for s in set_list.slots if s.filename or s.rig]
        if not filled:
            return None
        if self._since is None:
            self._since = now
            return None
        if now - self._since < max(float(interval), 1.0):
            return None
        idx = next((i for i, s in enumerate(filled)
                    if (s.filename, s.category) == self._current), -1)
        nxt = filled[(idx + 1) % len(filled)]
        self.note_loaded(nxt, now)
        return nxt

    def reset(self) -> None:
        self._since = None
        self._current = None
