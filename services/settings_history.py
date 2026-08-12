"""What changed in an archive's explore settings, and when.

settings.json is rewritten wholesale, so the value that admitted a given entry
is unrecoverable the moment a slider moves. This is the record that makes it
recoverable.

It never restores anything. The log is a record, not a mechanism - putting a
setting back is the user moving the slider.
"""
from __future__ import annotations


def diff(previous: dict, current: dict) -> dict:
    """-> {field: [old, new]} for every field whose value moved.

    A field absent from `current` is NOT a change: PERSISTED_FIELDS shrinking is
    a code change, and recording it would put a phantom row in every archive on
    the first run after it.
    """
    out = {}
    for key, new in current.items():
        old = previous.get(key)
        if key not in previous or old != new:
            out[key] = [old, new]
    return out


def replay(rows: list[dict]) -> dict:
    """Fold a history back into the settings in force after its last row.

    Version 0 carries the whole block; every later row is a diff, so this is
    what the search compares against when it did not write the earlier rows
    itself.
    """
    base: dict = {}
    for row in rows:
        if "full" in row:
            base = dict(row["full"])
            continue
        for key, pair in row.get("changed", {}).items():
            try:
                base[key] = pair[1]
            except (IndexError, TypeError):
                continue        # a hand-edited row is not worth raising over
    return base
