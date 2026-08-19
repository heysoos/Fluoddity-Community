"""Moving between neighbouring brain LAYOUTS, and bringing the genome along.

A layout move is two halves and neither is useful alone: propose a shape one
step from the parent's, then repack the parent's decoded genome into it. Both
are pure functions over a layout and an array - nothing here reaches the app,
the archive or the optimizer.

Generic implementations plus optional per-modality hooks, the pattern `mutate`,
`crossover` and `settings_of` already follow. The three unit-structured
modalities need no hook: their structure is one integer, and their unit carries
an amplitude the evaluation multiplies by.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.brains import (MAX_BRAIN_FLOATS, STRUCTURAL_KINDS, BrainLayout,
                             get, settings_of)


@dataclass(frozen=True)
class LayoutBounds:
    """How far a layout search may roam.

    `None` means "whatever the modality itself allows" - every modality already
    declares its own floor and ceiling in `settings_schema`, and a bound here
    can only narrow that, never widen it. `max_floats` is a hard ceiling
    because BrainLayout RAISES past MAX_BRAIN_FLOATS.

    `modalities` empty means "the running one only". A cross-modality jump is a
    restart, so opting into one is a decision rather than a default.
    """
    max_depth: int | None = None
    max_width: int | None = None
    max_floats: int = MAX_BRAIN_FLOATS
    modalities: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayoutMove:
    parent: BrainLayout
    child: BrainLayout
    operator: str

    @property
    def pair(self) -> tuple[str, str]:
        """What the ledger records and `banned` is keyed by."""
        return (self.parent.signature(), self.child.signature())


def _structural_int(m):
    """The one integer that IS this modality's structure, or None."""
    ints = [s for s in m.settings_schema() if s.kind in STRUCTURAL_KINDS]
    if len(ints) != 1 or ints[0].kind != "int":
        return None
    return ints[0]


def _generic_proposals(layout, bounds, m) -> list[tuple[str, dict]]:
    """+-1 on the structural integer. -> [(operator, settings), ...]"""
    s = _structural_int(m)
    if s is None:
        return []
    base = settings_of(layout)
    n = int(base.get(s.key, layout.shape[0] if layout.shape else 0))
    hi = int(s.hi) if bounds.max_width is None else min(int(s.hi),
                                                        int(bounds.max_width))
    lo = int(s.lo)
    out = []
    for delta, name in ((1, "grow"), (-1, "shrink")):
        k = n + delta
        if lo <= k <= hi:
            out.append((name, {**base, s.key: k}))
    return out


def _build(m, settings, bounds):
    """Rebuild a proposal, or None if it is not a legal layout.

    The float budget is checked BEFORE construction: BrainLayout.__post_init__
    raises past MAX_BRAIN_FLOATS, and a bounded search must not take the app
    down when it reaches its own ceiling.
    """
    try:
        child = m.layout_from_settings(dict(settings))
    except ValueError:
        return None
    if child.length > int(bounds.max_floats):
        return None
    return child


def candidate_moves(layout, bounds, modality=None) -> list[LayoutMove]:
    """Every legal one-step move from `layout`, rebuilt and verified.

    PROPOSED, REBUILT and COMPARED. `_shape_from_layers` clamps rather than
    raising - deliberately, because its input may be a config from a build with
    different limits - so a proposal that hits a limit comes back as a
    DIFFERENT layout, silently. Comparing the rebuilt settings against the
    proposal turns that into a rejected move rather than a move that did
    something else.
    """
    m = modality if modality is not None else get(layout.modality)
    fn = getattr(m, "layout_moves", None)
    proposals = (fn(layout, bounds) if fn is not None
                 else _generic_proposals(layout, bounds, m))
    out = []
    parent_sig = layout.signature()
    for op, settings in proposals:
        child = _build(m, settings, bounds)
        if child is None or child.signature() == parent_sig:
            continue
        # The round trip: what was asked for is what was built.
        if settings_of(child) != dict(settings):
            continue
        out.append(LayoutMove(parent=layout, child=child, operator=op))
    return out


def propose_layout_move(layout, bounds, rng, banned=frozenset()):
    """One legal move, drawn uniformly, or None.

    None means the caller runs an ordinary expedition: a cadence interval spent
    on one of those is worth more than one spent re-proposing a move already
    known to produce nothing. It deliberately does NOT fall back to a banned
    move.
    """
    moves = [mv for mv in candidate_moves(layout, bounds)
             if mv.pair not in banned]
    if not moves:
        return None
    return moves[int(rng.integers(len(moves)))]
