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

from services.brains import (MAX_BRAIN_FLOATS, REGISTRY, STRUCTURAL_KINDS,
                             BrainLayout, get, settings_of)


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


def parse_modalities(text) -> tuple[str, ...]:
    """A comma-separated bound into keys the registry knows. -> sorted keys.

    ONE string rather than one boolean per modality, so a fifth modality needs
    no new settings field and no second list of them. An unknown key is dropped
    rather than raising: the bound is written to disk and outlives the build
    that understood it.
    """
    if not text:
        return ()
    keys = {k.strip() for k in str(text).split(",")}
    return tuple(sorted(k for k in keys if k in REGISTRY))


def bounds_from(max_depth, max_width, max_floats, modalities,
                running) -> LayoutBounds:
    """Build the bounds a settings block describes.

    0 means "whatever the modality itself allows", which is what a fresh
    settings block holds - a literal bound of zero would forbid every layout.
    The RUNNING modality is dropped from the jump list: it is where the search
    already is, and a move to the layout it is standing on is not a move.
    """
    return LayoutBounds(
        max_depth=int(max_depth) or None,
        max_width=int(max_width) or None,
        max_floats=min(int(max_floats), MAX_BRAIN_FLOATS),
        modalities=tuple(k for k in parse_modalities(modalities)
                         if k != running.modality),
    )


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


def _modality_jumps(layout, bounds) -> list[LayoutMove]:
    """Every OTHER modality named in the bounds, at its own DEFAULT layout.

    There is no correspondence between a Fourier centre count and an MLP width,
    so a jump lands on the target's default rather than pretending to preserve
    a size. It carries no genome either - transfer_genome refuses one - which
    is what makes a jump a restart, and the reason it is opt-in.
    """
    out = []
    for key in bounds.modalities:
        if key == layout.modality or key not in REGISTRY:
            continue
        child = _build(REGISTRY[key], {}, bounds)
        if child is None:
            continue
        out.append(LayoutMove(parent=layout, child=child, operator="modality"))
    return out


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
    out.extend(_modality_jumps(layout, bounds))
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


def transfer_genome(params, parent, child, rng) -> np.ndarray:
    """Carry a DECODED brain from `parent`'s layout into `child`'s.

    This is what makes a layout move continuous rather than a restart. Whole
    units are copied; a new unit is DRAWN and then SILENCED - its amplitude
    zeroed - so the child evaluates exactly as its parent did at birth and any
    novelty it earns is earned. Drawing the incoming half matters as much:
    a unit zeroed on both sides is inert in a way sigma takes generations to
    undo.

    Shrinking is lossy by nature, and the units that remain remain unchanged.

    Only a cross-MODALITY jump has no transfer, and it raises rather than
    returning something that looks like a carried genome.
    """
    p = np.asarray(params, dtype=np.float32).reshape(-1)
    if parent.modality != child.modality:
        raise ValueError(
            f"cross-modality move {parent.signature()} -> {child.signature()} "
            "has no genome transfer; seed the expedition instead")
    m = get(child.modality)
    fn = getattr(m, "transfer_genome", None)
    if fn is not None:
        return np.asarray(fn(p, parent, child, rng),
                          dtype=np.float32).reshape(-1)

    stride = int(m.unit_floats(child) or 0)
    if not stride:
        raise ValueError(f"{child.modality} declares no unit and no transfer")
    out = np.asarray(m.random(rng, child), dtype=np.float32).reshape(-1)
    keep = min(p.size, out.size)
    keep -= keep % stride
    out[:keep] = p[:keep]
    lo, hi = getattr(m, "AMPLITUDE_SLICE", (0, 0))
    for start in range(keep, out.size, stride):
        out[start + lo:start + hi] = 0.0
    return out.astype(np.float32)
