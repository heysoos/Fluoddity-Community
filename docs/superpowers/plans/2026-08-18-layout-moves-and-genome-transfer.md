# Layout Moves and Genome Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the search two pure functions — propose a neighbouring brain
shape under bounds, and carry a genome into it so the child is bit-identical to
its parent at birth.

**Architecture:** One new module, `services/brains/layout_moves.py`, holding a
generic implementation plus dispatch to optional per-modality hooks — the
pattern `mutate`, `crossover`, `settings_of` and `shader_defines` already
follow. The three unit-structured modalities need no hook: their structure is
one integer and their unit carries an explicit amplitude slice. MLP declares
both hooks, because a layer stack fits neither "nudge one int" nor "copy whole
units". Nothing here touches the app, the driver or the archive — that is
sections 4–7 of the spec and is out of scope.

**Tech Stack:** Python 3.12, NumPy, ModernGL (for the phenotype proof), pytest.

## Global Constraints

- **Tests run on the venv interpreter, never bare `python`:**
  `.venv/Scripts/python.exe -m pytest -q`. Bare `python` is 3.10 with no pytest.
- **Comments state the rule, never the evidence.** No percentages, timings,
  dates, or before/after comparisons in comments, docstrings or tooltips.
  Measured facts live in `CLAUDE.md` or in
  `docs/superpowers/specs/2026-08-17-brain-layout-search-design.md`.
- **This tree is shared with concurrent sessions.** `git add` by explicit
  filename only. Never `git add -A` and never `git add <directory>`.
- **Never `git push`.** Commit locally; the user pushes after verifying by hand.
- **Do not restructure `sim.py`** — it is user-owned.
- **`MAX_BRAIN_FLOATS` is enforced by `BrainLayout.__post_init__`, which
  RAISES.** Every proposal must be checked before a `BrainLayout` is
  constructed, or a bounded search takes the app down.
- **A GPU test must skip cleanly where there is no context**, exactly as
  `tests/test_brain_modalities_gpu.py` does.

## Source of truth

The spec is
`docs/superpowers/specs/2026-08-17-brain-layout-search-design.md`, sections
**2** and **3** only. Sections 4–7 (the expedition wiring, keep-or-revert, the
ledger, the settings panel) are **out of scope** and must not be started here.

**This plan deliberately deviates from the spec in one place.** Spec §3 says the
generic transfer should "copy `min(n_old, n_new)` whole units and draw the
remainder". Drawing the remainder makes a grown child a different creature at
birth, which is exactly the artefact §3's MLP half exists to avoid. Every one of
the three unit modalities stores an explicit **amplitude** slice that its
evaluation multiplies by — `out += a * basis` in all three NumPy references in
`tests/test_brain_modalities_gpu.py` — so a new unit born with **zero
amplitude** contributes nothing and the child is bit-identical to its parent,
the same property the zeroed outgoing column gives MLP. Task 6 records the
deviation in the spec.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `services/brains/layout_moves.py` | *new* — bounds, proposal, transfer, dispatch | 1, 3 |
| `tests/test_layout_moves.py` | *new* — the generic proposal | 1 |
| `services/brains/mlp.py` | gains `layout_moves()` | 2 |
| `tests/test_layout_moves_mlp.py` | *new* — the stack operators | 2 |
| `tests/test_genome_transfer.py` | *new* — the generic transfer | 3 |
| `services/brains/mlp.py` | gains `transfer_genome()` | 4 |
| `tests/test_genome_transfer_mlp.py` | *new* — the repack | 4 |
| `tests/test_layout_move_phenotype_gpu.py` | *new* — bit-identical at birth | 5 |
| `CLAUDE.md`, the spec | the rules and the deviation | 6 |

---

### Task 1: Bounds and the generic proposal

**Files:**
- Create: `services/brains/layout_moves.py`
- Test: `tests/test_layout_moves.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `LayoutBounds(max_depth: int | None = None, max_width: int | None = None,
    max_floats: int = MAX_BRAIN_FLOATS, modalities: tuple[str, ...] = ())` —
    frozen dataclass. `None` means "whatever the modality itself allows".
  - `LayoutMove(parent: BrainLayout, child: BrainLayout, operator: str)` —
    frozen dataclass.
  - `candidate_moves(layout, bounds) -> list[LayoutMove]` — every legal move
    from `layout`, already rebuilt and verified.
  - `propose_layout_move(layout, bounds, rng, banned=frozenset()) -> LayoutMove | None`
    — one of them, drawn uniformly, or `None`.
  - `banned` is a set of `(parent_signature, child_signature)` string pairs.
  Tasks 2, 3 and 4 all build on these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout_moves.py`:

```python
"""Proposing a neighbouring brain shape, under bounds.

A move is PROPOSED, REBUILT and COMPARED - the round-trip discipline
layout_from_signature already uses - because _shape_from_layers CLAMPS rather
than raises, so a proposal that hits a limit comes back as a DIFFERENT layout
instead of an error.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.brains import REGISTRY, BrainLayout, default_layout
from services.brains.layout_moves import (LayoutBounds, LayoutMove,
                                          candidate_moves,
                                          propose_layout_move)

FOURIER = default_layout()                      # fourier-n10
GABOR = REGISTRY["gabor"].layout_from_settings({})
LENIA = REGISTRY["lenia"].layout_from_settings({})


def _rng():
    return np.random.default_rng(0)


def _ops(layout, bounds=None):
    return sorted(m.operator for m in
                  candidate_moves(layout, bounds or LayoutBounds()))


def _by_op(layout, op, bounds=None):
    for m in candidate_moves(layout, bounds or LayoutBounds()):
        if m.operator == op:
            return m
    raise AssertionError(f"no {op!r} move from {layout.signature()}")


def test_one_integer_of_structure_moves_by_one():
    """The three unit modalities need no hook at all: their structure IS one
    integer, so +-1 on it is the whole operator set."""
    assert _ops(FOURIER) == ["grow", "shrink"]
    grow = _by_op(FOURIER, "grow")
    assert grow.child.signature() == "fourier-n11"
    assert grow.child.length == 8 * 11
    shrink = _by_op(FOURIER, "shrink")
    assert shrink.child.signature() == "fourier-n9"


def test_every_unit_modality_moves():
    for lay, grown in ((GABOR, "gabor-n13"), (LENIA, "lenia-n13")):
        assert _by_op(lay, "grow").child.signature() == grown


def test_a_move_keeps_the_decode_scales():
    """Scales are not structure. A move that reset them would silently change
    what every transferred float MEANS."""
    tuned = BrainLayout("fourier", (10,), 80,
                        scales=(("freq_scale", 3.5), ("low_freq_bias", 0.25)))
    child = _by_op(tuned, "grow").child
    assert dict(child.scales) == {"freq_scale": 3.5, "low_freq_bias": 0.25}


def test_the_modality_own_floor_and_ceiling_are_respected():
    """Fourier declares centers 4..48, so neither end proposes past it."""
    at_floor = REGISTRY["fourier"].layout_from_settings({"centers": 4})
    assert _ops(at_floor) == ["grow"]
    at_ceiling = REGISTRY["fourier"].layout_from_settings({"centers": 48})
    assert _ops(at_ceiling) == ["shrink"]


def test_the_users_width_bound_narrows_it_further():
    bounds = LayoutBounds(max_width=11)
    assert _ops(FOURIER, bounds) == ["grow", "shrink"]
    at_bound = REGISTRY["fourier"].layout_from_settings({"centers": 11})
    assert _ops(at_bound, bounds) == ["shrink"]


def test_the_float_budget_is_checked_before_a_layout_is_built():
    """BrainLayout.__post_init__ RAISES past MAX_BRAIN_FLOATS, so a proposal
    that would cross it must never reach the constructor."""
    bounds = LayoutBounds(max_floats=8 * 10)      # exactly the parent's size
    assert _ops(FOURIER, bounds) == ["shrink"]


def test_a_banned_pair_is_skipped():
    banned = {("fourier-n10", "fourier-n11")}
    got = {propose_layout_move(FOURIER, LayoutBounds(), _rng(), banned)
           .child.signature() for _ in range(20)}
    assert got == {"fourier-n9"}


def test_when_every_move_is_banned_it_returns_nothing():
    """The caller runs an ordinary expedition instead. It must not fall back
    to a worse move and it must not raise."""
    banned = {("fourier-n10", "fourier-n11"), ("fourier-n10", "fourier-n9")}
    assert propose_layout_move(FOURIER, LayoutBounds(), _rng(), banned) is None


def test_a_move_never_returns_the_parent():
    for m in candidate_moves(FOURIER, LayoutBounds()):
        assert m.child.signature() != m.parent.signature()
        assert m.parent is FOURIER


def test_proposing_is_seeded_and_reproducible():
    a = propose_layout_move(FOURIER, LayoutBounds(), np.random.default_rng(4))
    b = propose_layout_move(FOURIER, LayoutBounds(), np.random.default_rng(4))
    assert a == b


def test_a_modality_with_no_structural_int_proposes_nothing():
    """Not a failure - it is how a modality opts out until it declares a hook."""
    class _Flat:
        name = "flat"

        def settings_schema(self):
            from services.brains import Setting
            return [Setting("gain", "Gain", "float", 0.0, 1.0, 0.5)]

    lay = BrainLayout("flat", (), 8)
    assert candidate_moves(lay, LayoutBounds(), modality=_Flat()) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_layout_moves.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.brains.layout_moves'`.

- [ ] **Step 3: Create the module**

Create `services/brains/layout_moves.py`:

```python
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

from dataclasses import dataclass, field

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_layout_moves.py -q`
Expected: PASS, 11 passed.

- [ ] **Step 5: Check nothing else moved**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout.py tests/test_brain_modalities.py tests/test_brain_scales.py -q`
Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add services/brains/layout_moves.py tests/test_layout_moves.py
git commit -m "feat: propose a neighbouring brain layout under bounds"
```

---

### Task 2: MLP's own move operators

**Files:**
- Modify: `services/brains/mlp.py` — add a `layout_moves` method to
  `MLPModality`, beside `settings_of`
- Test: `tests/test_layout_moves_mlp.py` (create)

**Interfaces:**
- Consumes: `LayoutBounds` and `candidate_moves` from Task 1.
- Produces: `MLPModality.layout_moves(layout, bounds) -> list[tuple[str, dict]]`,
  where each settings dict is `{"layers": [[w, a], ...], "w_scale": ..., "b_scale": ...}`.
  Operators are `"grow"`, `"add_layer"`, `"shrink"`, `"drop_layer"` and
  `"activation"`. Task 4 repacks for `"grow"`, `"add_layer"`, `"shrink"` and
  `"drop_layer"`; `"activation"` is a pure relabel with no repacking.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout_moves_mlp.py`:

```python
"""A layer stack does not fit "nudge one int", so mlp declares its own moves.

N is 1 by default: the point of the move is that the child is its parent plus
a little room, which is what makes the novelty it earns attributable. A large
jump is a restart wearing a growth move's name.
"""
from __future__ import annotations

import numpy as np

from services.brains import MAX_BRAIN_FLOATS, REGISTRY
from services.brains.layout_moves import (LayoutBounds, candidate_moves,
                                          propose_layout_move)
from services.brains.mlp import MAX_DEPTH, MAX_WIDTH, MIN_WIDTH

MLP = REGISTRY["mlp"]


def _lay(layers):
    return MLP.layout_from_settings({"layers": layers})


def _ops(layout, bounds=None):
    return sorted({m.operator for m in
                   candidate_moves(layout, bounds or LayoutBounds())})


def _children(layout, op, bounds=None):
    return sorted(m.child.signature() for m in
                  candidate_moves(layout, bounds or LayoutBounds())
                  if m.operator == op)


def test_a_single_layer_stack_offers_every_operator():
    lay = _lay([[16, 0]])
    assert _ops(lay) == ["activation", "add_layer", "grow", "shrink"]


def test_growing_adds_exactly_one_unit_to_one_layer():
    assert _children(_lay([[16, 0]]), "grow") == ["mlp-n17-a0"]
    # two layers, so two places to grow
    assert _children(_lay([[16, 0], [8, 0]]), "grow") == [
        "mlp-n16.9-a0.0", "mlp-n17.8-a0.0"]


def test_adding_a_layer_appends_one_at_the_default_width():
    """Appended, never inserted: an inserted layer renumbers every layer after
    it, so nothing downstream could be carried across."""
    assert _children(_lay([[16, 0]]), "add_layer") == ["mlp-n16.16-a0.0"]


def test_shrinking_and_dropping_are_separate_operators():
    lay = _lay([[16, 0], [8, 0]])
    assert _children(lay, "shrink") == ["mlp-n15.8-a0.0", "mlp-n16.7-a0.0"]
    assert _children(lay, "drop_layer") == ["mlp-n16-a0", "mlp-n8-a0"]


def test_the_last_layer_may_not_be_dropped():
    """A brain with no hidden layer is not a smaller brain, it is a different
    model."""
    assert _children(_lay([[16, 0]]), "drop_layer") == []


def test_changing_an_activation_keeps_every_width():
    got = _children(_lay([[16, 0]]), "activation")
    assert got == ["mlp-n16-a1", "mlp-n16-a2"]


def test_a_layer_at_the_floor_may_be_dropped_but_not_shrunk():
    """MIN_WIDTH is 1, and a width-0 layer is not a layer."""
    lay = _lay([[16, 0], [MIN_WIDTH, 0]])
    assert "mlp-n16-a0" in _children(lay, "drop_layer")
    # only the FIRST layer can still shrink
    assert _children(lay, "shrink") == ["mlp-n15.1-a0.0"]


def test_the_depth_bound_stops_add_layer():
    lay = _lay([[8, 0], [8, 0]])
    assert _children(lay, "add_layer", LayoutBounds(max_depth=2)) == []
    assert _children(lay, "add_layer", LayoutBounds(max_depth=3)) != []


def test_the_hard_depth_ceiling_holds_with_no_bound_set():
    lay = _lay([[MIN_WIDTH, 0]] * MAX_DEPTH)
    assert _children(lay, "add_layer") == []


def test_the_width_bound_stops_grow():
    lay = _lay([[8, 0]])
    assert _children(lay, "grow", LayoutBounds(max_width=8)) == []
    assert _children(lay, "grow", LayoutBounds(max_width=9)) == ["mlp-n9-a0"]


def test_a_move_that_the_clamp_would_change_is_rejected_not_silently_altered():
    """_shape_from_layers clamps to the float budget, so growing the second
    layer here asks for 1039 floats and comes back as mlp-n47.15 - a layout
    that also SHRANK the first layer. The round trip in candidate_moves is
    what turns that into no move at all rather than a move that did something
    else."""
    lay = _lay([[48, 0], [14, 0]])
    assert lay.signature() == "mlp-n48.14-a0.0", "the parent must not clamp"
    assert "mlp-n47.15-a0.0" not in _children(lay, "grow")
    assert "mlp-n48.15-a0.0" not in _children(lay, "grow")
    # growing the FIRST layer is over MAX_WIDTH, growing the second overflows,
    # so this stack cannot grow at all
    assert _children(lay, "grow") == []
    for m in candidate_moves(lay, LayoutBounds()):
        assert m.child.length <= MAX_BRAIN_FLOATS


def test_the_scales_survive_every_operator():
    lay = MLP.layout_from_settings({"layers": [[16, 0]], "w_scale": 4.0,
                                    "b_scale": 0.5})
    for m in candidate_moves(lay, LayoutBounds()):
        assert dict(m.child.scales) == {"w_scale": 4.0, "b_scale": 0.5}


def test_proposing_from_a_deep_stack_terminates():
    lay = _lay([[16, 0], [12, 1], [8, 2]])
    got = propose_layout_move(lay, LayoutBounds(), np.random.default_rng(1))
    assert got is not None and got.child.signature() != lay.signature()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_layout_moves_mlp.py -q`
Expected: FAIL — the generic `_structural_int` returns `None` for MLP (its one
structural setting is kind `"layers"`, not `"int"`), so `candidate_moves`
returns `[]` and every test that expects an operator fails.

- [ ] **Step 3: Add the hook to `MLPModality`**

In `services/brains/mlp.py`, add this method immediately after `settings_of`:

```python
    def layout_moves(self, layout: BrainLayout, bounds):
        """The stack's own operators. -> [(operator, settings), ...]

        Proposals only: layout_moves.candidate_moves rebuilds each one and
        rejects any that _shape_from_layers clamped, which is the check that
        makes a bounded proposal safe.

        Growth is by ONE unit and a new layer is APPENDED. An inserted layer
        would renumber every layer after it, leaving nothing for the transfer
        to carry across; a large jump is a restart wearing a growth move's
        name.
        """
        base = self.settings_of(layout)
        base.update({k: float(v) for k, v in layout.scales})
        layers = [list(p) for p in base["layers"]]
        depth = len(layers)
        max_depth = (MAX_DEPTH if bounds.max_depth is None
                     else min(MAX_DEPTH, int(bounds.max_depth)))
        max_width = (MAX_WIDTH if bounds.max_width is None
                     else min(MAX_WIDTH, int(bounds.max_width)))
        out = []

        def with_layers(new):
            return {**base, "layers": new}

        for i in range(depth):
            w, a = layers[i]
            if w + 1 <= max_width:
                grown = [list(p) for p in layers]
                grown[i][0] = w + 1
                out.append(("grow", with_layers(grown)))
            if w - 1 >= MIN_WIDTH:
                shrunk = [list(p) for p in layers]
                shrunk[i][0] = w - 1
                out.append(("shrink", with_layers(shrunk)))
            # The LAST hidden layer may not go: a brain with none is not a
            # smaller brain, it is a different model.
            if depth > 1:
                dropped = [list(p) for j, p in enumerate(layers) if j != i]
                out.append(("drop_layer", with_layers(dropped)))
            for act in range(len(ACTIVATIONS)):
                if act != a:
                    swapped = [list(p) for p in layers]
                    swapped[i][1] = act
                    out.append(("activation", with_layers(swapped)))

        if depth < max_depth:
            width = min(int(DEFAULT_NEW_LAYER_WIDTH), max_width)
            out.append(("add_layer",
                        with_layers([list(p) for p in layers] + [[width, 0]])))
        return out
```

- [ ] **Step 4: Name the new-layer width**

`DEFAULT_NEW_LAYER_WIDTH` must exist. In `services/brains/mlp.py`, beside
`MAX_DEPTH`, add:

```python
# A new layer's width. The `layers` Setting's own default, so the search adds
# what the `+ Add layer` button adds.
DEFAULT_NEW_LAYER_WIDTH = 16
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_layout_moves_mlp.py -q`
Expected: PASS, 12 passed.

- [ ] **Step 6: Run the MLP and layout suites**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mlp_layer_stack.py tests/test_brain_layer_ops.py tests/test_layout_moves.py -q`
Expected: PASS, no failures.

- [ ] **Step 7: Commit**

```bash
git add services/brains/mlp.py tests/test_layout_moves_mlp.py
git commit -m "feat: mlp grows, adds, shrinks and drops a layer"
```

---

### Task 3: The generic genome transfer

**Files:**
- Modify: `services/brains/layout_moves.py` — add `transfer_genome`
- Test: `tests/test_genome_transfer.py` (create)

**Interfaces:**
- Consumes: `LayoutMove` from Task 1.
- Produces: `transfer_genome(params, parent, child, rng) -> np.ndarray` —
  a DECODED genome of `child.length` float32. Raises `ValueError` on a
  cross-modality pair, which has no transfer. Task 4 overrides for MLP.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_genome_transfer.py`:

```python
"""Carrying a decoded brain into a neighbouring layout.

A grown child is BIT-IDENTICAL to its parent at birth. Every unit modality
stores an amplitude its evaluation multiplies by - `out += a * basis` - so a
new unit born with zero amplitude contributes nothing, and any novelty the
child earns is earned rather than an artefact of a random restart.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.brains import REGISTRY, default_layout
from services.brains.layout_moves import transfer_genome

FOURIER = default_layout()
GABOR = REGISTRY["gabor"].layout_from_settings({})
LENIA = REGISTRY["lenia"].layout_from_settings({})

# Where each modality's unit keeps the amplitude the evaluation multiplies by.
AMPLITUDE = {"fourier": (4, 8), "gabor": (8, 12), "lenia": (4, 8)}


def _parent(layout, seed=0):
    rng = np.random.default_rng(seed)
    return REGISTRY[layout.modality].random(rng, layout).reshape(-1)


def _grow(layout, key, n):
    return REGISTRY[layout.modality].layout_from_settings({key: n})


def test_growing_keeps_every_parent_unit_verbatim():
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(1))
    assert got.shape == (child_layout.length,)
    np.testing.assert_array_equal(got[:parent.size], parent)


def test_the_new_unit_is_born_silent():
    """Zero amplitude, so the child evaluates exactly as its parent does."""
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(1))
    new = got[parent.size:]
    lo, hi = AMPLITUDE["fourier"]
    np.testing.assert_array_equal(new[lo:hi], np.zeros(hi - lo, np.float32))


def test_the_new_unit_is_not_inert_on_both_sides():
    """A unit zeroed everywhere takes generations of sigma to wake up. Its
    INCOMING half is drawn, so it has something to say the moment the search
    moves its amplitude."""
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(1))
    new = got[parent.size:]
    lo, _hi = AMPLITUDE["fourier"]
    assert np.any(new[:lo] != 0.0), "the frequency half must be drawn"


@pytest.mark.parametrize("layout,key,grown", [
    (FOURIER, "centers", 11), (GABOR, "filters", 13), (LENIA, "bumps", 13)])
def test_every_unit_modality_transfers(layout, key, grown):
    parent = _parent(layout)
    child_layout = _grow(layout, key, grown)
    got = transfer_genome(parent, layout, child_layout,
                          np.random.default_rng(2))
    np.testing.assert_array_equal(got[:parent.size], parent)
    stride = REGISTRY[layout.modality].unit_floats(layout)
    new = got[parent.size:]
    assert new.size == stride
    lo, hi = AMPLITUDE[layout.modality]
    np.testing.assert_array_equal(new[lo:hi], np.zeros(hi - lo, np.float32))


def test_shrinking_keeps_the_units_that_remain():
    """Lossy by nature - a dropped unit's contribution goes with it - but what
    survives must survive unchanged."""
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 9)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(3))
    assert got.shape == (child_layout.length,)
    np.testing.assert_array_equal(got, parent[:child_layout.length])


def test_a_transfer_is_float32():
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(4))
    assert got.dtype == np.float32


@pytest.mark.parametrize("layout,key,grown", [
    (FOURIER, "centers", 11), (GABOR, "filters", 13), (LENIA, "bumps", 13)])
def test_the_transfer_re_encodes_without_clipping(layout, key, grown):
    """encode() is arctanh(p / scale) and clips at the rails. An AMPLITUDE is
    an offset-type float, so zero encodes to zero and a transferred genome
    makes a clean CMA-ES mean.

    Only the amplitude is zeroed, and that matters: every unit also carries a
    WIDTH (gabor's sigma, lenia's sigma) which does clip at zero, so a transfer
    that silenced the whole unit would start the search against a rail."""
    parent = _parent(layout)
    child_layout = _grow(layout, key, grown)
    got = transfer_genome(parent, layout, child_layout,
                          np.random.default_rng(5))
    _z, clipped = REGISTRY[layout.modality].encode(got, child_layout)
    assert clipped == 0


def test_a_cross_modality_pair_has_no_transfer():
    """A jump is a restart, and saying so is better than returning something
    that looks like a carried genome."""
    parent = _parent(FOURIER)
    with pytest.raises(ValueError, match="cross-modality"):
        transfer_genome(parent, FOURIER, GABOR, np.random.default_rng(6))


def test_an_unchanged_layout_returns_the_same_floats():
    parent = _parent(FOURIER)
    got = transfer_genome(parent, FOURIER, FOURIER, np.random.default_rng(7))
    np.testing.assert_array_equal(got, parent)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_genome_transfer.py -q`
Expected: FAIL — `ImportError: cannot import name 'transfer_genome'`.

- [ ] **Step 3: Declare the amplitude slice on each unit modality**

Where a unit keeps its outgoing weight is the modality's own fact, so it is
declared beside `UNIT_FLOATS` and `SCALE_OFFSETS` rather than mapped in
`layout_moves.py`. Add to `services/brains/fourier.py`, after
`SCALE_OFFSETS`:

```python
    # The OUTGOING half: the evaluation is `out += amplitude * basis`, so a
    # centre with zero amplitude contributes nothing and a grown brain is
    # bit-identical to its parent at birth.
    AMPLITUDE_SLICE = (4, 8)
```

Add to `services/brains/gabor.py`, after `SCALE_OFFSETS`:

```python
    # The OUTGOING half - see fourier.py. centre(4), frequency(4) come first.
    AMPLITUDE_SLICE = (8, 12)
```

Add to `services/brains/lenia.py`, after `SCALE_OFFSETS`:

```python
    # The OUTGOING half - see fourier.py. projection(4) comes first.
    AMPLITUDE_SLICE = (4, 8)
```

- [ ] **Step 4: Implement the generic transfer**

Append to `services/brains/layout_moves.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_genome_transfer.py -q`
Expected: PASS, 11 passed.

- [ ] **Step 6: Commit**

```bash
git add services/brains/layout_moves.py services/brains/fourier.py \
        services/brains/gabor.py services/brains/lenia.py \
        tests/test_genome_transfer.py
git commit -m "feat: a grown brain is born identical to its parent"
```

---

### Task 4: MLP's genome transfer

**Files:**
- Modify: `services/brains/mlp.py` — add a `transfer_genome` method
- Test: `tests/test_genome_transfer_mlp.py` (create)

**Interfaces:**
- Consumes: `transfer_genome` dispatch from Task 3.
- Produces: `MLPModality.transfer_genome(params, parent, child, rng) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_genome_transfer_mlp.py`:

```python
"""Repacking a layer stack into a neighbouring one.

A hidden unit is three separate regions of the buffer - its input weights, its
bias, and its COLUMN of the next layer's matrix - and W_out is stored
OUTPUT-MAJOR, so the new column is strided writes rather than a contiguous
append. Offsets come from layer_spans, the one definition mlp.glsl is checked
against.
"""
from __future__ import annotations

import numpy as np

from services.brains import REGISTRY
from services.brains.layout_moves import transfer_genome
from services.brains.mlp import layer_spans

MLP = REGISTRY["mlp"]


def _lay(layers):
    return MLP.layout_from_settings({"layers": layers})


def _parent(layout, seed=0):
    return np.asarray(MLP.random(np.random.default_rng(seed), layout),
                      dtype=np.float32).reshape(-1)


def _unpack(p, shape):
    hidden, out_w, out_b, n = layer_spans(shape)
    assert p.size == n
    layers = []
    for w_off, b_off, fan_in, w in hidden:
        layers.append((p[w_off:b_off].reshape(w, fan_in),
                       p[b_off:b_off + w].copy()))
    fan = hidden[-1][3] if hidden else 4
    return layers, p[out_w:out_b].reshape(4, fan), p[out_b:].copy()


def test_growing_the_only_layer_keeps_every_parent_weight():
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(1))
    assert got.size == ch.length

    lp, Wop, bop = _unpack(p, pa.shape)
    lc, Woc, boc = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0][:16], lp[0][0])
    np.testing.assert_array_equal(lc[0][1][:16], lp[0][1])
    np.testing.assert_array_equal(Woc[:, :16], Wop)
    np.testing.assert_array_equal(boc, bop)


def test_the_new_units_outgoing_column_is_zero():
    """W_out is output-major, so the column is one strided entry per output.
    Zero there is what makes the child identical to its parent at birth."""
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    got = transfer_genome(_parent(pa), pa, ch, np.random.default_rng(1))
    _lc, Woc, _boc = _unpack(got, ch.shape)
    np.testing.assert_array_equal(Woc[:, 16], np.zeros(4, np.float32))


def test_the_new_units_incoming_weights_are_drawn():
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    got = transfer_genome(_parent(pa), pa, ch, np.random.default_rng(1))
    lc, _Woc, _boc = _unpack(got, ch.shape)
    assert np.any(lc[0][0][16] != 0.0)


def test_growing_a_middle_layer_widens_the_NEXT_layers_fan_in():
    """Layer l gaining a unit adds a COLUMN to W_{l+1}, which is a hidden
    matrix here rather than W_out."""
    pa, ch = _lay([[16, 0], [8, 0]]), _lay([[17, 0], [8, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(2))
    lp, Wop, _ = _unpack(p, pa.shape)
    lc, Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[1][0][:, :16], lp[1][0])
    np.testing.assert_array_equal(lc[1][0][:, 16], np.zeros(8, np.float32))
    np.testing.assert_array_equal(Woc, Wop)


def test_adding_a_layer_leaves_the_parents_stack_untouched():
    pa, ch = _lay([[16, 0]]), _lay([[16, 0], [16, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(3))
    lp, _Wop, _ = _unpack(p, pa.shape)
    lc, _Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0], lp[0][0])
    np.testing.assert_array_equal(lc[0][1], lp[0][1])


def test_shrinking_keeps_the_units_that_remain():
    pa, ch = _lay([[16, 0]]), _lay([[15, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(4))
    lp, Wop, _ = _unpack(p, pa.shape)
    lc, Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0], lp[0][0][:15])
    np.testing.assert_array_equal(Woc, Wop[:, :15])


def test_dropping_a_layer_keeps_the_layers_that_remain():
    pa, ch = _lay([[16, 0], [8, 1]]), _lay([[16, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(5))
    lp, _Wop, _ = _unpack(p, pa.shape)
    lc, _Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0], lp[0][0])


def test_an_activation_change_moves_no_floats():
    """Same widths, same offsets - only what the shader does between them."""
    pa, ch = _lay([[16, 0]]), _lay([[16, 2]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(6))
    np.testing.assert_array_equal(got, p)


def test_the_transfer_re_encodes_without_clipping():
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    got = transfer_genome(_parent(pa), pa, ch, np.random.default_rng(7))
    _z, clipped = MLP.encode(got, ch)
    assert clipped == 0


def test_every_operator_produces_the_right_width():
    from services.brains.layout_moves import LayoutBounds, candidate_moves

    pa = _lay([[16, 0], [8, 1]])
    p = _parent(pa)
    for mv in candidate_moves(pa, LayoutBounds()):
        got = transfer_genome(p, pa, mv.child, np.random.default_rng(8))
        assert got.size == mv.child.length, mv.operator
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_genome_transfer_mlp.py -q`
Expected: FAIL — the generic path raises
`ValueError: mlp declares no unit and no transfer`, because
`MLPModality.UNIT_FLOATS` is `None`.

- [ ] **Step 3: Implement the repack**

In `services/brains/mlp.py`, add this method immediately after `layout_moves`:

```python
    def transfer_genome(self, params, parent: BrainLayout,
                        child: BrainLayout, rng) -> np.ndarray:
        """Repack a decoded stack into `child`'s shape.

        A hidden unit is three regions - input weights, bias, and a COLUMN of
        the next matrix - and W_out is OUTPUT-MAJOR, so that column is one
        strided entry per output rather than a contiguous append. Every offset
        comes from layer_spans, which is the definition mlp.glsl is checked
        against.

        Whatever a layer does not inherit is DRAWN, then its outgoing column is
        zeroed: the child evaluates exactly as its parent did at birth, while a
        new unit still has an incoming half to contribute the moment the search
        moves its output weight.
        """
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        out = np.asarray(self.random(rng, child),
                         dtype=np.float32).reshape(-1)
        p_hidden, p_ow, p_ob, _pn = layer_spans(parent.shape)
        c_hidden, c_ow, c_ob, _cn = layer_spans(child.shape)

        # Layer l of the child inherits from layer l of the parent, which is
        # what makes APPENDING a layer the only safe way to deepen a stack.
        for li, (w_off, b_off, fan_in, w) in enumerate(c_hidden):
            if li >= len(p_hidden):
                break
            pw_off, pb_off, p_fan, p_w = p_hidden[li]
            rows, cols = min(w, p_w), min(fan_in, p_fan)
            src = p[pw_off:pb_off].reshape(p_w, p_fan)
            dst = out[w_off:b_off].reshape(w, fan_in)
            dst[:rows, :cols] = src[:rows, :cols]
            # A widened fan-in means the layer BELOW grew: those columns are
            # the new unit's outgoing weights and start silent.
            dst[:rows, cols:] = 0.0
            out[b_off:b_off + rows] = p[pb_off:pb_off + rows]

        # W_out, output-major (OUT_DIM, fan_in of the last hidden layer).
        p_fan = p_hidden[-1][3] if p_hidden else IN_DIM
        c_fan = c_hidden[-1][3] if c_hidden else IN_DIM
        src = p[p_ow:p_ob].reshape(OUT_DIM, p_fan)
        dst = out[c_ow:c_ob].reshape(OUT_DIM, c_fan)
        cols = min(c_fan, p_fan)
        # Only when the LAST layer is the same one: a dropped or added layer
        # changes which units W_out reads, and those weights mean nothing.
        if len(c_hidden) == len(p_hidden):
            dst[:, :cols] = src[:, :cols]
            dst[:, cols:] = 0.0
            out[c_ob:c_ob + OUT_DIM] = p[p_ob:p_ob + OUT_DIM]
        return out.astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_genome_transfer_mlp.py -q`
Expected: PASS, 10 passed.

- [ ] **Step 5: Run the whole brain suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mlp_layer_stack.py tests/test_brain_genome_ops.py tests/test_genome_transfer.py tests/test_layout_moves_mlp.py -q`
Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add services/brains/mlp.py tests/test_genome_transfer_mlp.py
git commit -m "feat: an mlp carries its weights into a wider or deeper stack"
```

---

### Task 5: Prove the child is its parent, on the GPU

**Files:**
- Test: `tests/test_layout_move_phenotype_gpu.py` (create)

**Interfaces:**
- Consumes: `candidate_moves` and `transfer_genome` from Tasks 1–4.
- Produces: nothing. This is the guard the whole design rests on.

**Why a GPU test and not a NumPy one:** the phenotype is what `eval_brain`
computes, and `CLAUDE.md` is explicit that a modality is two hand-written
halves — Python decides what each float MEANS, GLSL decides where each float is
READ. A NumPy-only check would compare Python against itself and pass while the
shader read the new unit's floats from somewhere else. `evaluate()` in
`tests/test_brain_modalities_gpu.py` is the `purefn` harness this reuses: one
invocation per input point, no shared writes, bit-reproducible.

- [ ] **Step 1: Write the test**

Create `tests/test_layout_move_phenotype_gpu.py`:

```python
"""A grown brain must BE its parent until the search moves it.

The whole point of transferring rather than redrawing: if a child's novelty
came from a random restart, a layout move would look productive whatever shape
it proposed. This is the only test that can see it, because the phenotype is
what the SHADER computes and a NumPy check compares Python against itself.

Needs a real GL context, so it skips where there is none.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.brains import REGISTRY, default_layout
from services.brains.layout_moves import (LayoutBounds, candidate_moves,
                                          transfer_genome)
from tests.test_brain_modalities_gpu import evaluate, gl, inputs  # noqa: F401

# Operators that ADD capacity. Only these can be phenotype-preserving: a
# dropped unit takes its contribution with it, which is what makes shrinking a
# different kind of move rather than a broken one.
GROWTH = ("grow", "add_layer")

CASES = [
    default_layout(),
    REGISTRY["gabor"].layout_from_settings({}),
    REGISTRY["lenia"].layout_from_settings({}),
    REGISTRY["mlp"].layout_from_settings({"layers": [[16, 0]]}),
    REGISTRY["mlp"].layout_from_settings({"layers": [[12, 0], [8, 1]]}),
]


@pytest.mark.gpu
@pytest.mark.parametrize("parent", CASES,
                         ids=[c.signature() for c in CASES])
def test_a_growth_move_does_not_change_what_the_brain_computes(gl, parent):
    xs = inputs(3)
    p = np.asarray(REGISTRY[parent.modality].random(
        np.random.default_rng(11), parent), dtype=np.float32).reshape(-1)
    before = evaluate(gl, p, parent, xs)

    seen = 0
    for mv in candidate_moves(parent, LayoutBounds()):
        if mv.operator not in GROWTH:
            continue
        seen += 1
        child = transfer_genome(p, parent, mv.child, np.random.default_rng(12))
        after = evaluate(gl, child, mv.child, xs)
        np.testing.assert_array_equal(
            after, before,
            err_msg=f"{mv.operator}: {parent.signature()} -> "
                    f"{mv.child.signature()} changed the phenotype")
    assert seen, f"no growth move from {parent.signature()}"


@pytest.mark.gpu
def test_a_shrink_move_is_allowed_to_differ():
    """Stated so the equality above is read as a property of GROWTH rather
    than an accident of these layouts."""
    parent = default_layout()
    xs = inputs(4)
    p = np.asarray(REGISTRY["fourier"].random(
        np.random.default_rng(13), parent), dtype=np.float32).reshape(-1)
    before = evaluate(gl, p, parent, xs)
    child_layout = REGISTRY["fourier"].layout_from_settings({"centers": 9})
    child = transfer_genome(p, parent, child_layout, np.random.default_rng(14))
    after = evaluate(gl, child, child_layout, xs)
    assert not np.array_equal(after, before)
```

The import of `gl` is deliberate and it works: importing a `@pytest.fixture`
function into another test module makes it a fixture of that module. Reuse it
rather than copying the shader assembly — a second copy is a second thing to
forget when a modality is added.

- [ ] **Step 2: Run the test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_layout_move_phenotype_gpu.py -q -rs`
Expected: PASS on a machine with a GL context; SKIPPED with
`no standalone GL context` where there is none. A FAILURE here means the
transfer is writing a float the shader reads from a different offset — do not
weaken the assertion to `allclose`, find the offset.

- [ ] **Step 3: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no failures.

- [ ] **Step 4: Commit**

```bash
git add tests/test_layout_move_phenotype_gpu.py
git commit -m "test: a grown brain computes exactly what its parent did"
```

---

### Task 6: Record the rules and the deviation

**Files:**
- Modify: `CLAUDE.md` — two caveats under **Brains and modalities**
- Modify: `docs/superpowers/specs/2026-08-17-brain-layout-search-design.md` —
  correct §3's generic transfer

**Interfaces:**
- Consumes: everything above. Produces nothing.

- [ ] **Step 1: Correct the spec**

In `docs/superpowers/specs/2026-08-17-brain-layout-search-design.md`, in
section 3, replace the sentence

> **Generic:** for the unit-structured modalities, copy `min(n_old, n_new)` whole
> units and draw the remainder.

with:

```markdown
**Generic:** for the unit-structured modalities, copy `min(n_old, n_new)` whole
units, draw the remainder, and then ZERO the new unit's amplitude. Drawing it
outright was the first design and it is wrong for the same reason a randomly
initialised MLP unit is: all three unit modalities evaluate as
`out += amplitude * basis`, so an amplitude of zero makes the new unit silent
and the grown child bit-identical to its parent. Every growth move in every
modality is therefore phenotype-preserving, which is what makes the novelty a
move earns attributable to the move. `AMPLITUDE_SLICE` is declared by each
modality beside `UNIT_FLOATS`, because where a unit keeps its outgoing weight
is the modality's own fact.
```

- [ ] **Step 2: Add the caveats to CLAUDE.md**

In `CLAUDE.md`, under **Brains and modalities**, add:

```markdown
- **A GROWTH move is phenotype-preserving, and that is the whole point of
  transferring rather than redrawing.** A new unit is drawn and then SILENCED —
  its amplitude zeroed for the three unit modalities, its outgoing column
  zeroed for MLP — so the child computes bit-for-bit what its parent computed
  until the search moves it. Without that, a child's novelty comes from the
  random restart rather than from the extra capacity, and a layout move looks
  productive whatever shape it proposed. The INCOMING half is drawn, not
  zeroed: a unit silent on both sides is inert in a way `sigma` takes
  generations to undo. Zero decodes from `z = 0` exactly, so a transferred
  genome re-encodes without clipping and makes a clean CMA-ES mean. SHRINK and
  `drop_layer` are lossy by nature and preserve nothing — only what survives
  survives unchanged. Guarded by
  `tests/test_layout_move_phenotype_gpu.py`, which runs the real shader,
  because a NumPy check compares Python against itself and would pass while
  the shader read the new unit from a different offset.

- **A layout move is PROPOSED, REBUILT and COMPARED.** `_shape_from_layers`
  CLAMPS rather than raising — deliberately, because its input may be a config
  from a build with different limits — so a proposal that hits `MAX_DEPTH`,
  `MAX_WIDTH`, `MAX_BRAIN_FLOATS` or the user's own bound comes back as a
  DIFFERENT layout, silently. `candidate_moves` rebuilds each proposal and
  compares `settings_of` against it, which turns that into a rejected move
  rather than a move that did something else. The float budget is checked
  BEFORE the `BrainLayout` is constructed, because `__post_init__` raises past
  `MAX_BRAIN_FLOATS` and a bounded search must not take the app down at its own
  ceiling. A new MLP layer is APPENDED, never inserted: an inserted layer
  renumbers every layer after it, leaving nothing for the transfer to carry
  across. Fourier's phase offset is a function of the centre INDEX, so its new
  centre goes at the END for the same reason.
```

- [ ] **Step 3: Run the full suite one last time**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no failures.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-08-17-brain-layout-search-design.md
git commit -m "docs: growth moves preserve the phenotype, and why"
```

---

## Manual verification

Nothing in this plan reaches the app — there is no UI, no driver call and no
archive write, so there is nothing to click. The GPU test in Task 5 is the
verification. Confirm it actually RAN rather than skipped:

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_move_phenotype_gpu.py -q -rs
```

A line reading `SKIPPED [1] no standalone GL context` means the strongest
guard in this batch did not execute, and the batch is unverified.

## Out of scope

Sections 4–7 of the spec: the expedition wiring (`requested_layout`,
`start_expedition_at`), keep-or-revert, the ledger, and the `layout_search`
settings panel. `propose_layout_move` takes a `banned` set and nothing yet
fills it; `LayoutBounds` takes limits and nothing yet constructs it from
`ArchiveState`. Both are the seams those sections attach to, and neither may be
wired here.
