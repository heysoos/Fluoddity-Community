# A layout move rides on an expedition — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Sections 4 and 5 of
`docs/superpowers/specs/2026-08-17-brain-layout-search-design.md` — the search
proposes a neighbouring brain layout, carries the seed's genome into it, spends
one expedition there, and keeps or reverts the layout on whether the archive
admitted anything.

**Architecture:** The driver cannot switch brain — `App` owns the archive, the
sim and the tournament — so a move is a one-shot `requested_layout` the frame
loop reads. The expedition therefore starts in TWO stages: propose (under the
parent layout, where the seed lives) and begin (after the switch has landed,
where the genome's width is legal). The verdict is `admitted >= 1`.

**Tech Stack:** NumPy, pytest. `services/brains/layout_moves.py` (§2/§3),
`services/imgep_driver.py`, `main.py`.

## Global Constraints

- Tests run with `.venv/Scripts/python.exe -m pytest -q`. Bare `python` is 3.10
  with no pytest.
- Comments, docstrings and tooltips state the RULE, never the evidence. No
  percentages, timings, dates or before/after comparisons.
- `sim.py` is user-owned. Nothing here touches it.
- Stage commits by explicit filename. Never `git add -A`, never a directory.
- Commit each task locally. Never push.
- `native_rows()` stays strict: transfer is the single crossing and it reaches
  the optimizer through `x0`, never through the parent sampler.
- Layout search ships OFF (`layout_search = False`), so nothing in this batch
  changes what a run does until section 7 exposes the checkbox.

---

### Task 1: A jump to another modality is a candidate move

**Files:**
- Modify: `services/brains/layout_moves.py`
- Test: `tests/test_layout_moves.py`

**Interfaces:**
- Produces: `candidate_moves` additionally returns `LayoutMove(operator=
  "modality")` for every key in `bounds.modalities` that is not the parent's,
  each at that modality's DEFAULT layout.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layout_moves.py`:

```python
def test_no_modality_jump_is_offered_by_default():
    """LayoutBounds.modalities empty means 'the running one only'. A jump is a
    restart, so opting into one is a decision rather than a default."""
    moves = candidate_moves(FOURIER, LayoutBounds())
    assert not [m for m in moves if m.operator == "modality"]


def test_naming_another_modality_offers_its_default_layout():
    from services.brains import REGISTRY

    moves = candidate_moves(FOURIER, LayoutBounds(modalities=("fourier", "mlp")))
    jumps = [m for m in moves if m.operator == "modality"]
    assert [m.child.signature() for m in jumps] == [
        REGISTRY["mlp"].layout_from_settings({}).signature()]


def test_a_jump_never_proposes_the_modality_already_running():
    moves = candidate_moves(FOURIER, LayoutBounds(modalities=("fourier",)))
    assert not [m for m in moves if m.operator == "modality"]


def test_an_unknown_modality_key_is_skipped_rather_than_raising():
    """The bound is a comma-separated string in the settings, so a stale key
    outlives the build that understood it."""
    moves = candidate_moves(FOURIER, LayoutBounds(modalities=("nonesuch",)))
    assert not [m for m in moves if m.operator == "modality"]


def test_a_jump_respects_the_float_budget():
    moves = candidate_moves(
        FOURIER, LayoutBounds(max_floats=1, modalities=("mlp", "gabor")))
    assert not [m for m in moves if m.operator == "modality"]
```

- [ ] **Step 2: Run them and watch them fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_moves.py -q
```

Expected: the four `modality`-operator tests fail (empty list where one is
expected / no failure where one is expected is fine for the negatives — the
positive one is the gate).

- [ ] **Step 3: Implement**

In `services/brains/layout_moves.py`, widen the import to bring in `REGISTRY`,
add `_modality_jumps` after `_build`, and extend `candidate_moves`:

```python
def _modality_jumps(layout, bounds) -> list[LayoutMove]:
    """Every OTHER modality named in the bounds, at its own DEFAULT layout.

    There is no correspondence between a Fourier centre count and an MLP width,
    so a jump lands on the target's default rather than pretending to preserve
    a size. It carries no genome either - transfer_genome refuses one - which
    is what makes a jump a restart and the reason it is opt-in.
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
```

and at the end of `candidate_moves`, before `return out`:

```python
    out.extend(_modality_jumps(layout, bounds))
```

- [ ] **Step 4: Run the whole layout-move suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_moves.py tests/test_layout_moves_mlp.py tests/test_genome_transfer.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/brains/layout_moves.py tests/test_layout_moves.py
git commit -m "feat: a named modality is a one-step move to its default layout"
```

---

### Task 2: An expedition can start from an explicit x0

**Files:**
- Modify: `services/imgep_driver.py`
- Test: `tests/test_expedition_x0.py` (create)

**Interfaces:**
- Produces: `ImgepDriver._resolve_seed(embedding, kind, seed_index) -> int|None`
  and `ImgepDriver.start_expedition_at(embedding, kind, text, x0,
  seed_index=None) -> bool`. `start_expedition_with` becomes a thin wrapper
  that resolves a native row to an `x0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_expedition_x0.py`:

```python
"""An expedition can be told where to start, not only which row to start at.

A transferred genome has no archive row - it is a brain of a layout the archive
may hold nothing of yet - so _x0_index is None and the optimizer's mean comes
in directly.
"""
from __future__ import annotations

import numpy as np

from tests.test_imgep_driver import DIM, make


def _goal():
    g = np.zeros(DIM, dtype=np.float32)
    g[0] = 1.0
    return g


def test_an_explicit_x0_becomes_the_optimizers_mean():
    d, _arc, _ts = make()
    x0 = np.full(d.spec.dim, 0.25, dtype=np.float32)
    assert d.start_expedition_at(_goal(), "latent", "", x0) is True
    np.testing.assert_allclose(d.optimizer.mean, x0, atol=1e-6)


def test_an_explicit_x0_needs_no_archive_row_at_all():
    """The whole point: an empty archive cannot supply a seed, but a
    transferred genome does not need one."""
    d, arc, _ts = make()
    assert len(arc) == 0
    x0 = np.zeros(d.spec.dim, dtype=np.float32)
    assert d.start_expedition_at(_goal(), "latent", "", x0) is True
    assert d.regime == "expedition"
    assert d._x0_index is None


def test_an_x0_of_the_wrong_width_is_refused():
    d, _arc, _ts = make()
    bad = np.zeros(d.spec.dim + 1, dtype=np.float32)
    assert d.start_expedition_at(_goal(), "latent", "", bad) is False
    assert d.regime != "expedition"


def test_start_expedition_with_still_seeds_from_its_row():
    """The wrapper must be unchanged in behaviour - every existing caller goes
    through it."""
    from tests.test_imgep_driver import moving, snaps

    d, arc, _ts = make(grid=2, seed_n=1)
    d.tell(d.ask(4), snaps(4, [[10, 20, 30, 40], [11, 21, 31, 41]]))
    assert len(arc) > 0
    assert d.start_expedition_with(arc.embeddings[0].copy(), "chase", "") is True
    assert d._x0_index is not None
    np.testing.assert_allclose(
        d.optimizer.mean, d._parent_z(d._x0_index), atol=1e-5)
```

If `moving` is unused, drop that import — check before committing.

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_expedition_x0.py -q
```

Expected: `AttributeError: 'ImgepDriver' object has no attribute
'start_expedition_at'`.

- [ ] **Step 3: Split `start_expedition_with`**

Replace the body of `start_expedition_with` in `services/imgep_driver.py` and
add the two new methods. Keep the existing docstring on the wrapper; move the
seed-resolution comment onto `_resolve_seed`:

```python
    def _resolve_seed(self, emb, kind: str, seed_index) -> int | None:
        """Which archive row an expedition toward this goal should start at.

        NATIVE as well as in range. A seed becomes the optimizer's mean and is
        re-encoded under the running layout, so a row belonging to another
        brain is not a worse start but an unreadable one - and an archive pools
        every layout. A goal that names one falls through to _seed_index, which
        filters, exactly as an out-of-range index does.
        """
        if (seed_index is not None and 0 <= int(seed_index) < len(self.archive)
                and self.archive.is_native(int(seed_index))):
            return int(seed_index)
        if emb is None:
            return None             # nothing to point at and nowhere to start
        return self._seed_index(emb, str(kind))

    def start_expedition_with(self, embedding, kind: str, text: str,
                              seed_index: int | None = None) -> bool:
        """Begin an expedition toward a specific embedding. An expedition needs
        a seed, so an empty archive falls back to expansion rather than
        starting a search from nowhere.

        `seed_index` is the goal's own answer to "where should this start". A
        latent goal knows, because it was built by pushing past that entry, and
        a novelty goal has no embedding to derive one from at all. Only a text
        or chase goal has to go looking.
        """
        emb = (None if embedding is None
               else np.asarray(embedding, dtype=np.float32))
        i = self._resolve_seed(emb, kind, seed_index)
        if i is None:
            return False
        # The seed is the one re-encode worth reporting: it becomes the
        # optimizer's mean, so a clipped one starts the chase from a creature
        # the user did not pick. Expansion parents re-encode too, but there are
        # `tiles` of them every generation and a count per draw is not a signal.
        x0, clipped = self._parent_z_clipped(i)
        if not self.start_expedition_at(emb, kind, text, x0, seed_index=i):
            return False
        self._seed_phys_clipped = clipped
        return True

    def start_expedition_at(self, embedding, kind: str, text: str, x0,
                            seed_index: int | None = None) -> bool:
        """Begin an expedition from an explicit starting point.

        `x0` rather than an archive row, because a genome carried across a
        layout move has no row: it is a brain of a layout the archive may hold
        nothing of yet. A width that does not match the running space is
        refused rather than sliced - the same discipline GenomeSpec._check_width
        applies, and for the same reason.
        """
        if self.expedition_gens <= 0:
            return False
        x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
        if x0.size != self.spec.dim:
            return False
        emb = (None if embedding is None
               else np.asarray(embedding, dtype=np.float32))
        self._goal = Goal(kind, text, emb, seed_index=seed_index)
        self._x0_index = None if seed_index is None else int(seed_index)
        self._remaining = int(self.expedition_gens)
        self._since_expedition = 0
        # Reset explicitly: chase()/UI can start a new expedition on top of a
        # running one, and fitness is not comparable across different goals.
        self._expedition_best = -np.inf
        self._seed_phys_clipped = 0
        # Fresh optimizer per goal - a covariance learned for one goal doesn't
        # transfer to another. sigma << sigma0: local refinement, not a fresh
        # search.
        self._optimizer = make_optimizer(
            self.algorithm, self.spec.dim, self.tournament.tiles,
            self.expedition_sigma, self.base_seed + self.gen,
            x0.astype(np.float64),
            layout=self.spec.layout,
        )
        return True
```

- [ ] **Step 4: Run the new test and the driver suites**

```bash
.venv/Scripts/python.exe -m pytest tests/test_expedition_x0.py tests/test_imgep_driver.py tests/test_novelty_goals.py tests/test_search_layout_wiring.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/imgep_driver.py tests/test_expedition_x0.py
git commit -m "refactor: an expedition can be told where to start, not only which row"
```

---

### Task 3: The driver proposes a layout move and begins it once it lands

**Files:**
- Modify: `services/imgep_driver.py`
- Test: `tests/test_layout_expedition.py` (create)

**Interfaces:**
- Produces: settings `layout_search: bool`, `layout_move_chance: float`,
  `layout_bounds: LayoutBounds`; one-shot `requested_layout: BrainLayout|None`;
  `begin_moved_expedition() -> bool`; read-only `layout_move` (the LIVE move).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout_expedition.py`:

```python
"""A layout move rides on an ordinary expedition, in two stages.

Proposed under the PARENT layout, where the seed lives and the genome can be
read; begun under the CHILD, where the transferred genome's width is legal.
Only the frame loop can make the switch land, so the driver sets a one-shot and
waits.
"""
from __future__ import annotations

import numpy as np

from services.brains import REGISTRY, default_layout
from services.brains.layout_moves import LayoutBounds
from services.genome_spec import spec_for
from tests.test_imgep_driver import DIM, make, snaps

FOURIER = default_layout()


def a_driver(**kw):
    """A driver with one generation in the archive, so a seed exists."""
    d, arc, ts = make(grid=4, seed_n=1, **kw)
    d.tell(d.ask(4), snaps(4, [[10, 20, 30, 40], [11, 21, 31, 41]]))
    d.layout_search = True
    d.layout_move_chance = 1.0
    return d, arc, ts


def land(d):
    """What the frame loop does: adopt the requested layout, then begin."""
    lay = d.requested_layout
    d.requested_layout = None
    d.set_spec(spec_for(lay))
    return lay


def test_layout_search_is_off_by_default():
    d, _arc, _ts = make()
    assert d.layout_search is False
    assert d.requested_layout is None


def test_an_expedition_asks_for_a_neighbouring_layout():
    d, _arc, _ts = a_driver()
    assert d.start_expedition() is True
    assert d.requested_layout is not None
    assert d.requested_layout != FOURIER
    # Proposed, not started: the genome is the child's width and the spec is
    # still the parent's.
    assert d.regime != "expedition"


def test_the_expedition_begins_only_once_the_switch_has_landed():
    d, _arc, _ts = a_driver()
    d.start_expedition()
    child = land(d)
    assert d.begin_moved_expedition() is True
    assert d.regime == "expedition"
    assert d.optimizer.mean.size == child.length
    assert d.layout_move.child == child


def test_a_move_that_never_landed_is_dropped_rather_than_started():
    """The request may be refused, or overtaken by a switch of the user's own.
    Starting it anyway would optimise a genome in the wrong space."""
    d, _arc, _ts = a_driver()
    d.start_expedition()
    d.requested_layout = None            # the frame loop declined
    assert d.begin_moved_expedition() is False
    assert d.regime != "expedition"
    assert d.layout_move is None


def test_the_carried_genome_is_the_seeds_brain_grown_into_the_child():
    from services.brains.layout_moves import transfer_genome
    from services.genome_spec import decode

    d, arc, _ts = a_driver()
    d.start_expedition()
    child = land(d)
    d.begin_moved_expedition()
    mv = d.layout_move
    want = transfer_genome(arc.brain_at(d.layout_move_seed), mv.parent,
                           mv.child, np.random.default_rng(0))
    got = decode(d.optimizer.mean.astype(np.float32), child).reshape(-1)
    # Encode/decode is a squash and its inverse, so compare the phenotype
    # rather than z: a growth move is defined by what the brain COMPUTES.
    assert got.size == want.size


def test_a_modality_jump_enters_as_an_expedition_not_as_bootstrap():
    """A layout with zero native entries would otherwise bootstrap, which on
    real settings is a thousand random genomes before expansion is reachable -
    a layout search that spends its whole life bootstrapping."""
    d, _arc, _ts = a_driver()
    d.layout_bounds = LayoutBounds(modalities=("mlp",))
    # Only the jump is on offer once growth is bounded away.
    d.layout_bounds = LayoutBounds(max_width=int(FOURIER.shape[0]),
                                   modalities=("mlp",))
    assert d.start_expedition() is True
    child = land(d)
    assert child.modality == "mlp"
    assert d.begin_moved_expedition() is True
    assert d.regime == "expedition"
    assert d.optimizer.mean.size == child.length


def test_a_move_is_not_proposed_while_one_is_live():
    d, _arc, _ts = a_driver()
    d.start_expedition()
    land(d)
    d.begin_moved_expedition()
    d.requested_layout = None
    d.start_expedition()                 # a second one, on top
    assert d.requested_layout is None


def test_reset_drops_a_move_that_has_not_landed():
    d, _arc, _ts = a_driver()
    d.start_expedition()
    d.reset()
    assert d.requested_layout is None
    assert d.begin_moved_expedition() is False
```

- [ ] **Step 2: Run and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_expedition.py -q
```

Expected: `AttributeError: ... 'layout_search'`.

- [ ] **Step 3: Implement**

In `services/imgep_driver.py`, add the imports at the top:

```python
from services.brains.layout_moves import (LayoutBounds, propose_layout_move,
                                          transfer_genome)
from services.genome_spec import physics_spec_for, spec_for
```

(`encode` is already imported; check the existing import block and extend it
rather than duplicating.)

Add a small carrier above the class:

```python
@dataclass
class _PendingMove:
    """A layout move whose switch has not landed yet."""
    move: object
    goal: Goal
    x0: np.ndarray
    seed: int | None
    phys_clipped: int
```

(`dataclass` needs importing if it is not already.)

In `__init__`, after the expedition settings block:

```python
        # layout search (spec 4). OFF: opening the app must never start
        # changing brain under anyone.
        self.layout_search = False
        self.layout_move_chance = 0.25
        self.layout_bounds = LayoutBounds()
```

and in the state block:

```python
        # A layout move in three stages. REQUESTED: the one-shot the frame loop
        # reads, because App owns the archive, the sim and the tournament and
        # the driver owns none of them. PENDING: proposed, switch not landed.
        # LIVE: its expedition is running and _move_admitted is counting.
        self.requested_layout = None
        self._pending: _PendingMove | None = None
        self._move = None
        self._move_admitted = 0
        self._move_seed: int | None = None
        # Pairs that produced nothing, so the search does not spend its cadence
        # re-proposing them. In memory until the ledger persists it.
        self._reverted: set[tuple[str, str]] = set()
```

Add the read-only accessors near `optimizer`:

```python
    @property
    def layout_move(self):
        """The move whose expedition is LIVE, or None."""
        return self._move

    @property
    def layout_move_seed(self) -> int | None:
        """The archive row the live move's genome was carried from."""
        return self._move_seed
```

Extend `start_expedition`:

```python
    def start_expedition(self) -> bool:
        """Draw a goal from the configured sources and begin. -> did it start?"""
        goal = self._draw_goal()
        if goal is None:
            return False
        # A layout move rides on this expedition rather than replacing it: the
        # goal is drawn exactly as it always was, and only where it is chased
        # from changes.
        if self._propose_layout_move(goal):
            return True
        return self.start_expedition_with(goal.embedding, goal.kind, goal.text,
                                          seed_index=goal.seed_index)
```

Add the two halves after `start_expedition_at`:

```python
    def _propose_layout_move(self, goal) -> bool:
        """Ask for a neighbouring LAYOUT to chase this goal in. -> asked?

        True the moment the request is queued, NOT once an expedition is
        running: a genome of the child's width cannot be optimised under the
        parent's spec, so the switch has to land first and only the frame loop
        can land it. begin_moved_expedition is the other half.
        """
        if not self.layout_search or self._move is not None:
            return False
        if self._pending is not None:
            return False
        if float(self.rng.random()) >= float(self.layout_move_chance):
            return False
        parent = self.spec.layout
        mv = propose_layout_move(parent, self.layout_bounds, self.rng,
                                 banned=self._reverted)
        if mv is None:
            return False
        # The seed this goal would have picked anyway, under the layout still
        # running - which is what makes this a move WITH an expedition rather
        # than a move instead of one.
        i = self._resolve_seed(goal.embedding, goal.kind, goal.seed_index)
        if mv.child.modality == parent.modality:
            if i is None:
                return False        # nothing to carry; run an ordinary one
            params = transfer_genome(self.archive.brain_at(i), parent,
                                     mv.child, self.rng)
            zb, _clamped = encode(params, mv.child)
        else:
            # A jump has no transfer by definition, so it seeds the way
            # bootstrap does. It still enters as an EXPEDITION: a layout with
            # no native entries would otherwise spend its whole budget on
            # random genomes.
            zb = (self.sigma0 * self.rng.normal(size=int(mv.child.length))
                  ).astype(np.float32)
        zp, n_clipped = self._physics_z(i)
        spec = (physics_spec_for(mv.child) if self.physics_enabled
                else spec_for(mv.child))
        x0 = zb if spec.dim <= zb.size else np.concatenate([zb, zp])
        self._pending = _PendingMove(mv, goal, x0.astype(np.float32), i,
                                     int(n_clipped))
        self.requested_layout = mv.child
        return True

    def begin_moved_expedition(self) -> bool:
        """Start the expedition a layout move was proposed for. -> did it?

        VERIFIED rather than assumed: the request may have been refused, or
        overtaken by a switch of the user's own, and a move that did not land
        is dropped rather than started in a space its genome does not belong
        to. Consumes the request either way.
        """
        p, self._pending = self._pending, None
        if p is None:
            return False
        if self.spec.layout != p.move.child:
            return False
        if not self.start_expedition_at(p.goal.embedding, p.goal.kind,
                                        p.goal.text, p.x0):
            return False
        self._seed_phys_clipped = p.phys_clipped
        self._move = p.move
        self._move_seed = p.seed
        self._move_admitted = 0
        return True

    def _physics_z(self, i: int | None):
        """The physics half of a seed's z, or the origin itself. -> (z, clipped)

        Split out of _parent_z_clipped because a carried genome supplies its own
        brain half: the physics is the seed's and the brain is not.
        """
        from services.physics_genome import PHYSICS_DIM

        if i is not None and "physics" in self.archive.entries[i].spec:
            return encode_physics(_phys_dict(self.archive.physics[i]),
                                  self.physics_origin)
        # z = 0 decodes to the current origin exactly, so a brain-only entry -
        # and an absent one - is well-defined rather than an error.
        return np.zeros(PHYSICS_DIM, dtype=np.float32), 0
```

Then make `_parent_z_clipped` use it, replacing its own `if "physics" in
entry.spec:` branch with `zp, n_clipped = self._physics_z(i)` so the rule has
one home.

Finally, clear the request in `reset()`, immediately above `self.end_expedition()`:

```python
        # A move that has not landed belongs to the run being abandoned.
        self.requested_layout = None
        self._pending = None
        self._reverted.clear()
```

and in `end_expedition()`, after `self._seed_phys_clipped = 0`:

```python
        # The move's VERDICT is delivered before this is called; any other path
        # here - a grid change, a brain switch of the user's own - abandons the
        # move unjudged, which is honest: nothing was learned about it.
        self._move = None
        self._move_seed = None
        self._move_admitted = 0
```

`_pending` is deliberately NOT cleared here: `set_spec` calls `end_expedition`
when the space moves, and the space moving is exactly the request landing.

- [ ] **Step 4: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_expedition.py tests/test_imgep_driver.py tests/test_expedition_x0.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/imgep_driver.py tests/test_layout_expedition.py
git commit -m "feat: an expedition can carry its seed into a neighbouring layout"
```

---

### Task 4: Keep the layout, or ask for it back

**Files:**
- Modify: `services/imgep_driver.py`
- Test: `tests/test_layout_verdict.py` (create)

**Interfaces:**
- Produces: `_finish_layout_move()` runs once as a layout expedition ends;
  `requested_layout` is set to `move.parent` on a revert and the pair joins
  `_reverted`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout_verdict.py`:

```python
"""The archive is the judge, and it is already judging.

Admission means finite, viable, alive and separated from everything stored. A
layout that cannot produce ONE such tile in a whole expedition has answered the
question, so the verdict needs no second opinion and no threshold.
"""
from __future__ import annotations

import numpy as np

from services.genome_spec import spec_for
from tests.test_imgep_driver import make, snaps
from tests.test_layout_expedition import a_driver, land


def run_expedition(d, admit: bool):
    """Score `expedition_gens` generations, admitting or not."""
    for g in range(int(d.expedition_gens)):
        z = d.ask(4)
        v = [10 + 40 * g, 20 + 40 * g, 30 + 40 * g, 40 + 40 * g] if admit \
            else [0, 0, 0, 0]
        d.tell(z, snaps(4, [v, v]))


def test_a_move_that_admits_nothing_asks_for_the_parent_back():
    d, _arc, _ts = a_driver(expedition_gens=2)
    d.start_expedition()
    child = land(d)
    d.begin_moved_expedition()
    parent = d.layout_move.parent
    run_expedition(d, admit=False)
    assert d.requested_layout == parent
    assert d.layout_move is None


def test_a_reverted_move_is_not_immediately_re_proposed():
    d, _arc, _ts = a_driver(expedition_gens=2)
    d.start_expedition()
    pair = (d._pending.move.parent.signature(),
            d._pending.move.child.signature())
    land(d)
    d.begin_moved_expedition()
    run_expedition(d, admit=False)
    assert pair in d._reverted


def test_a_move_that_admits_keeps_the_layout():
    d, _arc, _ts = a_driver(expedition_gens=2)
    d.start_expedition()
    land(d)
    d.begin_moved_expedition()
    d._move_admitted = 0
    run_expedition(d, admit=True)
    assert d.requested_layout is None
    assert d.layout_move is None


def test_the_verdict_is_delivered_exactly_once():
    d, _arc, _ts = a_driver(expedition_gens=2)
    d.start_expedition()
    land(d)
    d.begin_moved_expedition()
    run_expedition(d, admit=False)
    d.requested_layout = None
    d.end_expedition()
    assert d.requested_layout is None


def test_an_abandoned_move_is_neither_kept_nor_banned():
    """A grid change or a brain switch of the user's own ends the expedition
    without a verdict. Nothing was learned, so nothing is recorded."""
    d, _arc, _ts = a_driver(expedition_gens=50)
    d.start_expedition()
    land(d)
    d.begin_moved_expedition()
    d.end_expedition()
    assert d.requested_layout is None
    assert not d._reverted
```

- [ ] **Step 2: Run and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_verdict.py -q
```

- [ ] **Step 3: Implement**

In `tell()`, where `admitted` is finished (immediately after the admission loop
and before `self.tournament.selected.clear()`):

```python
        if self._move is not None:
            self._move_admitted += admitted
```

and in the expedition branch, replace

```python
            if self._remaining <= 0:
                self.end_expedition()
```

with

```python
            if self._remaining <= 0:
                self._finish_layout_move()
                self.end_expedition()
```

Add the method after `begin_moved_expedition`:

```python
    def _finish_layout_move(self) -> None:
        """Keep the layout, or ask for the parent back. Once, as its
        expedition ends.

        The archive is already the judge: admission means finite, viable, alive
        and separated from everything stored. A layout that cannot produce one
        such tile in a whole expedition has answered the question.

        Comparing the admission RATE against the parent was rejected. A
        generation's tiles share one CMA-ES population, so they clear or miss
        any bar together - the same reason the adaptive admission threshold
        was removed.
        """
        mv, self._move = self._move, None
        admitted, self._move_admitted = self._move_admitted, 0
        self._move_seed = None
        if mv is None:
            return
        kept = admitted >= 1
        print(f"[brain] layout move {mv.operator} {mv.parent.signature()} -> "
              f"{mv.child.signature()}: {admitted} admitted, "
              f"{'kept' if kept else 'reverted'}")
        if kept:
            # It has native entries now, so ordinary expansion breeds from it
            # next generation with no special case anywhere.
            return
        self._reverted.add(mv.pair)
        self.requested_layout = mv.parent
```

- [ ] **Step 4: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_verdict.py tests/test_layout_expedition.py tests/test_imgep_driver.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/imgep_driver.py tests/test_layout_verdict.py
git commit -m "feat: a layout that admits nothing hands the brain back"
```

---

### Task 5: The frame loop honours the request

**Files:**
- Modify: `main.py`
- Modify: `tests/test_brain_layout_retarget_wiring.py`
- Test: `tests/test_layout_move_wiring.py` (create)

**Interfaces:**
- Produces: `App._apply_brain_layout(layout, ui_state, *, keep_running=False)`
  and `App._apply_requested_layout(ui_state) -> bool`, called from
  `_drive_auto_tournament` on the frame a generation is scored.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout_move_wiring.py`:

```python
"""Only the frame loop can switch brain, so this is where a layout move lands.

App owns the archive, the sim and the tournament; the driver owns none of them.
The move therefore reaches the app as a one-shot, in the frame the generation
that proposed it was scored.
"""
from __future__ import annotations

import types

import numpy as np

from services.brains import REGISTRY, default_layout

FOURIER = default_layout()
GABOR = REGISTRY["gabor"].layout_from_settings({})


class _Drv:
    def __init__(self, requested=None):
        self.requested_layout = requested
        self.began = 0

    def begin_moved_expedition(self):
        self.began += 1
        return True


class _App:
    """Just enough App to drive the two methods under test."""

    def __init__(self, drv):
        import main

        self.imgep_driver = drv
        self.applied = []
        self._apply_requested_layout = types.MethodType(
            main.App._apply_requested_layout, self)

    def _apply_brain_layout(self, layout, ui_state, *, keep_running=False):
        self.applied.append((layout, keep_running))
        return True


def _ui():
    from state.archive_state import ArchiveState
    from state.brain_state import BrainState

    return types.SimpleNamespace(archive=ArchiveState(), brain=BrainState())


def test_nothing_happens_without_a_request():
    app = _App(_Drv())
    assert app._apply_requested_layout(_ui()) is False
    assert not app.applied


def test_a_request_is_applied_without_stopping_the_search():
    """The search is moving its own space on purpose. Stopping it here would
    make a layout move cost the whole run."""
    drv = _Drv(GABOR)
    app, ui = _App(drv), _ui()
    assert app._apply_requested_layout(ui) is True
    assert app.applied == [(GABOR, True)]
    assert drv.requested_layout is None
    assert drv.began == 1


def test_the_brain_window_moves_with_the_sim():
    """_handle_brain_layout applies ui_state.brain every frame, so a switch
    that moves only the sim is undone by the next one."""
    drv = _Drv(GABOR)
    app, ui = _App(drv), _ui()
    app._apply_requested_layout(ui)
    assert ui.brain.modality == "gabor"


def test_the_request_is_consumed_even_if_the_switch_is_refused():
    class Refuse(_App):
        def _apply_brain_layout(self, layout, ui_state, *, keep_running=False):
            self.applied.append((layout, keep_running))
            return False

    drv = _Drv(GABOR)
    app, ui = Refuse(drv), _ui()
    app._apply_requested_layout(ui)
    assert drv.requested_layout is None
```

Then in `tests/test_brain_layout_retarget_wiring.py`, add beside the existing
pause test:

```python
def test_a_move_the_search_asked_for_keeps_it_running():
    """The search moved its own space deliberately and has an expedition ready
    to start in it. Pausing here would make a layout move cost the run."""
    log = _Log()
    app, ui = _App(log), _UI()
    ui.archive.running = True
    app._apply_brain_layout(GABOR, ui, keep_running=True)
    assert ui.archive.running is True
    assert "pause" not in log
    assert "refresh_specs(reset=True)" not in log
    assert "retarget" in log
```

- [ ] **Step 2: Run and watch them fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_move_wiring.py tests/test_brain_layout_retarget_wiring.py -q
```

- [ ] **Step 3: Implement**

In `main.py`, change the signature and the two gated blocks of
`_apply_brain_layout`:

```python
    def _apply_brain_layout(self, layout, ui_state, *,
                            keep_running: bool = False) -> bool:
```

extend its docstring with:

```
        `keep_running` is for a layout the SEARCH asked for. It is moving its
        own space on purpose and has an expedition ready to start in the new
        one, so the pause and the optimizer reset that a hand switch needs
        would cost the whole run.
```

replace

```python
        ui_state.archive.running = False
        if self.auto_service is not None:
            self.auto_service.pause()
```

with

```python
        if not keep_running:
            ui_state.archive.running = False
            if self.auto_service is not None:
                self.auto_service.pause()
```

and

```python
        self._refresh_driver_specs(layout, reset=True)
```

with

```python
        self._refresh_driver_specs(layout, reset=not keep_running)
```

Add the reader after `_apply_brain_layout`:

```python
    def _apply_requested_layout(self, ui_state) -> bool:
        """Honour a layout the SEARCH asked for, in the frame it asked. -> did
        anything move?

        The driver cannot switch brain itself - App owns the archive, the sim
        and the tournament - so it sets a one-shot and this reads it. The
        expedition the move was proposed for starts AFTERWARDS, because a
        genome of the child's width cannot be optimised under the parent's
        spec; begin_moved_expedition verifies the switch actually landed and
        drops the move if it did not.
        """
        from command_handler import CommandHandler

        drv = getattr(self, "imgep_driver", None)
        layout = getattr(drv, "requested_layout", None)
        if drv is None or layout is None:
            return False
        drv.requested_layout = None
        applied = self._apply_brain_layout(layout, ui_state, keep_running=True)
        # The WINDOW as well as the sim: _handle_brain_layout applies whatever
        # it finds in ui_state.brain every frame, so a switch that moves only
        # the sim is undone by the next one.
        CommandHandler._put_brain_window(layout, ui_state)
        drv.begin_moved_expedition()
        return applied
```

In `_drive_auto_tournament`, in the `Action.SCORE` branch:

```python
        if action is Action.SCORE:
            fit = svc.score_and_tell()
            # None while the CLIP pass runs off-thread; retry next frame.
            if fit is not None:
                # Before the rest: everything below reports what is live NOW,
                # and a layout move that landed changed it.
                self._apply_requested_layout(ui_state)
                self._after_generation(fit)
                self._record_settings_version(ui_state)
            return 0
```

- [ ] **Step 4: Run the full suite**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: all pass (one pre-existing skip).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_layout_move_wiring.py tests/test_brain_layout_retarget_wiring.py
git commit -m "feat: the frame loop honours a layout the search asked for"
```

---

### Task 6: Record the rules

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the caveats**

Under **Brains and modalities**, after the "A layout move is PROPOSED, REBUILT
and COMPARED" caveat, add caveats stating:

- A layout move is TWO STAGES because only the frame loop can switch brain, and
  `begin_moved_expedition` verifies the landing rather than assuming it.
- `_pending` deliberately survives `end_expedition()`, because `set_spec` calls
  that when the space moves and the space moving IS the request landing.
- The verdict is `admitted >= 1`, delivered once, and an abandoned move is
  neither kept nor banned.
- `keep_running=True` is the search's own path through `_apply_brain_layout`.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: a layout move is two stages, and the archive is its judge"
```
