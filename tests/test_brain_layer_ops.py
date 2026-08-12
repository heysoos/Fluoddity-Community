"""Right-clicking a layer: reroll, rescale, reset - and who is allowed to.

A layer's weights are a GENOME edit: free, undoable with Z, and structural in
nothing. Its width, activation or count is a LAYOUT edit, which resets the
search and switches archive. The two live on the same row, so which is which
has to be exact.

The other half is WHICH brain is being edited. There is not always one to point
at - with no rule loaded every cohort has its own, and under a tournament every
tile does - and picking the wrong one silently edits a brain nobody is looking
at.
"""
import numpy as np
import pytest

from command_handler import CommandHandler
from services.brains import REGISTRY, default_layout
from services.brains.mlp import DISTRIBUTIONS, layer_spans
from state import UIState
from ui.brain_window import BrainWindowMixin

MLP = REGISTRY["mlp"]
LAYERS = [[8, 0], [6, 1], [4, 2]]


def a_layout(layers=None):
    return MLP.layout_from_settings({"layers": layers or LAYERS})


def a_genome(layout, seed=3):
    m = REGISTRY[layout.modality]
    return np.asarray(m.random(np.random.default_rng(seed), layout),
                      dtype=np.float32).reshape(-1)


class _Sim:
    def __init__(self, layout, per_cohort=False, tournament=False, grid=4):
        self.brain_layout = layout
        self.brain_per_cohort = per_cohort
        self.tournament_enabled = tournament
        self.tournament_grid = grid
        self.applied = []
        self._cohorts = [a_genome(layout, s) for s in range(8)]

    def apply_rule(self, params):
        self.applied.append(np.asarray(params, dtype=np.float32).reshape(-1))

    def cohort_brain(self, i):
        return self._cohorts[i] if self.brain_per_cohort else None

    def write_cohort_brain(self, i, params):
        if self.brain_per_cohort:
            self._cohorts[i] = np.asarray(params, dtype=np.float32).reshape(-1)


class _RuleManager:
    def __init__(self, rule):
        self.stack = [np.asarray(rule, dtype=np.float32)]

    def push_rule(self, rule, seed):
        self.stack.append(np.asarray(rule, dtype=np.float32).reshape(-1))

    def get_current_rule(self):
        return self.stack[-1]


def handler(sim, rule=None):
    h = object.__new__(CommandHandler)
    h.sim = sim
    h.rule_manager = _RuleManager(
        a_genome(sim.brain_layout) if rule is None else rule)
    h._borrow = None
    h.apply_brain_layout = None
    h.archive = None
    ui_state = UIState()
    ui_state.brain.modality = "mlp"
    ui_state.brain.settings = {"layers": LAYERS}
    ui_state.brain.preview_per_cohort = sim.brain_per_cohort
    ui_state.brain.preview_tile0 = sim.tournament_enabled
    return h, ui_state


def run(h, ui_state, *ops):
    for op in ops:
        ui_state.brain.layer_op = op
        h._handle_brain_source(ui_state)


def spans(layout, i):
    return MLP.layer_parts(layout, i)


def outside(layout, i):
    """A mask of every float NOT in layer i, which no op may touch."""
    m = np.ones(layout.length, dtype=bool)
    for lo, hi in spans(layout, i).values():
        m[lo:hi] = False
    return m


# ---- what the ops do -------------------------------------------------------

@pytest.mark.parametrize("i", range(len(LAYERS)))
def test_reroll_weights_changes_only_that_layers_weights(i):
    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    before = h.rule_manager.get_current_rule().copy()

    run(h, ui_state, (i, "scale_begin", None), (i, "reroll_weights", 0))

    after = h.rule_manager.get_current_rule()
    assert len(h.rule_manager.stack) == 2, "the reroll was not undoable"
    w_lo, w_hi = spans(layout, i)["weights"]
    b_lo, b_hi = spans(layout, i)["biases"]
    assert not np.array_equal(after[w_lo:w_hi], before[w_lo:w_hi])
    assert np.array_equal(after[b_lo:b_hi], before[b_lo:b_hi]), "biases moved"
    off = outside(layout, i)
    assert np.array_equal(after[off], before[off]), "another layer moved"
    assert np.array_equal(sim.applied[-1], after), "the GPU has something else"


def test_reroll_biases_leaves_the_weights_alone():
    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    before = h.rule_manager.get_current_rule().copy()

    run(h, ui_state, (1, "scale_begin", None), (1, "reroll_biases", 0))

    after = h.rule_manager.get_current_rule()
    w_lo, w_hi = spans(layout, 1)["weights"]
    b_lo, b_hi = spans(layout, 1)["biases"]
    assert np.array_equal(after[w_lo:w_hi], before[w_lo:w_hi])
    assert not np.array_equal(after[b_lo:b_hi], before[b_lo:b_hi])


@pytest.mark.parametrize("dist", range(len(DISTRIBUTIONS)))
def test_every_distribution_stays_inside_the_rails(dist):
    """A reroll draws z and then decodes, so it is always a genome the search
    could also have produced - heavy-tail included, which is unbounded in z."""
    from services.brains.mlp import W_SCALE

    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    run(h, ui_state, (0, "scale_begin", None), (0, "reroll_weights", dist))

    lo, hi = spans(layout, 0)["weights"]
    got = h.rule_manager.get_current_rule()[lo:hi]
    assert np.all(np.isfinite(got))
    assert np.abs(got).max() <= W_SCALE + 1e-5


def test_sparse_actually_zeroes_most_of_the_layer():
    layout = MLP.layout_from_settings({"layers": [[8, 0], [8, 0]]})
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    run(h, ui_state, (1, "scale_begin", None),
        (1, "reroll_weights", DISTRIBUTIONS.index("sparse")))

    lo, hi = spans(layout, 1)["weights"]
    got = h.rule_manager.get_current_rule()[lo:hi]
    assert float(np.mean(got == 0.0)) > 0.5


def test_scale_multiplies_the_snapshot_rather_than_compounding():
    """Applied to the live value it would compound over a drag and the layer
    would explode - the same shape as the width slider's draft."""
    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    base = h.rule_manager.get_current_rule().copy()
    lo, hi = spans(layout, 1)["weights"]

    run(h, ui_state, (1, "scale_begin", None), (1, "scale", 2.0),
        (1, "scale", 2.0), (1, "scale", 1.5))

    assert np.allclose(sim.applied[-1][lo:hi], base[lo:hi] * 1.5, atol=1e-5)
    assert len(h.rule_manager.stack) == 1, "a drag must not push per frame"


def test_a_scale_drag_pushes_once_on_close():
    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    base = h.rule_manager.get_current_rule().copy()
    lo, hi = spans(layout, 0)["biases"]

    run(h, ui_state, (0, "scale_begin", None), (0, "scale", 0.5),
        (0, "scale_end", None))

    assert len(h.rule_manager.stack) == 2
    assert np.allclose(h.rule_manager.get_current_rule()[lo:hi],
                       base[lo:hi] * 0.5, atol=1e-5)


def test_opening_and_closing_without_touching_anything_pushes_nothing():
    """Otherwise every right-click would fill the undo history."""
    sim = _Sim(a_layout())
    h, ui_state = handler(sim)
    run(h, ui_state, (0, "scale_begin", None), (0, "scale_end", None))
    assert len(h.rule_manager.stack) == 1


def test_scale_is_unclamped():
    """The archive stores DECODED params, so a rescaled layer saves and plays
    back exactly as it looks. What it costs is only SEEDING a search from that
    creature, because encode clips - which the row warns about instead."""
    from services.brains.mlp import W_SCALE

    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    lo, hi = spans(layout, 0)["weights"]
    run(h, ui_state, (0, "scale_begin", None), (0, "scale", 4.0))
    assert np.abs(sim.applied[-1][lo:hi]).max() > W_SCALE


def test_reset_returns_the_layer_to_the_menus_opening_state():
    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    base = h.rule_manager.get_current_rule().copy()

    run(h, ui_state, (2, "scale_begin", None), (2, "reroll_weights", 0),
        (2, "reroll_biases", 1), (2, "reset", None))

    assert np.allclose(h.rule_manager.get_current_rule(), base, atol=1e-6)


def test_a_reroll_rebases_the_scale():
    """Otherwise the next Scale would multiply the brain from before the
    reroll, and the layer would jump."""
    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    lo, hi = spans(layout, 0)["weights"]

    run(h, ui_state, (0, "scale_begin", None), (0, "reroll_weights", 0))
    rerolled = h.rule_manager.get_current_rule().copy()
    run(h, ui_state, (0, "scale", 2.0))

    assert np.allclose(sim.applied[-1][lo:hi], rerolled[lo:hi] * 2.0, atol=1e-5)


# ---- which brain, and who may edit it --------------------------------------

def test_the_source_kind_follows_the_sim():
    bst = UIState().brain
    assert BrainWindowMixin.source_kind(bst) == "rule"
    bst.preview_per_cohort = True
    assert BrainWindowMixin.source_kind(bst) == "cohort"
    # A rule-less startup leaves the cohort flag set underneath a tournament,
    # so the tile case has to win.
    bst.preview_tile0 = True
    assert BrainWindowMixin.source_kind(bst) == "tile"


def test_editing_a_cohort_writes_that_cohort_and_no_rule():
    """Visible at once and saved by nothing - File > Save writes slot 0, which
    is what the Adopt button is for."""
    layout = a_layout()
    sim = _Sim(layout, per_cohort=True)
    h, ui_state = handler(sim)
    ui_state.brain.source_index = 5
    before = [c.copy() for c in sim._cohorts]

    run(h, ui_state, (0, "scale_begin", None), (0, "reroll_weights", 0))

    assert not np.array_equal(sim._cohorts[5], before[5])
    for k in range(len(before)):
        if k != 5:
            assert np.array_equal(sim._cohorts[k], before[k]), f"cohort {k}"
    assert sim.applied == [], "a cohort edit must not become the loaded rule"
    assert len(h.rule_manager.stack) == 1


def test_adopt_promotes_the_named_cohort():
    layout = a_layout()
    sim = _Sim(layout, per_cohort=True)
    h, ui_state = handler(sim)
    ui_state.brain.source_index = 3
    ui_state.brain.adopt_requested = True
    h._handle_brain_source(ui_state)

    assert np.array_equal(sim.applied[-1], sim._cohorts[3])
    assert len(h.rule_manager.stack) == 2
    assert ui_state.brain.adopt_requested is False, "the one-shot never cleared"


def test_a_tournament_tile_is_read_only():
    """The grid owns those slots and rewrites them every generation, so there is
    nothing an edit could survive."""
    layout = a_layout()
    sim = _Sim(layout, tournament=True, grid=4)
    h, ui_state = handler(sim)
    before = h.rule_manager.get_current_rule().copy()

    run(h, ui_state, (0, "scale_begin", None), (0, "reroll_weights", 0))

    assert np.array_equal(h.rule_manager.get_current_rule(), before)
    assert sim.applied == []
    assert ui_state.brain.source_count == 16


def test_a_borrowed_brain_is_not_edited():
    """Slot 0 holds someone else's brain for as long as the pointer sits on a
    menu item, and the commit path would keep the edit."""
    layout = a_layout()
    sim = _Sim(layout)
    h, ui_state = handler(sim)
    h._borrow = object()
    before = h.rule_manager.get_current_rule().copy()

    run(h, ui_state, (0, "scale_begin", None), (0, "reroll_weights", 0))

    assert np.array_equal(h.rule_manager.get_current_rule(), before)
    assert sim.applied == []
    assert ui_state.brain.layer_op is None, "the one-shot never cleared"


def test_the_cohort_count_reaches_the_window():
    sim = _Sim(a_layout(), per_cohort=True)
    h, ui_state = handler(sim)
    ui_state.sim.num_cohorts = 24
    h._handle_brain_source(ui_state)
    assert ui_state.brain.source_count == 24


def test_a_modality_without_layers_is_left_alone():
    """Only MLP declares layer_parts. The others must not crash into it."""
    sim = _Sim(default_layout())
    h, ui_state = handler(sim)
    ui_state.brain.modality = "fourier"
    before = h.rule_manager.get_current_rule().copy()
    run(h, ui_state, (0, "scale_begin", None), (0, "reroll_weights", 0))
    assert np.array_equal(h.rule_manager.get_current_rule(), before)


def test_an_op_naming_a_layer_that_is_gone_is_ignored():
    """The stack can be edited while the menu is open."""
    sim = _Sim(a_layout([[8, 0]]))
    h, ui_state = handler(sim)
    before = h.rule_manager.get_current_rule().copy()
    run(h, ui_state, (0, "scale_begin", None), (7, "reroll_weights", 0))
    assert np.array_equal(h.rule_manager.get_current_rule(), before)


def test_the_layer_spans_cover_the_hidden_layers_and_nothing_else():
    """The output layer has no row, so rerolling the last hidden layer must
    leave W_out where it is."""
    layout = a_layout()
    hidden, out_w, out_b, n = layer_spans(layout.shape)
    seen = np.zeros(n, dtype=bool)
    for i in range(len(LAYERS)):
        for lo, hi in spans(layout, i).values():
            assert not seen[lo:hi].any(), "two layers claim the same floats"
            seen[lo:hi] = True
    assert seen[:out_w].all(), "a hidden float belongs to no layer"
    assert not seen[out_w:].any(), "the output layer is addressable"
