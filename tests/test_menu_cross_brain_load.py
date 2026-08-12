"""File > Load, driven the way the MENU drives it: hover, then click.

The other cross-brain tests stop at _restore_brain_settings, and the direct
paths (Ctrl+V, a click with no hover) reach it. The menu does not: hovering
sets preview_rule_active, and the click then takes the 'preview already applied
it' branch - which applied no brain at all, so the preset silently did not load
while the console reported that it had.

Hovering now BORROWS the config's layout, the same mechanism the archive
gallery uses, and the click turns the borrow into the real switch.
"""
from __future__ import annotations

import numpy as np
import pytest

from command_handler import CommandHandler
from services.brains import default_layout, get
from ui.brain_window import layout_for

FOURIER = default_layout()


def _rule_for(layout, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(get(layout.modality).random(rng, layout),
                      dtype=np.float32).reshape(-1)


class _Config:
    rule_seed = 0.0

    def __init__(self, layout):
        self.rule = _rule_for(layout)
        self.brain_layout = layout.signature()
        self.brain_settings = {}


class _Sim:
    def __init__(self, layout):
        self.brain_layout = layout
        self.applied = []

    def realloc_brain_buffers(self, layout):
        self.brain_layout = layout

    def apply_rule(self, rule):
        # WITH the live layout: a genome applied under the wrong one is the
        # whole failure, and the rule alone cannot show it.
        self.applied.append((rule, self.brain_layout))


class _RuleManager:
    def __init__(self, base):
        self.stack = [(base, 0.0)]

    def push_rule(self, rule, seed):
        self.stack.append((np.array(rule), seed))

    def pop_rule(self):
        if len(self.stack) > 1:
            self.stack.pop()
            return self.stack[-1]
        self.stack = []
        return None, None

    def get_current_rule(self):
        return self.stack[-1][0] if self.stack else None


class _MenuUI:
    """The menu's own half of a hover: ui/menu_bar.py applies the physics."""

    def __init__(self, configs):
        self.configs = configs
        self.loaded_defaults = []

    def _get_config_path(self, filename, category):
        return filename                        # the key into self.configs

    def update_physics_defaults(self, filename):
        self.loaded_defaults.append(filename)


class _Saver:
    def __init__(self, ui):
        self.ui = ui

    def load_from_file(self, path):
        return self.ui.configs.get(path)

    def apply_config(self, config, sim_state, watercolor_override=None):
        """The real one writes the physics into SimState and returns the rule.
        Only the rule matters here; the physics are ConfigSaver's own tests."""
        return config.rule


def _handler(sim, configs):
    from state import UIState

    h = object.__new__(CommandHandler)
    for a in ("field_handler", "param_lock_service", "archive", "archive_store",
              "auto_service", "imgep_driver", "multi_load_service"):
        setattr(h, a, None)
    h.sim = sim
    h.rule_manager = _RuleManager(np.zeros(sim.brain_layout.length, np.float32))
    h.ui = _MenuUI(configs)
    h.config_saver = _Saver(h.ui)
    h._pending_brain_rule = None
    h._borrow = None
    h.preview_rule_active = False
    h._preview_rule_was_pushed = False
    h.switches = []

    def _apply(layout, _ui_state):
        # App._apply_brain_layout's half of the handoff: it takes the stashed
        # genome, matched on signature. Recording it here is what proves the
        # creature arrived, since _handle_brain_layout clears the stash after.
        h.switches.append((layout,
                           h.take_pending_brain_rule(layout.signature())))

    h.apply_brain_layout = _apply

    ui_state = UIState()
    ui_state.brain.modality = "fourier"
    ui_state.brain.settings = {}
    return h, ui_state


def _hover(h, ui_state, name):
    ui_state.request_clear_preview = False
    ui_state.request_preview_config = True
    ui_state.preview_filename, ui_state.preview_category = name, "custom"
    h._handle_preview_commands(ui_state)
    ui_state.request_preview_config = False


def _unhover(h, ui_state):
    ui_state.request_clear_preview = True
    h._handle_preview_commands(ui_state)
    ui_state.request_clear_preview = False


def _click(h, ui_state, name):
    ui_state.request_load_file = True
    ui_state.load_filename, ui_state.load_category = name, "custom"
    h._handle_config_commands(ui_state)
    ui_state.request_load_file = False


@pytest.mark.parametrize("modality", ["gabor", "lenia", "mlp"])
def test_hover_then_click_lands_on_the_config_s_own_brain(modality):
    """The defect. The click used to 'finalize' a preview that had applied
    nothing, so the preset did not load and the console said it did."""
    layout = layout_for(modality, {})
    cfg = _Config(layout)
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": cfg})

    _hover(h, ui_state, "p")
    _click(h, ui_state, "p")
    h._handle_brain_layout(ui_state)

    assert len(h.switches) == 1, "the click must perform the real switch"
    got_layout, handed = h.switches[0]
    assert got_layout == layout
    assert ui_state.brain.modality == modality
    assert handed is not None, "the switch was handed no creature"
    assert np.allclose(handed, cfg.rule)


@pytest.mark.parametrize("modality", ["gabor", "lenia", "mlp"])
def test_hovering_shows_the_creature_without_switching(modality):
    """A hover fires per menu item and a switch rebuilds the archive, so the
    layout is borrowed rather than adopted - but the genome still runs."""
    layout = layout_for(modality, {})
    cfg = _Config(layout)
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": cfg})

    _hover(h, ui_state, "p")
    h._handle_brain_layout(ui_state)

    rule, live = sim.applied[-1]
    assert live == layout, "the genome reached the GPU under the wrong brain"
    assert np.allclose(rule, cfg.rule)
    assert h.switches == [], "hovering must not switch"
    assert ui_state.brain.modality == "fourier", "nor touch the Brain window"


def test_moving_off_the_menu_gives_the_user_s_brain_back():
    layout = layout_for("gabor", {})
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": _Config(layout)})

    _hover(h, ui_state, "p")
    _unhover(h, ui_state)

    assert sim.brain_layout == FOURIER
    assert h._borrow is None
    rule, live = sim.applied[-1]
    assert live == FOURIER and rule.size == FOURIER.length


def test_sliding_across_two_brains_still_restores_the_user_s_own():
    """Two borrows with no return in between. Remembering the second base
    would restore the user to a layout they never chose."""
    gabor, mlp = layout_for("gabor", {}), layout_for("mlp", {})
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"g": _Config(gabor), "m": _Config(mlp)})

    _hover(h, ui_state, "g")
    _hover(h, ui_state, "m")
    assert sim.brain_layout == mlp, "the second hover must re-point the sim"
    _unhover(h, ui_state)

    assert sim.brain_layout == FOURIER


def test_a_same_brain_config_borrows_nothing():
    """The ordinary case has to be untouched by all of this."""
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": _Config(FOURIER)})

    _hover(h, ui_state, "p")

    assert h._borrow is None
    assert sim.brain_layout == FOURIER
    _click(h, ui_state, "p")
    h._handle_brain_layout(ui_state)
    assert [lay for lay, _ in h.switches] == [FOURIER], (
        "the per-frame apply still runs; it just finds nothing to change")


def test_a_signature_this_build_cannot_rebuild_borrows_nothing():
    layout = layout_for("gabor", {})
    cfg = _Config(layout)
    cfg.brain_layout = "quantum-n4"
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": cfg})

    _hover(h, ui_state, "p")

    assert h._borrow is None
    assert sim.brain_layout == FOURIER
    assert sim.applied == [], "a rule of the wrong width must not be applied"


def test_a_click_with_no_hover_still_loads():
    """The direct path, which always worked - the fix must not disturb it."""
    layout = layout_for("mlp", {})
    cfg = _Config(layout)
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": cfg})

    _click(h, ui_state, "p")
    h._handle_brain_layout(ui_state)

    assert [lay for lay, _ in h.switches] == [layout]
    assert ui_state.brain.modality == "mlp"


# ---- the borrow has an OWNER --------------------------------------------
#
# Two things borrow a layout now: the Load menu and the archive gallery. They
# run in the same frame, the gallery second, and its teardown used to return
# whatever was borrowed - so the menu's borrow was gone before the click could
# turn it into a switch, and the preset silently did not load. Reproduced here
# by running the frame in its real order.

def _archive_frame(h, ui_state):
    """_handle_archive_preview, which process_commands runs every frame after
    the menu preview - whether or not an archive is open."""
    h.archive = None
    h._archive_preview_id = -1
    h._archive_preview_pushed = False
    h._archive_preview_physics = None
    h._archive_preview_arc = None
    h._handle_archive_preview(ui_state)


def test_the_gallery_teardown_does_not_steal_the_menu_s_borrow():
    layout = layout_for("mlp", {})
    cfg = _Config(layout)
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": cfg})

    _hover(h, ui_state, "p")
    _archive_frame(h, ui_state)              # the rest of the same frame
    assert h._borrow is not None, (
        "the archive teardown returned a layout it does not own")
    assert sim.brain_layout == layout, "the hover preview was undone"

    _click(h, ui_state, "p")
    h._handle_brain_layout(ui_state)

    assert len(h.switches) == 1, "the click did not switch"
    got_layout, handed = h.switches[0]
    assert got_layout == layout
    assert handed is not None and np.allclose(handed, cfg.rule)


def test_a_frame_of_gallery_teardown_between_hover_and_click():
    """Several frames pass while the pointer sits on the menu item."""
    layout = layout_for("gabor", {})
    cfg = _Config(layout)
    sim = _Sim(FOURIER)
    h, ui_state = _handler(sim, {"p": cfg})

    _hover(h, ui_state, "p")
    for _ in range(5):
        _archive_frame(h, ui_state)
    assert sim.brain_layout == layout

    _click(h, ui_state, "p")
    h._handle_brain_layout(ui_state)
    assert [lay for lay, _ in h.switches] == [layout]
