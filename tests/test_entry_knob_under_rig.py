"""An archive entry of the DEAF brain, loaded under a rig, lands widened.

The rig re-imposes its input count every frame, so the sim runs
fourier-n10+c2+aK while the archive's Fourier entries are fourier-n10+c2. That
makes every one of them "foreign" to the running layout, and adopting one as
a foreign brain switches to the deaf layout - which the rig then widens back
in the next frame. The entry has to be what lands, exactly as a same-brain
deaf preset does through File > Load.
"""
import numpy as np

from services.brains import default_layout
from services.brains.layout_moves import grow_inputs, transfer_audio_inputs
from tests.test_brain_audio_inputs_apply import _menu_handler, _rule, _ui


class _Entry:
    def __init__(self, i, layout):
        self.id = i
        self.layout = layout
        self.novelty = 0.5
        self.liveness = 0.02
        self.pinned = False
        self.source = "expansion"
        self.goal = ""
        self.ts = float(i)
        self.spec = "brain:80"
        self.run_id = ""


class _Archive:
    def __init__(self, signature, layouts, brains):
        self.signature = signature
        self.entries = [_Entry(i, lay) for i, lay in enumerate(layouts)]
        self._brains = brains
        self.retargets = []

    def layout_at(self, i):
        return self.entries[i].layout or self.signature

    def is_native(self, i):
        return self.layout_at(i) == self.signature

    def brain_at(self, i):
        return self._brains[i]

    def __len__(self):
        return len(self.entries)


def _frame(h, ui_state):
    """The frame loop's order: the brain layout, then the archive one-shots."""
    h._handle_brain_layout(ui_state)
    h._handle_archive_preview(ui_state)


def test_a_deaf_archive_entry_lands_widened_under_the_rig():
    base = default_layout()
    wide = grow_inputs(base, 2)
    old = _rule(wide, seed=1)
    app, h = _menu_handler(wide, old)
    sim = app.sim
    # The real sim keeps slot 0; _live_rule reads it to carry a rule across.
    sim.slot0_params = lambda: app.applied[-1] if app.applied else old
    ui_state = _ui()
    ui_state.audio.audio_inputs = 2
    ui_state.audio.audio_seed = 0.3
    ui_state.brain.settings = {"audio_inputs": 2}

    entry_rule = _rule(base, seed=7)
    h.archive = _Archive(wide.signature(), [base.signature()], [entry_rule])
    h._knob_entry_row = -1
    h._archive_preview_id = -1
    h._archive_preview_pushed = False
    h._archive_preview_physics = None
    h._archive_preview_arc = None
    h._run_config_cache = {}
    h._run_physics_written = ""
    ui_state.archive.live_preview = False

    ui_state.archive.step_entry = 1
    for _ in range(3):
        _frame(h, ui_state)

    want = transfer_audio_inputs(entry_rule, base, wide, 0.3)
    assert sim.brain_layout == wide
    np.testing.assert_array_equal(app.applied[-1], want)
    np.testing.assert_array_equal(h.rule_manager.get_current_rule(), want)
    # The same brain: widened in place, never switched to and back.
    assert "realloc" not in sim.calls
