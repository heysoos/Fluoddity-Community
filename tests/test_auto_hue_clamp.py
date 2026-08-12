"""An automatic mode caps the hue gain, whatever preset seeded the run.

Hue is a readout of the axial force term through a modulo, so at a high gain a
mutation far too small to change the pattern slides the colour through whole
revolutions - and CLIP reads a hue rotation as a new creature. Roughly half the
preset library ships hue_sensitivity 0.5, so without the cap how often a
hue-only mutation clears min_separation depends on which preset the run started
from. See CLAUDE.md.
"""
import pytest

from main import AUTO_HUE_MAX, clamp_auto_hue, put_back_auto_overrides
from state.sim_state import SimState


class FakeUIState:
    def __init__(self, hue, auto=False, explore=False):
        self.sim = SimState()
        self.sim.hue_sensitivity = hue
        self.auto_tournament = type("A", (), {"enabled": auto})()
        self.archive = type("B", (), {"enabled": explore})()
        self.preferences = type("P", (), {
            "canvas_aspect_ratio": "1:1", "speedmult": 1, "motion_blur": False,
        })()
        self.request_world_size_change = False


# -- the cap -----------------------------------------------------------------

@pytest.mark.parametrize("mode", ["auto", "explore"])
def test_a_high_gain_preset_is_capped(mode):
    """Butterflies, Growth, Zipper and a dozen others ship 0.5."""
    ui = FakeUIState(0.5, auto=(mode == "auto"), explore=(mode == "explore"))
    clamp_auto_hue(ui)
    assert ui.sim.hue_sensitivity == AUTO_HUE_MAX


@pytest.mark.parametrize("mode", ["auto", "explore"])
def test_a_gain_already_below_the_cap_is_left_alone(mode):
    """_Default ships 0.09 and is already in the good range - the cap is a
    ceiling, never a target."""
    ui = FakeUIState(0.09, auto=(mode == "auto"), explore=(mode == "explore"))
    clamp_auto_hue(ui)
    assert ui.sim.hue_sensitivity == pytest.approx(0.09)


def test_the_cap_does_not_apply_outside_an_automatic_mode():
    """Manual tournament and ordinary single-sim runs are what the user sees;
    nothing is being scored, so nothing needs capping."""
    ui = FakeUIState(0.5)
    clamp_auto_hue(ui)
    assert ui.sim.hue_sensitivity == pytest.approx(0.5)


def test_the_cap_reapplies_after_a_preset_is_loaded_mid_run():
    """The reason this runs every frame rather than on the mode-enable edge:
    loading a config mid-run writes that preset's own gain into SimState."""
    ui = FakeUIState(0.09, auto=True)
    clamp_auto_hue(ui)
    ui.sim.hue_sensitivity = 0.5          # a preset load lands here
    clamp_auto_hue(ui)
    assert ui.sim.hue_sensitivity == AUTO_HUE_MAX


def test_the_cap_is_in_the_range_the_measurement_supports():
    """Above ~0.15 the nuisance climbs steeply; below ~0.08 between-genome hue
    variety starts collapsing. See CLAUDE.md."""
    assert 0.08 <= AUTO_HUE_MAX <= 0.15


# -- putting it back ---------------------------------------------------------

def test_the_users_gain_comes_back_when_the_mode_ends():
    ui = FakeUIState(AUTO_HUE_MAX, auto=True)
    put_back_auto_overrides(ui, None, None, None, prev_hue=0.5)
    assert ui.sim.hue_sensitivity == pytest.approx(0.5)


def test_restoring_without_a_captured_gain_touches_nothing():
    """Nothing was captured yet, so there is nothing to put back - and
    ui_state may not even carry a sim in that case."""
    ui = FakeUIState(AUTO_HUE_MAX, auto=True)
    put_back_auto_overrides(ui, None, None, None)
    assert ui.sim.hue_sensitivity == pytest.approx(AUTO_HUE_MAX)
