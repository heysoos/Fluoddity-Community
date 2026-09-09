"""File > Load previews on hover, unless the setting says otherwise.

Performing to an audience means scrolling the preset list without the picture
moving, so the hover is skipped OUTRIGHT rather than applied and undone - a
preview that is applied and reverted is still a flash on the projector.

The 450-line menu body cannot be driven from a test, which is why the decision
lives in `_handle_load_hover` and this drives that.
"""
from __future__ import annotations

from state.preferences_state import PreferencesState
from ui.menu_bar import MenuBarMixin


class _Sim:
    watercolor_mode = False


class _State:
    def __init__(self):
        self.preferences = PreferencesState()
        self.sim = _Sim()


class _Config:
    force_field_strength = 1.5
    strafe_field_strength = 0.5


class Harness(MenuBarMixin):
    """Everything `_handle_load_hover` touches, and nothing else."""

    def __init__(self):
        self.state = _State()
        self.param_lock_service = None
        self.currently_previewing = None
        self.currently_previewing_category = None
        self.cached_configs = {"Core/reef": _Config()}
        self.cached_config = '{"cached": true}'
        self._cached_field_strengths = (0.1, 0.2)
        self._request_preview_config = False
        self._request_clear_preview = False
        self._preview_filename = None
        self._preview_category = None
        self.applied = []
        self.restored = []

    def _apply_config_locked(self, config, watercolor_override=None):
        self.applied.append(config)

    def _load_from_string_locked(self, s, watercolor_override=None):
        self.restored.append(s)


def test_a_hover_previews_when_the_setting_is_on():
    h = Harness()
    h._handle_load_hover(("reef", "Core"), False)
    assert h.applied == [h.cached_configs["Core/reef"]]
    assert h._request_preview_config
    assert h.currently_previewing == "reef"


def test_a_hover_applies_nothing_when_the_setting_is_off():
    h = Harness()
    h.state.preferences.load_preview_on_hover = False
    h._handle_load_hover(("reef", "Core"), False)
    assert h.applied == [], "the preset was applied to the running sim"
    assert not h._request_preview_config
    assert h.currently_previewing is None


def test_turning_it_off_mid_menu_hands_the_preview_back_once():
    h = Harness()
    h._handle_load_hover(("reef", "Core"), False)
    h.state.preferences.load_preview_on_hover = False

    h._handle_load_hover(("other", "Core"), False)
    assert h._request_clear_preview
    assert h.restored == [h.cached_config]
    assert h.currently_previewing is None

    h._request_clear_preview = False
    h._handle_load_hover(("other", "Core"), False)
    assert not h._request_clear_preview, "it kept handing back a gone preview"
    assert h.restored == [h.cached_config]


def test_leaving_every_item_reverts_to_the_state_the_menu_opened_on():
    h = Harness()
    h._handle_load_hover(("reef", "Core"), False)
    h._handle_load_hover(None, False)
    assert h.restored == [h.cached_config]
    assert h.state.preferences.force_field_strength == 0.1
    assert h.currently_previewing is None
