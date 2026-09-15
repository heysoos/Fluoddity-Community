from dataclasses import dataclass, field

from .set_list import SetList


@dataclass
class PerformState:
    """Perform mode: the picture on a second display, the controls on this one.

    `enabled` is deliberately not persisted - opening the app must not light up
    a projector, the same rule ArchiveState.enabled follows for not resuming a
    search. The remembered monitor lives in PreferencesState.
    """

    enabled: bool = False
    # Which monitor the window is actually on, "" when closed. A readout.
    active_monitor: str = ""
    notice: str = ""

    # Corner-pin calibration for the display currently being performed on.
    # None means the letterbox; `default_corners` turns that into four points.
    # Corners are GL coordinates, v = 1 at the TOP, ordered TL TR BR BL - the
    # ImGui proxy canvas flips v, and nothing else does.
    #
    # None of this is persisted. The saved copy lives in
    # PreferencesState.perform_calibrations, keyed by the display's device_key,
    # and main.py moves it in and out.
    calibrating: bool = False
    corners: tuple | None = None
    held_corner: int = -1     # which corner the pointer has, -1 for none
    reset_corners_requested: bool = False

    # Exhibition: a set list of presets on keys, and knobs on sliders. The
    # lock is not persisted, for the reason `enabled` is not: opening the app
    # must not take the keyboard away.
    exhibition_locked: bool = False
    set_list: SetList = field(default_factory=SetList.default)
    # One-shots. `slot_load` is (filename, category) for the key just pressed;
    # `recentre_knobs` is set by whatever applied a config, so a push on an
    # encoder returns its slider to the value that preset loaded with.
    slot_load: tuple | None = None
    recentre_knobs: bool = False
    midi_status: str = ""     # a readout

    def slot_for_key(self, name: str):
        return self.set_list.slot_for_key(name)
