from dataclasses import dataclass


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
