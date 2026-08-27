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
