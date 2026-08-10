"""State for the interactive tournament feature."""
from dataclasses import dataclass


@dataclass
class TournamentState:
    """UI-facing tournament state + one-shot command flags."""
    # Persistent controls
    enabled: bool = False
    mutation_strength: float = 0.15   # 0.0 - 0.5
    inject_randoms: int = 1           # 0 - 4 fresh randoms per generation
    crossover_enabled: bool = False

    # One-shot flags (reset after UI.get_state each frame)
    clicked_tile: int = -1            # -1 = no click this frame; else tile 0..15
    next_gen_requested: bool = False
    undo_requested: bool = False
    reset_requested: bool = False
    # No save flag: Save Selected opens the shared name dialog, which comes
    # back as UIState.request_save_file with kind "tournament_tile".

    # Persistent, dismissed by the user: a save lands in a folder that is not
    # on screen, so a console print is not feedback.
    notice: str = ""
