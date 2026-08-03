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
    save_requested: bool = False
