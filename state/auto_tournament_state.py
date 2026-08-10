"""UI state for the CLIP-guided automatic tournament.

One-shot request flags are set by the UI and cleared by the consuming side in
CommandHandler - never inside UI.get_state(), which returns the live object.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoTournamentState:
    # persistent settings
    enabled: bool = False
    running: bool = False
    prompt: str = ""
    algorithm: str = "CMA-ES"
    grid: int = 4
    steps_per_gen: int = 300
    sim_steps_per_frame: int = 10
    snapshots_per_gen: int = 4
    sigma0: float = 0.5
    autosave_every: int = 10
    physics_enabled: bool = False
    tile_mutation_enabled: bool = False
    variants_per_tile: int = 4
    tile_mutation_strength: float = 0.1

    # persistent, not a one-shot: dismissed explicitly by the user
    warning: str = ""
    # Same, for things that went RIGHT. A save lands in a folder that is not on
    # screen, so a console print is not feedback.
    notice: str = ""

    # one-shot request flags, cleared by CommandHandler
    start_requested: bool = False
    pause_requested: bool = False
    reset_requested: bool = False
    prompt_changed: bool = False
    grid_changed: bool = False
    save_checkpoint_requested: bool = False
    # No save_best flag: the button opens the shared name dialog directly.
    save_tile_requested: int = -1
    # Set by CommandHandler, consumed by the UI: "ask for a name for this
    # tile". The round trip exists because only the orchestrator knows whether
    # Explore mode has claimed the right-click for 'chase this tile'.
    pending_save_tile: int = -1
    load_checkpoint_path: str = ""
    load_genome_path: str = ""
    download_model_requested: bool = False
