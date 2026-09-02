"""UI state for the vision-guided automatic tournament.

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
    # Which encoder scores the prompt. Free, unlike Explore's: nothing is
    # stored, so there are no vectors a change could invalidate.
    model_key: str = "clip-b32"
    prompt: str = ""
    # The goal is the prompt OR a picture, by `goal_kind`; never both.
    goal_kind: str = "text"
    goal_image: str = ""
    goal_distractors: bool = True
    # The square crop of the picture: where it sits on each axis and how
    # tight it is. The defaults are the centre crop.
    goal_crop_x: float = 0.5
    goal_crop_y: float = 0.5
    goal_crop_zoom: float = 1.0
    # Score the capture as particle density, with no colour term.
    grayscale: bool = False
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
    # Kind switched, picture chosen or cleared, distractors toggled.
    goal_changed: bool = False
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

    def goal_crop(self) -> tuple:
        return (self.goal_crop_x, self.goal_crop_y, self.goal_crop_zoom)

    def has_goal(self) -> bool:
        """Is there something to start a search toward, under the selected
        kind? A prompt is not a goal while Image is selected."""
        if self.goal_kind == "image":
            return bool(self.goal_image)
        return bool(self.prompt.strip())
