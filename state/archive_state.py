"""UI state for Explore (IMGEP) mode and the archive browser.

One-shot request flags are set by the UI and cleared by the consuming side in
CommandHandler - never inside UI.get_state(), which returns the live object.

Defaults are spec 7.4. Note steps_per_gen=2000 and snapshots_per_gen=6: at 2000
steps a generation is ~2.8 s, so 6 snapshots land ~333 steps apart, which is
enough for a slow pattern to visibly change. Six snapshots is 96 CLIP images at
N=4, about 7 ms against 2800 ms of simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchiveState:
    # persistent settings
    enabled: bool = False
    running: bool = False
    show_browser: bool = False

    # which archive is loaded (mirrors preferences.archive_name, for display)
    archive_name: str = "default"
    # cached services.archive_library.list_archives(); NOT recomputed per frame -
    # it stats every thumbnail, and ImGui re-renders this tab every frame.
    archive_list: list = field(default_factory=list)

    # rollout (shared with Auto mode's widgets, own defaults)
    grid: int = 4
    steps_per_gen: int = 2000
    sim_steps_per_frame: int = 10
    snapshots_per_gen: int = 6
    physics_enabled: bool = False
    tile_mutation_enabled: bool = False
    variants_per_tile: int = 4
    tile_mutation_strength: float = 0.1

    # exploration
    sigma0: float = 0.5              # bootstrap scatter only
    sigma_expand: float = 0.15
    alpha: float = 4.0
    k: int = 10
    seed_n: int = 256
    # MEASURED 2026-08-07 over all 131 presets at 2000 steps / 6 snapshots:
    # liveness runs 0.0034 (Branes2) to 0.0792 (Sandcrabs), median 0.0241. The
    # originally guessed 0.02 sits just under that median and would have
    # rejected 43 of 131 hand-curated presets. This floor clears the quietest
    # of them by 1.7x while still rejecting a frozen canvas, which scores 0.
    liveness_min: float = 0.002
    target_rate: float = 0.15
    capacity: int = 20000
    refresh_per_gen: int = 64

    # expeditions
    expansion_between: int = 25      # 0 disables expeditions
    expedition_gens: int = 50
    expedition_sigma: float = 0.1
    latent_share: float = 0.5
    goal_order: str = "round_robin"  # or "least_matched"

    # browser view
    sort_by: str = "novelty"         # novelty | recency | liveness
    pinned_only: bool = False
    selected_entry_id: int = -1

    # persistent, not a one-shot: dismissed explicitly by the user
    warning: str = ""

    # editing buffer for the goal list
    new_goal_text: str = ""

    # modal text buffers
    new_archive_name: str = ""
    confirm_delete_text: str = ""

    # one-shot request flags, cleared by CommandHandler
    start_requested: bool = False
    pause_requested: bool = False
    reset_requested: bool = False
    add_goal_requested: bool = False
    remove_goal_index: int = -1
    move_goal_index: int = -1
    move_goal_delta: int = 0
    grid_changed: bool = False
    cancel_expedition_requested: bool = False
    chase_tile: int = -1
    pin_tile: int = -1
    export_entry_id: int = -1
    seed_entry_id: int = -1
    delete_entry_id: int = -1
    refit_projection_requested: bool = False
    switch_archive_name: str = ""
    new_archive_requested: bool = False
    clear_archive_requested: bool = False
    delete_archive_requested: bool = False
    refresh_archive_list_requested: bool = False
