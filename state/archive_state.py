"""UI state for Explore (IMGEP) mode and the archive browser.

One-shot request flags are set by the UI and cleared by the consuming side in
CommandHandler - never inside UI.get_state(), which returns the live object.

What each setting does: docs/imgep.md. Why a default is what it is: CLAUDE.md.
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

    # Which encoder this archive's vectors are in. Read back from encoder.json,
    # never chosen here: the choice is made once, in the New Archive modal.
    # See CLAUDE.md.
    encoder_key: str = "clip-b32"
    # View buffers, deliberately NOT persisted: the count is what the browser
    # prints, and the rows are read off disk when the history section is
    # opened.
    archive_entry_count: int = 0
    history_rows: list = field(default_factory=list)
    show_history: bool = False
    # One-shot: the orchestrator reads the log and fills history_rows.
    request_history_reload: bool = False

    # rollout (shared with Auto mode's widgets, own defaults)
    grid: int = 4
    steps_per_gen: int = 2000
    sim_steps_per_frame: int = 10
    snapshots_per_gen: int = 6
    # Sub-crops averaged into each tile's embedding; 1 is the raw frame and is
    # fully position-dependent. See VisionScorer.embed_mean.
    n_views: int = 3
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
    # A ceiling on each layout's bootstrap, in generations, because seed_n is a
    # native COUNT and separation is pooled across every layout - so a layout
    # born into a full archive takes ever longer to reach the same count.
    # Bootstrap ends at whichever of the two comes first.
    bootstrap_gens: int = 30
    # A FLOOR on the bulk, deliberately far below what looks reasonable - a
    # frozen canvas scores 0, and real presets run much lower than they seem.
    # Re-run tools.calibrate_imgep --liveness before changing it.
    liveness_min: float = 0.002
    # Capacity is the long-run cap; min_separation is what stops a converged
    # expedition filling it with its own endpoint. 0 disables it.
    capacity: int = 20000
    min_separation: float = 0.02
    # Generations for a full novelty sweep, NOT entries per generation: the
    # bound that matters is how stale novelty may get, and it has to hold as
    # the archive grows. See ImgepDriver._refresh_count.
    refresh_sweep_gens: int = 10

    # expeditions
    expansion_between: int = 25      # 0 disables expeditions
    expedition_gens: int = 50
    expedition_sigma: float = 0.1
    latent_share: float = 0.5
    # Three kinds of expedition goal: novelty, latent, and text with
    # whatever is left. See ImgepDriver._draw_goal.
    novelty_share: float = 0.25
    goal_order: str = "round_robin"  # or "least_matched"
    # How many archive entries are effectively in the running as an expedition
    # seed. A BAND, not a target - the spread across goals is signal, and only
    # the degenerate ends need stopping. See services/novelty.banded_alpha.
    seed_ess_min: float = 8.0
    seed_ess_max: float = 512.0

    # brain layout search. OFF: opening the app must never start changing
    # brain under anyone. See services/brains/layout_moves.py.
    layout_search: bool = False
    layout_move_chance: float = 0.2
    # ONE convention for all three: 0 is "the limit this build already
    # allows". A literal bound of zero would forbid every layout, so it can
    # never mean itself - and it is what lets these defaults be stated here
    # without importing services, which nothing else in state/ does.
    layout_max_depth: int = 0
    layout_max_width: int = 0
    layout_max_floats: int = 0
    # Comma-separated registry keys, so a fifth modality needs no new field.
    # Empty is the running modality alone: a jump carries no genome, and
    # opting into a restart should be a decision.
    layout_modalities: str = ""

    # browser view
    sort_by: str = "novelty"         # see GALLERY_SORTS
    sort_desc: bool = True
    pinned_only: bool = False
    selected_entry_id: int = -1
    # Gallery tile size in pixels, and the list row height below the threshold.
    # See ui/archive_window.py for the threshold and why the grid has a floor.
    thumb_size: int = 96

    # Live preview: run an archive entry in the single sim, the way hovering
    # File > Load previews a preset. Only outside tournament mode, where the
    # canvas is one simulation rather than a grid of them.
    live_preview: bool = False
    # The brain this archive was last SEARCHED under. A readout on the way out
    # and a command on the way in: the app remembers which archive you were in,
    # and without this it does not remember which brain you were working on in
    # it - so an archive of one modality reopens under another with no native
    # entries. Empty means not recorded, which is every archive written before
    # this one.
    layout_signature: str = ""
    # CONTINUOUS, not a one-shot: the entry under the pointer right now, or -1.
    # CommandHandler compares it against what it is already showing, so the UI
    # does not have to track transitions itself.
    preview_entry_id: int = -1

    # Map view. Pure display state, so the UI writes it directly - there is no
    # command for "the user scrolled". Centre is in normalised projection
    # units, where the whole archive spans [0, 1].
    map_zoom: float = 1.0
    map_center_x: float = 0.5
    map_center_y: float = 0.5
    # How the map draws. Every default is the historical behaviour: these are
    # additions, and the plain scatter coloured by regime stays one combo away.
    # See services/map_view for what each mode means and why they exist.
    map_color_by: str = "source"     # source | novelty | liveness
    map_filter: str = "all"          # all | recent | novel | kept | goal | source
    map_render: str = "points"       # points | density | points+density
    # Which engine lays the map out. See services/map_layout.py; "pca" is the
    # default and the fallback, so a build without umap-learn is unchanged.
    map_layout: str = "pca"          # pca | umap
    # The atlas REPLACES the scatter while it is on: a picture and a dot
    # competing for the same entry is what made it flicker on every pan.
    # On by default - a map of pictures is what the map is FOR, and the dots
    # are the fallback.
    map_thumbs: bool = True
    map_thumb_px: int = 32
    # Low, because a large archive spans few generations: a default that shows
    # everything is a filter that filters nothing.
    map_recent_gens: int = 50        # for filter "recent"
    map_novel_pct: int = 25          # for filter "novel", top N%
    map_filter_goal: str = ""        # for filter "goal"
    map_filter_source: str = ""      # for filter "source"

    # persistent, not a one-shot: dismissed explicitly by the user
    warning: str = ""
    # Same, for things that went RIGHT. Export writes a file somewhere the user
    # cannot see, and a console print is not feedback in a GUI.
    notice: str = ""

    # editing buffer for the goal list
    new_goal_text: str = ""

    # modal buffers. The encoder rides with the name because both are the new
    # archive's identity, fixed at creation.
    new_archive_name: str = ""
    new_archive_encoder: str = "clip-b32"
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
    # No export flag: "Save as config" opens the shared name dialog, which
    # comes back as UIState.request_save_file with kind "archive_entry".
    seed_entry_id: int = -1
    delete_entry_id: int = -1
    # Commit what is being previewed, so it survives the pointer leaving.
    load_entry_id: int = -1
    refit_projection_requested: bool = False
    switch_archive_name: str = ""
    new_archive_requested: bool = False
    clear_archive_requested: bool = False
    delete_archive_requested: bool = False
    refresh_archive_list_requested: bool = False
    # Extras > Archive Browser. Its own flag rather than watching show_browser,
    # because opening from the menu may need the archive LOADED first and that
    # is the orchestrator's job, not the UI's.
    open_browser_requested: bool = False
    # Fetch this archive's own encoder. Its own flag rather than Auto's,
    # because the key comes from encoder.json and no combo here can change it.
    download_model_requested: bool = False

    def to_settings(self) -> dict:
        """The tuning knobs, for `settings.json` in the archive's own folder."""
        return {name: getattr(self, name) for name in PERSISTED_FIELDS}

    def apply_settings(self, data: dict) -> list[str]:
        """Overwrite the persisted fields from `data`; -> the names applied.

        Anything missing keeps its current value, so a settings.json written
        before a field existed still loads. A value of the wrong type is
        skipped rather than raising: this runs during an archive switch, and a
        hand-edited or truncated file must not take the app down.
        """
        applied = []
        # A RECORD of the archive, not a setting to carry over. Everything else
        # keeps its current value when absent - which is what lets a brand new
        # archive inherit the settings you were just using - but inheriting
        # this one would tell an archive it was searched under a brain it has
        # never held, and the restore would then move the user onto a layout
        # with no native rows.
        if "layout_signature" not in data:
            self.layout_signature = ""
        for name in PERSISTED_FIELDS:
            if name not in data:
                continue
            want = type(getattr(self, name))
            value = data[name]
            try:
                # bool BEFORE int, because bool is a subclass of int: checked
                # the other way round, every checkbox would come back as 0/1.
                if want is bool:
                    value = bool(value)
                elif want is int:
                    value = int(value)
                elif want is float:
                    value = float(value)
                elif want is str:
                    value = str(value)
                else:
                    continue
            except (TypeError, ValueError):
                continue
            setattr(self, name, value)
            applied.append(name)
        return applied


# What follows an archive from one session to the next, written to
# `settings.json` beside its `goals.json`.
#
# An explicit ALLOWLIST, not "every field that is not a flag". Two thirds of
# ArchiveState is one-shot commands, transient view state and text buffers, and
# persisting `start_requested` or `delete_entry_id` would replay a command on
# load. A new field is therefore NOT persisted until it is named here, which is
# the safe default for a class whose main job is carrying one-shot flags.
#
# `enabled` is deliberately absent: opening the app should not resume a search
# because one was running when it closed. `archive_name` is absent because it
# lives in preferences and identifies which of these files to read in the first
# place.
PERSISTED_FIELDS = (
    # the encoder this archive's vectors are in
    "encoder_key", "show_history",
    # rollout
    "grid", "steps_per_gen", "sim_steps_per_frame", "snapshots_per_gen",
    "n_views", "physics_enabled", "tile_mutation_enabled",
    "variants_per_tile", "tile_mutation_strength",
    # exploration
    "sigma0", "sigma_expand", "alpha", "k", "seed_n", "bootstrap_gens",
    "liveness_min",
    "capacity", "min_separation", "refresh_sweep_gens",
    # expeditions
    "expansion_between", "expedition_gens", "expedition_sigma",
    "latent_share", "novelty_share", "goal_order",
    "seed_ess_min", "seed_ess_max",
    # brain layout search
    "layout_search", "layout_move_chance", "layout_max_depth",
    "layout_max_width", "layout_max_floats", "layout_modalities",
    # browser and map view: per-archive, and restoring where you were looking
    # is most of what "open it in its last state" means once the archive is
    # large enough that the map does not fit on screen.
    "show_browser", "sort_by", "sort_desc", "pinned_only", "live_preview",
    "layout_signature", "thumb_size",
    "map_zoom", "map_center_x", "map_center_y",
    "map_color_by", "map_filter", "map_render",
    "map_layout", "map_thumbs", "map_thumb_px",
    "map_recent_gens", "map_novel_pct", "map_filter_goal", "map_filter_source",
)
