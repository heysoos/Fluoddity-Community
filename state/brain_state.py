"""UI state for the brain modality selector.

`request_layout_change` is a one-shot flag: the UI sets it, the orchestrator
reads and clears it, exactly as `request_reset` works. It is a request rather
than a direct call because applying a layout tears down and rebuilds the
archive, which the UI has no business doing.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BrainState:
    enabled: bool = False            # is the window open

    # Which brain the particles run. The name, not the modality object, so this
    # stays a plain serialisable dataclass like every other state group.
    modality: str = "fourier"
    # Only the settings the user has actually touched; anything absent falls
    # back to the modality's own default, so a new setting added later does not
    # need a migration.
    settings: dict = field(default_factory=dict)

    # Set by the UI when a setting that changes the PARAMETER COUNT is applied.
    # Cleared by CommandHandler.
    request_layout_change: bool = False

    # Inspector view settings. Display only - none of these touch the brain,
    # they choose which 2D slice of the 4D sensor space is drawn and how it is
    # scaled on screen.
    # Both DEFAULT to a random projection - the RGB one for the output, which
    # shows all four at once. An axis-aligned view is a special case with no
    # claim to being the neutral one: it shows the brain along one arbitrary
    # frame, and a unit acting diagonally is invisible in every one of them.
    preview_axes: int = 4        # index into services.brain_preview.AXES
    preview_channel: int = 7     # CHANNEL_RANDOM_RGB in services.brain_preview
    preview_range: float = 2.0   # half-extent of the swept plane
    preview_gain: float = 1.0    # contrast only
    # Which random plane, when preview_axes selects the random projection. Held
    # in state rather than drawn per frame: the basis is recomputed every frame,
    # so a fresh draw each time would strobe instead of showing anything. The
    # Reseed button bumps it.
    preview_seed: int = 0

    # Display only, pushed in by the orchestrator each frame.
    archive_entries: int = 0
    # The best search vector the active driver has found, or None when no search
    # has produced one. The saturation readout is its only reader, and None is
    # not the same as an unsaturated z - it means there is nothing to report.
    best_z: object = None
    # True when no rule is loaded, so each cohort is running its own generated
    # brain. Not a warning - it is the normal startup state and the reason the
    # canvas holds several different behaviours at once. The Inspector shows
    # cohort 0's and says so.
    preview_per_cohort: bool = False
    # True while the tournament grid is running, when slot 0 is TILE 0's genome
    # rather than any cohort's. The Inspector draws slot 0 either way, so
    # without this it names the wrong thing - and a grid of 16 or 64 distinct
    # brains showing one of them unlabelled reads as the whole grid.
    preview_tile0: bool = False

    # ---- which brain the Inspector draws, and the layer menu edits ---------
    #
    # There is not always ONE brain to point at: with no rule loaded every
    # cohort has its own, and under a tournament every tile does. Naming which
    # is what makes editing a layer mean something. The KIND follows the sim -
    # only one of the three exists at a time - so only the index is a choice.
    source_index: int = 0
    # How many cohorts or tiles there are, pushed in by the orchestrator.
    source_count: int = 1
    # One-shot: promote the named brain into slot 0, the collapse
    # click-to-adopt performs, as a button rather than a side effect.
    adopt_requested: bool = False

    # One-shot layer operation, as (layer index, op, argument). Read and
    # cleared by CommandHandler, which owns the rule stack.
    layer_op: object = None
    # One-shot: redraw the brain's audio weights from a fresh seed. A genome
    # edit like the layer operations, and refused in the same two states.
    reroll_audio_requested: bool = False
    # The audio channels cohort 0 hears this frame, pushed in by the
    # orchestrator; empty when silent. The Inspector evaluates at these.
    audio_live: tuple = ()
    # Which distribution each layer's Reroll draws from, by layer index. A UI
    # preference, not part of the genome, the layout or the signature - the
    # weights are what a preset saves, so there is no round trip to break.
    layer_dist: dict = field(default_factory=dict)
    # True while a hover preview holds someone else's brain in slot 0. Pushed
    # in by the orchestrator; the layer menu refuses to edit through it.
    borrow_active: bool = False
