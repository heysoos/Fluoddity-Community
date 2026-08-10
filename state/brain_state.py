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
    # Both DEFAULT to the random projection - the last entry of AXES and of
    # CHANNELS. An axis-aligned view is a special case with no claim to being
    # the neutral one: it shows the brain along one arbitrary frame, and a unit
    # acting diagonally is invisible in every one of them.
    preview_axes: int = 4        # index into services.brain_preview.AXES
    preview_channel: int = 6     # index into services.brain_preview.CHANNELS
    preview_range: float = 2.0   # half-extent of the swept plane
    preview_gain: float = 1.0    # contrast only
    # Which random plane, when preview_axes selects the random projection. Held
    # in state rather than drawn per frame: the basis is recomputed every frame,
    # so a fresh draw each time would strobe instead of showing anything. The
    # Reseed button bumps it.
    preview_seed: int = 0

    # Display only, pushed in by the orchestrator each frame.
    archive_entries: int = 0
    # True when no rule is loaded, so each cohort is running its own generated
    # brain. Not a warning - it is the normal startup state and the reason the
    # canvas holds several different behaviours at once. The Inspector shows
    # cohort 0's and says so.
    preview_per_cohort: bool = False
