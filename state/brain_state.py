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

    # Display only, pushed in by the orchestrator each frame.
    archive_entries: int = 0
