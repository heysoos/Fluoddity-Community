"""A journal of the live configuration, for undo and redo.

Pure data: no GL, no ImGui, no file I/O. A snapshot is built by comparing
declared fields, so any code path that writes one is covered without naming
this module.

`ui.physics_params` and `services.brains` are imported inside the functions
that need them. At module scope either one closes an import cycle through
services/__init__ -> config_saver -> ui/__init__ -> ui.core.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from state import preferences_state, sim_state

# UIState attribute -> (module holding the declaration, panel name for a label)
CONTAINERS: tuple[tuple[str, object, str], ...] = (
    ("sim", sim_state, "Physics"),
    ("preferences", preferences_state, "Render"),
)


@dataclass(frozen=True)
class Snapshot:
    """One undoable state of the app."""
    fields: dict[str, dict] = field(default_factory=dict)
    rule: np.ndarray | None = None
    brain_signature: str = ""
    brain_settings: dict = field(default_factory=dict)
    label: str = ""


def capture(ui_state, rule, layout) -> Snapshot:
    """Snapshot the declared fields, the rule, and the brain it belongs to.

    `rule` and `layout` may be None - a session with no rule loaded yet.
    """
    from services.brains import settings_of

    values = {
        name: {f: copy.deepcopy(getattr(getattr(ui_state, name), f))
               for f in module.UNDOABLE_FIELDS}
        for name, module, _panel in CONTAINERS
    }
    return Snapshot(
        fields=values,
        rule=None if rule is None else np.array(rule, copy=True),
        brain_signature="" if layout is None else layout.signature(),
        brain_settings={} if layout is None else dict(settings_of(layout)),
    )


def same(a: Snapshot, b: Snapshot) -> bool:
    """Do two snapshots hold the same state? `label` is not state."""
    if a.brain_signature != b.brain_signature:
        return False
    if a.brain_settings != b.brain_settings:
        return False
    for name, module, _panel in CONTAINERS:
        av, bv = a.fields.get(name, {}), b.fields.get(name, {})
        for f in module.UNDOABLE_FIELDS:
            if av.get(f) != bv.get(f):
                return False
    if (a.rule is None) != (b.rule is None):
        return False
    return a.rule is None or np.array_equal(a.rule, b.rule)


def _field_label(name: str) -> str:
    """A physics parameter's UI label, or a titled field name."""
    from services.config_saver import PARAM_TO_LABEL

    return PARAM_TO_LABEL.get(name, name.replace("_", " ").title())


def describe(old: Snapshot, new: Snapshot) -> str:
    """A short name for what changed between two snapshots."""
    if old.brain_signature != new.brain_signature:
        return f"Brain: {new.brain_signature}"

    moved: list[tuple[str, str]] = []
    for name, module, panel in CONTAINERS:
        ov, nv = old.fields.get(name, {}), new.fields.get(name, {})
        for f in module.UNDOABLE_FIELDS:
            if ov.get(f) != nv.get(f):
                moved.append((panel, f))

    rule_moved = not (
        (old.rule is None and new.rule is None)
        or (old.rule is not None and new.rule is not None
            and np.array_equal(old.rule, new.rule)))

    if not moved:
        return "Rule" if rule_moved else "Change"
    if len(moved) == 1 and not rule_moved:
        return _field_label(moved[0][1])
    panels = {panel for panel, _f in moved}
    if rule_moved:
        panels.add("Rule")
    return panels.pop() if len(panels) == 1 else "Settings"
