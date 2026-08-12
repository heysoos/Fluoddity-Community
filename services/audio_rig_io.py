"""Reading and writing the audio rig.

One rig, global to the application. Not written into physics configs: preset
hopping is the workflow, and a rig that vanished on every load would be
unusable.
"""
from __future__ import annotations

import json
from pathlib import Path

from state.audio_in_state import apply_dict, to_dict
from utilities.paths import get_user_data_dir


def rig_path() -> Path:
    return get_user_data_dir() / "audio_rig.json"


def save_rig(state, path: Path | None = None) -> bool:
    target = Path(path) if path is not None else rig_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(to_dict(state), indent=2),
                          encoding="utf-8")
        return True
    except Exception:
        return False


def load_rig(state, path: Path | None = None) -> bool:
    """False for a missing or unreadable rig, leaving `state` as it was."""
    target = Path(path) if path is not None else rig_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    apply_dict(state, data)
    return True
