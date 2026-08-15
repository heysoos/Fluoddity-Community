"""Reading and writing the audio rig.

One rig, global to the application. Not written into physics configs: preset
hopping is the workflow, and a rig that vanished on every load would be
unusable.
"""
from __future__ import annotations

import json
from pathlib import Path

from services.save_targets import safe_stem
from state.audio_in_state import apply_dict, to_dict
from utilities.paths import get_user_data_dir


def rig_path() -> Path:
    return get_user_data_dir() / "audio_rig.json"


def rigs_dir() -> Path:
    """Where named rigs live.

    Deliberately not the user configs folder: everything there appears under
    File > Load > Custom, and a rig is not a physics config.
    """
    return get_user_data_dir() / "audio_rigs"


def list_rigs() -> list[str]:
    """The names on disk, sorted. Empty for a folder that is not there yet."""
    try:
        return sorted(p.stem for p in rigs_dir().glob("*.json"))
    except OSError:
        return []


def preset_path(name: str) -> Path:
    """The file one named rig lives in.

    `safe_stem` again rather than trusting the caller: a separator in the name
    would otherwise write outside the folder.
    """
    return rigs_dir() / f"{safe_stem(name)}.json"


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
