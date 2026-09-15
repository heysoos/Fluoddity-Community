"""Set lists on disk: Documents/Fluoddity/setlists/<name>.json."""
import json
from pathlib import Path

from state.set_list import SetList
from utilities.paths import get_set_lists_dir
from services.save_targets import safe_stem


def set_list_path(name: str, root: Path | None = None) -> Path:
    root = Path(root) if root is not None else get_set_lists_dir()
    return root / f"{safe_stem(name)}.json"


def list_set_lists(root: Path | None = None) -> list[str]:
    root = Path(root) if root is not None else get_set_lists_dir()
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.json"))


def save_set_list(sl: SetList, root: Path | None = None) -> Path:
    """Write `sl` under its name. Raises ValueError on an empty name."""
    if not safe_stem(sl.name):
        raise ValueError("A set list needs a name.")
    path = set_list_path(sl.name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sl.to_dict(), indent=2), encoding="utf-8")
    return path


def load_set_list(name: str, root: Path | None = None) -> SetList | None:
    """The set list called `name`, or None when there is no such file."""
    path = set_list_path(name, root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    sl = SetList.from_dict(data)
    sl.name = path.stem
    return sl
