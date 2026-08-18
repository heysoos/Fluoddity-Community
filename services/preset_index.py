"""A stable, numbered list of physics configs, for selection over OSC.

vvvv needs to pick a preset by index, which means the index has to mean the
same thing on both sides. This builds one flat ordered list from the same
directories the Load menu scans, and hands out (filename, category) pairs.

Ordering is Core, then Advanced, then Custom, alphabetical within each. Custom
is last on purpose: it is the only category that grows during a session, so
saving a new config appends rather than renumbering everything the patch
already refers to.
"""
from pathlib import Path

# Matches the Load menu's category order (ui/config_browser.py).
CATEGORIES = ("Core", "Advanced", "Custom")


class PresetIndex:
    """Enumerates physics configs in a stable order.

    Args:
        app_configs_dir: bundled configs, containing Core/ and Advanced/.
        user_configs_dir: user-saved configs, listed as Custom.
    """

    def __init__(self, app_configs_dir: Path, user_configs_dir: Path):
        self.app_configs_dir = Path(app_configs_dir)
        self.user_configs_dir = Path(user_configs_dir)
        self.entries: list[tuple[str, str]] = []   # (filename, category)
        self.refresh()

    def refresh(self) -> int:
        """Rescan the directories. Returns the number of presets found."""
        entries: list[tuple[str, str]] = []
        for category in CATEGORIES:
            if category == "Custom":
                directory = self.user_configs_dir
            else:
                directory = self.app_configs_dir / category
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                entries.append((path.stem, category))
        self.entries = entries
        return len(entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.entries]

    def name_list(self) -> str:
        """Comma-separated names, for one OSC string payload.

        Commas are stripped from names rather than escaped -- a comma in a
        config filename would silently split one entry into two on the vvvv
        side, shifting every index after it.
        """
        return ",".join(name.replace(",", " ") for name in self.names)

    def get(self, index) -> tuple[str, str] | None:
        """Look up one preset. Returns (filename, category), or None if the
        index is out of range -- callers should treat that as 'do nothing'
        rather than clamping, so a stray value cannot load a random preset."""
        try:
            i = int(round(float(index)))
        except (TypeError, ValueError):
            return None
        if 0 <= i < len(self.entries):
            return self.entries[i]
        return None

    def index_of(self, filename: str) -> int:
        """Index of a preset by filename, or -1. Used to report the current
        selection back to vvvv after a load from Fluoddity's own UI."""
        for i, (name, _) in enumerate(self.entries):
            if name == filename:
                return i
        return -1
