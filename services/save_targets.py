"""What a "Save" is about, for every place in the app that can save one.

A save subject is (kind, arg), and everything else - the suggested name, the
label the dialog shows, the files it will touch - is derived here. Pure
functions over plain data, so the naming and collision rules are testable
without an ImGui context or a GPU.

The one rule that governs all of it: a tournament save must land where an
ordinary File > Save lands, under the same name the user typed, and go through
the same overwrite confirmation. A saved genome is an ordinary config file (see
services/genome_io) - there is no second format and no second folder to look in.
"""
from __future__ import annotations

# kind constants. "" is the ordinary File > Save of the live configuration and
# must keep behaving exactly as it did.
CONFIG = ""
TOURNAMENT_TILE = "tournament_tile"
AUTO_BEST = "auto_best"
AUTO_TILE = "auto_tile"
ARCHIVE_ENTRY = "archive_entry"

_ILLEGAL = '<>:"/\\|?*'


def safe_stem(name: str) -> str:
    """A filename stem that cannot escape the configs directory.

    Separators are stripped so a typed name cannot write outside the folder.
    Trailing dots and spaces are stripped because Windows silently drops them,
    which would make the existence check disagree with what lands on disk.
    """
    cleaned = "".join("_" if c in _ILLEGAL else c for c in str(name))
    cleaned = "".join(c for c in cleaned if c.isprintable())
    return cleaned.strip().rstrip(". ").strip()


def suggested_name(kind: str, arg: int = -1, generation: int = 0,
                   tiles=(), current_project: str = "") -> str:
    """The name the dialog opens with. Always editable."""
    tiles = tuple(tiles)
    if kind == TOURNAMENT_TILE:
        return f"tile{tiles[0]}" if len(tiles) == 1 else "tournament"
    if kind == AUTO_BEST:
        return f"best_gen{int(generation):04d}"
    if kind == AUTO_TILE:
        return f"tile{int(arg)}_gen{int(generation):04d}"
    if kind == ARCHIVE_ENTRY:
        return f"archive_{int(arg):06d}"
    return safe_stem(current_project)


def subject_label(kind: str, arg: int = -1, tiles=()) -> str:
    """One line naming what is about to be written, shown in the dialog.

    "Save Config" over an unlabelled text box is the same dialog whether it is
    about the live sliders or about tile 11 of a tournament, and those produce
    very different files.
    """
    tiles = tuple(tiles)
    if kind == TOURNAMENT_TILE:
        if len(tiles) == 1:
            return f"Tournament tile {tiles[0]}"
        return f"{len(tiles)} tournament tiles: {', '.join(str(t) for t in tiles)}"
    if kind == AUTO_BEST:
        return "Auto mode's best genome so far"
    if kind == AUTO_TILE:
        return f"Auto mode tile {int(arg)}"
    if kind == ARCHIVE_ENTRY:
        return f"Archive entry #{int(arg)}"
    return "Current physics and rule"


def target_stems(kind: str, base: str, tiles=()) -> list[str]:
    """Every file stem this save will write, in order.

    Only one subject writes more than one file: several selected tournament
    tiles. They are suffixed rather than sharing a name, since one shared name
    would clobber itself across tiles within a single click.
    """
    stem = safe_stem(base)
    if not stem:
        return []
    tiles = tuple(tiles)
    if kind == TOURNAMENT_TILE and len(tiles) > 1:
        return [f"{stem}_tile{t}" for t in tiles]
    return [stem]
