"""Named archive directories on disk.

Which archive is loaded IS an experimental variable: novelty is measured
against the archive, so its contents are the search's memory. This module owns
the directory-level half of that - naming, listing, creating, emptying and
deleting - and nothing else.

It knows about folders, not about Archive, ImgepDriver or GL, which is what
makes it testable against tmp_path with no GPU and no CLIP. It also imports
nothing beyond the standard library, so it is safe to import at module scope
anywhere, including in ui/ where onnxruntime must never be pulled in.

Nothing here raises for a user error. Every operation returns a Result the
orchestrator can put in front of the user; a disk problem must never stop the
search.
"""
from __future__ import annotations

import os
import shutil
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MAX_NAME = 64
CLEARED_MARK = ".cleared-"

# Windows-reserved punctuation plus both separators. Removing these is what
# guarantees a name cannot change WHICH directory a path refers to.
_FORBIDDEN = set('\\/:*?"<>|')


@dataclass
class Result:
    ok: bool
    name: str = ""
    message: str = ""      # user-facing; empty when ok


def safe_name(raw) -> str:
    """A directory name that cannot redirect a path.

    Case and unicode survive - 'Dense Trails' and 'gustoy' in Cyrillic are both
    fine names. What is removed is separators, drive-letter colons, the
    Windows-reserved punctuation, control characters, and leading or trailing
    dots and spaces: Windows silently trims trailing dots and spaces from
    directory names, so a name ending in one would not round-trip.

    -> "" for anything that sanitises away to nothing. Callers treat "" as a
    refusal rather than inventing a name.
    """
    if not isinstance(raw, str):
        return ""
    kept = [
        ch for ch in raw
        if ch not in _FORBIDDEN and unicodedata.category(ch)[0] != "C"
    ]
    name = "".join(kept).strip(" .")
    # Truncation can expose a new trailing dot, so strip again after cutting.
    return name[:MAX_NAME].strip(" .")


def resolve(root, name: str) -> Path:
    """The directory for `name` under `root`, guaranteed to be inside it.

    Raises ValueError if it is not. Unreachable through the UI, because
    safe_name has already run - but delete() rmtree's what this returns, so a
    future caller that skips safe_name must fail loudly rather than quietly
    outside archives/.
    """
    base = Path(root).resolve()
    safe = safe_name(name)
    if not safe:
        raise ValueError(f"not a usable archive name: {name!r}")
    path = (base / safe).resolve()
    if base not in path.parents:
        raise ValueError(f"{path} escapes {base}")
    return path


def list_archives(root) -> list[dict]:
    """One dict per archive directory, sorted by name.

    NOT cheap enough to call per frame: an archive with 4,800 thumbnails is
    4,800 stat calls, and ImGui re-renders every frame. The orchestrator calls
    this on demand and caches the result on ArchiveState.archive_list.
    """
    base = Path(root)
    out: list[dict] = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return out                      # no root yet is an empty list, not an error
    for child in children:
        try:
            if not child.is_dir() or CLEARED_MARK in child.name:
                continue
        except OSError:
            continue                    # one bad folder must not blank the dropdown
        out.append({
            "name": child.name,
            "entries": _count_entries(child),
            "mtime": _mtime(child),
            "size_mb": _size_mb(child),
        })
    return out


def _count_entries(path: Path) -> int:
    """How many entries a load would actually yield, across every brain layout.

    Entries live in <archive>/<layout-signature>/, so one named archive can hold
    several layouts side by side. This reports what the FOLDER holds - the live
    count for the layout currently loaded comes from the Archive itself. Looking
    only at the top level reports every migrated archive as empty.
    """
    subs = []
    try:
        subs = [p for p in path.iterdir() if p.is_dir() and p.name != "thumbs"]
    except OSError:
        pass
    if subs:
        total = sum(_count_one(p) for p in subs)
        if total:
            return total
    return _count_one(path)             # not yet migrated


def _count_one(path: Path) -> int:
    """vectors.npz is the honest source. index.jsonl is APPEND-ONLY - Archive
    never rewrites it - so after an eviction or a user delete the row survives
    while the id is gone from the arrays, and load_from_store keeps only the
    intersection. Counting lines would have this number drift above the
    browser's, without bound, once an archive reaches capacity.

    The index is the fallback for an archive with no arrays yet: vectors.npz is
    only rewritten every 200 admissions, so a young one has rows and no file.
    """
    vectors = path / "vectors.npz"
    if vectors.is_file():
        try:
            # Local: this module is imported into ui/, and stays importable
            # before numpy would otherwise be needed.
            import numpy as np

            with np.load(vectors, allow_pickle=False) as z:
                return int(z["ids"].shape[0])
        except (OSError, ValueError, KeyError):
            pass                        # quarantined or truncated
    return _count_index_rows(path)


def _count_index_rows(path: Path) -> int:
    n = 0
    try:
        with open(path / "index.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n


def _size_mb(path: Path) -> float:
    """Bytes under `path`, in MB.

    os.scandir, NOT Path.rglob: rglob pays a separate stat() syscall per entry,
    where a DirEntry already carries its type. This number is displayed once,
    per archive, when Explore mode opens - see CLAUDE.md.
    """
    total = 0
    stack = [str(path)]
    while stack:
        try:
            it = os.scandir(stack.pop())
        except OSError:
            continue        # one unreadable subdirectory must not zero the total
        with it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    return total / (1024.0 * 1024.0)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ---- operations that change the disk ------------------------------------

def create(root, name: str, encoder: str | None = None) -> Result:
    """Make an empty archive. `encoder` names its embedding space FOREVER.

    Pinned here, at the one moment an archive holds nothing: after this the
    vectors are in that space and no other, so every later control over it is
    a readout rather than a choice.
    """
    from services.archive_io import pin_encoder
    from services.vision_models import DEFAULT_KEY

    base = Path(root)
    safe = safe_name(name)
    if not safe:
        return Result(False, "", "That name has no usable characters.")
    path = resolve(base, safe)
    if path.exists():
        return Result(False, safe, f"An archive named '{safe}' already exists.")
    try:
        (path / "thumbs").mkdir(parents=True)
    except OSError as exc:
        return Result(False, safe, f"Could not create '{safe}': {exc}")
    pin_encoder(path, encoder or DEFAULT_KEY)
    return Result(True, safe)


def clear(root, name: str) -> Result:
    """Empty an archive WITHOUT deleting anything.

    The contents move to <name>.cleared-<unix-ts> and the live directory comes
    back empty. An archive is hours of exploration; 'empty it' should be one
    rename away from being undone, not an rmtree. The snapshots are never
    garbage-collected - they are the user's data, and the app does not get to
    decide when their undo expires.
    """
    base = Path(root)
    safe = safe_name(name)
    if not safe:
        return Result(False, "", "That name has no usable characters.")
    path = resolve(base, safe)
    if not path.is_dir():
        return Result(False, safe, f"'{safe}' is no longer on disk.")

    stamp = int(time.time())
    aside = base / f"{safe}{CLEARED_MARK}{stamp}"
    n = 1
    while aside.exists():
        # Two clicks inside one second must not rename onto each other.
        n += 1
        aside = base / f"{safe}{CLEARED_MARK}{stamp}-{n}"
    try:
        path.rename(aside)
        (path / "thumbs").mkdir(parents=True)
    except OSError as exc:
        return Result(False, safe, f"Could not empty '{safe}': {exc}")
    return Result(True, safe)


def delete(root, name: str) -> Result:
    """The only irreversible operation here. The UI puts it behind a
    typed-name confirmation."""
    base = Path(root)
    safe = safe_name(name)
    if not safe:
        return Result(False, "", "That name has no usable characters.")
    path = resolve(base, safe)
    if not path.is_dir():
        return Result(False, safe, f"'{safe}' is no longer on disk.")
    if len(list_archives(base)) <= 1:
        return Result(False, safe, "This is the only archive.")
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return Result(False, safe, f"Could not delete '{safe}': {exc}")
    return Result(True, safe)
