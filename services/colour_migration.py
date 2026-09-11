"""The one-time move of archives written before brains had colour outputs.

A layout directory whose signature carries no `+c2` holds genomes of the old
width. Each such archive is copied into archives_legacy/ before anything in it
is touched; then every legacy directory is widened and renamed, and the root
files that name layouts are brought to the new names. A directory whose new
name is already taken is left alone: ids restart per directory, so two cannot
be merged without reissuing them.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

# Root files that name layouts, beside the layout directories.
_ROOT_FILES = ("settings.json", "layouts.jsonl", "map_layout.npz")


def _legacy_layout(name: str):
    from services.brains import layout_from_signature

    layout = layout_from_signature(name)
    if layout is None or layout.colour_outputs:
        return None
    return layout


def legacy_dirs(archive) -> list:
    """-> [(directory, legacy layout)] under one archive."""
    from services.archive_io import signature_dirs

    out = []
    for d in signature_dirs(archive):
        layout = _legacy_layout(d.name)
        if layout is not None:
            out.append((d, layout))
    return out


def _save_npz(path: Path, arrays: dict, allow_pickle: bool = False) -> None:
    """Atomic: a torn file here would be an archive's vectors."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, allow_pickle=allow_pickle, **arrays)
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _widened_vectors(path: Path, legacy) -> dict | None:
    """vectors.npz's arrays with every genome widened, or None if absent."""
    from services.brains import COLOUR_OUTPUTS, widen_colour, with_colour_outputs

    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    brains = np.asarray(arrays["brains"], dtype=np.float32)
    n = len(brains)
    width = with_colour_outputs(legacy, COLOUR_OUTPUTS).length
    if not n:
        arrays["brains"] = np.zeros((0, width), dtype=np.float32)
        return arrays
    brains = brains.reshape(n, -1)
    if brains.shape[1] != legacy.length:
        raise ValueError(f"{path} holds {brains.shape[1]}-float genomes, "
                         f"{legacy.signature()} is {legacy.length}")
    arrays["brains"] = np.stack([widen_colour(b, legacy)[1] for b in brains])
    return arrays


def _migrate_dir(archive: Path, d: Path, legacy) -> bool:
    """Widen one layout directory and give it its new name. -> moved."""
    from services.brains import upgrade_signature

    target = archive / upgrade_signature(d.name)
    if target.exists():
        print(f"[archive] {archive.name}/{d.name}: {target.name} already "
              "exists, so it is left as it is")
        return False
    arrays = _widened_vectors(d / "vectors.npz", legacy)
    # Staged beside the old file and swapped in only after the rename, so a
    # rename refused by a held handle leaves the legacy directory untouched.
    staged = d / "vectors.npz.colour"
    if arrays is not None:
        with open(staged, "wb") as fh:
            np.savez(fh, **arrays)
    try:
        os.replace(d, target)
    except OSError:
        staged.unlink(missing_ok=True)
        raise
    if arrays is not None:
        os.replace(target / staged.name, target / "vectors.npz")
    return True


def _rewrite_root_files(archive: Path) -> None:
    """Bring every layout name the root files hold to its current name."""
    from services.brains import upgrade_signature

    settings = archive / "settings.json"
    if settings.is_file():
        data = json.loads(settings.read_text(encoding="utf-8"))
        sig = data.get("layout_signature") if isinstance(data, dict) else None
        if isinstance(sig, str) and sig and upgrade_signature(sig) != sig:
            data["layout_signature"] = upgrade_signature(sig)
            _write_text(settings, json.dumps(data, indent=2))

    ledger = archive / "layouts.jsonl"
    if ledger.is_file():
        lines, changed = [], False
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            for key in ("parent", "child"):
                sig = row.get(key) if isinstance(row, dict) else None
                if isinstance(sig, str) and upgrade_signature(sig) != sig:
                    row[key] = upgrade_signature(sig)
                    changed = True
            lines.append(json.dumps(row))
        if changed:
            _write_text(ledger, "\n".join(lines) + "\n")

    cache = archive / "map_layout.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=True) as z:
            arrays = {k: z[k] for k in z.files}
        sigs = [str(s) for s in arrays.get("sigs", [])]
        new = [upgrade_signature(s) for s in sigs]
        if new != sigs:
            arrays["sigs"] = np.array(new, dtype=object)
            _save_npz(cache, arrays, allow_pickle=True)


def _back_up(archive: Path, legacy_root: Path, dirs) -> None:
    """Copy what is about to change. Never overwrites: the first copy of an
    archive is the one worth keeping."""
    dest = legacy_root / archive.name
    if not dest.exists():
        shutil.copytree(archive, dest)
        print(f"[archive] backed up {archive.name} to {dest}")
        return
    for d, _layout in dirs:
        if not (dest / d.name).exists():
            shutil.copytree(d, dest / d.name)
    for name in _ROOT_FILES:
        if (archive / name).is_file() and not (dest / name).exists():
            shutil.copy2(archive / name, dest / name)


def migrate_colour_outputs(root, legacy_root) -> int:
    """Migrate every archive under `root`. -> layout directories moved.

    One archive failing is reported and skipped; it is tried again at the next
    launch, since its legacy directories are still there.
    """
    from services.archive_io import migrate_archive

    root, legacy_root = Path(root), Path(legacy_root)
    if not root.is_dir():
        return 0
    moved = 0
    for archive in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            migrate_archive(archive)
            dirs = legacy_dirs(archive)
            if dirs:
                _back_up(archive, legacy_root, dirs)
                for d, legacy in dirs:
                    moved += _migrate_dir(archive, d, legacy)
            _rewrite_root_files(archive)
        except Exception as exc:  # noqa: BLE001 - one archive, not the launch
            print(f"[archive] {archive.name}: colour-output migration "
                  f"stopped ({exc})")
    if moved:
        print(f"[archive] {moved} layout directories now carry colour outputs")
    return moved
