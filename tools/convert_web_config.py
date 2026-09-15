"""Rewrite configs downloaded from fluoddity.com as desktop configs.

    python -m tools.convert_web_config <file-or-folder>... [--backup DIR]

Each web file is replaced by one desktop file per config it holds (a file
with several gets a `_1`, `_2`... suffix from the second on). The originals are
moved to --backup (default: `web_originals` beside the first input). Files
already in the desktop format are left alone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import ui  # noqa: F401,E402  prime the services/ui import cycle
from services.web_config import is_web_config, translate_web_config


def convert_file(path: Path, backup: Path) -> list[Path]:
    """Convert one web file in place. -> the desktop files written."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not is_web_config(data):
        return []
    configs = translate_web_config(data)
    backup.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(backup / path.name))
    written = []
    for i, cfg in enumerate(configs):
        out = path if i == 0 else path.with_name(f"{path.stem}_{i}{path.suffix}")
        out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        written.append(out)
        if cfg["notes"]:
            print(f"  {out.name}: {cfg['notes'].splitlines()[-1]}")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--backup", type=Path, default=None,
                    help="where the web originals are moved")
    args = ap.parse_args(argv)

    files = []
    for p in args.paths:
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    backup = args.backup or files[0].parent / "web_originals"

    total = 0
    for f in files:
        written = convert_file(f, backup)
        if written:
            total += len(written)
            print(f"{f.name} -> {', '.join(w.name for w in written)}")
    print(f"{total} config(s) written; originals in {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
