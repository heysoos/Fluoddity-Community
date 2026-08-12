"""Persistence for the exploration archive.

index.jsonl is the AUTHORITY for which entries exist; vectors.npz supplies their
arrays. They are reconciled on load, so a crash between the last flush and quit
costs the trailing entries rather than the archive.

Write strategies mirror the two already in this codebase: append-and-flush per
generation for the index (RunLogger), tmp + os.replace for the vectors
(run_checkpoint). Never a pickle; np.load always with allow_pickle=False.

A disk problem must never block evolution. Every method degrades to a no-op with
one printed warning instead of raising.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from services.save_targets import safe_stem

FORMAT_VERSION = 1
THUMB_PX = 160
THUMB_QUALITY = 85

_ARRAY_KEYS = ("ids", "embeddings", "brains", "physics")
# Newer field, absent from older archives. NOT in _ARRAY_KEYS and NOT a
# format_version bump on purpose: both would quarantine every existing archive
# on first open. A file without it simply loads without it, and
# Archive.load_from_store rescores from the embeddings anyway.
_OPTIONAL_ARRAY_KEYS = ("novelty",)


class ArchiveStore:
    def __init__(self, root, layout=None, signature: str | None = None):
        """`root` is the named archive; ENTRIES live one level down, in a
        directory named for the BRAIN LAYOUT signature.

        A stored genome is a bare float vector and the layout is the only thing
        that says what those floats mean. Two layouts sharing a directory does
        not error - it decodes a Gabor brain through the Fourier squash and
        scores the result, which is worse than a crash because every number
        along the way looks reasonable. The signature in the path makes the
        mistake unrepresentable rather than merely unlikely.

        Everything that belongs to the ARCHIVE rather than to one of its brains
        - the goals, the Explore settings, each run's physics - stays at
        `root`. One archive holds entries of every brain, so a per-brain copy
        of those would be several answers to a question with one.

        `signature` names the layout directory when there is no BrainLayout to
        hand, which is how the archive reaches a sibling brain's entries.
        """
        from services.brains import default_layout

        self.base = Path(root)
        if signature is None:
            self.layout = layout or default_layout()
            self.signature = self.layout.signature()
        else:
            self.layout = layout
            self.signature = str(signature)
        self.root = self.base / self.signature
        self.enabled = True
        self._fh = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "thumbs").mkdir(exist_ok=True)
            self._fh = open(self.index_path, "a", encoding="utf-8")
        except OSError as exc:
            self.enabled = False
            print(f"[Archive] persistence disabled ({exc}); the run continues")

    # ---- paths ---------------------------------------------------------
    # Per LAYOUT: the entries and their pictures.

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.root / "vectors.npz"

    def thumb_path(self, name: str) -> Path:
        return self.root / "thumbs" / name

    # Per ARCHIVE: everything that is not a genome.

    @property
    def goals_path(self) -> Path:
        return self.base / "goals.json"

    @property
    def settings_path(self) -> Path:
        return self.base / "settings.json"

    def run_config_path(self, run_id: str) -> Path:
        return self.base / "runs" / f"{safe_stem(run_id)}.json"

    # ---- writing -------------------------------------------------------

    def append_index(self, row: dict) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()  # a crash must lose at most one entry
        except (OSError, TypeError) as exc:
            self.enabled = False
            print(f"[Archive] index write failed ({exc}); persistence disabled")

    def flush_vectors(self, ids, embeddings, brains, physics,
                      novelty=None) -> None:
        """Rewrite vectors.npz atomically. Embeddings go to disk as fp16 - half
        the bytes, well below the precision any novelty decision needs.

        novelty belongs HERE rather than in index.jsonl because it is the one
        stored field that CHANGES after admission: refresh() re-scores entries
        against the grown archive, and index.jsonl is append-only so it can
        only ever hold the at-admission value. See CLAUDE.md."""
        if not self.enabled:
            return
        tmp = self.vectors_path.with_suffix(self.vectors_path.suffix + ".tmp")
        try:
            # A file handle, not a path: np.savez appends ".npz" to any path
            # that does not already end in it, which would produce
            # "vectors.npz.tmp.npz" and leave os.replace with nothing to move.
            with open(tmp, "wb") as fh:
                np.savez(
                    fh,
                    format_version=np.array(FORMAT_VERSION),
                    ids=np.asarray(ids, dtype=np.int64),
                    embeddings=np.asarray(embeddings, dtype=np.float16),
                    brains=np.asarray(brains, dtype=np.float32),
                    physics=np.asarray(physics, dtype=np.float32),
                    novelty=np.asarray(
                        np.zeros(len(np.asarray(ids)))
                        if novelty is None else novelty,
                        dtype=np.float32),
                )
            os.replace(tmp, self.vectors_path)
        except (OSError, ValueError) as exc:
            print(f"[Archive] vector flush failed ({exc}); previous file kept")
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def write_thumb(self, entry_id: int, crop: np.ndarray) -> str:
        """-> the filename, or "" if it could not be written. A missing
        thumbnail costs a gallery placeholder, never an admission."""
        if not self.enabled:
            return ""
        name = f"{int(entry_id):06d}.jpg"
        try:
            from PIL import Image

            img = Image.fromarray(np.asarray(crop, dtype=np.uint8))
            img = img.resize((THUMB_PX, THUMB_PX), Image.BILINEAR)
            img.save(self.thumb_path(name), quality=THUMB_QUALITY)
        except (OSError, ValueError) as exc:
            print(f"[Archive] thumbnail {name} failed ({exc})")
            return ""
        return name

    def delete_thumb(self, name: str) -> bool:
        """Remove one thumbnail. -> did a file go?

        Load-bearing now that capacity is the only pruning rule: without this,
        every eviction orphans a JPEG nothing can ever reach again -
        index.jsonl is append-only and the entry is gone from vectors.npz.
        """
        if not self.enabled or not name:
            return False
        try:
            self.thumb_path(str(name)).unlink()
            return True
        except OSError:
            # Already gone, or locked by the gallery's loader. Neither is worth
            # interrupting a generation over.
            return False

    def save_goals(self, items: list[dict]) -> None:
        if not self.enabled:
            return
        tmp = self.goals_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(list(items), indent=2), encoding="utf-8")
            os.replace(tmp, self.goals_path)
        except OSError as exc:
            print(f"[Archive] goal list not saved ({exc})")

    def save_settings(self, data: dict) -> None:
        """The Explore settings this archive was last worked with.

        Beside the goal list, and for the same reason: the settings that suit a
        20000-entry archive are not the ones that suit an empty one, so they
        belong to the archive rather than to the app. Written whole and
        atomically, like goals.json - a partially-written settings file would
        load as defaults, which is precisely the failure it exists to prevent.
        """
        if not self.enabled:
            return
        tmp = self.settings_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(dict(data), indent=2, sort_keys=True),
                           encoding="utf-8")
            os.replace(tmp, self.settings_path)
        except (OSError, TypeError) as exc:
            print(f"[Archive] settings not saved ({exc})")

    def save_run_config(self, run_id: str, config_json: str) -> bool:
        """Record the physics a run is about to be carried out under.

        An entry stores its brain and, when physics is searched, the deltas the
        optimizer moved. It has never stored the config those deltas are
        relative to - so replaying an entry used whatever the sliders happened
        to say, and with physics search OFF that is the entry's physics
        entirely. Written once per run, because with search off every entry in
        the run shares one config; `run_id` is already on every index row.

        Never overwrites: a run id names one set of physics, and a resume that
        wrote a second would silently reinterpret the entries already filed
        under it. -> whether a file is now on disk for this run.
        """
        if not self.enabled or not run_id:
            return False
        path = self.run_config_path(run_id)
        if path.exists():
            return True
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(config_json, encoding="utf-8")
            os.replace(tmp, path)
            return True
        except OSError as exc:
            print(f"[Archive] run config not saved ({exc}); "
                  f"entries from this run will replay under the live sliders")
            return False

    # ---- reading -------------------------------------------------------

    def load_run_config(self, run_id: str) -> str | None:
        """-> the run's config JSON, or None for a run recorded before this
        existed. None means "leave the sliders alone", which is what every
        entry did before run configs were written."""
        if not run_id:
            return None
        try:
            return self.run_config_path(run_id).read_text(encoding="utf-8")
        except OSError:
            return None

    def load_goals(self) -> list[dict]:
        try:
            return json.loads(self.goals_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def load_settings(self) -> dict:
        """-> the saved settings, or {} for an archive that has none yet.

        {} means "keep what is on screen", not "reset to defaults": a brand new
        archive inherits the settings you were just using, which is the useful
        behaviour when you make one to try a variation.
        """
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def load(self) -> tuple[list[dict], dict]:
        """-> (index rows, arrays). Either may be empty; the caller reconciles."""
        rows: list[dict] = []
        try:
            text = self.index_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A torn trailing line is one lost entry, not a lost archive.
                continue

        arrays: dict = {}
        if self.vectors_path.is_file():
            try:
                with np.load(self.vectors_path, allow_pickle=False) as z:
                    got = int(z["format_version"])
                    if got != FORMAT_VERSION:
                        raise ValueError(
                            f"vectors.npz format_version is {got}, "
                            f"this build expects {FORMAT_VERSION}")
                    arrays = {k: z[k] for k in _ARRAY_KEYS}
                    arrays.update({k: z[k] for k in _OPTIONAL_ARRAY_KEYS
                                   if k in z.files})
            except (OSError, ValueError, KeyError) as exc:
                self._quarantine(exc)
                arrays = {}
        return rows, arrays

    def _quarantine(self, exc) -> None:
        """Move a bad vectors file aside. Never overwrite it - if the archive
        represents hours of exploration, a recoverable file is worth more than
        a tidy directory."""
        bad = self.vectors_path.with_name(f"vectors.npz.bad-{int(time.time())}")
        try:
            os.replace(self.vectors_path, bad)
            print(f"[Archive] {exc}; moved to {bad.name}, starting empty")
        except OSError:
            print(f"[Archive] {exc}; could not quarantine {self.vectors_path}")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# What belongs to a LAYOUT: the entries and their pictures. These move DOWN.
_LAYOUT_MEMBERS = ("index.jsonl", "vectors.npz", "thumbs")
# What belongs to the ARCHIVE: everything that is not a genome. These move UP.
_ARCHIVE_MEMBERS = ("goals.json", "settings.json", "runs")


def signature_dirs(archive_dir) -> list[Path]:
    """Every layout directory under one named archive, oldest name first.

    A directory is a layout's if it holds an index; that is the one member a
    store always creates, and it keeps `thumbs` from an unmigrated archive out
    of the list.
    """
    try:
        return sorted((p for p in Path(archive_dir).iterdir()
                       if p.is_dir() and (p / "index.jsonl").is_file()),
                      key=lambda p: p.name)
    except OSError:
        return []


def _count_rows(sig_dir) -> int:
    """Index rows in one layout directory. Cheap, and only used to decide which
    layout's shared files the archive keeps."""
    try:
        with open(sig_dir / "index.jsonl", "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def migrate_archive(archive_dir) -> Path | None:
    """Put one named archive into the layout it is read in. -> what moved, or None.

    Two directions, because the split has two halves:

    DOWN - a pre-modality archive held index.jsonl at the top level. Everything
    written before brains were swappable is Fourier at 10 centres, so that is
    the signature it lands under.

    UP - goals, settings and each run's physics belong to the ARCHIVE, and an
    earlier build filed them beside the entries. One archive holds every brain,
    so a per-brain copy of those is several answers to a question that has one.

    Moves, never copies: an archive runs to hundreds of megabytes of thumbnails.

    Distinct from utilities.paths.migrate_legacy_archive, which moves the whole
    pre-2026-08-08 archive under archives/default; this one is the level below.
    """
    from services.brains import default_layout

    base = Path(archive_dir)
    moved = None
    try:
        present = [m for m in _LAYOUT_MEMBERS if (base / m).exists()]
        if present:
            target = base / default_layout().signature()
            if target.exists():
                # A restored backup sitting beside an already-migrated archive.
                # Swallowing it into a directory that has contents would merge
                # two unrelated runs, so leave both exactly where they are.
                print(f"[Archive] {base.name}: legacy files found beside an "
                      f"existing {target.name}; left in place")
            else:
                target.mkdir(parents=True)
                for name in present:
                    os.replace(base / name, target / name)
                print(f"[Archive] {base.name}: entries moved into {target.name}")
                moved = target

        # BUSIEST layout first, then newest. Not mtime alone: switching brain
        # creates an empty sibling and writes its settings, so the directories
        # with nothing in them are reliably the most recently touched, and
        # mtime hands the archive the settings of a layout never worked in.
        # The losers stay put and say so - merging two settings files silently
        # is worse than picking one loudly.
        skipped = []
        for sub in sorted(signature_dirs(base),
                          key=lambda p: (_count_rows(p), p.stat().st_mtime),
                          reverse=True):
            for name in _ARCHIVE_MEMBERS:
                src = sub / name
                if not src.exists():
                    continue
                if (base / name).exists():
                    skipped.append(f"{sub.name}/{name}")
                    continue
                os.replace(src, base / name)
                print(f"[Archive] {base.name}: {name} moved up from {sub.name}")
                moved = moved or base
        if skipped:
            print(f"[Archive] {base.name}: the archive already had one, so "
                  f"{len(skipped)} spare copies stay where they are "
                  f"({', '.join(skipped[:4])}"
                  f"{', ...' if len(skipped) > 4 else ''})")
    except OSError as exc:
        print(f"[Archive] could not migrate {base.name} ({exc}); left in place")
        return None
    return moved
