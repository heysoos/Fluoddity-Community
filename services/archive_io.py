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

FORMAT_VERSION = 1
THUMB_PX = 160
THUMB_QUALITY = 85

_ARRAY_KEYS = ("ids", "embeddings", "brains", "physics")
# Written since 2026-08-08, absent from every archive saved before it. NOT in
# _ARRAY_KEYS and NOT a format_version bump on purpose: both would quarantine
# every existing archive on first open. A file without it simply loads without
# it, and Archive.load_from_store rescores from the embeddings anyway.
_OPTIONAL_ARRAY_KEYS = ("novelty",)


class ArchiveStore:
    def __init__(self, root, layout=None):
        """`root` is the named archive's directory; the store lives in a
        subdirectory named for the BRAIN LAYOUT signature.

        A stored genome is a bare float vector and the layout is the only thing
        that says what those floats mean. Two layouts sharing a directory does
        not error - it decodes a Gabor brain through the Fourier squash and
        scores the result, which is worse than a crash because every number
        along the way looks reasonable. The signature in the path makes the
        mistake unrepresentable rather than merely unlikely.
        """
        from services.brains import default_layout

        self.layout = layout or default_layout()
        self.root = Path(root) / self.layout.signature()
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

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.root / "vectors.npz"

    @property
    def goals_path(self) -> Path:
        return self.root / "goals.json"

    def thumb_path(self, name: str) -> Path:
        return self.root / "thumbs" / name

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
        the bytes, and the precision loss is far below the scale any novelty
        decision turns on.

        novelty belongs HERE rather than in index.jsonl because it is the one
        stored field that CHANGES after admission: refresh() re-scores entries
        against the grown archive, and index.jsonl is append-only, so the index
        can only ever hold the at-admission value. Measured 2026-08-08 on the
        default archive, that value correlates 0.075 with the truth - which
        made parent sampling, latent-goal anchoring and eviction all run on a
        column that was very nearly noise."""
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

        Load-bearing now that capacity is the only pruning rule: at 64 tiles a
        generation a full archive evicts 64 entries every ~2.8 s, and an
        orphaned 9 KB JPEG each would be ~12 MB a minute of files nothing can
        ever reach again - index.jsonl is append-only and the entry is gone
        from vectors.npz, so nothing on reload would even name them.
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

    # ---- reading -------------------------------------------------------

    def load_goals(self) -> list[dict]:
        try:
            return json.loads(self.goals_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

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


_LEGACY_MEMBERS = ("index.jsonl", "vectors.npz", "goals.json", "thumbs")


def migrate_to_signature_dir(archive_dir) -> Path | None:
    """Move a pre-modality archive down into `<archive_dir>/fourier-n10/`.

    `archive_dir` is one NAMED archive - the directory that used to hold
    index.jsonl directly. Everything written before brain modalities was Fourier
    at 10 centres, so that is the signature it lands under.

    A move, not a copy: an archive runs to hundreds of megabytes of thumbnails.

    -> the new directory, or None if there was nothing to do. Distinct from
    utilities.paths.migrate_legacy_archive, which moves the whole pre-2026-08-08
    archive under archives/default; this one is the level below.
    """
    from services.brains import default_layout

    base = Path(archive_dir)
    target = base / default_layout().signature()
    try:
        present = [m for m in _LEGACY_MEMBERS if (base / m).exists()]
        if not present:
            return None                 # already migrated, or a fresh archive
        if target.exists():
            # A restored backup sitting beside an already-migrated archive.
            # Swallowing it into a directory that has contents would merge two
            # unrelated runs, so leave both exactly where they are.
            print(f"[Archive] {base.name}: legacy files found beside an "
                  f"existing {target.name}; left in place")
            return None
        target.mkdir(parents=True)
        for name in present:
            os.replace(base / name, target / name)
    except OSError as exc:
        print(f"[Archive] could not migrate {base.name} ({exc}); left in place")
        return None
    print(f"[Archive] {base.name}: moved into {target.name}")
    return target
