"""Full run state save/load.

A checkpoint answers 'resume this exact search'. It carries optimizer state,
RNG, history and settings, and is inseparable from its grid size, because
cmaes.CMA fixes its population at construction.

For 'keep this creature', see services/genome_io.py instead.

Format is a single .npz: arrays at the top level, everything else as one JSON
string under 'meta'. Explicitly never a pickle - allow_pickle stays False on
load, so a checkpoint cannot execute code and is not tied to a library version.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1

_ARRAY_PREFIX = "arr__"
_OPT_PREFIX = "opt__"

_META_KEYS = (
    "genome_spec_signature", "generation", "optimizer_name", "base_seed",
    "prompt", "distractors", "settings", "history", "best_fitness",
)


class CheckpointError(Exception):
    pass


def _split(state: dict):
    """Separate numpy arrays (npz top level) from JSON-able scalars (meta)."""
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {"format_version": FORMAT_VERSION}

    for key in _META_KEYS:
        if key in state:
            meta[key] = state[key]

    if "best_z" in state:
        arrays[_ARRAY_PREFIX + "best_z"] = np.asarray(
            state["best_z"], dtype=np.float32
        )

    opt_meta = {}
    for k, v in (state.get("optimizer_state") or {}).items():
        if isinstance(v, np.ndarray):
            arrays[_OPT_PREFIX + k] = v
        elif isinstance(v, (bool, int, float, str)):
            opt_meta[k] = v
        elif isinstance(v, (list, tuple)):
            opt_meta[k] = list(v)
        else:
            raise CheckpointError(
                f"optimizer_state[{k!r}] is {type(v).__name__}; "
                "checkpoints store explicit arrays and scalars, never pickles"
            )
    meta["optimizer_meta"] = opt_meta
    return arrays, meta


def save_checkpoint(path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = _split(state)          # raises before any file is touched
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        # Write through a file handle, not a path: np.savez appends ".npz" to
        # any path that does not already end in it, which would produce
        # "<name>.npz.tmp.npz" and leave the rename below with nothing to move.
        with open(tmp, "wb") as fh:
            np.savez(fh, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(path, expect_signature: str | None = None) -> dict:
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"no checkpoint at {p}")

    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}

    got = int(meta.get("format_version", -1))
    if got != FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint format_version is {got}, "
            f"this build expects {FORMAT_VERSION}"
        )

    sig = meta.get("genome_spec_signature")
    if expect_signature is not None and sig != expect_signature:
        raise CheckpointError(
            f"checkpoint genome is {sig!r}, this build expects {expect_signature!r}"
        )

    opt_state = dict(meta.get("optimizer_meta", {}))
    for k, v in arrays.items():
        if k.startswith(_OPT_PREFIX):
            opt_state[k[len(_OPT_PREFIX):]] = v

    out = {k: v for k, v in meta.items() if k != "optimizer_meta"}
    out["optimizer_state"] = opt_state
    if _ARRAY_PREFIX + "best_z" in arrays:
        out["best_z"] = arrays[_ARRAY_PREFIX + "best_z"]
    return out
