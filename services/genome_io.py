"""Genome export/import using the existing Fluoddity config format.

Deliberately not a new file format. PhysicsConfig already carries the (10, 8)
rule alongside every physics slider, so an exported genome opens in the normal
single-simulation view at any resolution through the existing config browser,
and is copy-pasteable through the existing clipboard string.

Provenance is stored under a single extra top-level key. PhysicsConfig.from_dict
reads with .get(), so the extra key is ignored by every existing reader.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from services.config_saver import ConfigSaver, PhysicsConfig
from services.genome_spec import decode, encode

META_KEY = "fluoddity_evolution"


def export_genome(path, z: np.ndarray, sim_state, meta: dict | None = None) -> None:
    """Write z as an ordinary config file, plus a provenance block."""
    rule = decode(np.asarray(z, dtype=np.float32))
    config = ConfigSaver().create_config(sim_state, rule)
    data = json.loads(config.to_json())
    data[META_KEY] = dict(meta or {})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def import_genome(path) -> tuple[np.ndarray, int, dict]:
    """Read any config file as a search starting point.

    Returns (z, n_clamped, meta). Only x0 comes from the file - sigma, algorithm
    and grid are always taken from the current UI, which is what makes this
    'load the model, not the optimizer'.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text())
    meta = data.pop(META_KEY, {})
    config = PhysicsConfig.from_dict(data)
    z, clamped = encode(config.rule)
    return z, clamped, meta
