"""Behaviour descriptor and liveness, from one generation's snapshot embeddings.

The descriptor is the TRAJECTORY CENTROID, not the final frame. E&E uses the
final state; Fluoddity patterns are watched in motion, and a single frame
discards what the pattern does - which is most of what distinguishes one
creature from another here. The centroid is also insensitive to the exact
stopping step, which matters because steps_per_gen is user-adjustable.

Known failure mode, accepted: a pattern oscillating between two dissimilar
states gets a centroid resembling neither, and may land next to an unrelated
pattern. Liveness is stored alongside precisely so this case is visible.

Liveness is ASAL's open-endedness objective (arXiv:2412.17799 Eq. 3),
sign-flipped so that higher is better like everything else in this codebase.
"""
from __future__ import annotations

import numpy as np


def stack_snapshots(per_snapshot: list[np.ndarray]) -> np.ndarray:
    """[(n, dim)] * S -> (S, n, dim) float32."""
    return np.stack([np.asarray(e, dtype=np.float32) for e in per_snapshot], axis=0)


def _check(snaps: np.ndarray) -> np.ndarray:
    a = np.asarray(snaps, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected (S, n, dim), got {a.shape}")
    return a


def descriptor(snaps: np.ndarray) -> np.ndarray:
    """(S, n, dim) unit-norm embeddings -> (n, dim) unit-norm centroid."""
    a = _check(snaps)
    m = a.mean(axis=0)
    return (m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8)
            ).astype(np.float32)


def liveness(snaps: np.ndarray) -> np.ndarray:
    """(S, n, dim) -> (n,) float32.

        L = 1 - mean_{s>0} max_{s'<s} <e_s, e_s'>

    A frozen pattern scores 0; a pattern whose every state is orthogonal to all
    its predecessors scores 1. The max is over ALL earlier snapshots, not just
    the previous one, so returning to an earlier state is correctly not novel.

    S < 2 returns zeros: one snapshot is no evidence of change. Callers must
    disable the liveness gate at snapshots_per_gen == 1 rather than reject
    everything.
    """
    a = _check(snaps)
    s, n, _ = a.shape
    if s < 2:
        return np.zeros(n, dtype=np.float32)
    e = np.transpose(a, (1, 0, 2))              # (n, S, dim)
    sim = np.einsum("nsd,ntd->nst", e, e)       # (n, S, S)
    worst = np.empty((n, s - 1), dtype=np.float32)
    for i in range(1, s):
        worst[:, i - 1] = sim[:, i, :i].max(axis=1)
    return (1.0 - worst.mean(axis=1)).astype(np.float32)
