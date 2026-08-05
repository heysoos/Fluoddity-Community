"""Mapping between the optimizer's search space and real genomes.

The optimizer searches an unbounded z in R^80. Decoding applies a bounded tanh
squash, so it can never wander to freq=50, and - unlike clipping - no repair
bias is introduced at the boundary. Clipping would map many distinct z to the
same genome, which distorts CMA-ES's covariance estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.genome import N_CENTERS

FREQ_SCALE = 3.0
AMP_SCALE = 1.0
EPS = 1e-4

DIM = N_CENTERS * 8  # 80


def decode(z: np.ndarray) -> np.ndarray:
    """(80,) -> (10, 8) float32, matching services.genome.random_genome ranges."""
    z = np.asarray(z, dtype=np.float32).reshape(N_CENTERS * 2, 4)
    freq = FREQ_SCALE * np.tanh(z[:N_CENTERS])
    amp = AMP_SCALE * np.tanh(z[N_CENTERS:])
    return np.concatenate([freq, amp], axis=1).astype(np.float32)


def encode(genome: np.ndarray) -> tuple[np.ndarray, int]:
    """(10, 8) -> ((80,) float32, n_clamped).

    A config file may hold values outside the squash range (legacy generator,
    hand editing). Those are clamped, which is lossy at the extremes but keeps
    x0 finite. The clamp count is returned so a badly out-of-range file is
    visible rather than silent.
    """
    g = np.asarray(genome, dtype=np.float32).reshape(N_CENTERS, 8)
    freq = g[:, :4] / FREQ_SCALE
    amp = g[:, 4:] / AMP_SCALE
    raw = np.concatenate([freq, amp], axis=0)
    n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
    z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
    return z.reshape(-1).astype(np.float32), n_clamped


@dataclass(frozen=True)
class Block:
    name: str
    size: int


class GenomeSpec:
    """Ordered named blocks of the search vector.

    Adding physics later means appending Block("physics", 12); the optimizer,
    scorer, logger and loop are all dimension-agnostic.
    """

    def __init__(self, blocks: list[Block]):
        self.blocks = list(blocks)

    @property
    def dim(self) -> int:
        return sum(b.size for b in self.blocks)

    def signature(self) -> str:
        """Stable string used to reject incompatible checkpoints."""
        return ",".join(f"{b.name}:{b.size}" for b in self.blocks)

    def decode(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z = np.asarray(z, dtype=np.float32)
        out: dict[str, np.ndarray] = {}
        off = 0
        for b in self.blocks:
            chunk = z[off:off + b.size]
            out[b.name] = decode(chunk) if b.name == "brain" else chunk.copy()
            off += b.size
        return out

    def encode(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        pieces = []
        for b in self.blocks:
            v = parts[b.name]
            pieces.append(
                encode(v)[0] if b.name == "brain" else np.asarray(v).reshape(-1)
            )
        return np.concatenate(pieces).astype(np.float32)


BRAIN_SPEC = GenomeSpec([Block("brain", DIM)])
