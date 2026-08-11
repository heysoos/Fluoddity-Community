"""Mapping between the optimizer's search space and real genomes.

The optimizer searches an unbounded z; decoding applies a bounded squash, so it
can never wander to freq=50, and - unlike clipping - no repair bias is
introduced at the boundary. Clipping would map many distinct z to the same
genome, which distorts CMA-ES's covariance estimate.

The squash itself belongs to the MODALITY, not to this module: each brain has
its own parameter ranges, and Fourier's flat 3.0 was measurably mismatched
against the low-frequency bias its own random generator uses. This module now
only routes to the active modality and describes block layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.brains import BrainLayout, default_layout, get
from services.genome import N_CENTERS

FREQ_SCALE = 3.0
AMP_SCALE = 1.0
EPS = 1e-4

DIM = N_CENTERS * 8  # 80


def _active(layout: BrainLayout | None):
    layout = layout or default_layout()
    return get(layout.modality), layout


def present(flat, layout: BrainLayout) -> np.ndarray:
    """The shape callers expect a brain of this modality to arrive in.

    Shape (N, 8) is preserved for Fourier so legacy callers that reshape or
    index by centre are unaffected. Other modalities have no such 2D structure
    and stay flat.
    """
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if layout.modality == "fourier":
        return flat.reshape(layout.shape[0], 8)
    return flat


def decode(z: np.ndarray, layout: BrainLayout | None = None) -> np.ndarray:
    """(dim,) -> params."""
    m, layout = _active(layout)
    return present(m.decode(z, layout), layout)


def random_genome_for(rng, layout: BrainLayout | None = None) -> np.ndarray:
    """A fresh brain of `layout`, in the same presentation decode() produces.

    The generic replacement for services.genome.random_genome, which is
    hardcoded to Fourier's (10, 8) and was what the interactive tournament bred
    whatever modality was selected.
    """
    m, layout = _active(layout)
    return present(m.random(rng, layout), layout)


def encode(genome: np.ndarray, layout: BrainLayout | None = None):
    """params -> ((dim,) float32, n_clamped)."""
    m, layout = _active(layout)
    return m.encode(np.asarray(genome, dtype=np.float32).reshape(-1), layout)


@dataclass(frozen=True)
class Block:
    name: str
    size: int


class GenomeSpec:
    """Ordered named blocks of the search vector.

    Carries the BrainLayout so the brain block knows how to decode itself; the
    optimizer, scorer, logger and loop stay dimension-agnostic.
    """

    def __init__(self, blocks: list[Block], layout: BrainLayout | None = None):
        self.blocks = list(blocks)
        self.layout = layout or default_layout()

    @property
    def dim(self) -> int:
        return sum(b.size for b in self.blocks)

    def signature(self) -> str:
        """Stable string used to reject incompatible checkpoints."""
        return ",".join(f"{b.name}:{b.size}" for b in self.blocks)

    def same_space_as(self, other) -> bool:
        """Would an optimizer built for `other` still be valid here?

        Not `is`, and not signature() either. Signature is widths alone, and
        Fourier at 21 centres is 168 floats exactly as Gabor at 12 filters is -
        the same dimension over completely different meanings. The layout is
        what separates them.

        BrainLayout compares with `scales` excluded, so a decode-scale change
        counts as the SAME space on purpose: it changes what a z means, not how
        many there are, and the covariance a run has spent generations learning
        is still about the right axes. Callers hold specs by value, not by
        identity - main._refresh_driver_specs builds a fresh one every time it
        is called - so comparing objects reset the search on every scale tweak.
        """
        return self.blocks == other.blocks and self.layout == other.layout

    def decode(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z = np.asarray(z, dtype=np.float32)
        out: dict[str, np.ndarray] = {}
        off = 0
        for b in self.blocks:
            chunk = z[off:off + b.size]
            # Routes through the module-level decode, NOT the modality directly:
            # that is what preserves the (N, 8) shape for Fourier. Callers all
            # over the app index brains by centre, and handing them a flat
            # vector fails far from here.
            out[b.name] = decode(chunk, self.layout) if b.name == "brain" else chunk.copy()
            off += b.size
        return out

    def split(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """The raw sub-vectors of z, undecoded.

        decode() gives phenotypes; this gives the pieces of z itself, which is
        what a caller needs when it hands one block to something that will
        decode it again. Passing a whole 88-wide brain+physics vector to a
        brain decoder is a reshape error, and inside a command handler that
        takes the app down.
        """
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        out: dict[str, np.ndarray] = {}
        off = 0
        for b in self.blocks:
            out[b.name] = z[off:off + b.size].copy()
            off += b.size
        return out

    def encode(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        pieces = []
        for b in self.blocks:
            v = parts[b.name]
            pieces.append(
                get(self.layout.modality).encode(
                    np.asarray(v, dtype=np.float32).reshape(-1), self.layout)[0]
                if b.name == "brain" else np.asarray(v).reshape(-1)
            )
        return np.concatenate(pieces).astype(np.float32)


def layout_of(tournament) -> BrainLayout:
    """The layout a search should adopt when it was not given a spec.

    The TournamentService is the object that already knows which brain is
    running, and every search is built around one, so asking it removes the
    chance of the two disagreeing at construction. Defaulting to Fourier
    instead is what let a search built after a brain switch produce genomes of
    the wrong width.

    Falls back rather than raising: the archive window can outlive its
    tournament, and a driver that refuses to build takes a panel down with it.
    """
    return getattr(tournament, "layout", None) or default_layout()


def spec_for(layout: BrainLayout) -> GenomeSpec:
    return GenomeSpec([Block("brain", layout.length)], layout)


def physics_spec_for(layout: BrainLayout) -> GenomeSpec:
    from services.physics_genome import PHYSICS_DIM

    return GenomeSpec(
        [Block("brain", layout.length), Block("physics", PHYSICS_DIM)], layout)


BRAIN_SPEC = GenomeSpec([Block("brain", DIM)])

# Physics in the search space. Measured: brain-only search only improves when
# the starting preset already sits near the goal, because the preset fixes gross
# morphology and the brain only shapes local behaviour.
from services.physics_genome import PHYSICS_DIM  # noqa: E402

BRAIN_PHYSICS_SPEC = GenomeSpec([Block("brain", DIM), Block("physics", PHYSICS_DIM)])
