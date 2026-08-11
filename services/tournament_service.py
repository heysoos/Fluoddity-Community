"""TournamentService — interactive evolutionary selection over 16 brain genomes.

Human-in-the-loop: the user selects interesting tiles; next_generation() keeps
selected genomes pinned to their tiles and breeds the rest by mutating the
selected pool (plus a few fresh randoms for diversity). No fitness function.

Every genome is of the ACTIVE brain layout. This used to call
services.genome.random_genome/mutate/crossover directly, all of which are
hardcoded to Fourier's (10, 8) - so under any other modality the tournament bred
80-float genomes for a GPU expecting 168, 120 or 148, and
sim.write_tournament_rules re-sliced the upload at layout.length into genomes
that straddled the originals.
"""
import numpy as np

from services.brains import (BrainLayout, brain_rng, crossover, default_layout,
                             mutate)
from services.genome_spec import random_genome_for


class TournamentService:
    def __init__(self, grid: int = 4, rng: np.random.Generator | None = None,
                 layout: BrainLayout | None = None, seed: float = 0.0):
        self.grid = int(grid)
        self.tiles = self.grid * self.grid
        # The app's own randomness control, kept live: main pushes SimState's
        # rule_seed onto it, and it is what decides the tiles a modality switch
        # lands on. Turning it must change them, so it is an attribute rather
        # than something baked into the generator at construction.
        self.seed = float(seed)
        # A SEPARATE stream for breeding and rerolls, spawned from the seed's.
        # It must not start where _seeded_population starts, or the first Reset
        # would hand back the very grid the session opened on.
        self._rng = rng if rng is not None else brain_rng(self.seed).spawn(2)[1]
        self._layout = layout or default_layout()
        self.population: list[np.ndarray] = [
            self._blank() for _ in range(self.tiles)
        ]
        self.selected: set[int] = set()
        self.mutation_strength: float = 0.15
        self.inject_randoms: int = 1
        self.crossover_enabled: bool = False
        self.initialized: bool = False
        self._undo_stack: list[tuple[list[np.ndarray], set[int]]] = []
        self._dirty: bool = False

    # --- the active brain ---
    @property
    def layout(self):
        return self._layout

    def _blank(self) -> np.ndarray:
        from services.genome_spec import present

        return present(np.zeros(self._layout.length, dtype=np.float32),
                       self._layout)

    def _random(self) -> np.ndarray:
        return random_genome_for(self._rng, self._layout)

    def _seeded_population(self) -> list[np.ndarray]:
        """The tiles this (layout, seed) means, from a FRESH generator.

        Drawn afresh rather than from the instance generator so that a modality
        round trip returns to where it started instead of to wherever breeding
        and resets had since carried the stream. Same formula and same seed
        mapping as generated_brains, so a tile and a no-rule cohort brain are
        the same draw.
        """
        rng = brain_rng(self.seed)
        return [random_genome_for(rng, self._layout) for _ in range(self.tiles)]

    def set_layout(self, layout) -> None:
        """Repopulate for a new brain layout.

        A scales-only change is a no-op: it changes what a genome MEANS, not how
        wide it is, and rerolling would throw away the tiles the user is part
        way through selecting. BrainLayout's == already excludes scales.
        """
        if layout == self._layout:
            self._layout = layout
            return
        self._layout = layout
        self.selected.clear()
        self._undo_stack.clear()
        self.population = self._seeded_population()
        self.mark_dirty()

    # --- lifecycle ---
    def init_population(self) -> None:
        # Seeded, so the grid a session opens on is the seed's, not the clock's.
        self.population = self._seeded_population()
        self.selected.clear()
        self._undo_stack.clear()
        self.initialized = True
        self.mark_dirty()

    def set_grid(self, grid: int) -> None:
        """Resize the population. Callers must only do this between generations
        - cmaes.CMA fixes popsize at construction, so the optimizer is reset
        separately by AutoTournamentService."""
        grid = int(grid)
        if grid == self.grid:
            return
        self.grid = grid
        self.tiles = grid * grid
        self.selected.clear()
        self._undo_stack.clear()
        self.population = [self._random() for _ in range(self.tiles)]
        self.mark_dirty()

    def reset(self) -> None:
        self._push_undo()
        self.population = [self._random() for _ in range(self.tiles)]
        self.selected.clear()
        self.mark_dirty()

    # --- selection ---
    def toggle_select(self, tile: int) -> None:
        if not (0 <= tile < self.tiles):
            return
        if tile in self.selected:
            self.selected.remove(tile)
        else:
            self.selected.add(tile)

    # --- breeding ---
    def next_generation(self) -> None:
        self._push_undo()
        parents = ([self.population[i] for i in sorted(self.selected)]
                   if self.selected else list(self.population))

        new: list[np.ndarray | None] = [None] * self.tiles
        # Survivors stay pinned to their own tiles.
        for i in self.selected:
            new[i] = self.population[i].copy()

        empty = [i for i in range(self.tiles) if new[i] is None]
        self._rng.shuffle(empty)
        n_random = max(0, min(self.inject_randoms, len(empty)))

        for j, tile in enumerate(empty):
            if j < n_random:
                new[tile] = self._random()
            else:
                p = parents[self._rng.integers(len(parents))]
                if self.crossover_enabled and len(parents) >= 2:
                    q = parents[self._rng.integers(len(parents))]
                    child = crossover(p, q, self._rng, self._layout)
                else:
                    child = p
                new[tile] = mutate(child, self.mutation_strength, self._rng,
                                   self._layout)

        self.population = [g for g in new]  # all slots filled
        # Start each round with a clean slate; undo() restores the prior selection.
        self.selected.clear()
        self.mark_dirty()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        pop, sel = self._undo_stack.pop()
        self.population = [g.copy() for g in pop]
        self.selected = set(sel)
        self.mark_dirty()

    def _push_undo(self) -> None:
        self._undo_stack.append(([g.copy() for g in self.population], set(self.selected)))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    # --- GPU packing ---
    def pack_rule_bytes(self) -> bytes:
        data = bytearray()
        for g in self.population:
            data.extend(np.ascontiguousarray(g, dtype=np.float32).tobytes())
        return bytes(data)

    # --- dirty flag ---
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    def clear_dirty(self) -> None:
        self._dirty = False
