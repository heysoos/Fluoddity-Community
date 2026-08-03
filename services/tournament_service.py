"""TournamentService — interactive evolutionary selection over 16 brain genomes.

Human-in-the-loop: the user selects interesting tiles; next_generation() keeps
selected genomes pinned to their tiles and breeds the rest by mutating the
selected pool (plus a few fresh randoms for diversity). No fitness function.
"""
import numpy as np
from services.genome import random_genome, mutate, crossover, GENOME_SHAPE


class TournamentService:
    TILES = 16

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng if rng is not None else np.random.default_rng()
        self.population: list[np.ndarray] = [
            np.zeros(GENOME_SHAPE, dtype=np.float32) for _ in range(self.TILES)
        ]
        self.selected: set[int] = set()
        self.mutation_strength: float = 0.15
        self.inject_randoms: int = 1
        self.crossover_enabled: bool = False
        self.initialized: bool = False
        self._undo_stack: list[tuple[list[np.ndarray], set[int]]] = []
        self._dirty: bool = False

    # --- lifecycle ---
    def init_population(self) -> None:
        self.population = [random_genome(self._rng) for _ in range(self.TILES)]
        self.selected.clear()
        self._undo_stack.clear()
        self.initialized = True
        self.mark_dirty()

    def reset(self) -> None:
        self._push_undo()
        self.population = [random_genome(self._rng) for _ in range(self.TILES)]
        self.selected.clear()
        self.mark_dirty()

    # --- selection ---
    def toggle_select(self, tile: int) -> None:
        if not (0 <= tile < self.TILES):
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

        new: list[np.ndarray | None] = [None] * self.TILES
        # Survivors stay pinned to their own tiles.
        for i in self.selected:
            new[i] = self.population[i].copy()

        empty = [i for i in range(self.TILES) if new[i] is None]
        self._rng.shuffle(empty)
        n_random = max(0, min(self.inject_randoms, len(empty)))

        for j, tile in enumerate(empty):
            if j < n_random:
                new[tile] = random_genome(self._rng)
            else:
                p = parents[self._rng.integers(len(parents))]
                if self.crossover_enabled and len(parents) >= 2:
                    q = parents[self._rng.integers(len(parents))]
                    child = crossover(p, q, self._rng)
                else:
                    child = p
                new[tile] = mutate(child, self.mutation_strength, self._rng)

        self.population = [g for g in new]  # all slots filled
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
