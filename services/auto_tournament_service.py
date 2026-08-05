"""The automatic tournament generation state machine.

Holds an Optimizer, a CLIPScorer and a RunLogger, all injected. Contains no GL
calls and no ImGui, so the whole loop is testable with fakes.

The rollout is spread across real application frames - update() returns one
action per call and never loops internally. That is the mechanism by which the
app stays responsive; it is not an optimization to be added later.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from services.genome_spec import BRAIN_SPEC, decode
from services.optimizers import make_optimizer


class Action(str, Enum):
    NONE = "none"
    WRITE_RULES = "write_rules"
    STEP = "step"
    CAPTURE = "capture"
    SCORE = "score"


class Phase(str, Enum):
    IDLE = "idle"
    ROLLOUT = "rollout"
    SCORE = "score"
    PAUSED = "paused"


def snapshot_steps(steps_per_gen: int, n: int) -> list[int]:
    """Evenly spaced step counts, always ending on the last step."""
    n = max(1, min(int(n), int(steps_per_gen)))
    return [int(round(steps_per_gen * (i + 1) / n)) for i in range(n)]


class AutoTournamentService:
    def __init__(self, tournament_service, scorer=None, logger=None,
                 spec=BRAIN_SPEC, base_seed: int = 1000):
        self.tournament = tournament_service
        self.scorer = scorer
        self.logger = logger
        self.spec = spec
        self.base_seed = int(base_seed)

        self.steps_per_gen = 300
        self.snapshots_per_gen = 4
        self.sim_steps_per_frame = 10
        self.sigma0 = 0.5
        self.algorithm = "CMA-ES"
        self.tile_mutation_enabled = False
        self.variants_per_tile = 4
        self.tile_mutation_strength = 0.1
        self.autosave_every = 10

        self.phase = Phase.IDLE
        self.generation = 0
        self.step_in_gen = 0
        self.fitness: np.ndarray | None = None
        self.prompt = ""
        self.optimizer = None

        self._z = None
        self._snaps: list[int] = []
        self._next_snap = 0
        self._buffer: list[np.ndarray] = []
        self._needs_write = False

    # ---- configuration -------------------------------------------------

    def configure(self, **kw) -> None:
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(f"unknown setting {k!r}")
            setattr(self, k, v)

    @property
    def popsize(self) -> int:
        return self.tournament.tiles

    @property
    def gen_seed(self) -> int:
        return self.base_seed + self.generation

    @property
    def current_z(self):
        """This generation's search vectors, or None before the first ask."""
        return self._z

    @property
    def sigma(self) -> float:
        return float(self.optimizer.sigma) if self.optimizer else self.sigma0

    def _ensure_optimizer(self, x0=None) -> None:
        if self.optimizer is None:
            self.optimizer = make_optimizer(
                self.algorithm, self.spec.dim, self.popsize,
                self.sigma0, self.base_seed, x0,
            )

    def set_prompt(self, text: str) -> None:
        """Changing the prompt keeps the optimizer's learned covariance and
        simply starts climbing a new landscape."""
        self.prompt = text
        if self.scorer is not None:
            self.scorer.set_prompt(text)

    def set_x0(self, z: np.ndarray) -> None:
        """Load a genome as the search starting point. Discards optimizer
        state; sigma, algorithm and grid come from the UI, not the file."""
        self.optimizer = None
        self._ensure_optimizer(np.asarray(z, dtype=np.float64))

    # ---- lifecycle -----------------------------------------------------

    def start(self, prompt: str | None = None) -> None:
        if prompt is not None:
            self.set_prompt(prompt)
        self._ensure_optimizer()
        if self.phase is Phase.IDLE:
            self._begin_generation()
        if self.phase in (Phase.IDLE, Phase.PAUSED):
            self.phase = Phase.ROLLOUT

    def pause(self) -> None:
        if self.phase is not Phase.IDLE:
            self.phase = Phase.PAUSED

    def reset(self) -> None:
        self.optimizer = None
        self.generation = 0
        self.step_in_gen = 0
        self.fitness = None
        self._z = None
        self._buffer.clear()
        self._needs_write = False
        self.phase = Phase.IDLE

    def abort_generation(self) -> None:
        """A partial rollout is not a valid fitness sample. Used on resize,
        shader reload, and grid change."""
        self.step_in_gen = 0
        self._buffer.clear()
        self._next_snap = 0
        if self.phase is not Phase.IDLE:
            self._needs_write = True

    def _begin_generation(self) -> None:
        # The population size is fixed at optimizer construction (cmaes asserts
        # on it in tell()). If the grid changed by any route that did not reset
        # us, rebuild rather than crash on the next tell.
        if self.optimizer is not None and self.optimizer.popsize != self.popsize:
            self.optimizer = None
        self._ensure_optimizer()
        self._z = self.optimizer.ask(self.popsize)
        self.tournament.population = [decode(z) for z in self._z]
        self.tournament.mark_dirty()
        self._snaps = snapshot_steps(self.steps_per_gen, self.snapshots_per_gen)
        self._next_snap = 0
        self.step_in_gen = 0
        self._buffer.clear()
        self._needs_write = True

    # ---- per-frame driver ----------------------------------------------

    def update(self) -> Action:
        if self.phase in (Phase.IDLE, Phase.PAUSED):
            return Action.NONE

        if self._needs_write:
            self._needs_write = False
            return Action.WRITE_RULES

        if (self._next_snap < len(self._snaps)
                and self.step_in_gen >= self._snaps[self._next_snap]):
            self._next_snap += 1
            return Action.CAPTURE

        if self._next_snap >= len(self._snaps):
            self.phase = Phase.SCORE
            return Action.SCORE

        self.step_in_gen += self.sim_steps_per_frame
        return Action.STEP

    def submit_frames(self, crops: np.ndarray) -> None:
        self._buffer.append(np.asarray(crops, dtype=np.uint8))

    # ---- scoring -------------------------------------------------------

    def score_and_tell(self) -> np.ndarray:
        n = self.popsize
        if self.scorer is None or not self._buffer:
            fit = np.zeros(n, dtype=np.float32)
        else:
            per_snap = [np.asarray(self.scorer.score(c), dtype=np.float32)
                        for c in self._buffer]
            fit = np.mean(np.stack(per_snap, axis=0), axis=0).astype(np.float32)

        bad = ~np.isfinite(fit)
        if bad.any():
            finite = fit[~bad]
            fit[bad] = float(finite.min()) if finite.size else 0.0

        # Elite injection: selected tiles are already in this population, so
        # this overwrites their fitness to force them to the top ranks.
        selected = sorted(self.tournament.selected)
        if selected:
            top = float(fit.max())
            for rank, tile in enumerate(selected):
                if 0 <= tile < n:
                    fit[tile] = top + 1e-3 * (len(selected) - rank)

        self.optimizer.tell(self._z, fit)
        self.fitness = fit
        self.generation += 1
        self.tournament.selected.clear()

        if self.logger is not None:
            self.logger.log_generation({
                "gen": self.generation,
                "prompt": self.prompt,
                "algorithm": self.algorithm,
                "sigma": self.sigma,
                "popsize": n,
                "grid": self.tournament.grid,
                "seed": self.gen_seed,
                "fit_best": float(fit.max()),
                "fit_mean": float(fit.mean()),
                "fit_median": float(np.median(fit)),
                "fit_min": float(fit.min()),
                "fit_std": float(fit.std()),
                "best_tile": int(np.argmax(fit)),
                "steps_per_gen": self.steps_per_gen,
                "snapshots": self.snapshots_per_gen,
                "elites_injected": len(selected),
                "tile_mutation": bool(self.tile_mutation_enabled),
                "nan_replaced": int(bad.sum()),
            })

        self._begin_generation()
        self.phase = Phase.ROLLOUT
        return fit

    # ---- checkpointing -------------------------------------------------

    def checkpoint_state(self) -> dict:
        best_z, best_f = (self.optimizer.best() if self.optimizer
                          else (np.zeros(self.spec.dim, np.float32), -np.inf))
        return {
            "genome_spec_signature": self.spec.signature(),
            "generation": self.generation,
            "optimizer_name": self.algorithm,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else {},
            "base_seed": self.base_seed,
            "prompt": self.prompt,
            "distractors": [],
            "settings": {
                "grid": self.tournament.grid,
                "steps_per_gen": self.steps_per_gen,
                "snapshots_per_gen": self.snapshots_per_gen,
                "sim_steps_per_frame": self.sim_steps_per_frame,
                "sigma0": self.sigma0,
                "autosave_every": self.autosave_every,
                "tile_mutation_enabled": self.tile_mutation_enabled,
                "variants_per_tile": self.variants_per_tile,
                "tile_mutation_strength": self.tile_mutation_strength,
            },
            "history": self.logger.history() if self.logger else {},
            "best_z": best_z,
            "best_fitness": float(best_f) if np.isfinite(best_f) else 0.0,
        }

    def restore(self, state: dict) -> None:
        s = state["settings"]
        # Grid and popsize are inseparable from the checkpoint: cmaes.CMA fixes
        # its population at construction.
        self.tournament.set_grid(int(s["grid"]))
        self.configure(
            steps_per_gen=int(s["steps_per_gen"]),
            snapshots_per_gen=int(s["snapshots_per_gen"]),
            sim_steps_per_frame=int(s["sim_steps_per_frame"]),
            sigma0=float(s["sigma0"]),
            autosave_every=int(s.get("autosave_every", 10)),
            tile_mutation_enabled=bool(s.get("tile_mutation_enabled", False)),
            variants_per_tile=int(s.get("variants_per_tile", 4)),
            tile_mutation_strength=float(s.get("tile_mutation_strength", 0.1)),
            algorithm=str(state["optimizer_name"]),
        )
        self.base_seed = int(state["base_seed"])
        self.generation = int(state["generation"])
        self.optimizer = None
        self._ensure_optimizer()
        if state.get("optimizer_state"):
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.logger is not None and state.get("history"):
            self.logger.load_history(state["history"])
        self.set_prompt(str(state["prompt"]))
        self.phase = Phase.PAUSED
