"""Today's Auto (Prompt) behaviour, extracted behind SearchDriver.

This is a mechanical extraction of AutoTournamentService's optimizer half, not a
rewrite. Its correctness gate is that the eight pre-existing auto-mode test
files pass UNMODIFIED once the service is rewired onto it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from services.genome_spec import layout_of, spec_for
from services.optimizers import make_optimizer
from services.vision_scorer import load_goal_image


class PromptDriver:
    name = "prompt"

    def __init__(self, tournament, scorer=None, spec=None):
        self.tournament = tournament
        self.scorer = scorer
        # From the tournament when unnamed, never Fourier by default - see
        # genome_spec.layout_of.
        self.spec = spec if spec is not None else spec_for(layout_of(tournament))
        self.algorithm = "CMA-ES"
        self.sigma0 = 0.5
        self.base_seed = 1000
        self.prompt = ""
        # The goal is the prompt OR a picture, never both: a set path is what
        # makes the goal an image, and set_prompt clears it.
        self.goal_image = ""
        self.goal_distractors = True
        self._optimizer = None
        self._x0 = None
        self._n_nan = 0
        self._n_elites = 0

    # ---- state exposed to the service and the UI -----------------------

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def sigma(self) -> float:
        return float(self._optimizer.sigma) if self._optimizer else float(self.sigma0)

    @property
    def goal_kind(self) -> str:
        return "image" if self.goal_image else "text"

    @property
    def goal_label(self) -> str:
        """What the run log and the tab call the goal."""
        if self.goal_image:
            return f"image: {Path(self.goal_image).name}"
        return self.prompt

    def status(self) -> dict:
        cosine = self.goal_kind == "image" and not self.goal_distractors
        return {
            "prompt": self.goal_label,
            "goal_kind": self.goal_kind,
            "algorithm": self.algorithm,
            "sigma": self.sigma,
            "nan_replaced": int(self._n_nan),
            "elites_injected": int(self._n_elites),
            "score_label": "cosine" if cosine else "fitness",
        }

    # ---- configuration -------------------------------------------------

    def set_spec(self, spec) -> None:
        # Always adopt the object - its scales decide what decode() produces -
        # but only discard the optimizer when the SPACE actually changed.
        same = self.spec.same_space_as(spec)
        self.spec = spec
        if not same:
            self._optimizer = None

    def set_prompt(self, text: str) -> None:
        """Changing the prompt keeps the learned covariance and simply starts
        climbing a new landscape."""
        self.prompt = text
        self.goal_image = ""
        if self.scorer is not None:
            self.scorer.set_prompt(text)

    def set_image_goal(self, path: str, distractors: bool = True) -> None:
        """Make a picture on disk the goal. Raises OSError on a file that
        cannot be read, and then changes nothing - the previous goal stands."""
        if self.scorer is not None:
            image = load_goal_image(path, self.scorer.model.px)
            self.scorer.set_image_goal(image, distractors=bool(distractors))
        elif not Path(path).is_file():
            raise FileNotFoundError(path)
        self.goal_image = str(path)
        self.goal_distractors = bool(distractors)
        self.prompt = ""

    def set_x0(self, z: np.ndarray) -> None:
        """Load a genome as the search starting point. Discards optimizer state;
        sigma, algorithm and grid come from the UI, not the file."""
        self._optimizer = None
        self._x0 = np.asarray(z, dtype=np.float64).reshape(-1)
        self._ensure(self.tournament.tiles)

    def reset(self) -> None:
        self._optimizer = None
        self._x0 = None

    # ---- the driver interface ------------------------------------------

    def _ensure(self, popsize: int) -> None:
        # The population size is fixed at construction (cmaes asserts on it in
        # tell()). If the grid changed by any route that did not reset us,
        # rebuild rather than crash on the next tell.
        if self._optimizer is not None and self._optimizer.popsize != popsize:
            self._optimizer = None
        if self._optimizer is None:
            self._optimizer = make_optimizer(
                self.algorithm, self.spec.dim, popsize,
                self.sigma0, self.base_seed, self._x0,
                layout=self.spec.layout,
            )

    def ask(self, n: int) -> np.ndarray:
        self._ensure(int(n))
        return self._optimizer.ask(int(n))

    def precompute(self, snapshots: list[np.ndarray]):
        """CLIP, and nothing else. Runs OFF the main thread; see
        ImgepDriver.precompute for why the split is here."""
        if self.scorer is None or not snapshots:
            return None
        return [np.asarray(self.scorer.score(c), dtype=np.float32)
                for c in snapshots]

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray],
             pre=None) -> np.ndarray:
        n = len(z)
        if self.scorer is None or not snapshots:
            fit = np.zeros(n, dtype=np.float32)
        else:
            per_snap = pre if pre is not None else self.precompute(snapshots)
            fit = np.mean(np.stack(per_snap, axis=0), axis=0).astype(np.float32)

        bad = ~np.isfinite(fit)
        self._n_nan = int(bad.sum())
        if bad.any():
            finite = fit[~bad]
            fit[bad] = float(finite.min()) if finite.size else 0.0

        # Elite injection: selected tiles are already in this population, so
        # this overwrites their fitness to force them to the top ranks.
        selected = sorted(self.tournament.selected)
        self._n_elites = len(selected)
        if selected:
            top = float(fit.max())
            for rank, tile in enumerate(selected):
                if 0 <= tile < n:
                    fit[tile] = top + 1e-3 * (len(selected) - rank)

        self._ensure(n)
        self._optimizer.tell(z, fit)
        self.tournament.selected.clear()
        return fit

    # ---- checkpointing -------------------------------------------------

    def checkpoint_state(self) -> dict:
        best_z, best_f = (self._optimizer.best() if self._optimizer
                          else (np.zeros(self.spec.dim, np.float32), -np.inf))
        return {
            "optimizer_name": self.algorithm,
            "optimizer_state": self._optimizer.state_dict() if self._optimizer else {},
            "prompt": self.prompt,
            "goal_kind": self.goal_kind,
            "goal_image": self.goal_image,
            "goal_distractors": bool(self.goal_distractors),
            "distractors": [],
            "best_z": best_z,
            "best_fitness": float(best_f) if np.isfinite(best_f) else 0.0,
        }

    def restore(self, d: dict) -> None:
        self.algorithm = str(d.get("optimizer_name", self.algorithm))
        self._optimizer = None
        self._x0 = None
        self._ensure(self.tournament.tiles)
        if d.get("optimizer_state"):
            self._optimizer.load_state_dict(d["optimizer_state"])
        # The goal last, so a picture that has gone leaves the optimizer
        # restored and the previous goal standing.
        if d.get("goal_kind", "text") == "image" and d.get("goal_image"):
            self.set_image_goal(str(d["goal_image"]),
                                bool(d.get("goal_distractors", True)))
        else:
            self.set_prompt(str(d.get("prompt", "")))
