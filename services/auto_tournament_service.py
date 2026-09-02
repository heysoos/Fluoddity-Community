"""The automatic tournament generation state machine.

Owns the ROLLOUT ONLY. Which genomes to run, and what the resulting images
mean, both belong to an injected SearchDriver - PromptDriver for Auto (Prompt),
ImgepDriver for Explore. The split exists because the rollout machine is
identical for every search while the optimizer and scoring are not.

Contains no GL calls, no ImGui, and - since the driver extraction - no CLIP,
so the whole loop is testable with fakes.

The rollout is spread across real application frames - update() returns one
action per call and never loops internally. That is the mechanism by which the
app stays responsive; it is not an optimization to be added later.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import numpy as np

from services.genome_spec import layout_of, physics_spec_for, spec_for
from services.physics_genome import decode_physics
from services.prompt_driver import PromptDriver


class Action(str, Enum):
    NONE = "none"
    WRITE_RULES = "write_rules"
    STEP = "step"
    CAPTURE = "capture"
    SCORE = "score"


_PENDING = object()      # the CLIP pass has not finished; no fitness this frame


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
                 spec=None, base_seed: int = 1000, driver=None):
        self.tournament = tournament_service
        self.logger = logger
        # The search space is DERIVED from the active brain. Holding the layout
        # rather than the spec is what lets _resolve_spec re-derive it when the
        # physics block is toggled without forgetting which brain it is for.
        #
        # With no spec the layout comes from the TOURNAMENT, which is the object
        # that already knows which brain is running. It used to default to
        # Fourier, and main builds this lazily the first time Auto mode is
        # opened - so a brain selected before that was never delivered, and
        # _resolve_spec re-derived from the stale layout every generation rather
        # than correcting it. The service then wrote Fourier-width genomes into
        # the tournament's population, which the next manual generation could
        # not breed.
        self._layout = (spec.layout if spec is not None
                        else layout_of(tournament_service))
        self.spec = spec if spec is not None else spec_for(self._layout)
        self.base_seed = int(base_seed)
        # The service owns the rollout machine; the driver owns what to run and
        # what the pictures mean. Injecting it is how Explore mode reuses this
        # whole state machine rather than keeping a second copy of it.
        self.driver = (driver if driver is not None
                       else PromptDriver(tournament_service, scorer, spec))

        self.algorithm = "CMA-ES"
        self.sigma0 = 0.5
        self.steps_per_gen = 300
        self.snapshots_per_gen = 4
        self.sim_steps_per_frame = 10
        self.physics_enabled = False
        self.tile_physics: list[dict] = []
        # Current physics, so z=0 decodes to the LOADED PRESET rather than the
        # slider midpoint - a dead, propulsion-less configuration CMA-ES would
        # otherwise centre its search on.
        self.physics_origin: dict[str, float] = {}
        self.tile_mutation_enabled = False
        self.variants_per_tile = 4
        self.tile_mutation_strength = 0.1
        # Score the capture as density alone. See CLAUDE.md.
        self.grayscale = False
        self.autosave_every = 10

        self.phase = Phase.IDLE
        self.generation = 0
        self.step_in_gen = 0
        self.fitness: np.ndarray | None = None

        self._z = None
        self._snaps: list[int] = []
        self._next_snap = 0
        self._buffer: list[np.ndarray] = []
        self._needs_write = False
        # A reset's write, which is honoured whatever the phase. See update().
        self._force_write = False

        # Scoring runs off the frame loop. See _precomputed().
        self.async_scoring = True
        self._pool: ThreadPoolExecutor | None = None
        self._pre_future = None

    # ---- delegated to the driver ---------------------------------------
    # These exist so the service's public surface survived the extraction
    # unchanged. command_handler.py and every pre-existing test reach through
    # them; a rename here is a silent break there.

    @property
    def optimizer(self):
        """Read-only on purpose: any leftover `self.optimizer = None` in this
        file must fail loudly rather than silently shadow the driver's."""
        return getattr(self.driver, "optimizer", None)

    @property
    def scorer(self):
        return getattr(self.driver, "scorer", None)

    @scorer.setter
    def scorer(self, value):
        self.driver.scorer = value

    @property
    def prompt(self) -> str:
        return getattr(self.driver, "prompt", "")

    @property
    def sigma(self) -> float:
        return float(getattr(self.driver, "sigma", self.sigma0))

    @property
    def run_id(self) -> str:
        return getattr(self.logger, "run_id", "") if self.logger else ""

    def set_prompt(self, text: str) -> None:
        self.driver.set_prompt(text)

    def set_image_goal(self, path: str, distractors: bool = True,
                       crop=(0.5, 0.5, 1.0)) -> None:
        self.driver.set_image_goal(path, distractors, crop,
                                   grayscale=self.grayscale)

    @property
    def goal_crop(self):
        return tuple(getattr(self.driver, "goal_crop", (0.5, 0.5, 1.0)))

    @property
    def goal_kind(self) -> str:
        return getattr(self.driver, "goal_kind", "text")

    @property
    def goal_image(self) -> str:
        return getattr(self.driver, "goal_image", "")

    @property
    def goal_distractors(self) -> bool:
        return bool(getattr(self.driver, "goal_distractors", True))

    def set_x0(self, z) -> None:
        self.driver.set_x0(z)

    def _sync_driver(self) -> None:
        """Push the UI-owned settings the driver understands, and the current
        genome layout. Only attributes the driver already has are set, so a
        driver may ignore settings that mean nothing to it."""
        for k in ("algorithm", "sigma0", "base_seed", "grayscale",
                  "physics_origin", "physics_enabled", "run_id"):
            if hasattr(self.driver, k):
                setattr(self.driver, k, getattr(self, k))
        self.driver.set_spec(self.spec)

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

    def set_layout(self, layout) -> None:
        """The active brain changed, so the search space did too.

        Nothing else can do this. _resolve_spec runs on every generation, so a
        spec set from outside is overwritten within one - which is exactly how
        the search stayed on Fourier's 80 floats while the GPU ran a 168-float
        Gabor brain.

        A generation already rolling out belongs to the OLD space, and so does
        any scoring pass in flight for it - which would otherwise land on the
        thread pool after the switch and be told about the new one. Dropped
        only when the space actually moved: BrainLayout keeps scales out of
        ==, so a decode-scale tweak still does not restart the search.
        """
        moved = self._layout != layout
        self._layout = layout
        self._resolve_spec()
        self._sync_driver()
        if moved:
            # The population in flight belongs to the OLD space, and a z does
            # not convert between spaces. abort_generation() restarts the
            # rollout WITHOUT re-asking, so leaving it standing meant the old
            # genomes reached tell() under the spec now in force - the crash
            # this drops it to avoid. update() asks for a new one.
            self._z = None
            self.tile_physics = []
            self.abort_generation()

    def _resolve_spec(self) -> None:
        """Point self.spec at the space the current settings imply.

        Rebuilt rather than cached: a decode-scale change produces an equal-but-
        distinct spec that must be adopted, since it is the layout's scales that
        decide what a z decodes to. The drivers compare with same_space_as, so
        rebuilding here does not disturb an optimizer mid-run.
        """
        self.spec = (physics_spec_for(self._layout) if self.physics_enabled
                     else spec_for(self._layout))

    # ---- lifecycle -----------------------------------------------------

    def start(self, prompt: str | None = None) -> None:
        if prompt is not None:
            self.set_prompt(prompt)
        self._resolve_spec()
        self._sync_driver()
        if self.phase is Phase.IDLE:
            self._begin_generation()
        if self.phase in (Phase.IDLE, Phase.PAUSED):
            self.phase = Phase.ROLLOUT

    def pause(self) -> None:
        if self.phase is not Phase.IDLE:
            self.phase = Phase.PAUSED

    def reset(self) -> None:
        # The plot reads logger.history(); leaving it would draw the abandoned
        # search's curve in front of the new one.
        if self.logger is not None:
            self.logger.start_new_run()
        # A run that replays the last one is not a reset. base_seed feeds BOTH
        # the optimizer and gen_seed, and generation goes back to 0 below, so
        # leaving it fixed handed back the identical population in the identical
        # tiles over the identical particle field - a new prompt only reranked
        # creatures the user had already watched. Advanced PAST the generations
        # just spent, so the two runs' gen_seeds cannot overlap either, and read
        # before the counter is cleared.
        self.base_seed += self.generation + 1
        self.driver.reset()
        self.generation = 0
        self.step_in_gen = 0
        self.fitness = None
        self._z = None
        self.tile_physics = []
        self._buffer.clear()
        self._drop_score()
        # Reset has to be VISIBLE. Nothing rewrote the grid until
        # _begin_generation, which only runs from start(), so the abandoned
        # search's creatures stayed on screen - and on the GPU - until Start:
        # a button that appeared to do nothing. A fresh random grid is what
        # "the search is abandoned" looks like, and Start then replaces it with
        # the optimizer's first generation.
        self.tournament.reset()
        self._needs_write = False
        self._force_write = True
        self.phase = Phase.IDLE

    def abort_generation(self) -> None:
        """A partial rollout is not a valid fitness sample. Used on resize,
        shader reload, and grid change."""
        self.step_in_gen = 0
        self._buffer.clear()
        self._drop_score()
        self._next_snap = 0
        if self.phase is not Phase.IDLE:
            self._needs_write = True

    def _begin_generation(self) -> None:
        # Turning physics search on or off changes the dimension of the search
        # space; _sync_driver hands the new spec to the driver, which discards
        # any optimizer built for the old one.
        self._resolve_spec()
        self._sync_driver()
        self._z = self.driver.ask(self.popsize)
        parts = [self.spec.decode(z) for z in self._z]
        self.tournament.population = [p["brain"] for p in parts]
        self.tile_physics = (
            [decode_physics(p["physics"], self.physics_origin) for p in parts]
            if self.physics_enabled else [])
        self.tournament.mark_dirty()
        self._snaps = snapshot_steps(self.steps_per_gen, self.snapshots_per_gen)
        self._next_snap = 0
        self.step_in_gen = 0
        self._buffer.clear()
        self._drop_score()
        self._needs_write = True

    # ---- per-frame driver ----------------------------------------------

    def update(self) -> Action:
        # A RESET's write outranks the phase gate: what the GPU is holding
        # belongs to a search that has been abandoned, and the user must see
        # that without pressing Start. A GENERATION's write does not - it is
        # queued behind the pause, or pausing between generations would
        # advance the picture to the next one instead of holding it.
        if self._force_write:
            self._force_write = False
            return Action.WRITE_RULES

        if self.phase in (Phase.IDLE, Phase.PAUSED):
            return Action.NONE

        # No population to score: the search space moved under a rollout that
        # was already in flight. Ask in the space now in force rather than
        # scoring genomes the spec cannot read.
        if self._z is None:
            self._begin_generation()

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

    def _score_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="score")
        return self._pool

    def _precomputed(self):
        """This generation's CLIP result, or _PENDING while it is still running.

        The first SCORE frame submits and hands the frame straight back; every
        later one polls. update() keeps returning Action.SCORE in the meantime,
        because the phase and the snapshot counter are both unchanged, so this
        needs no new state machine - only somewhere to put the future.

        -> None for a driver with no precompute() (the test fakes), which makes
        tell() do the whole thing inline exactly as before.

        The buffer is COPIED into the worker: abort_generation clears the list,
        and the worker must not see it emptied halfway through.
        """
        fn = getattr(self.driver, "precompute", None)
        if fn is None or not self.async_scoring:
            return None
        if self._pre_future is None:
            self._pre_future = self._score_pool().submit(fn, list(self._buffer))
            return _PENDING
        if not self._pre_future.done():
            return _PENDING
        f, self._pre_future = self._pre_future, None
        return f.result()               # re-raises on the main thread

    def _drop_score(self) -> None:
        """Abandon an in-flight CLIP pass.

        Its result describes a rollout that is no longer going to be scored,
        and self._buffer is about to be cleared underneath it. cancel() only
        bites if the worker has not started; either way the future is dropped,
        so a stale result can never reach tell().
        """
        f, self._pre_future = self._pre_future, None
        if f is not None:
            f.cancel()

    def close(self) -> None:
        """Let go of the scoring thread. Called from App.cleanup()."""
        self._drop_score()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def score_and_tell(self):
        """-> fitness, or None while the CLIP pass is still off-thread.

        None is not a failure and not the end of the generation: the caller
        gives the frame back to the UI and tries again next frame.
        """
        pre = self._precomputed()
        if pre is _PENDING:
            return None
        # Only pass it when there is one: a driver fake with a two-argument
        # tell() must keep working.
        raw = (self.driver.tell(self._z, self._buffer) if pre is None
               else self.driver.tell(self._z, self._buffer, pre))
        fit = np.asarray(raw, dtype=np.float32)
        self.fitness = fit
        self.generation += 1
        st = self.driver.status()

        if self.logger is not None:
            self.logger.log_generation({
                "gen": self.generation,
                "prompt": st.get("prompt", ""),
                "algorithm": st.get("algorithm", self.algorithm),
                "sigma": st.get("sigma", self.sigma),
                "popsize": len(fit),
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
                "elites_injected": int(st.get("elites_injected", 0)),
                "tile_mutation": bool(self.tile_mutation_enabled),
                "physics_search": bool(self.physics_enabled),
                "nan_replaced": int(st.get("nan_replaced", 0)),
                "driver": self.driver.name,
            })

        self._begin_generation()
        self.phase = Phase.ROLLOUT
        return fit

    # ---- checkpointing -------------------------------------------------

    def checkpoint_state(self) -> dict:
        d = self.driver.checkpoint_state()
        return {
            "genome_spec_signature": self.spec.signature(),
            # Widths alone cannot identify a search: Fourier at 21 centres and
            # Gabor at 12 filters are both "brain:168". Restoring one under the
            # other would hand CMA-ES a covariance about the wrong axes.
            "brain_layout_signature": self.spec.layout.signature(),
            "generation": self.generation,
            "optimizer_name": d.get("optimizer_name", self.algorithm),
            "optimizer_state": d.get("optimizer_state", {}),
            "base_seed": self.base_seed,
            "prompt": d.get("prompt", ""),
            "goal_kind": d.get("goal_kind", "text"),
            "goal_image": d.get("goal_image", ""),
            "goal_distractors": bool(d.get("goal_distractors", True)),
            "goal_crop": list(d.get("goal_crop", [0.5, 0.5, 1.0])),
            "distractors": d.get("distractors", []),
            "settings": {
                "grid": self.tournament.grid,
                "steps_per_gen": self.steps_per_gen,
                "snapshots_per_gen": self.snapshots_per_gen,
                "sim_steps_per_frame": self.sim_steps_per_frame,
                "sigma0": self.sigma0,
                "autosave_every": self.autosave_every,
                "tile_mutation_enabled": self.tile_mutation_enabled,
                "physics_enabled": self.physics_enabled,
                "variants_per_tile": self.variants_per_tile,
                "tile_mutation_strength": self.tile_mutation_strength,
                "grayscale": bool(self.grayscale),
            },
            "history": self.logger.history() if self.logger else {},
            "best_z": d.get("best_z", np.zeros(self.spec.dim, np.float32)),
            "best_fitness": float(d.get("best_fitness", 0.0)),
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
            physics_enabled=bool(s.get("physics_enabled", False)),
            variants_per_tile=int(s.get("variants_per_tile", 4)),
            tile_mutation_strength=float(s.get("tile_mutation_strength", 0.1)),
            grayscale=bool(s.get("grayscale", False)),
            algorithm=str(state["optimizer_name"]),
        )
        self.base_seed = int(state["base_seed"])
        self.generation = int(state["generation"])
        self._resolve_spec()
        self._sync_driver()
        self.driver.reset()
        self.driver.restore(state)
        if self.logger is not None and state.get("history"):
            self.logger.load_history(state["history"])
        self.phase = Phase.PAUSED
