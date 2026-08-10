"""Evolution strategies for the automatic tournament.

Pure numpy plus the `cmaes` library. No OpenGL, no CLIP, no app state.

Sign convention: `tell` takes fitness where HIGHER IS BETTER. Implementations
negate internally (cmaes minimizes). Keeping the convention in one place stops
it leaking into the loop.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from services.genome import crossover, mutate


class Optimizer(Protocol):
    name: str

    def ask(self, n: int) -> np.ndarray: ...
    def tell(self, z: np.ndarray, fitness: np.ndarray) -> None: ...
    def best(self) -> tuple[np.ndarray, float]: ...
    @property
    def popsize(self) -> int: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...
    @property
    def sigma(self) -> float: ...


def _rng_state(rng) -> dict:
    """Capture a numpy RandomState or Generator without pickling."""
    if hasattr(rng, "get_state"):  # RandomState, as used by cmaes
        kind, keys, pos, has_gauss, cached = rng.get_state()
        return {
            "rng_kind": str(kind),
            "rng_keys": np.asarray(keys, dtype=np.uint32),
            "rng_pos": int(pos),
            "rng_has_gauss": int(has_gauss),
            "rng_cached": float(cached),
        }
    st = rng.bit_generator.state  # Generator (PCG64)
    return {
        "bg_state": str(st["state"]["state"]),
        "bg_inc": str(st["state"]["inc"]),
        "bg_has_uint32": int(st["has_uint32"]),
        "bg_uinteger": int(st["uinteger"]),
    }


def _restore_rng(rng, d: dict) -> None:
    if "rng_keys" in d:
        rng.set_state((
            str(d["rng_kind"]),
            np.asarray(d["rng_keys"], dtype=np.uint32),
            int(d["rng_pos"]),
            int(d["rng_has_gauss"]),
            float(d["rng_cached"]),
        ))
        return
    st = rng.bit_generator.state
    st["state"]["state"] = int(d["bg_state"])
    st["state"]["inc"] = int(d["bg_inc"])
    st["has_uint32"] = int(d["bg_has_uint32"])
    st["uinteger"] = int(d["bg_uinteger"])
    rng.bit_generator.state = st


class _BaseOptimizer:
    name = "base"

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None, layout=None):
        # Only the GA reads the layout - it is the one operator with structure.
        # It lives on the base anyway so make_optimizer can forward it without
        # knowing which algorithm it is building.
        self._layout = layout
        self._dim = int(dim)
        self._popsize = int(popsize)
        self._sigma0 = float(sigma0)
        self._seed = int(seed)
        self._x0 = (
            np.zeros(self._dim, dtype=np.float64)
            if x0 is None
            else np.asarray(x0, dtype=np.float64).reshape(-1)
        )
        self._best_z = None
        self._best_f = -np.inf

    def _track_best(self, z, fitness):
        i = int(np.argmax(fitness))
        if float(fitness[i]) > self._best_f:
            self._best_f = float(fitness[i])
            self._best_z = np.asarray(z[i], dtype=np.float32).copy()

    @property
    def popsize(self) -> int:
        return self._popsize

    def best(self):
        if self._best_z is None:
            return np.zeros(self._dim, dtype=np.float32), -np.inf
        return self._best_z.copy(), self._best_f

    def _base_state(self):
        return {
            "dim": self._dim,
            "popsize": self._popsize,
            "sigma0": self._sigma0,
            "seed": self._seed,
            "best_z": (
                self._best_z
                if self._best_z is not None
                else np.zeros(self._dim, dtype=np.float32)
            ),
            "best_f": self._best_f if np.isfinite(self._best_f) else 0.0,
            "has_best": int(self._best_z is not None),
        }

    def _load_base(self, d):
        self._dim = int(d["dim"])
        self._popsize = int(d["popsize"])
        self._sigma0 = float(d["sigma0"])
        self._seed = int(d["seed"])
        self._best_z = (
            np.asarray(d["best_z"], dtype=np.float32) if int(d["has_best"]) else None
        )
        self._best_f = float(d["best_f"]) if int(d["has_best"]) else -np.inf


class _CMAFamily(_BaseOptimizer):
    """Wraps cmaes.CMA / cmaes.SepCMA.

    state_dict captures every numpy array and scalar attribute rather than an
    allowlist of the ones believed mutable. Constants restore to the same value
    harmlessly, and nothing can be silently missed when the library changes.
    Never a pickle - checkpoints must not be tied to a library version.
    """

    _cls = None

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None, layout=None):
        super().__init__(dim, popsize, sigma0, seed, x0, layout)
        self._cma = self._new()

    def _new(self):
        return self._cls(
            mean=self._x0.copy(),
            sigma=self._sigma0,
            population_size=self._popsize,
            seed=self._seed,
        )

    @property
    def sigma(self) -> float:
        return float(getattr(self._cma, "_sigma", self._sigma0))

    def ask(self, n: int) -> np.ndarray:
        return np.array(
            [self._cma.ask() for _ in range(n)], dtype=np.float32
        ).reshape(n, self._dim)

    def tell(self, z: np.ndarray, fitness: np.ndarray) -> None:
        self._track_best(z, fitness)
        # cmaes minimizes, so negate.
        solutions = [
            (np.asarray(zi, dtype=np.float64), -float(f)) for zi, f in zip(z, fitness)
        ]
        self._cma.tell(solutions)

    def state_dict(self) -> dict:
        d = self._base_state()
        arrays, ints, floats, bools = [], [], [], []
        for k, v in vars(self._cma).items():
            if k == "_rng":
                continue
            if isinstance(v, np.ndarray):
                d["cma" + k] = v
                arrays.append(k)
            elif isinstance(v, (bool, np.bool_)):
                d["cma" + k] = int(v)
                bools.append(k)
            elif isinstance(v, (int, np.integer)):
                d["cma" + k] = int(v)
                ints.append(k)
            elif isinstance(v, (float, np.floating)):
                d["cma" + k] = float(v)
                floats.append(k)
            # None and other objects (e.g. _bounds) are left at their fresh value
        d["cma_arrays"] = arrays
        d["cma_ints"] = ints
        d["cma_floats"] = floats
        d["cma_bools"] = bools
        d.update(_rng_state(self._cma._rng))
        return d

    def load_state_dict(self, d: dict) -> None:
        self._load_base(d)
        self._cma = self._new()
        for k in list(d["cma_arrays"]):
            setattr(self._cma, str(k), np.asarray(d["cma" + str(k)]))
        for k in list(d["cma_ints"]):
            setattr(self._cma, str(k), int(d["cma" + str(k)]))
        for k in list(d["cma_floats"]):
            setattr(self._cma, str(k), float(d["cma" + str(k)]))
        for k in list(d["cma_bools"]):
            setattr(self._cma, str(k), bool(int(d["cma" + str(k)])))
        _restore_rng(self._cma._rng, d)


class CMAESOptimizer(_CMAFamily):
    name = "CMA-ES"

    def __init__(self, *a, **kw):
        from cmaes import CMA

        self._cls = CMA
        super().__init__(*a, **kw)


class SepCMAESOptimizer(_CMAFamily):
    name = "Sep-CMA-ES"

    def __init__(self, *a, **kw):
        from cmaes import SepCMA

        self._cls = SepCMA
        super().__init__(*a, **kw)


class RandomSearchOptimizer(_BaseOptimizer):
    """The control. If CMA-ES cannot beat this, the fitness signal is not real."""

    name = "Random Search"

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None, layout=None):
        super().__init__(dim, popsize, sigma0, seed, x0, layout)
        self._rng = np.random.default_rng(seed)

    @property
    def sigma(self) -> float:
        return self._sigma0

    def ask(self, n: int) -> np.ndarray:
        return (
            self._x0 + self._sigma0 * self._rng.normal(0, 1, (n, self._dim))
        ).astype(np.float32)

    def tell(self, z, fitness) -> None:
        self._track_best(z, fitness)

    def state_dict(self) -> dict:
        d = self._base_state()
        d.update(_rng_state(self._rng))
        return d

    def load_state_dict(self, d: dict) -> None:
        self._load_base(d)
        self._rng = np.random.default_rng(self._seed)
        _restore_rng(self._rng, d)


class GAOptimizer(_BaseOptimizer):
    """Manual mode's operator, driven by CLIP instead of by a human.

    For Fourier it reuses services.genome.mutate/crossover, applied in z-space
    by reshaping to the (10, 8) layout those operators expect. Other modalities
    have no such row structure and get a generic per-gene operator - see
    _breed().
    """

    name = "GA"
    ELITES = 4
    MUT = 0.25

    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None, layout=None):
        super().__init__(dim, popsize, sigma0, seed, x0, layout)
        self._rng = np.random.default_rng(seed)
        self._pop = self._fresh(popsize)
        self._told = False

    def _fresh(self, n):
        return (
            self._x0 + self._sigma0 * self._rng.normal(0, 1, (n, self._dim))
        ).astype(np.float32)

    @property
    def sigma(self) -> float:
        return float(self._pop.std()) if self._told else self._sigma0

    def ask(self, n: int) -> np.ndarray:
        if len(self._pop) != n:
            self._pop = self._fresh(n)
        return self._pop.copy()

    def tell(self, z, fitness) -> None:
        z = np.asarray(z, dtype=np.float32)
        fitness = np.asarray(fitness, dtype=np.float32)
        self._track_best(z, fitness)
        self._told = True
        order = np.argsort(-fitness)
        n = len(z)
        elites = z[order[: min(self.ELITES, n)]]
        nxt = [e.copy() for e in elites]
        while len(nxt) < n:
            a = elites[self._rng.integers(len(elites))]
            b = elites[self._rng.integers(len(elites))]
            nxt.append(self._breed(a, b))
        self._pop = np.array(nxt[:n], dtype=np.float32)

    def _breed(self, a, b):
        """One child from two elites, in z-space.

        Fourier keeps per-CENTRE crossover, because a centre is a unit: its
        eight floats are one frequency vector and one amplitude, and splitting
        them apart makes a child that is neither parent's feature. No other
        modality has that row structure - Gabor is 14 floats, MLP is not a grid
        - so they get uniform per-gene crossover, which is the honest generic
        operator. Blending would be worse than either: it invents values neither
        parent held.

        reshape(-1, 8) was also a hard error for any length not divisible by 8,
        which MLP at H=16 (148 floats) is.
        """
        if self._layout is None or self._layout.modality == "fourier":
            child = crossover(a.reshape(-1, 8), b.reshape(-1, 8), self._rng)
            return mutate(child, self.MUT, self._rng).reshape(-1).astype(np.float32)
        take_a = self._rng.random(a.shape) < 0.5
        child = np.where(take_a, a, b)
        child = child + self.MUT * self._rng.normal(0, 1, a.shape)
        return child.astype(np.float32)

    def state_dict(self) -> dict:
        d = self._base_state()
        d["pop"] = self._pop
        d["told"] = int(self._told)
        d.update(_rng_state(self._rng))
        return d

    def load_state_dict(self, d: dict) -> None:
        self._load_base(d)
        self._pop = np.asarray(d["pop"], dtype=np.float32)
        self._told = bool(int(d["told"]))
        self._rng = np.random.default_rng(self._seed)
        _restore_rng(self._rng, d)


ALGORITHMS: dict[str, type] = {
    "CMA-ES": CMAESOptimizer,
    "Sep-CMA-ES": SepCMAESOptimizer,
    "GA": GAOptimizer,
    "Random Search": RandomSearchOptimizer,
}


def make_optimizer(name, dim, popsize, sigma0=0.5, seed=0, x0=None,
                   layout=None) -> Optimizer:
    """`layout` is the BrainLayout the brain block of z decodes under.

    Only the GA reads it, but it must be forwarded unconditionally: this is the
    single route the app uses, and while it did not take the argument at all,
    GAOptimizer._breed fell back to its Fourier branch for every modality and
    raised on three of the four.
    """
    return ALGORITHMS[name](dim, popsize, sigma0, seed, x0, layout)
