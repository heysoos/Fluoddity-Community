"""Intrinsically-motivated goal exploration over the archive.

Expedition & Expansion (arXiv:2509.03863) on a substrate that runs N^2 rollouts
per generation. E&E's loop is serial - one theta, one rollout, one embedding -
so one generation here is a batch of 4 to 64 IMGEP samples at no extra cost.

Three regimes, checked in order:

  BOOTSTRAP    archive < seed_n, or empty. z ~ N(0, sigma0): a scattered
               population, and the archive admits on viability and liveness
               only (5.2.2).
  EXPEDITION   an expedition is active: optimizer.ask, fitness <b, g>.
  EXPANSION    otherwise: per tile, sample a parent with p ~ NOV^alpha,
               re-encode it under the CURRENT physics origin, add Gaussian
               noise. No optimizer, no covariance, no shared state between
               tiles - the population is a sample, not a distribution.

Deliberate deviation from E&E: every expedition generation's tiles also go
through the admission gate, not just the endpoint. E&E archives only the final
optimised solution because its expeditions are a serial inner loop; here the
embeddings are already computed and the path toward a goal is itself territory.
"""
from __future__ import annotations

import numpy as np

from services.archive import Candidate
from services.capture_health import is_viable_tile
from services.descriptor import descriptor, liveness, stack_snapshots
from services.genome_spec import BRAIN_SPEC, encode
from services.novelty import sample_by_novelty
from services.physics_genome import PHYSICS_DIM, PHYSICS_PARAMS, encode_physics


def _phys_dict(vec) -> dict[str, float]:
    return {n: float(v) for (n, _g, _lo, _hi), v in zip(PHYSICS_PARAMS, vec)}


class ImgepDriver:
    name = "imgep"

    def __init__(self, tournament, scorer, archive, goals=None,
                 spec=BRAIN_SPEC, rng=None):
        self.tournament = tournament
        self.scorer = scorer
        self.archive = archive
        self.goals = goals
        self.spec = spec
        self.rng = rng if rng is not None else np.random.default_rng()

        # pushed by AutoTournamentService._sync_driver
        self.algorithm = "CMA-ES"
        self.sigma0 = 0.5
        self.base_seed = 1000
        self.physics_origin: dict[str, float] = {}
        self.physics_enabled = False
        self.run_id = ""

        # exploration settings (spec 7.4)
        self.sigma_expand = 0.15
        self.alpha = 4.0
        self.k = 10
        self.seed_n = 256
        self.liveness_min = 0.02
        self.refresh_per_gen = 64
        self.flush_every = 200

        self.gen = 0
        self._last_score_label = "novelty"

    # ---- state ---------------------------------------------------------

    @property
    def regime(self) -> str:
        # max(1, ...): an empty archive has nothing to expand FROM, so it stays
        # in bootstrap whatever seed_n says. Without this, parent sampling
        # raises on the first ask when seed_n is 0.
        if len(self.archive) < max(1, self.seed_n):
            return "bootstrap"
        return "expansion"

    @property
    def optimizer(self):
        return None

    @property
    def sigma(self) -> float:
        return float(self.sigma_expand)

    @property
    def goal_label(self) -> str:
        return ""

    def status(self) -> dict:
        st = self.archive.stats()
        return {
            "regime": self.regime,
            "goal": self.goal_label,
            "archive_size": st["size"],
            "threshold": st["threshold"],
            "admission_rate": st["admission_rate"],
            "n_pinned": st["n_pinned"],
            "blocked_by_pins": st["blocked_by_pins"],
            "score_label": self._last_score_label,
            "sigma": self.sigma,
            "algorithm": self.algorithm,
            "prompt": self.goal_label,
        }

    # ---- driver interface ----------------------------------------------

    def set_spec(self, spec) -> None:
        self.spec = spec

    def reset(self) -> None:
        """Clears the SEARCH, never the archive. The archive is the product;
        Reset is about abandoning the current trajectory through it."""
        self.gen = 0

    def ask(self, n: int) -> np.ndarray:
        n = int(n)
        if self.regime == "bootstrap":
            return self._ask_bootstrap(n)
        return self._ask_expansion(n)

    def _ask_bootstrap(self, n: int) -> np.ndarray:
        return (self.sigma0 * self.rng.normal(size=(n, self.spec.dim))
                ).astype(np.float32)

    def _ask_expansion(self, n: int) -> np.ndarray:
        nov = np.array([e.novelty for e in self.archive.entries], dtype=np.float32)
        idx = sample_by_novelty(nov, n, self.rng, self.alpha)
        out = np.empty((n, self.spec.dim), dtype=np.float32)
        for j, i in enumerate(idx):
            out[j] = self._parent_z(int(i))
        out += (self.sigma_expand * self.rng.normal(size=out.shape)).astype(np.float32)
        return out

    def _parent_z(self, i: int) -> np.ndarray:
        """Re-encode an archived PHENOTYPE into z under the CURRENT origin.

        The archive stores decoded values precisely so this works: a z archived
        under one preset would mean a different creature under another."""
        zb, _clamped = encode(self.archive.brains[i])
        if self.spec.dim <= len(zb):
            return zb[: self.spec.dim].astype(np.float32)
        entry = self.archive.entries[i]
        if "physics" in entry.spec:
            zp = encode_physics(_phys_dict(self.archive.physics[i]),
                                self.physics_origin)
        else:
            # z = 0 decodes to the current origin exactly, so a brain-only
            # entry is well-defined rather than an error.
            zp = np.zeros(PHYSICS_DIM, dtype=np.float32)
        return np.concatenate([zb, zp]).astype(np.float32)

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray]) -> np.ndarray:
        n = len(z)
        if not snapshots:
            return np.zeros(n, dtype=np.float32)

        per_snap = [np.asarray(self.scorer.embed(c, n_views=1), dtype=np.float32)
                    for c in snapshots]
        snaps = stack_snapshots(per_snap)
        b = descriptor(snaps)
        live = liveness(snaps)
        nov = self.archive.novelty_of(b)

        # One snapshot is no evidence of change; rejecting everything would be
        # wrong rather than conservative.
        self.archive.liveness_min = (
            float(self.liveness_min) if len(snapshots) >= 2 else 0.0)
        self.archive.k = int(self.k)
        self.archive.seed_n = int(self.seed_n)

        last = snapshots[-1]
        pinned = set(self.tournament.selected)
        source = self.regime
        goal_text = self.goal_label
        parts = [self.spec.decode(zi) for zi in z]

        for i in range(n):
            phys = parts[i].get("physics")
            if phys is not None and self.physics_enabled:
                from services.physics_genome import decode_physics

                values = decode_physics(phys, self.physics_origin)
                phys_vec = np.array([values[nm] for nm, _g, _lo, _hi in PHYSICS_PARAMS],
                                    dtype=np.float32)
            else:
                phys_vec = np.zeros(PHYSICS_DIM, dtype=np.float32)

            self.archive.consider(
                Candidate(
                    brain=np.asarray(parts[i]["brain"], dtype=np.float32),
                    physics=phys_vec,
                    embedding=b[i],
                    liveness=float(live[i]),
                    viable=bool(is_viable_tile(last[i])),
                    spec=self.spec.signature(),
                    gen=int(self.gen),
                    tile=int(i),
                    run_id=self.run_id,
                    goal=goal_text,
                ),
                float(nov[i]),
                pinned=(i in pinned),
                source=source,
                thumb_crop=last[i],
            )

        self.tournament.selected.clear()
        self.gen += 1
        self.archive.refresh(self.refresh_per_gen)
        self.archive.maybe_flush(every=self.flush_every)

        self._last_score_label = "novelty"
        return np.asarray(nov, dtype=np.float32)

    # ---- checkpointing -------------------------------------------------

    def checkpoint_state(self) -> dict:
        return {
            "optimizer_name": self.algorithm,
            "optimizer_state": {},
            "prompt": self.goal_label,
            "distractors": [],
            "best_z": np.zeros(self.spec.dim, dtype=np.float32),
            "best_fitness": 0.0,
            "imgep_gen": int(self.gen),
            "imgep_threshold": float(self.archive.threshold.value),
        }

    def restore(self, d: dict) -> None:
        self.gen = int(d.get("imgep_gen", 0))
        if "imgep_threshold" in d:
            self.archive.threshold.value = float(d["imgep_threshold"])
