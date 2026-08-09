"""Intrinsically-motivated goal exploration over the archive.

Expedition & Expansion (arXiv:2509.03863) on a substrate that runs N^2 rollouts
per generation. E&E's loop is serial - one theta, one rollout, one embedding -
so one generation here is a batch of 4 to 64 IMGEP samples at no extra cost.

Three regimes, checked in this order:

  EXPEDITION   an expedition is active: optimizer.ask, fitness <b, g>. First,
               because a chase must pre-empt the cadence immediately.
  BOOTSTRAP    archive < seed_n, or empty. z ~ N(0, sigma0): a scattered
               population, and the archive admits on viability and liveness
               only (5.2.2).
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
from services.archive_projection import Projection
from services.capture_health import is_viable_tile
from services.descriptor import descriptor, liveness, stack_snapshots
from services.expedition_fitness import (
    IMAGE_LOGIT_SCALE,
    TEXT_LOGIT_SCALE,
    contrastive,
)
from services.genome_spec import BRAIN_SPEC, encode
from services.goal_source import LATENT_DIMS, Goal, latent_goal
from services.novelty import (
    banded_alpha,
    effective_sample_size,
    sample_by_novelty,
)
from services.optimizers import make_optimizer
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
        self.liveness_min = 0.002    # measured; see state/archive_state.py
        self.refresh_sweep_gens = 10
        self.flush_every = 200

        # expedition settings (spec 7.4)
        self.expansion_between = 25      # 0 disables expeditions entirely
        self.expedition_gens = 50        # E&E uses 350; that is ~16 min at 2000 steps
        self.expedition_sigma = 0.1      # E&E's value; deliberately << sigma0
        self.latent_share = 0.5
        self.goal_order = "round_robin"  # or "least_matched"
        # Band on the seed pool, not a target - the spread of ESS across goals
        # is real information about the archive. See novelty.banded_alpha.
        self.seed_ess_min = 8.0
        self.seed_ess_max = 512.0

        # The search's OWN projection, separate from the map's 2-component one:
        # refitting between 2 and 8 components every frame would thrash both.
        self.projection = Projection(LATENT_DIMS)
        self._distractors = None

        self._optimizer = None
        self._goal: Goal | None = None
        self._remaining = 0
        self._since_expedition = 0
        self._x0_index: int | None = None
        self._last_descriptors: np.ndarray | None = None
        self._last_seed_ess = 0.0
        self._last_seed_alpha = 0.0

        self.gen = 0
        self._last_score_label = "novelty"
        self.trace = {k: [] for k in ("gen", "regime", "archive_size",
                                      "mean_novelty", "admitted", "tiles",
                                      "fit_best", "fit_mean")}

    # One row per generation, ~2.8 s each, so this is about 8 hours of run. The
    # plot subsamples anyway; the cap is only here so a machine left running
    # over a weekend does not grow a list without bound.
    TRACE_CAP = 10000

    # ---- state ---------------------------------------------------------

    @property
    def regime(self) -> str:
        if self._remaining > 0 and self._goal is not None:
            return "expedition"
        # max(1, ...): an empty archive has nothing to expand FROM, so it stays
        # in bootstrap whatever seed_n says. Without this, parent sampling
        # raises on the first ask when seed_n is 0.
        if len(self.archive) < max(1, self.seed_n):
            return "bootstrap"
        return "expansion"

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def sigma(self) -> float:
        if self._optimizer is not None:
            return float(self._optimizer.sigma)
        return float(self.sigma_expand)

    @property
    def goal_label(self) -> str:
        return self._goal.label if self._goal is not None else ""

    def phase(self) -> dict:
        """What this regime is working toward, and how far along it is.

        Every regime has a finish line, and none of them were visible: the tab
        showed the regime's NAME and nothing about where in it the search was.
        -> (label, done, total, note); total 0 means there is no finish line to
        show, which is the honest answer when expeditions are switched off.
        """
        r = self.regime
        if r == "expedition":
            total = max(1, int(self.expedition_gens))
            return {"label": f"expedition: {self.goal_label or 'latent goal'}",
                    "done": total - int(self._remaining), "total": total,
                    "unit": "generations",
                    "note": f"{int(self._remaining)} left"}
        if r == "bootstrap":
            total = max(1, int(self.seed_n))
            done = min(int(len(self.archive)), total)
            return {"label": "bootstrap: scattering to fill the archive",
                    "done": done, "total": total, "unit": "entries",
                    "note": f"{total - done} more before expansion starts"}
        if int(self.expansion_between) <= 0:
            return {"label": "expansion: no expeditions (Expansion Between = 0)",
                    "done": 0, "total": 0, "unit": "", "note": ""}
        total = int(self.expansion_between)
        done = min(int(self._since_expedition), total)
        return {"label": "expansion: breeding from novel parents",
                "done": done, "total": total, "unit": "generations",
                "note": f"next expedition in {total - done}"}

    def status(self) -> dict:
        st = self.archive.stats()
        return {
            "regime": self.regime,
            "goal": self.goal_label,
            "archive_size": st["size"],
            "capacity": st["capacity"],
            "n_evicted": st["n_evicted"],
            "admission_rate": st["admission_rate"],
            "n_pinned": st["n_pinned"],
            "blocked_by_pins": st["blocked_by_pins"],
            "n_rejected_dead": st["n_rejected_dead"],
            "n_rejected_close": st["n_rejected_close"],
            "min_separation": st["min_separation"],
            "mean_novelty": st["mean_novelty"],
            "score_label": self._last_score_label,
            "sigma": self.sigma,
            "algorithm": self.algorithm,
            "prompt": self.goal_label,
            "seed_ess": float(self._last_seed_ess),
            "seed_alpha": float(self._last_seed_alpha),
            "phase": self.phase(),
            "last_admitted": (self.trace["admitted"][-1]
                              if self.trace["admitted"] else 0),
            "last_tiles": (self.trace["tiles"][-1]
                           if self.trace["tiles"] else 0),
        }

    # ---- driver interface ----------------------------------------------

    def set_spec(self, spec) -> None:
        if spec is not self.spec:
            self.spec = spec
            # The search dimension changed; an optimizer for the old one is
            # meaningless, and the archive is unaffected because it stores
            # phenotypes rather than z.
            self.end_expedition()

    def reset(self) -> None:
        """Clears the SEARCH, never the archive. The archive is the product;
        Reset is about abandoning the current trajectory through it.

        The traces go with it. They are plotted against a generation counter
        that restarts at 0 here, so keeping them would draw the new run on top
        of the old one with no way to tell which was which.
        """
        self.gen = 0
        self._since_expedition = 0
        for v in self.trace.values():
            v.clear()
        self.end_expedition()

    def end_expedition(self) -> None:
        self._optimizer = None
        self._goal = None
        self._remaining = 0
        self._x0_index = None

    def ask(self, n: int) -> np.ndarray:
        n = int(n)
        if self.regime == "expedition":
            return self._ask_expedition(n)
        if self.regime == "bootstrap":
            return self._ask_bootstrap(n)
        return self._ask_expansion(n)

    def _ask_expedition(self, n: int) -> np.ndarray:
        # The population size is fixed at construction (cmaes asserts on it in
        # tell()). A grid change mid-expedition ends the expedition rather than
        # crashing on the next tell.
        if self._optimizer is not None and self._optimizer.popsize != n:
            self.end_expedition()
            return self._ask_expansion(n) if len(self.archive) else self._ask_bootstrap(n)
        return self._optimizer.ask(n)

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

    # ---- expeditions ---------------------------------------------------

    def start_expedition(self) -> bool:
        """Draw a goal from the configured sources and begin. -> did it start?"""
        goal = self._draw_goal()
        if goal is None:
            return False
        return self.start_expedition_with(goal.embedding, goal.kind, goal.text)

    def start_expedition_with(self, embedding, kind: str, text: str) -> bool:
        """Begin an expedition toward a specific embedding. An expedition needs
        a seed, so an empty archive falls back to expansion rather than
        starting a search from nowhere."""
        i = self._seed_index(np.asarray(embedding, dtype=np.float32), str(kind))
        if i is None or self.expedition_gens <= 0:
            return False
        self._goal = Goal(kind, text, np.asarray(embedding, dtype=np.float32))
        self._x0_index = int(i)
        self._remaining = int(self.expedition_gens)
        self._since_expedition = 0
        # A FRESH optimizer per goal: a covariance learned climbing toward
        # "coral reef" is not informative about "lightning". sigma is
        # deliberately much smaller than sigma0 - an expedition is a local
        # refinement from an already-relevant seed, not a fresh search.
        self._optimizer = make_optimizer(
            self.algorithm, self.spec.dim, self.tournament.tiles,
            self.expedition_sigma, self.base_seed + self.gen,
            self._parent_z(self._x0_index).astype(np.float64),
        )
        return True

    def _draw_goal(self) -> Goal | None:
        want_latent = float(self.rng.random()) < float(self.latent_share)
        text_goal = None
        if self.goals is not None:
            self.goals.ensure_embedded(self.scorer)
            text_goal = (self.goals.least_matched(self.archive.embeddings)
                         if self.goal_order == "least_matched"
                         else self.goals.next_goal())
        if want_latent or text_goal is None:
            return (latent_goal(self.archive, self.rng, self.projection,
                                self.alpha) or text_goal)
        return text_goal

    def chase(self, tile: int) -> bool:
        """Start an expedition toward one tile's own descriptor.

        The direct analogue of manual mode's 'more like that one', expressed as
        a goal rather than a selection. The descriptor is already computed for
        that generation, so this costs no extra CLIP work."""
        d = self._last_descriptors
        if d is None or not (0 <= int(tile) < len(d)):
            return False
        return self.start_expedition_with(d[int(tile)].copy(), "chase", "")

    # ---- rollout -------------------------------------------------------

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

        last = snapshots[-1]
        pinned = set(self.tournament.selected)
        source = self.regime
        goal_text = self.goal_label
        parts = [self.spec.decode(zi) for zi in z]

        viable = np.array([bool(is_viable_tile(last[i]))
                           and float(live[i]) >= float(self.archive.liveness_min)
                           for i in range(n)])
        # One matmul for the whole batch instead of n matrix-vector products.
        sep = self.archive.separation_of(b)
        # ALWAYS TAKE ONE. The most novel tile that is a picture of something
        # gets in whatever the separation rule says, so a converged expedition
        # still leaves a trail through the archive and a generation is never
        # silently absent from the record. If nothing is viable there is
        # nothing worth forcing, and the whole generation is dropped.
        keeper = -1
        if viable.any():
            cand_nov = np.where(viable, nov, -np.inf)
            keeper = int(np.argmax(cand_nov))
        admitted = 0

        for i in range(n):
            phys = parts[i].get("physics")
            if phys is not None and self.physics_enabled:
                from services.physics_genome import decode_physics

                values = decode_physics(phys, self.physics_origin)
                phys_vec = np.array([values[nm] for nm, _g, _lo, _hi in PHYSICS_PARAMS],
                                    dtype=np.float32)
            else:
                phys_vec = np.zeros(PHYSICS_DIM, dtype=np.float32)

            entry = self.archive.consider(
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
                separation=float(sep[i]),
                force=(i == keeper),
            )
            if entry is not None:
                admitted += 1
                # The batch has to separate from ITSELF too. Without this, 64
                # tiles that have all converged onto the same pattern would
                # each measure against the pre-generation archive, find nothing
                # within l, and all go in - which is the exact failure the rule
                # exists to stop.
                sep = np.minimum(sep, 1.0 - b @ b[i])

        self.tournament.selected.clear()
        self._last_descriptors = b
        self.gen += 1
        self.archive.refresh(self._refresh_count())
        # AFTER refresh, so eviction ranks on the freshest novelty available,
        # and once per generation rather than per admission - the whole point
        # of admitting generously is that the ranking happens on the batch.
        self.archive.prune_to_capacity()
        self.archive.maybe_flush(every=self.flush_every)

        if source == "expedition":
            fit = self._expedition_fitness(snaps)
            self._optimizer.tell(z, fit)
            self._remaining -= 1
            self._record(source, admitted, n, fit)
            if self._remaining <= 0:
                self.end_expedition()
            self._last_score_label = "goal match"
            return fit

        self._record(source, admitted, n, None)
        self._since_expedition += 1
        if (self.expansion_between > 0
                and self._since_expedition >= self.expansion_between
                and len(self.archive) >= self.seed_n):
            self.start_expedition()

        self._last_score_label = "novelty"
        return np.asarray(nov, dtype=np.float32)

    # ---- traces ---------------------------------------------------------

    def _record(self, regime: str, admitted: int, tiles: int, fit) -> None:
        """One row per generation, for the Explore tab's plots.

        Two separate stories, deliberately in one buffer so they share an x
        axis: what the ARCHIVE is doing (size and mean novelty - is the search
        still finding new territory, or filling in ground it already has?) and
        what the current EXPEDITION is doing (best and mean goal match - is it
        still climbing, or has it converged and started producing the same tile
        64 times?). Reading either one alone is what made a stalled expedition
        invisible.

        Fitness is NaN outside an expedition rather than absent, so the two
        series stay index-aligned and a gap in the plot is a real gap.
        """
        t = self.trace
        t["gen"].append(int(self.gen))
        t["regime"].append(regime)
        t["archive_size"].append(int(len(self.archive)))
        t["mean_novelty"].append(float(self.archive.mean_novelty()))
        t["admitted"].append(int(admitted))
        t["tiles"].append(int(tiles))
        if fit is None or not len(fit):
            t["fit_best"].append(float("nan"))
            t["fit_mean"].append(float("nan"))
        else:
            f = np.asarray(fit, dtype=np.float64)
            t["fit_best"].append(float(np.nanmax(f)))
            t["fit_mean"].append(float(np.nanmean(f)))
        if len(t["gen"]) > self.TRACE_CAP:
            for v in t.values():
                del v[: len(v) - self.TRACE_CAP]

    def expedition_trace(self) -> dict:
        """The CURRENT expedition's fitness, or the last one if it has ended.

        Kept after the expedition finishes rather than cleared: the question a
        user asks is "did that get anywhere", and it is only askable afterwards.
        """
        reg = self.trace["regime"]
        end = len(reg)
        while end > 0 and reg[end - 1] != "expedition":
            end -= 1
        start = end
        while start > 0 and reg[start - 1] == "expedition":
            start -= 1
        return {"best": self.trace["fit_best"][start:end],
                "mean": self.trace["fit_mean"][start:end],
                "gens": end - start}

    # ---- expedition fitness ---------------------------------------------

    def _refresh_count(self) -> int:
        """How many entries to re-score this generation.

        A FRACTION of the archive, not a fixed count, because the quantity that
        matters is how many generations a full sweep takes - i.e. how stale
        novelty is allowed to get - and that has to hold as the archive grows.
        The old fixed 64 gave a sweep of 75 generations at 4808 entries and 312
        (14.6 minutes) at the 20000 capacity, where 64 tiles a generation are
        also being ADMITTED: the sweep took exactly as long as a complete
        turnover, so an entry's novelty could be a whole archive-lifetime old.

        Staleness is not symmetric, which is why it matters. Expansion breeds
        locally, so new entries land near old ones and a true novelty only ever
        falls; a stale value is therefore systematically too HIGH - 41-59% of
        entries measured inflated - and an inflated novelty makes an entry both
        likelier to be chosen as a parent and likelier to survive eviction.

        Measured cost at sweep=10: 0.9% of a 2.8 s generation at 4808 entries,
        12.7% at 20000.
        """
        g = int(self.refresh_sweep_gens)
        if g <= 0:
            return 0
        return int(-(-len(self.archive) // g))     # ceil, so a sweep completes

    def _expedition_fitness(self, snaps: np.ndarray) -> np.ndarray:
        """Contrastive, from the PER-SNAPSHOT embeddings.

        Deliberately not `descriptor(snaps) @ goal`. That was the old objective
        and it failed twice over: raw cosine saturates in a cone whose mean
        pairwise similarity is 0.897, and descriptor() renormalises the
        trajectory centroid, so 1/||m|| paid a bonus for decorrelated snapshots
        rather than for matching the goal. The descriptor is still the right
        thing to ARCHIVE - novelty needs unit vectors - it was only ever wrong
        as a fitness.
        """
        refs, scale = self._references()
        if refs is None or len(refs) == 0:
            # No centroid means an empty archive, which cannot happen inside an
            # expedition. Fall back to raw alignment rather than raising in the
            # middle of a generation.
            return (descriptor(snaps) @ self._goal.embedding).astype(np.float32)
        return contrastive(snaps, self._goal.embedding, refs, logit_scale=scale)

    def _seed_index(self, goal_emb: np.ndarray, kind: str) -> int | None:
        """Where the expedition starts: the archive entry that best matches the
        goal UNDER THE SAME OBJECTIVE the expedition will be scored on.

        Not Archive.nearest(), which is argmax(embeddings @ goal). For a text
        goal that ranking is degenerate, and its winner is a NOISE TEXTURE:
        measured 2026-08-08 over 15 unrelated prompts - galaxy, flowing water, a
        human face, ocean waves, fire, stained glass, smoke - raw cosine
        returned just 6 distinct seeds, and one cyan static tile won 7 of them.
        High-frequency noise has energy everywhere, so it carries a decent
        cosine to every phrase, and the modality gap leaves nothing else to
        separate entries by. The same 15 prompts give 14 distinct seeds here.

        That is why a text expedition visibly started from a bad image and
        climbed nowhere: the fitness was fixed, the STARTING POINT was not.

        SAMPLED with p proportional to fit^alpha, not argmaxed - the same rule
        E&E uses to pick a parent, and the same `alpha`. An argmax would send
        every expedition toward a given goal from the identical entry, so
        repeating a goal could only ever retrace one trajectory.

        alpha is then BANDED, not fixed and not solved to a target: see
        novelty.banded_alpha. How concentrated a goal's matches are is real
        information about the archive, so the band only clips the ends.
        """
        e = self.archive.embeddings
        if len(e) == 0:
            return None
        refs, scale = self._references(kind)
        if refs is None or len(refs) == 0:
            return self.archive.nearest(goal_emb)
        fit = contrastive(e[None, :, :], goal_emb, refs, logit_scale=scale)
        a = banded_alpha(fit, self.alpha, self.seed_ess_min, self.seed_ess_max)
        self._last_seed_ess = effective_sample_size(fit, a)
        self._last_seed_alpha = a
        # A goal every tile floors on leaves fit all-zero; sample_by_novelty
        # falls back to uniform there rather than dividing by zero.
        return int(sample_by_novelty(fit, 1, self.rng, a)[0])

    def _references(self, kind: str | None = None):
        """-> (references, logit_scale) for a goal of this kind.

        The scale differs by MODALITY, not by taste: CLIP's 100 is tuned for the
        narrow band that text-image similarity occupies, and applying it to
        image-image similarity above 0.9 floors 59.6% of a generation to zero.

        `kind` is explicit because seed selection has to ask this question
        BEFORE self._goal is assigned.
        """
        if kind is None:
            kind = self._goal.kind if self._goal is not None else ""
        if kind == "text":
            return self._distractor_embeddings(), TEXT_LOGIT_SCALE
        c = self.archive.centroid()
        return (None if c is None else c[None, :]), IMAGE_LOGIT_SCALE

    def _distractor_embeddings(self):
        """The Auto tab's distractor set, embedded once for the whole run.

        Deliberately NOT via scorer.set_prompt(): that writes scorer._text_emb,
        which the Auto tab owns, and the two modes must not clobber each other.
        """
        if self._distractors is None and self.scorer is not None:
            from services.clip_scorer import DEFAULT_DISTRACTORS

            self._distractors = np.asarray(
                self.scorer.embed_text(list(DEFAULT_DISTRACTORS)),
                dtype=np.float32)
        return self._distractors

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
            "imgep_since_expedition": int(self._since_expedition),
        }

    def restore(self, d: dict) -> None:
        # An expedition is deliberately NOT resumed: its optimizer is one
        # goal's local refinement, and the archive - which is the thing worth
        # preserving - is on disk independently of any checkpoint.
        self.end_expedition()
        self.gen = int(d.get("imgep_gen", 0))
        self._since_expedition = int(d.get("imgep_since_expedition", 0))
        # "imgep_threshold" appears in checkpoints written before 2026-08-08.
        # Ignored rather than rejected: there is no threshold to restore it to,
        # and an old checkpoint is still perfectly good for gen and cadence.
