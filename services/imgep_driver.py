"""Intrinsically-motivated goal exploration over the archive.

Based on Expedition & Expansion (arXiv:2509.03863), adapted to a tournament
grid: one generation is a batch of N tiles rather than E&E's serial rollouts,
and every expedition generation's tiles go through the admission gate, not
just the endpoint. See docs/imgep.md for the loop, regimes and fitness
equations.
"""
from __future__ import annotations

import numpy as np

from services.archive import Candidate
from services.archive_projection import Projection
from services.capture_health import is_viable_tile, structure
from services.descriptor import descriptor, liveness, stack_snapshots
from services.expedition_fitness import contrastive
from services.genome_spec import encode, layout_of, spec_for
from services.goal_source import (
    LATENT_DIMS,
    Goal,
    latent_goal,
    novelty_goal,
)
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
                 spec=None, rng=None):
        self.tournament = tournament
        self.scorer = scorer
        self.archive = archive
        self.goals = goals
        # From the tournament when unnamed, never Fourier by default - see
        # genome_spec.layout_of. main builds this the first time Explore is
        # opened, which can be long after the brain was switched.
        self.spec = spec if spec is not None else spec_for(layout_of(tournament))
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
        self.liveness_min = 0.002    # see CLAUDE.md
        self.n_views = 3             # see VisionScorer.embed_mean
        self.refresh_sweep_gens = 10
        self.flush_every = 200

        # expedition settings (spec 7.4)
        self.expansion_between = 25      # 0 disables expeditions entirely
        self.expedition_gens = 50
        self.expedition_sigma = 0.1      # deliberately << sigma0: local refinement
        self.latent_share = 0.5
        self.novelty_share = 0.25        # text takes what's left; see _draw_goal
        self.goal_order = "round_robin"  # or "least_matched"
        # Band on the seed pool, not a target. See novelty.banded_alpha.
        self.seed_ess_min = 8.0
        self.seed_ess_max = 512.0

        # The search's own projection, separate from the map's - refitting a
        # shared one every frame would thrash both.
        self.projection = Projection(LATENT_DIMS)
        self._distractors = None

        self._optimizer = None
        self._goal: Goal | None = None
        self._remaining = 0
        self._since_expedition = 0
        self._x0_index: int | None = None
        # High-water mark for the CURRENT expedition. Reset with the
        # expedition: fitness is contrastive against a different goal each
        # time, so the numbers aren't comparable across expeditions.
        self._expedition_best = -np.inf
        self._n_summits = 0
        # Cumulative over the run: how many times a tile beat the archive's
        # best match for one of the text goals.
        self._n_records = 0
        self._last_descriptors: np.ndarray | None = None
        self._last_seed_ess = 0.0
        self._last_seed_alpha = 0.0

        self.gen = 0
        self._last_score_label = "novelty"
        self.trace = {k: [] for k in ("gen", "regime", "archive_size",
                                      "mean_novelty", "admitted", "tiles",
                                      "fit_best", "fit_mean")}

    # One row per generation; caps trace memory on a long-running machine.
    TRACE_CAP = 10000

    # ---- state ---------------------------------------------------------

    @property
    def _native_n(self) -> int:
        """Entries the RUNNING brain can be seeded or bred from.

        Every regime decision counts these rather than the archive, because an
        archive pools layouts: switching brain inside a full one leaves the new
        brain with nothing to expand FROM, and counting the whole thing put the
        driver in expansion, where each expedition then declined for want of a
        native seed and wasted its cadence interval. len(archive) still sizes
        anything that pools - novelty, the refresh sweep, the map.
        """
        return int(len(self.archive.native_rows()))

    @property
    def regime(self) -> str:
        if self._remaining > 0 and self._goal is not None:
            return "expedition"
        # max(1, ...): nothing to expand FROM stays in bootstrap whatever
        # seed_n says. Without this, parent sampling raises on the first ask
        # when seed_n is 0.
        if self._native_n < max(1, self.seed_n):
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

        -> (label, done, total, note); total 0 means there is no finish line,
        which is the honest answer when expeditions are switched off.
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
            done = min(self._native_n, total)
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
            "n_summits": int(self._n_summits),
            "n_records": int(self._n_records),
            "expedition_best": (float(self._expedition_best)
                                if np.isfinite(self._expedition_best) else None),
            "phase": self.phase(),
            "last_admitted": (self.trace["admitted"][-1]
                              if self.trace["admitted"] else 0),
            "last_tiles": (self.trace["tiles"][-1]
                           if self.trace["tiles"] else 0),
        }

    # ---- driver interface ----------------------------------------------

    def set_spec(self, spec) -> None:
        # Adopt the object either way; abandon the expedition only if the space
        # moved under it. A scale tweak mid-expedition would otherwise throw
        # away the goal it was climbing toward.
        same = self.spec.same_space_as(spec)
        self.spec = spec
        if not same:
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
        self._n_summits = 0
        self._n_records = 0
        for v in self.trace.values():
            v.clear()
        self.end_expedition()

    def end_expedition(self) -> None:
        self._optimizer = None
        self._goal = None
        self._remaining = 0
        self._x0_index = None
        self._expedition_best = -np.inf

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
            return (self._ask_expansion(n) if self._native_n
                    else self._ask_bootstrap(n))
        return self._optimizer.ask(n)

    def _ask_bootstrap(self, n: int) -> np.ndarray:
        return (self.sigma0 * self.rng.normal(size=(n, self.spec.dim))
                ).astype(np.float32)

    def _ask_expansion(self, n: int) -> np.ndarray:
        # NATIVE rows only. Novelty ranks across every brain in the archive,
        # because novelty is about pictures - but a parent has to be decodable
        # by the optimizer that is about to mutate it, and another brain's
        # floats are a different creature under this one's squash.
        rows = self.archive.native_rows()
        if not len(rows):
            return self._ask_bootstrap(n)
        nov = np.array([self.archive.entries[i].novelty for i in rows],
                       dtype=np.float32)
        idx = sample_by_novelty(nov, n, self.rng, self.alpha)
        out = np.empty((n, self.spec.dim), dtype=np.float32)
        for j, i in enumerate(idx):
            out[j] = self._parent_z(int(rows[int(i)]))
        out += (self.sigma_expand * self.rng.normal(size=out.shape)).astype(np.float32)
        return out

    def _parent_z(self, i: int) -> np.ndarray:
        """Re-encode an archived PHENOTYPE into z under the CURRENT origin.

        The archive stores decoded values precisely so this works: a z archived
        under one preset would mean a different creature under another.

        Encoded under the SPEC's layout - the brain that is running - and only
        ever called on a native row, because that is the only kind of genome
        this z is going to be mutated as.

        Stated as a check because it used to be stated only as a comment: a
        foreign row reached this and surfaced as a reshape error inside
        whichever modality happened to be running, several frames of stack away
        from the sampler that chose it.
        """
        if not self.archive.is_native(i):
            raise ValueError(
                f"seed row {i} is a {self.archive.layout_at(i)} genome, but "
                f"{self.spec.layout.signature()} is running; the sampler that "
                f"chose it must filter through Archive.native_rows()")
        zb, _clamped = encode(self.archive.brain_at(i), self.spec.layout)
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
        return self.start_expedition_with(goal.embedding, goal.kind, goal.text,
                                          seed_index=goal.seed_index)

    def start_expedition_with(self, embedding, kind: str, text: str,
                              seed_index: int | None = None) -> bool:
        """Begin an expedition toward a specific embedding. An expedition needs
        a seed, so an empty archive falls back to expansion rather than
        starting a search from nowhere.

        `seed_index` is the goal's own answer to "where should this start". A
        latent goal knows, because it was built by pushing past that entry, and
        a novelty goal has no embedding to derive one from at all. Only a text
        or chase goal has to go looking.
        """
        emb = (None if embedding is None
               else np.asarray(embedding, dtype=np.float32))
        # NATIVE as well as in range. A seed becomes the optimizer's mean and is
        # re-encoded under the running layout, so a row belonging to another
        # brain is not a worse start but an unreadable one - and an archive
        # pools every layout. A goal that names one falls through to
        # _seed_index, which filters, exactly as an out-of-range index does.
        if (seed_index is not None and 0 <= int(seed_index) < len(self.archive)
                and self.archive.is_native(int(seed_index))):
            i = int(seed_index)
        elif emb is None:
            return False            # nothing to point at and nowhere to start
        else:
            i = self._seed_index(emb, str(kind))
        if i is None or self.expedition_gens <= 0:
            return False
        self._goal = Goal(kind, text, emb, seed_index=i)
        self._x0_index = int(i)
        self._remaining = int(self.expedition_gens)
        self._since_expedition = 0
        # Reset explicitly: chase()/UI can start a new expedition on top of a
        # running one, and fitness is not comparable across different goals.
        self._expedition_best = -np.inf
        # Fresh optimizer per goal - a covariance learned for one goal doesn't
        # transfer to another. sigma << sigma0: local refinement, not a fresh
        # search.
        self._optimizer = make_optimizer(
            self.algorithm, self.spec.dim, self.tournament.tiles,
            self.expedition_sigma, self.base_seed + self.gen,
            self._parent_z(self._x0_index).astype(np.float64),
            layout=self.spec.layout,
        )
        return True

    def _draw_goal(self) -> Goal | None:
        """One of three kinds, by share. Text takes whatever is left over.

        Ordered novelty, latent, text and falling THROUGH rather than failing:
        each source can decline (no archive, unfittable projection, empty goal
        list), and an expedition that does not start is a whole cadence
        interval wasted.
        """
        u = float(self.rng.random())
        nov_share = max(0.0, float(self.novelty_share))
        lat_share = max(0.0, float(self.latent_share))
        # Clamped rather than normalised: the two sliders are independent, and
        # silently rescaling one because the other moved would make neither
        # mean what it says.
        if nov_share + lat_share > 1.0:
            lat_share = max(0.0, 1.0 - nov_share)

        text_goal = None
        if self.goals is not None:
            self.goals.ensure_embedded(self.scorer)
            text_goal = (self.goals.least_matched(self.archive.embeddings)
                         if self.goal_order == "least_matched"
                         else self.goals.next_goal())

        def latent():
            return latent_goal(self.archive, self.rng, self.projection, self.alpha)

        def novelty():
            return novelty_goal(self.archive, self.rng, self.alpha)

        if u < nov_share:
            order = (novelty, latent, lambda: text_goal)
        elif u < nov_share + lat_share:
            order = (latent, novelty, lambda: text_goal)
        elif text_goal is not None:
            return text_goal
        else:
            order = (latent, novelty, lambda: None)
        for make in order:
            g = make()
            if g is not None:
                return g
        return None

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

    def precompute(self, snapshots: list[np.ndarray]):
        """CLIP, and nothing else. Runs OFF the main thread - see CLAUDE.md.

        Split here rather than running the whole of tell() on the worker,
        because everything after this point mutates the archive, which the UI
        reads every frame. This half touches only its own arguments and the
        scorer, so it needs no locking.
        """
        return [np.asarray(self._embed(c), dtype=np.float32) for c in snapshots]

    def tell(self, z: np.ndarray, snapshots: list[np.ndarray],
             pre=None) -> np.ndarray:
        n = len(z)
        if not snapshots:
            return np.zeros(n, dtype=np.float32)

        per_snap = pre if pre is not None else self.precompute(snapshots)
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

        # Two different questions, and the summit answers only the first.
        # `shows_something` is "is this a picture at all" - not black, not
        # blown out. `viable` adds "and it is still changing", which is what
        # the archive's ordinary admission gate applies.
        shows_something = np.array([bool(is_viable_tile(last[i]))
                                    for i in range(n)])
        alive = np.asarray(live, dtype=np.float64) >= float(self.archive.liveness_min)
        viable = shows_something & alive
        # One matmul for the whole batch instead of n matrix-vector products.
        sep = self.archive.separation_of(b)
        # Always take one: the most novel viable tile gets in regardless of
        # separation, so a generation is never silently absent from the record.
        keeper = -1
        if viable.any():
            cand_nov = np.where(viable, nov, -np.inf)
            keeper = int(np.argmax(cand_nov))
        # One pass over the crops, reused by the fitness for every goal kind.
        coherence = structure(last)
        # BEFORE the admission loop, not after it, so the summit ratchet can
        # see this generation's fitness.
        fit = (self._expedition_fitness(snaps, nov, coherence)
               if source == "expedition" else None)
        summit = self._summit(fit, shows_something)
        # Every regime, not just an expedition toward that goal: a run chasing
        # one phrase routinely produces the best thing the archive has ever
        # had for a DIFFERENT one, and nothing else would keep it.
        records = self._goal_records(b, shows_something)
        self._n_records += len(records)
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

            record = records.get(i)
            # The goal a record entry BEAT, not the one the run happens to be
            # chasing - that is the only way to find it again afterwards.
            tile_goal = record[0] if (record and i != summit) else goal_text
            if i == summit:
                tile_source = "summit"
            elif record:
                tile_source = "record"
            else:
                tile_source = source

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
                    goal=tile_goal,
                ),
                float(nov[i]),
                pinned=(i in pinned),
                source=tile_source,
                thumb_crop=last[i],
                separation=float(sep[i]),
                force=(i == keeper or i == summit or record is not None),
                ignore_liveness=(i == summit or record is not None),
            )
            if entry is not None:
                admitted += 1
                # The batch has to separate from itself too, or converged
                # tiles would all pass separation against the pre-generation
                # archive and all go in.
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
                and self._native_n >= self.seed_n):
            self.start_expedition()

        self._last_score_label = "novelty"
        return np.asarray(nov, dtype=np.float32)

    def _goal_records(self, b, shows_something) -> dict[int, tuple[str, float]]:
        """Tiles that match a text goal better than ANYTHING in the archive.

        -> {tile: (goal text, margin)}, at most one tile per goal.

        Runs in EVERY regime - the goal being chased and the goal being beaten
        are unrelated, and nothing else keeps this record. See CLAUDE.md.

        Scored on descriptors on both sides via contrastive() with the
        distractor set, the same objective `_seed_index` uses (archive entries
        have no snapshots, so the fitness's per-snapshot averaging doesn't
        apply here). A tile breaking several records is credited with the
        largest margin and admitted once.
        """
        if self.goals is None or len(self.archive) == 0:
            return {}
        if not np.any(shows_something):
            return {}
        self.goals.ensure_embedded(self.scorer)
        live = self.goals.enabled_goals()
        refs = self._distractor_embeddings()
        if not live or refs is None or not len(refs):
            return {}

        arc = self.archive.embeddings[None, :, :]
        tiles = np.asarray(b, dtype=np.float32)[None, :, :]
        out: dict[int, tuple[str, float]] = {}
        for g in live:
            held = float(contrastive(arc, g.embedding, refs,
                                     logit_scale=self._text_scale).max())
            cand = contrastive(tiles, g.embedding, refs,
                               logit_scale=self._text_scale).astype(np.float64)
            cand = np.where(shows_something, cand, -np.inf)
            i = int(np.argmax(cand))
            margin = float(cand[i]) - held
            if margin > 0.0 and (i not in out or margin > out[i][1]):
                out[i] = (g.text, margin)
        return out

    def _summit(self, fit, shows_something) -> int:
        """The tile that sets a new best fitness for this expedition, or -1.

        A RATCHET: separation discards an expedition's best-matching tile
        (it necessarily resembles recent tiles), and `keeper` is the most
        novel viable tile, not the best-matching one. See CLAUDE.md. A
        converged expedition stops improving and therefore stops admitting -
        at most one extra entry per generation.

        Gated on `shows_something` (viability), NOT liveness - a settled
        attractor is what a converging chase produces. See
        Archive.consider(ignore_liveness=).
        """
        if fit is None or not len(fit) or not np.any(shows_something):
            return -1
        f = np.where(shows_something, np.asarray(fit, dtype=np.float64), -np.inf)
        i = int(np.argmax(f))
        if not np.isfinite(f[i]) or float(f[i]) <= self._expedition_best:
            return -1
        self._expedition_best = float(f[i])
        self._n_summits += 1
        return i

    # ---- traces ---------------------------------------------------------

    def _record(self, regime: str, admitted: int, tiles: int, fit) -> None:
        """One row per generation, for the Explore tab's plots.

        Two stories in one buffer so they share an x axis: what the ARCHIVE is
        doing (size, mean novelty) and what the EXPEDITION is doing (best and
        mean goal match). Fitness is NaN outside an expedition rather than
        absent, so the series stay index-aligned and a gap is a real gap.
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

    def _embed(self, crops):
        """One embedding per tile, averaged over n_views random sub-crops.

        Falls back to the plain call for any scorer without embed_mean (the
        tests' fakes, and tools/ scripts), so this is not a hard dependency.
        """
        v = max(1, int(self.n_views))
        fn = getattr(self.scorer, "embed_mean", None)
        if fn is None or v == 1:
            return self.scorer.embed(crops, n_views=1)
        return fn(crops, v)

    def _refresh_count(self) -> int:
        """How many entries to re-score this generation.

        A FRACTION of the archive, not a fixed count, so a full sweep takes a
        constant number of generations as the archive grows. See CLAUDE.md.
        """
        g = int(self.refresh_sweep_gens)
        if g <= 0:
            return 0
        return int(-(-len(self.archive) // g))     # ceil, so a sweep completes

    def _expedition_fitness(self, snaps: np.ndarray, nov, coherence) -> np.ndarray:
        """What the optimizer climbs, scaled by how much of it is a pattern.

        Contrastive, from the PER-SNAPSHOT embeddings - not
        `descriptor(snaps) @ goal`, whose raw cosine saturates and whose
        renormalised centroid rewards decorrelated snapshots rather than goal
        match. See CLAUDE.md. `descriptor()` is still the right thing to
        ARCHIVE (novelty needs unit vectors); it was only ever wrong as a
        fitness.

        A NOVELTY expedition has no goal to match, so its fitness is the kNN
        novelty already computed for admission this generation.

        Multiplied by `coherence` (~0 for static, ~1 for a real pattern) so
        noise cannot maximise a single-reference contrastive score. See
        capture_health.structure.
        """
        c = np.asarray(coherence, dtype=np.float32)
        if self._goal is not None and self._goal.kind == "novelty":
            return (np.asarray(nov, dtype=np.float32) * c).astype(np.float32)

        refs, scale = self._references()
        if refs is None or len(refs) == 0:
            # No centroid means an empty archive, which cannot happen inside an
            # expedition. Fall back to raw alignment rather than raising in the
            # middle of a generation.
            base = (descriptor(snaps) @ self._goal.embedding).astype(np.float32)
        else:
            base = contrastive(snaps, self._goal.embedding, refs,
                               logit_scale=scale)
        return (base * c).astype(np.float32)

    def _seed_index(self, goal_emb: np.ndarray, kind: str) -> int | None:
        """Where the expedition starts: the archive entry that best matches the
        goal UNDER THE SAME OBJECTIVE the expedition will be scored on.

        Not Archive.nearest() (argmax(embeddings @ goal)): for a text goal
        that ranking is degenerate and its winner is a noise texture. See
        CLAUDE.md.

        SAMPLED with p proportional to fit^alpha, not argmaxed, so repeating a
        goal explores a different trajectory each time. alpha is then BANDED,
        not fixed - see novelty.banded_alpha.

        NATIVE rows only, for _ask_expansion's reason: a seed becomes the
        optimizer's mean, so it has to be a genome this brain can decode.
        """
        rows = self.archive.native_rows()
        if not len(rows):
            return None
        e = self.archive.embeddings[rows]
        refs, scale = self._references(kind)
        if refs is None or len(refs) == 0:
            return int(rows[int(np.argmax(e @ np.asarray(goal_emb, np.float32)))])
        fit = contrastive(e[None, :, :], goal_emb, refs, logit_scale=scale)
        a = banded_alpha(fit, self.alpha, self.seed_ess_min, self.seed_ess_max)
        self._last_seed_ess = effective_sample_size(fit, a)
        self._last_seed_alpha = a
        # A goal every tile floors on leaves fit all-zero; sample_by_novelty
        # falls back to uniform there rather than dividing by zero.
        return int(rows[int(sample_by_novelty(fit, 1, self.rng, a)[0])])

    def _references(self, kind: str | None = None):
        """-> (references, logit_scale) for a goal of this kind.

        The scale differs by MODALITY - see CLAUDE.md. `kind` is explicit
        because seed selection has to ask this before self._goal is assigned.
        """
        if kind is None:
            kind = self._goal.kind if self._goal is not None else ""
        if kind == "text":
            return self._distractor_embeddings(), self._text_scale
        c = self.archive.centroid()
        return (None if c is None else c[None, :]), self._image_scale

    @property
    def _text_scale(self) -> float:
        return self.scorer.model.text_logit_scale

    @property
    def _image_scale(self) -> float:
        return self.scorer.model.image_logit_scale

    def _distractor_embeddings(self):
        """The Auto tab's distractor set, embedded once for the whole run.

        Deliberately NOT via scorer.set_prompt(): that writes scorer._text_emb,
        which the Auto tab owns, and the two modes must not clobber each other.
        """
        if self._distractors is None and self.scorer is not None:
            from services.vision_scorer import DEFAULT_DISTRACTORS

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
        # An expedition is not resumed: its optimizer is one goal's local
        # refinement, and the archive is on disk independently of any
        # checkpoint.
        self.end_expedition()
        self.gen = int(d.get("imgep_gen", 0))
        self._since_expedition = int(d.get("imgep_since_expedition", 0))
        # "imgep_threshold" may appear in older checkpoints; ignored, not
        # restored.
