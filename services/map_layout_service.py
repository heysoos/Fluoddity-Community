"""Owns the map's layout: which engine, when it refits, and the cache.

The UMAP fit is seconds, so it runs in a worker PROCESS and the map keeps
drawing the layout it already has until the new one lands. A process, not a
thread: numba's JIT holds the GIL, so a thread stalled the frame loop for the
whole fit (see CLAUDE.md). The worker is handed a COPY of the embeddings and
fits a SEPARATE layout object, which is what keeps it off the one the frame
loop is reading.

Both the layout factory and the executor are injected, so the decision rules
are testable without paying for a real fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from services.map_layout import (
    STALE_AT,
    LayoutCache,
    make_layout,
    staleness,
)

# PCA is cheap and inline, and refits on the rule it always had.
PCA_REFIT_GROWTH = 500


def fit_layout(make, engine: str, x, keys, prev):
    """Fit a fresh `engine` layout to `x`, seeded from `prev`. -> it, or None.

    Module-level so it pickles by reference into the worker process. The seed
    positions are computed here too, off the frame loop.
    """
    init = None
    if prev is not None and getattr(prev, "fitted", False):
        init = np.array(prev.transform(x, keys), dtype=np.float32)
    fresh = make(engine)
    if not fresh.fit(x, keys, init_pos=init):
        return None
    return fresh


class MapLayoutService:
    def __init__(self, make=make_layout, executor=None):
        self._make = make
        self._pool = executor
        self._owns_pool = executor is None
        self.engine = "pca"
        self._layouts: dict = {}
        self._archive = None
        self._path: Path | None = None
        self._encoder = ""
        self._future = None
        self._refit_requested = False
        self._pca_at = 0
        self.error = ""

    # ---- lifecycle ------------------------------------------------------

    @property
    def executor(self):
        if self._pool is None:
            from utilities.process_executor import ProcessExecutor

            # `ui` first: services imports it back, and the child unpickles
            # fit_layout by module path.
            self._pool = ProcessExecutor(prime=("ui",))
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None and self._owns_pool:
            self._pool.shutdown(wait=False)
            self._pool = None

    def bind(self, archive, path, encoder: str) -> None:
        """Point at a different archive. Drops every layout it was holding.

        Entry ids restart in every archive, so keeping positions would draw the
        previous archive's map under this one's entries.

        The DIRECTORY is what identifies them, not the Archive object: a brain
        switch rebuilds that object over the same directory, with the same ids
        in the same positions. Dropping the layout there leaves the map with
        nothing to draw for a frame, which collapses the browser's contents
        and clamps its scroll to the top - and throws away a UMAP fit that
        costs seconds.
        """
        path = Path(path) if path is not None else None
        same = (self._archive is not None and path == self._path
                and str(encoder or "") == self._encoder)
        self._archive = archive
        self._path = path
        self._encoder = str(encoder or "")
        if same:
            return
        self._layouts = {}
        self._future = None
        self._refit_requested = False
        self._pca_at = 0
        self.error = ""

    def configure(self, engine: str) -> None:
        self.engine = "umap" if str(engine) == "umap" else "pca"

    def request_refit(self) -> None:
        self._refit_requested = True

    # ---- readouts -------------------------------------------------------

    @property
    def layout(self):
        got = self._layouts.get(self.engine)
        if got is None:
            got = self._layouts[self.engine] = self._make(self.engine)
        return got

    @property
    def _pca(self):
        got = self._layouts.get("pca")
        if got is None:
            got = self._layouts["pca"] = self._make("pca")
        return got

    @property
    def fitting(self) -> bool:
        return self._future is not None

    @property
    def fitted(self) -> bool:
        return bool(self._drawing.fitted)

    @property
    def fitted_n(self) -> int:
        return int(getattr(self._drawing, "fitted_n", 0))

    @property
    def version(self) -> int:
        # The engine is part of it: switching engine changes every position
        # without either layout having been refitted.
        return int(getattr(self._drawing, "version", 0)) * 4 + (
            1 if self._drawing is not self._pca else 0)

    @property
    def unavailable(self) -> str:
        return str(getattr(self._layouts.get("umap"), "unavailable", "") or "")

    @property
    def staleness(self) -> float:
        n = len(self._archive) if self._archive is not None else 0
        return staleness(n, self.fitted_n)

    @property
    def _drawing(self):
        """The layout the map actually draws: the chosen one once it has a fit,
        PCA until then. A blank map while UMAP thinks is not an option."""
        got = self._layouts.get(self.engine)
        if got is not None and got.fitted:
            return got
        return self._pca

    def status(self) -> str:
        from services.map_layout import ENGINE_LABELS
        name = ENGINE_LABELS.get(self.engine, self.engine)
        if self.fitting:
            return f"{name} - fitting..."
        if self.engine == "umap" and not self._layouts.get("umap", None):
            return f"{name} - not fitted"
        drawing = self._drawing
        if drawing is self._pca and self.engine == "umap":
            return f"PCA - {self.error or self.unavailable or 'no UMAP layout yet'}"
        n = len(self._archive) if self._archive is not None else 0
        if self.fitted_n and n != self.fitted_n:
            return f"{name} - fitted at {self.fitted_n} of {n}"
        return f"{name} - fitted at {self.fitted_n}"

    # ---- the frame ------------------------------------------------------

    def update(self, archive) -> None:
        """Collect a finished fit, then decide whether to start one."""
        self._archive = archive
        if archive is None:
            return
        self._collect()
        # Relayout reaches PCA too: it is the engine most of the time, and
        # a button that does nothing under the default is worse than none.
        forced = self._refit_requested
        self._keep_pca_current(archive, force=forced)
        if self.engine != "umap":
            self._refit_requested = False
            return
        if self.fitting:
            return

        umap = self.layout
        if not getattr(umap, "available", lambda: True)():
            if not umap.unavailable:
                umap.unavailable = "umap-learn is not installed"
            self._refit_requested = False
            return

        # A forced relayout refits rather than adopting what is on disk.
        if not umap.fitted and not forced and self._adopt_cache(archive):
            self._refit_requested = False
            return

        if forced or not umap.fitted or self.staleness >= STALE_AT:
            self._refit_requested = False
            self._start(archive)

    def _keep_pca_current(self, archive, force: bool = False) -> None:
        """PCA is cheap, inline, and on the rule it always had."""
        pca = self._pca
        grown = len(archive) - self._pca_at
        if force or grown >= PCA_REFIT_GROWTH or (
                not pca.fitted and len(archive) > 2):
            if pca.fit(archive.embeddings, self._keys(archive)):
                self._pca_at = len(archive)

    def _start(self, archive) -> None:
        x = np.array(archive.embeddings, dtype=np.float32, copy=True)
        keys = self._keys(archive)
        umap = self.layout
        prev = umap if umap.fitted else None
        self._future = self.executor.submit(
            fit_layout, self._make, "umap", x, keys, prev)

    def _collect(self) -> None:
        fut = self._future
        if fut is None or not fut.done():
            return
        self._future = None
        try:
            fresh = fut.result()
        except Exception as exc:                      # noqa: BLE001
            self.error = str(exc) or exc.__class__.__name__
            return
        if fresh is None:
            self.error = (getattr(self._layouts.get("umap"), "unavailable", "")
                          or "the layout did not fit")
            return
        self.error = ""
        self._layouts["umap"] = fresh
        self._save_cache(fresh)

    # ---- the cache ------------------------------------------------------

    def _keys(self, archive):
        return [(archive.layout_at(i), int(e.id))
                for i, e in enumerate(archive.entries)]

    def _adopt_cache(self, archive) -> bool:
        if self._path is None:
            return False
        cache = LayoutCache.load(self._path, self._encoder)
        if cache is None or cache.engine != "umap":
            return False
        try:
            return bool(self.layout.adopt(cache, archive.embeddings,
                                          self._keys(archive)))
        except Exception:                             # noqa: BLE001
            return False

    def _save_cache(self, layout) -> None:
        if self._path is None:
            return
        try:
            cache = layout.to_cache(self._encoder)
            if cache is not None:
                cache.save(self._path)
        except Exception:                             # noqa: BLE001
            # A view. Failing to cache costs one refit next time, never a frame.
            pass

    # ---- what the map asks for ------------------------------------------

    def transform_rows(self, archive, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if archive is None or not len(idx):
            return np.zeros((0, 2), dtype=np.float32)
        keys = self._keys(archive)
        return self._drawing.transform(archive.embeddings[idx],
                                       [keys[i] for i in idx])

    def transform(self, embeddings) -> np.ndarray:
        """For the goal marker, which is not an archive row."""
        return self._drawing.transform(np.asarray(embeddings), None)
