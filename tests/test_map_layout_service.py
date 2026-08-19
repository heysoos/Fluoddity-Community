"""Who decides when the map is laid out again, and what the map draws meanwhile.

The fit runs on a worker thread, so the rule that matters is that the map keeps
drawing the layout it already has and swaps only when a fit lands. Both the
layout factory and the executor are injected, so none of this pays for a real
UMAP fit.
"""
import numpy as np
import pytest

from services.map_layout import PcaLayout, staleness
from services.map_layout_service import MapLayoutService


class _Inline:
    """An executor that runs now and hands back something future-shaped."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *a, **kw):
        self.submitted += 1

        class _F:
            def __init__(self, value, exc=None):
                self._v, self._e = value, exc

            def done(self):
                return True

            def result(self):
                if self._e:
                    raise self._e
                return self._v

        try:
            return _F(fn(*a, **kw))
        except Exception as exc:            # noqa: BLE001
            return _F(None, exc)

    def shutdown(self, wait=False):
        pass


class _FakeUmap:
    """Stands in for UmapLayout: fits instantly, positions are the first 2 dims."""
    engine = "umap"
    fits = 0

    def __init__(self):
        self.version = 0
        self.fitted_n = 0
        self.unavailable = ""
        self._pos = None
        self._keys = []
        self.init_seen = None

    @property
    def fitted(self):
        return self._pos is not None

    @staticmethod
    def available():
        return True

    def fit(self, x, keys=None, init_pos=None):
        type(self).fits += 1
        self.init_seen = None if init_pos is None else np.array(init_pos)
        x = np.asarray(x, dtype=np.float32)
        self._pos = x[:, :2].copy() + 100.0     # unmistakably not PCA
        self._keys = list(keys or [])
        self.fitted_n = len(x)
        self.version += 1
        return True

    def transform(self, x, keys=None):
        x = np.asarray(x, dtype=np.float32)
        if not self.fitted:
            return np.zeros((len(x), 2), dtype=np.float32)
        return x[:, :2] + 100.0

    def to_cache(self, encoder):
        from services.map_layout import LayoutCache
        return LayoutCache(pos=self._pos, keys=self._keys, encoder=str(encoder),
                           reduction=np.eye(2, 8, dtype=np.float32),
                           reduction_mean=np.zeros(8, dtype=np.float32),
                           fitted_n=self.fitted_n, engine="umap")

    def adopt(self, cache, x, keys):
        self._pos = np.asarray(cache.pos, np.float32)
        self._keys = list(cache.keys)
        self.fitted_n = int(cache.fitted_n)
        self.version += 1
        return True


class _Arc:
    signature = "fourier-n10"
    encoder = "clip-b32"

    def __init__(self, n=40, dim=8):
        rs = np.random.RandomState(0)
        self.embeddings = rs.randn(n, dim).astype(np.float32)
        self.entries = [type("E", (), {"id": i})() for i in range(n)]

    def __len__(self):
        return len(self.entries)

    def layout_at(self, i):
        return self.signature

    def grow(self, k):
        rs = np.random.RandomState(len(self.entries))
        more = rs.randn(k, self.embeddings.shape[1]).astype(np.float32)
        self.embeddings = np.concatenate([self.embeddings, more])
        base = len(self.entries)
        self.entries += [type("E", (), {"id": base + i})() for i in range(k)]


def _svc(tmp_path, engine="umap"):
    _FakeUmap.fits = 0
    svc = MapLayoutService(
        make=lambda e: _FakeUmap() if e == "umap" else PcaLayout(),
        executor=_Inline())
    arc = _Arc()
    svc.bind(arc, tmp_path / "map_layout.npz", arc.encoder)
    svc.configure(engine)
    return svc, arc


# ---- the default is exactly what shipped --------------------------------

def test_pca_is_the_default_and_never_goes_near_the_worker(tmp_path):
    svc, arc = _svc(tmp_path, engine="pca")
    svc.update(arc)
    assert svc.engine == "pca"
    assert svc.executor.submitted == 0
    assert svc.fitted


def test_pca_positions_are_the_projection_it_always_was(tmp_path):
    svc, arc = _svc(tmp_path, engine="pca")
    svc.update(arc)
    want = PcaLayout()
    want.fit(arc.embeddings)
    got = svc.transform_rows(arc, np.arange(len(arc)))
    assert np.allclose(got, want.transform(arc.embeddings), atol=1e-4)


# ---- the first fit -------------------------------------------------------

def test_an_archive_with_no_cached_layout_fits_itself_once(tmp_path):
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    assert _FakeUmap.fits == 1
    svc.update(arc)
    svc.update(arc)
    assert _FakeUmap.fits == 1, "it refitted with nothing having changed"


def test_the_fit_writes_a_cache_next_to_the_archive(tmp_path):
    svc, arc = _svc(tmp_path)
    svc.update(arc)                        # submits
    svc.update(arc)                        # a later frame collects it
    assert (tmp_path / "map_layout.npz").exists()


def test_reopening_adopts_the_cache_instead_of_fitting_again(tmp_path):
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    svc.update(arc)                        # collect, which is what caches
    assert _FakeUmap.fits == 1

    again, arc2 = _svc(tmp_path)          # resets the counter, same directory
    again.update(arc2)
    assert _FakeUmap.fits == 0, "a cached layout still paid for a fit"
    assert again.fitted


# ---- staleness -----------------------------------------------------------

def test_growth_past_a_quarter_refits(tmp_path):
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    assert _FakeUmap.fits == 1
    arc.grow(9)                            # 40 -> 49, under 25%
    svc.update(arc)
    assert _FakeUmap.fits == 1
    arc.grow(2)                            # 51, over
    svc.update(arc)
    assert _FakeUmap.fits == 2


def test_the_staleness_threshold_is_what_the_service_uses(tmp_path):
    """Derived from the same function the readout shows, so a changed constant
    cannot move one without the other."""
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    n = svc.fitted_n
    assert staleness(n, n) == 0.0
    assert svc.staleness == 0.0


def test_a_manual_relayout_refits_even_when_nothing_is_stale(tmp_path):
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    svc.request_refit()
    svc.update(arc)
    assert _FakeUmap.fits == 2


def test_a_refit_is_seeded_from_the_layout_already_on_screen(tmp_path):
    """Otherwise a refit rearranges the map the user had learned, which is why
    UMAP was rejected the first time."""
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    svc.update(arc)                        # collect the first layout
    arc.grow(20)
    svc.update(arc)                        # stale, so this seeds from it
    svc.update(arc)                        # collect the reseeded one
    seeded = svc.layout.init_seen
    assert seeded is not None and len(seeded) == len(arc)


# ---- while a fit is in flight -------------------------------------------

class _Never:
    """A future that never finishes, so the map has to cope."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *a, **kw):
        self.submitted += 1

        class _F:
            def done(self):
                return False

            def result(self):
                raise AssertionError("not done")

        return _F()

    def shutdown(self, wait=False):
        pass


def test_the_map_keeps_drawing_while_a_fit_is_running(tmp_path):
    svc = MapLayoutService(make=lambda e: _FakeUmap() if e == "umap" else PcaLayout(),
                           executor=_Never())
    arc = _Arc()
    svc.bind(arc, tmp_path / "map_layout.npz", arc.encoder)
    svc.configure("umap")
    svc.update(arc)
    assert svc.fitting
    out = svc.transform_rows(arc, np.arange(len(arc)))
    assert out.shape == (len(arc), 2)
    assert np.all(np.isfinite(out))


def test_only_one_fit_is_ever_in_flight(tmp_path):
    svc = MapLayoutService(make=lambda e: _FakeUmap() if e == "umap" else PcaLayout(),
                           executor=_Never())
    arc = _Arc()
    svc.bind(arc, tmp_path / "map_layout.npz", arc.encoder)
    svc.configure("umap")
    for _ in range(5):
        svc.request_refit()
        svc.update(arc)
    assert svc.executor.submitted == 1


# ---- when umap is not there ---------------------------------------------

class _Missing(_FakeUmap):
    @staticmethod
    def available():
        return False

    def fit(self, x, keys=None, init_pos=None):
        self.unavailable = "umap-learn is not installed"
        return False


def test_a_missing_engine_falls_back_to_pca_and_says_why(tmp_path):
    svc = MapLayoutService(make=lambda e: _Missing() if e == "umap" else PcaLayout(),
                           executor=_Inline())
    arc = _Arc()
    svc.bind(arc, tmp_path / "map_layout.npz", arc.encoder)
    svc.configure("umap")
    svc.update(arc)
    out = svc.transform_rows(arc, np.arange(len(arc)))
    assert out.shape == (len(arc), 2)
    assert svc.unavailable
    assert svc.fitted, "with umap gone the map must still draw something"


def test_a_worker_that_raises_does_not_take_the_frame_down(tmp_path):
    class _Boom(_FakeUmap):
        def fit(self, x, keys=None, init_pos=None):
            raise RuntimeError("numba exploded")

    svc = MapLayoutService(make=lambda e: _Boom() if e == "umap" else PcaLayout(),
                           executor=_Inline())
    arc = _Arc()
    svc.bind(arc, tmp_path / "map_layout.npz", arc.encoder)
    svc.configure("umap")
    svc.update(arc)
    svc.update(arc)
    assert svc.error
    assert svc.transform_rows(arc, np.arange(len(arc))).shape == (len(arc), 2)


# ---- switching archives --------------------------------------------------

def test_binding_a_new_archive_drops_the_old_layout(tmp_path):
    """Entry ids restart in every archive; keeping positions would draw the
    previous archive's map."""
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    v = svc.version
    other = _Arc(n=30)
    svc.bind(other, tmp_path / "other.npz", other.encoder)
    assert not svc.fitted or svc.version != v


def test_rebinding_the_same_directory_keeps_the_layout(tmp_path):
    """A brain switch rebuilds the Archive OBJECT but the directory, the ids
    and the positions are all unchanged - and dropping the fit leaves the map
    with nothing to draw for a frame, which collapses the browser's contents
    and clamps its scroll to the top."""
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    assert svc.fitted

    same = _Arc()                             # what _build_archive_set makes
    svc.bind(same, tmp_path / "map_layout.npz", same.encoder)
    assert svc.fitted, "a same-directory rebind must not drop the layout"


def test_rebinding_a_different_directory_still_drops_it(tmp_path):
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    assert svc.fitted

    svc.bind(_Arc(), tmp_path / "other.npz", arc.encoder)
    assert not svc.fitted


def test_rebinding_under_another_encoder_drops_it(tmp_path):
    svc, arc = _svc(tmp_path)
    svc.update(arc)
    assert svc.fitted

    svc.bind(_Arc(), tmp_path / "map_layout.npz", "siglip2-b16")
    assert not svc.fitted
