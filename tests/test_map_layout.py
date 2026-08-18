"""The map's 2-D layout: PCA, UMAP, and everything that keeps a refit stable.

UMAP lays out a SNAPSHOT. Entries admitted after a fit are placed by kNN
interpolation rather than by umap.transform, so most of what is tested here is
placement, alignment and the cache - none of which need umap installed.
"""
import pathlib

import numpy as np
import pytest

from services.map_layout import (
    LayoutCache,
    PcaLayout,
    UmapLayout,
    knn_place,
    procrustes_align,
    staleness,
)


def _ids(n, sig="fourier-n10"):
    return [(sig, i) for i in range(n)]


def _blobs(n=60, dim=8, seed=0):
    """Three separated clusters, so a layout has something real to find."""
    rs = np.random.RandomState(seed)
    centres = rs.randn(3, dim).astype(np.float32) * 4.0
    x = np.concatenate([centres[i % 3] + rs.randn(dim).astype(np.float32) * 0.3
                        for i in range(n)]).reshape(n, dim)
    return x.astype(np.float32)


# ---- PCA stays exactly what it was --------------------------------------

def test_the_pca_layout_reproduces_the_projection_it_wraps():
    """PCA is the default and the fallback; this change must not move it."""
    from services.archive_projection import Projection

    x = _blobs()
    want = Projection(2)
    want.fit(x)

    got = PcaLayout()
    got.fit(x, _ids(len(x)))

    assert np.allclose(got.transform(x, _ids(len(x))), want.transform(x))


def test_an_unfitted_layout_answers_zeros_rather_than_raising():
    p = PcaLayout()
    assert not p.fitted
    assert p.transform(_blobs(4), _ids(4)).shape == (4, 2)


def test_a_fit_bumps_the_version_so_a_view_can_cache_on_it():
    p = PcaLayout()
    v = p.version
    p.fit(_blobs(), _ids(60))
    assert p.version == v + 1


# ---- placing entries admitted after the fit ------------------------------

def test_knn_places_a_duplicate_on_top_of_its_twin():
    """The whole point: a new entry lands where its neighbours already are."""
    ref = _blobs(30)
    pos = np.random.RandomState(1).randn(30, 2).astype(np.float32)
    twin = ref[7:8].copy()
    out = knn_place(twin, ref, pos, k=1)
    assert np.allclose(out[0], pos[7], atol=1e-5)


def test_knn_places_a_point_between_the_neighbours_it_sits_between():
    ref = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    pos = np.array([[10.0, 0.0], [-10.0, 0.0]], dtype=np.float32)
    out = knn_place(np.array([[0.0, 1.0]], dtype=np.float32), ref, pos, k=2)
    assert abs(float(out[0, 0])) < 1e-4


def test_knn_placement_with_no_reference_is_the_origin_not_a_crash():
    out = knn_place(_blobs(3), np.zeros((0, 8), np.float32),
                    np.zeros((0, 2), np.float32), k=5)
    assert out.shape == (3, 2)
    assert np.all(out == 0.0)


def test_knn_placement_asks_for_no_more_neighbours_than_exist():
    ref = _blobs(2)
    pos = np.zeros((2, 2), dtype=np.float32)
    assert knn_place(_blobs(1, seed=3), ref, pos, k=10).shape == (1, 2)


# ---- keeping a refit from rearranging the map ----------------------------

def test_procrustes_undoes_a_rotation():
    rs = np.random.RandomState(2)
    a = rs.randn(40, 2).astype(np.float32)
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th), np.cos(th)]], dtype=np.float32)
    b = a @ rot.T
    out = procrustes_align(b, a)
    assert np.linalg.norm(out - a) < np.linalg.norm(b - a) * 0.05


def test_procrustes_undoes_a_reflection():
    """UMAP can mirror as well as rotate, and a mirrored map is a new map."""
    rs = np.random.RandomState(3)
    a = rs.randn(40, 2).astype(np.float32)
    b = a * np.array([1.0, -1.0], dtype=np.float32)
    out = procrustes_align(b, a)
    assert np.linalg.norm(out - a) < np.linalg.norm(b - a) * 0.05


def test_procrustes_leaves_an_already_aligned_layout_alone():
    rs = np.random.RandomState(4)
    a = rs.randn(30, 2).astype(np.float32)
    assert np.allclose(procrustes_align(a.copy(), a), a, atol=1e-4)


def test_procrustes_with_nothing_to_align_against_returns_its_input():
    a = np.random.RandomState(5).randn(10, 2).astype(np.float32)
    assert np.allclose(procrustes_align(a, np.zeros((0, 2), np.float32)), a)


# ---- when to refit -------------------------------------------------------

@pytest.mark.parametrize("size,fitted_n,want", [
    (100, 100, 0.0),
    (125, 100, 0.25),
    (200, 100, 1.0),
    (90, 100, 0.0),        # eviction shrank it; that is not staleness
])
def test_staleness_is_growth_over_the_size_it_was_fitted_at(size, fitted_n, want):
    assert staleness(size, fitted_n) == pytest.approx(want)


def test_an_unfitted_layout_is_maximally_stale_rather_than_dividing_by_zero():
    assert staleness(10, 0) >= 1.0


# ---- the cache -----------------------------------------------------------

def _cache(n=12):
    return LayoutCache(
        pos=np.random.RandomState(6).randn(n, 2).astype(np.float32),
        keys=_ids(n), encoder="clip-b32",
        reduction=np.eye(4, 8, dtype=np.float32),
        reduction_mean=np.zeros(8, dtype=np.float32),
        fitted_n=n, engine="umap")


def test_a_cache_round_trips_through_a_file(tmp_path):
    c = _cache()
    p = tmp_path / "map_layout.npz"
    c.save(p)
    back = LayoutCache.load(p, encoder="clip-b32")
    assert back is not None
    assert np.allclose(back.pos, c.pos)
    assert back.keys == c.keys
    assert back.fitted_n == c.fitted_n
    assert back.engine == "umap"


def test_a_cache_from_another_encoder_is_refused():
    """At equal width a foreign vector is silently wrong, not an error - which
    is why load_from_store already refuses one."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "map_layout.npz"
        _cache().save(p)
        assert LayoutCache.load(p, encoder="siglip2-b16") is None


def test_a_corrupt_cache_is_refused_rather_than_raising(tmp_path):
    """It is a view. The fallback is a refit, never a crash on open."""
    p = tmp_path / "map_layout.npz"
    p.write_bytes(b"not an npz at all")
    assert LayoutCache.load(p, encoder="clip-b32") is None


def test_a_missing_cache_is_simply_absent(tmp_path):
    assert LayoutCache.load(tmp_path / "nope.npz", encoder="clip-b32") is None


def test_the_cache_looks_up_positions_by_layout_and_id():
    """An id is unique only inside one brain's directory, and the map pools
    every layout - so a bare id would hand back another brain's position."""
    c = _cache()
    keys = [("fourier-n10", 3), ("mlp-n16-a0", 3), ("fourier-n10", 999)]
    pos, hit = c.lookup(keys)
    assert hit.tolist() == [True, False, False]
    assert np.allclose(pos[0], c.pos[3])


# ---- the engine, when it is not installed --------------------------------

def test_a_missing_umap_leaves_the_layout_unavailable_and_quiet(monkeypatch):
    """Absent means PCA only with a note, never an exception mid-frame."""
    import builtins
    real = builtins.__import__

    def no_umap(name, *a, **kw):
        if name == "umap" or name.startswith("umap."):
            raise ImportError("no umap here")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_umap)
    u = UmapLayout()
    assert u.fit(_blobs(), _ids(60)) is False
    assert not u.fitted
    assert u.unavailable


# ---- the real engine ----------------------------------------------------
#
# Marked slow: umap pays a numba JIT on first use. The QUALITY claim (roughly
# double PCA's kNN preservation over the real archives) is measured by
# tools/measure_map_layout.py, not asserted here.

pytestmark_slow = pytest.mark.slow


@pytest.mark.slow
def test_a_seeded_refit_stays_near_the_layout_it_started_from():
    """The reason UMAP is usable at all: a refit must refine the map, not
    rearrange it. Without init= and the Procrustes alignment, the same points
    come back somewhere else entirely."""
    umap = pytest.importorskip("umap")          # noqa: F841
    x = _blobs(n=120, dim=12, seed=7)
    keys = _ids(len(x))

    first = UmapLayout(seed=1)
    assert first.fit(x, keys)
    before = first.transform(x, keys)

    second = UmapLayout(seed=2)                 # a DIFFERENT seed on purpose
    assert second.fit(x, keys, init_pos=before)
    after = second.transform(x, keys)

    loose = UmapLayout(seed=2)
    assert loose.fit(x, keys)                   # same seed, no seeding
    adrift = procrustes_align(loose.transform(x, keys), before)

    def spread(p):
        return float(np.linalg.norm(p - p.mean(axis=0), axis=1).mean())

    seeded_move = float(np.linalg.norm(after - before, axis=1).mean())
    adrift_move = float(np.linalg.norm(adrift - before, axis=1).mean())
    assert seeded_move < adrift_move, (seeded_move, adrift_move)
    assert seeded_move < spread(before)


@pytest.mark.slow
def test_a_cache_replays_the_positions_it_stored(tmp_path):
    """Reopening an archive must show the map it had, without refitting."""
    pytest.importorskip("umap")
    x = _blobs(n=80, dim=10, seed=3)
    keys = _ids(len(x))
    fitted = UmapLayout(seed=5)
    assert fitted.fit(x, keys)
    want = fitted.transform(x, keys)

    p = tmp_path / "map_layout.npz"
    fitted.to_cache("clip-b32").save(p)

    back = UmapLayout()
    assert back.adopt(LayoutCache.load(p, "clip-b32"), x, keys)
    assert np.allclose(back.transform(x, keys), want, atol=1e-4)


def test_asking_whether_umap_exists_does_not_import_it():
    """`import umap` pulls in numba and costs seconds. The Layout combo asks
    this every frame to decide whether to offer UMAP at all, so the first Map
    frame froze the whole app while it loaded a library nobody had chosen."""
    import subprocess
    import sys as _sys
    code = (
        "import sys, pathlib;"
        "sys.path.insert(0, r'" + str(pathlib.Path(__file__).resolve().parents[1]) + "');"
        "import ui;"   # prime the services/ui import cycle
        "from services.map_layout import UmapLayout;"
        "ok = UmapLayout.available();"
        "print('AVAILABLE', ok, 'IMPORTED', 'umap' in sys.modules)"
    )
    r = subprocess.run([_sys.executable, "-c", code],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert "AVAILABLE True" in out, out
    assert "IMPORTED False" in out, out
