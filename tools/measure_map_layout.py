"""Is UMAP actually a better map than PCA? Read-only over the real archives.

The yardstick is kNN(10) PRESERVATION: of the ten nearest entries in the full
embedding space, how many are still among the ten nearest on the 2-D map. PCA
is recorded at 27.0% over 500 entries and 3.1% over 13049.

    python -m tools.measure_map_layout
    python -m tools.measure_map_layout --sizes 500 2000 --archive debug09

Opens nothing through ArchiveStore: the app is usually running, and the store
creates directories and appends on construction. This reads vectors.npz
directly and writes nothing at all.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import ui  # noqa: F401,E402  prime the services/ui import cycle
from services.map_layout import PcaLayout, UmapLayout
from services.novelty import knn_distances
from utilities.paths import get_archives_root


def load_vectors(archive_dir: Path) -> np.ndarray | None:
    """Every layout's embeddings under one archive, pooled as the map does."""
    rows = []
    for npz in sorted(archive_dir.rglob("vectors.npz")):
        try:
            with np.load(npz, allow_pickle=False) as z:
                for key in ("embeddings", "emb", "vectors"):
                    if key in z:
                        rows.append(np.asarray(z[key], dtype=np.float32))
                        break
        except Exception:
            continue
    if not rows:
        return None
    width = max(r.shape[1] for r in rows)
    rows = [r for r in rows if r.shape[1] == width]
    return np.concatenate(rows, axis=0) if rows else None


def knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    """The k nearest rows of x for each row of x, excluding itself."""
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(x), k), dtype=np.int64)
    block = max(1, 4_000_000 // max(1, len(x)))
    for lo in range(0, len(x), block):
        sims = xn[lo:lo + block] @ xn.T
        for r in range(sims.shape[0]):
            sims[r, lo + r] = -2.0          # never your own row
        out[lo:lo + block] = np.argpartition(-sims, k, axis=1)[:, :k]
    return out


def preservation(high: np.ndarray, low: np.ndarray, k: int = 10) -> float:
    """Fraction of each row's k nearest that survive the projection."""
    a, b = knn_indices(high, k), knn_indices(low, k)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))]))


def run(x: np.ndarray, k: int) -> None:
    print(f"  {'engine':<8} {'kNN(10) kept':>13} {'fit s':>8}")
    for name, layout in (("pca", PcaLayout()), ("umap", UmapLayout())):
        t0 = time.perf_counter()
        ok = layout.fit(x, [("m", i) for i in range(len(x))])
        dt = time.perf_counter() - t0
        if not ok:
            why = getattr(layout, "unavailable", "") or "not enough data"
            print(f"  {name:<8} {'-':>13} {'-':>8}   ({why})")
            continue
        pos = layout.transform(x, [("m", i) for i in range(len(x))])
        print(f"  {name:<8} {preservation(x, pos, k):>12.1%} {dt:>8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None, help="name under archives/")
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[500, 2000, 5000, 13049])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    root = get_archives_root()
    names = ([args.archive] if args.archive
             else sorted(p.name for p in root.iterdir() if p.is_dir()))
    if not names:
        print(f"no archives under {root}")
        return

    for name in names:
        x = load_vectors(root / name)
        if x is None or len(x) < 100:
            print(f"\n{name}: no usable vectors")
            continue
        print(f"\n{name}: {len(x)} entries, dim {x.shape[1]}")
        rs = np.random.RandomState(0)
        for n in args.sizes:
            if n > len(x):
                continue
            sub = x[rs.choice(len(x), n, replace=False)]
            print(f" n={n}")
            run(sub, args.k)


if __name__ == "__main__":
    main()
