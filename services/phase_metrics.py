"""Scalar summaries of one run, for a phase diagram.

Pure numpy over arrays already read back from the GPU. Nothing here imports
moderngl or Sim, so every feature is testable against a synthetic field whose
answer is known by construction; `tools/phase_diagram.py` owns the GL side.

Lenia's survival plots do not transcribe directly: its mass is a free variable
with two failure modes, and Fluoddity's particle count is fixed. What is free
here is how the trail CONCENTRATES - the same particles piled into a few texels
or spread across all of them - and that carries the same two failure modes.
`participation_ratio` is the analogue, and `alive_steps` is the transcription.

See docs/superpowers/specs/2026-08-14-phase-diagram-sweep-design.md.
"""
from __future__ import annotations

import numpy as np

from services.capture_health import structure as _structure

# Computed at every probe. Cheap enough to run every 50 steps: reductions only,
# no FFT and no autocorrelation.
PROBE_NAMES: tuple[str, ...] = (
    "participation_ratio",
    "coverage",
    "rho_mean",
    "change",
)

# Computed at each stored snapshot. The final snapshot's values are the
# diagram's instantaneous channels.
FRAME_NAMES: tuple[str, ...] = (
    "speed_p50",
    "speed_p90",
    "polar_order",
    "particle_pr",
    "participation_ratio",
    "coverage",
    "structure",
    "spec_peak_wavelen",
    "spec_entropy",
    "field_order",
)

# One row of the phase diagram: the final frame, plus the two the series owns.
CELL_NAMES: tuple[str, ...] = FRAME_NAMES + ("change_rate", "alive_steps")


def rho_of(field: np.ndarray) -> np.ndarray:
    """(H, W, 2) velocity field -> (H, W) magnitude, float64.

    The canvas is RG32F because the trail IS a velocity field; every field
    feature below is a function of this magnitude or of the raw vectors.
    """
    f = np.asarray(field, dtype=np.float64)
    return np.sqrt((f * f).sum(axis=-1))


def participation_ratio(a: np.ndarray) -> float:
    """Non-negative array -> (0, 1]. The fraction of cells effectively occupied.

    `(sum a)^2 / (N * sum a^2)`. Mass in m of N cells scores m/N, so a blob
    tends to 0 and a uniform field to 1, with no threshold to choose. This is
    the stand-in for Lenia's mass: particle count is fixed here, but how that
    fixed mass concentrates is not.
    """
    x = np.asarray(a, dtype=np.float64)
    s1 = x.sum()
    s2 = (x * x).sum()
    if s2 <= 0.0:
        return 0.0
    return float(s1 * s1 / (x.size * s2))


def coverage(rho: np.ndarray, threshold: float) -> float:
    """Fraction of texels above an ABSOLUTE threshold.

    Absolute, never a fraction of this frame's own max: a per-frame
    normalisation would make the number incomparable between the very cells the
    diagram exists to compare.
    """
    return float((np.asarray(rho) > float(threshold)).mean())


def polar_order(vel: np.ndarray) -> float:
    """(n, 2) particle velocities -> [0, 1]. The Vicsek order parameter.

    Magnitude of the mean UNIT velocity: one coherent stream scores 1 and an
    isotropic swarm scores 0, whatever the speed. Particles at rest carry no
    direction and are dropped rather than counted as agreeing.
    """
    v = np.asarray(vel, dtype=np.float64)
    mag = np.sqrt((v * v).sum(axis=-1))
    moving = mag > 0.0
    if not moving.any():
        return 0.0
    u = v[moving] / mag[moving, None]
    return float(np.sqrt((u.mean(axis=0) ** 2).sum()))


def field_order(field: np.ndarray) -> float:
    """(H, W, 2) -> [0, 1]. `polar_order`'s field-side companion.

    The rho-weighted mean of the unit vector is `sum(v) / sum(|v|)`, so no texel
    is ever divided by its own near-zero magnitude.
    """
    f = np.asarray(field, dtype=np.float64)
    total = np.sqrt((f * f).sum(axis=-1)).sum()
    if total <= 0.0:
        return 0.0
    return float(np.sqrt((f.sum(axis=(0, 1)) ** 2).sum()) / total)


def speed_percentiles(vel: np.ndarray) -> tuple[float, float]:
    """(n, 2) -> (p50, p90) of |v|. Separates frozen from active."""
    v = np.asarray(vel, dtype=np.float64)
    mag = np.sqrt((v * v).sum(axis=-1))
    mag = mag[np.isfinite(mag)]
    if not mag.size:
        return 0.0, 0.0
    p50, p90 = np.percentile(mag, [50.0, 90.0])
    return float(p50), float(p90)


def particle_pr(pos: np.ndarray, bins: int = 64, extent: float = 1.0) -> float:
    """(n, 2) positions -> (0, 1]. Clustering, via the participation ratio of a
    coarse occupancy histogram.

    A nearest-neighbour index would say the same thing better, and would need a
    KD-tree; scipy is not a dependency of this project. Positions outside
    [-extent, extent] are dropped, which under a wrap boundary is nothing.
    """
    p = np.asarray(pos, dtype=np.float64)
    if not p.size:
        return 0.0
    rng = [[-extent, extent], [-extent, extent]]
    hist, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=int(bins), range=rng)
    return participation_ratio(hist)


def radial_spectrum(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) -> (wavenumbers, power), azimuthally averaged, DC removed.

    Wavenumbers are integer cycles-per-image; bin 0 is forced to zero because
    the mean is subtracted and the near-DC bin would otherwise collect the
    residual and win every argmax.

    Each annulus is AVERAGED over its modes, not summed. An annulus at radius k
    holds a number of modes proportional to k, so a sum makes white noise slope
    upwards - which puts a spurious peak at the Nyquist bin and costs a flat
    spectrum its entropy.
    """
    g = np.asarray(rho, dtype=np.float64)
    g = g - g.mean()
    power2d = np.abs(np.fft.fft2(g)) ** 2
    h, w = g.shape
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    k = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    kmax = min(h, w) // 2
    idx = np.clip(np.rint(k).astype(np.int64), 0, kmax)
    total = np.bincount(idx.ravel(), weights=power2d.ravel(), minlength=kmax + 1)
    modes = np.bincount(idx.ravel(), minlength=kmax + 1)
    power = total[: kmax + 1] / np.maximum(modes[: kmax + 1], 1)
    power[0] = 0.0
    return np.arange(kmax + 1, dtype=np.float64), power


def _interpolated_peak(power: np.ndarray) -> float:
    """Sub-bin position of the spectrum's peak, by a parabola through its log.

    Wavelength is `width / k`, so the bins are crowded at large scales: at a
    228-texel canvas the first three bins are 228, 114 and 76 texels, and a
    whole family of distinguishable patterns collapses onto one of them. The
    interpolation is what makes the feature continuous enough to show a
    gradient rather than three terraces.
    """
    i = int(np.argmax(power))
    if i <= 0 or i >= len(power) - 1:
        return float(i)
    trio = power[i - 1:i + 2]
    if np.any(trio <= 0.0):
        return float(i)
    y0, y1, y2 = np.log(trio)
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(i)
    return float(i) + float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))


def spectral_features(rho: np.ndarray) -> tuple[float, float]:
    """(H, W) -> (peak wavelength in TEXELS, normalised spectral entropy).

    Wavelength rather than wavenumber because `world_size` changes the canvas
    size: every length in the shader scales by 1/sqrt(world_size) too, so a
    pattern's size in texels is invariant while its cycles-per-image is not, and
    the pilot compares world sizes directly.

    Entropy is normalised by log(number of bins), so 1 is a flat (broadband)
    spectrum and a single sharp peak tends to 0.

    A field with no spectrum at all - dead, or perfectly flat - returns NaN
    rather than 0. Zero is a legal-looking wavelength, and a sentinel that
    collides with data reads as a real measurement of the finest possible
    scale, which is the opposite of what it means.
    """
    k, power = radial_spectrum(rho)
    total = power.sum()
    if total <= 0.0:
        return float("nan"), float("nan")
    peak = _interpolated_peak(power)
    wavelen = float(rho.shape[1]) / peak if peak > 0.0 else float("nan")
    q = power / total
    q = q[q > 0.0]
    entropy = float(-(q * np.log(q)).sum() / np.log(len(power)))
    return wavelen, entropy


def structure(rho: np.ndarray) -> float:
    """(H, W) -> [0, 1]. Coherent 1, white noise 0.

    `capture_health.structure` averages over its last axis before doing anything
    else, so a single-channel float field needs no adaptation - only the batch
    and channel axes. Reused rather than reimplemented so its lag choice stays
    in one place.
    """
    a = np.asarray(rho, dtype=np.float32)[None, :, :, None]
    return float(_structure(a)[0])


def change(rho: np.ndarray, previous: np.ndarray | None) -> float:
    """L1 change since the previous probe, as a fraction of current L1 mass.

    Frozen scores 0 and a fully redrawn field scores about 2. This is
    `descriptor.liveness()`'s question without CLIP's answer: it compares
    fields, not embeddings, so it sees a rigid translation as change - which is
    correct here, since a travelling structure is alive.
    """
    if previous is None:
        return 0.0
    a = np.asarray(rho, dtype=np.float64)
    b = np.asarray(previous, dtype=np.float64)
    denom = np.abs(a).sum()
    if denom <= 0.0:
        return 0.0
    return float(np.abs(a - b).sum() / denom)


def probe_row(rho: np.ndarray, previous: np.ndarray | None,
              coverage_threshold: float) -> np.ndarray:
    """One row of the dense series, in PROBE_NAMES order."""
    return np.array([
        participation_ratio(rho),
        coverage(rho, coverage_threshold),
        float(np.asarray(rho, dtype=np.float64).mean()),
        change(rho, previous),
    ], dtype=np.float64)


def frame_row(pos: np.ndarray, vel: np.ndarray, field: np.ndarray,
              coverage_threshold: float) -> np.ndarray:
    """One snapshot's features, in FRAME_NAMES order."""
    rho = rho_of(field)
    p50, p90 = speed_percentiles(vel)
    wavelen, entropy = spectral_features(rho)
    return np.array([
        p50,
        p90,
        polar_order(vel),
        particle_pr(pos),
        participation_ratio(rho),
        coverage(rho, coverage_threshold),
        structure(rho),
        wavelen,
        entropy,
        field_order(field),
    ], dtype=np.float64)


def alive_steps(steps: np.ndarray, pr: np.ndarray, lo: float, hi: float,
                budget: int) -> float:
    """First step at which participation ratio leaves [lo, hi], else `budget`.

    The Lenia transcription. The bracket is not guessable in advance - the
    caller reads it off a pilot and writes it into the sidecar, so a diagram
    always carries the definition that produced it.
    """
    s = np.asarray(steps)
    v = np.asarray(pr, dtype=np.float64)
    outside = (v < float(lo)) | (v > float(hi))
    if not outside.any():
        return float(budget)
    return float(s[int(np.argmax(outside))])


def cell_row(frames: np.ndarray, steps: np.ndarray, series: np.ndarray,
             pr_lo: float, pr_hi: float, budget: int,
             tail_fraction: float = 0.25) -> np.ndarray:
    """The diagram's row for one cell, in CELL_NAMES order.

    `frames` is (M, len(FRAME_NAMES)) taken at the LAST M probe times and
    AVERAGED, not read off the final frame. Measured on `fish soup`: past about
    2500 steps a cell's participation ratio stops converging and simply
    fluctuates, at a fifth to two fifths of the whole diagram's spread. A single
    frame therefore samples that fluctuation rather than the state, and no step
    budget fixes it - only averaging does. See the design doc.

    `change_rate` averages the tail of the dense series for the same reason,
    and skips the opening transient, which is large in every cell and would
    otherwise swamp the difference between a settled pattern and a churning one.
    """
    last = np.nanmean(np.atleast_2d(np.asarray(frames, dtype=np.float64)), axis=0)
    ser = np.asarray(series, dtype=np.float64)
    ch = ser[:, PROBE_NAMES.index("change")]
    tail = max(1, int(round(len(ch) * float(tail_fraction))))
    change_rate = float(ch[-tail:].mean()) if len(ch) else 0.0
    pr = ser[:, PROBE_NAMES.index("participation_ratio")]
    return np.concatenate([
        last,
        [change_rate, alive_steps(steps, pr, pr_lo, pr_hi, budget)],
    ])
