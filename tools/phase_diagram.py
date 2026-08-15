"""Phase diagram of one preset: sweep two physics parameters, one row per cell.

    python -m tools.phase_diagram --preset "fish soup" --pilot
    python -m tools.phase_diagram --preset "fish soup" --grid 128 --raw-stride 4

Each cell runs the preset to `--steps` with two parameters overridden, probing
cheap scalars every `--probe-every` steps and storing full fields at a handful
of log-spaced snapshots. Never renders: features come off the entity buffer and
the RG32F canvas directly, so there is no Camera, no bloom and no CLIP.

Two knobs, because the two costs differ by three orders of magnitude. Features
are ~5 KB a cell, so grid resolution is nearly free; raw snapshots are ~1 MiB a
cell and are written only on a nested sub-lattice at `--raw-stride`.

`world_size` shrinks the world without shrinking it RELATIVE TO THE PATTERN -
every length in entity_update.glsl scales by 1/sqrt(WORLD_SIZE) - so a small
world is a smaller window onto the same phase. It is a poor SPEED knob even so:
a cell is dominated by per-step Python dispatch, not by GPU work, so an 8x cut
in particles buys well under 40%. Prefer the largest world the storage budget
allows. `particle_density` is not a speed knob at all - it changes trail
intensity, which feeds back through the sensors into speed and moves the
diagram.

See docs/superpowers/specs/2026-08-14-phase-diagram-sweep-design.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import ui  # noqa: F401  prime the services/ui import cycle

from services import phase_metrics as pm

# One particle's deposit into the trail, so a texel above this has been visited.
DEFAULT_COVERAGE_THRESHOLD = 0.01

# Outside this the run has collapsed to a blob or gassed out. Provisional until
# a pilot has been read; whatever is used is written into the sidecar.
DEFAULT_PR_BRACKET = (0.02, 0.98)

PARTICLE_SUBSAMPLE = 4096
SUBSAMPLE_SEED = 20260814


def find_preset(name: str) -> Path:
    """User configs first, then the shipped library. Accepts a path too."""
    from utilities.paths import (get_app_physics_configs_dir,
                                 get_user_physics_configs_dir)

    direct = Path(name)
    if direct.suffix == ".json" and direct.exists():
        return direct

    roots = [get_user_physics_configs_dir(),
             get_app_physics_configs_dir() / "Core",
             get_app_physics_configs_dir() / "Advanced"]
    for root in roots:
        candidate = root / f"{name}.json"
        if candidate.exists():
            return candidate
    raise SystemExit(f"no preset named {name!r} under {[str(r) for r in roots]}")


def snapshot_steps(steps: int, count: int) -> np.ndarray:
    """Log-spaced, ending exactly on `steps`.

    Transients are fast early; linear spacing would spend most of the snapshots
    on the part that has already settled.
    """
    first = max(1, steps // 50)
    raw = np.geomspace(first, steps, max(1, count))
    out = np.unique(np.rint(raw).astype(np.int64))
    out[-1] = steps
    return np.unique(out)


def progressive_order(r: int) -> list[tuple[int, int]]:
    """Cell order that fills a COMPLETE coarse diagram before refining it.

    Raster order leaves an interrupted sweep as a half-height picture, which
    answers nothing about the half it never reached. Halving the stride instead
    means every checkpoint holds a full-extent diagram at some resolution, and
    the run can be stopped at any moment rather than at a planned one.
    """
    stride = 1 << max(0, int(np.ceil(np.log2(max(1, r)))))
    seen: set[tuple[int, int]] = set()
    order: list[tuple[int, int]] = []
    while stride >= 1:
        for iy in range(0, r, stride):
            for ix in range(0, r, stride):
                if (iy, ix) not in seen:
                    seen.add((iy, ix))
                    order.append((iy, ix))
        stride //= 2
    return order


class Harness:
    """One GL context and one Sim, reused across every cell.

    Rebuilding the Sim per cell would recompile shaders; `reset()` clears both
    canvas buffers and the frame counter, which is what makes the next update
    re-seed the particles.
    """

    def __init__(self, preset_path: Path, world_size: float, density: float):
        import moderngl

        from services.config_saver import ConfigSaver
        from sim import SIZE_OF_ENTITY_STRUCT, Sim
        from state import SimState

        self.stride = SIZE_OF_ENTITY_STRUCT // 4
        self.ctx = moderngl.create_standalone_context(require=430)

        saver = ConfigSaver()
        self.raw_config = json.loads(preset_path.read_text(encoding="utf-8"))
        config = saver.load_from_file(preset_path)
        if config is None:
            raise SystemExit(f"could not load {preset_path}")
        self.state = SimState()
        rule = saver.apply_config(config, self.state)

        if getattr(self.state, "parameter_sweeps_enabled", False):
            raise SystemExit(
                "this preset has parameter sweeps enabled, which makes a "
                "parameter vary WITHIN a cell by position - every comparison "
                "the diagram makes would be confounded.")

        self.sim = Sim(self.ctx, world_size=world_size,
                       canvas_aspect_ratio="1:1", particle_density=density)

        # The layout must be live BEFORE apply_rule: it measures a rule against
        # whatever layout is current and silently refuses a mismatch, which
        # would leave the sweep running a generated brain of the wrong modality.
        self._adopt_layout()
        self.sim.apply_state(self.state)
        self.sim.apply_rule(rule)
        if self.sim.slot0_params is None:
            raise SystemExit("the preset's rule was refused - wrong brain layout")
        if not np.any(self.sim.slot0_params):
            raise SystemExit("the preset's rule decoded to all zeros")

        self.width, self.height = self.sim.get_canvas_dimensions()
        n = self.sim.entity_count
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        k = min(PARTICLE_SUBSAMPLE, n)
        self.sub = np.sort(rng.choice(n, size=k, replace=False))

    def _adopt_layout(self) -> None:
        from services.brains import layout_from_signature

        sig = self.raw_config.get("brain_layout")
        if not sig:
            return
        layout = layout_from_signature(sig, self.raw_config.get("brain_settings"))
        if layout is None:
            raise SystemExit(f"cannot rebuild brain layout {sig!r}")
        if layout.signature() != self.sim.brain_layout.signature():
            self.sim.realloc_brain_buffers(layout)

    def read_field(self) -> np.ndarray:
        """The canvas as (H, W, 2) float32. RG32F: the trail is a velocity
        field, and both update branches leave the current image in the texture
        `can_read_index` names."""
        tex = self.sim.can_textures[self.sim.can_read_index]
        raw = np.frombuffer(tex.read(), dtype=np.float32)
        return raw.reshape(self.height, self.width, 2)

    def read_particles(self) -> tuple[np.ndarray, np.ndarray]:
        """(pos, vel) for the fixed subsample. The struct is pos:2, vel:2,
        size:1, pad:3, color:4."""
        raw = np.frombuffer(self.sim.entities.read(), dtype=np.float32)
        rows = raw.reshape(-1, self.stride)[self.sub]
        return rows[:, 0:2].copy(), rows[:, 2:4].copy()

    def run_cell(self, overrides: dict[str, float], seed: float, steps: int,
                 probe_every: int, snaps: np.ndarray, measures: np.ndarray,
                 cov: float, want_raw: bool):
        """-> (series, probe steps, frames, canvas stack | None, particles | None).

        Three independent schedules, deliberately decoupled:
          - PROBES, every `probe_every`, cheap reductions only;
          - MEASURES, the last few probe times, where the full frame features
            are computed and later averaged - one late frame is a sample of the
            fluctuation, not of the state;
          - SNAPSHOTS, log-spaced, where raw fields are kept for storage.
        """
        for name, value in overrides.items():
            setattr(self.state, name, float(value))
        self.sim.reset_seed = float(seed)
        self.sim.reset()
        self.sim.apply_state(self.state)

        snap_set = set(int(s) for s in snaps)
        measure_set = set(int(s) for s in measures)
        series, steps_probed, frames = [], [], []
        canvases, particles = [], []
        previous = None

        for step in range(1, steps + 1):
            self.sim.update(self.ctx)

            is_snap = step in snap_set
            is_measure = step in measure_set
            if step % probe_every and not is_snap and not is_measure:
                continue

            self.ctx.finish()
            field = self.read_field()
            rho = pm.rho_of(field)
            series.append(pm.probe_row(rho, previous, cov))
            steps_probed.append(step)
            previous = rho

            if is_measure or (is_snap and want_raw):
                pos, vel = self.read_particles()
                if is_measure:
                    frames.append(pm.frame_row(pos, vel, field, cov))
                if is_snap and want_raw:
                    canvases.append(field.astype(np.float16))
                    particles.append(
                        np.concatenate([pos, vel], axis=1).astype(np.float16))

        return (np.asarray(series), np.asarray(steps_probed),
                np.asarray(frames),
                np.asarray(canvases) if want_raw else None,
                np.asarray(particles) if want_raw else None)

    def release(self) -> None:
        self.ctx.release()


def _write_sidecar(out, args, harness, path, feats, frames, series, done,
                   strip, xs, ys, probe_steps, snaps, measures, raw_idx) -> None:
    """The features file. Rewritten wholesale, cheaply enough to run mid-sweep."""
    np.savez_compressed(
        out / "features.npz",
        cell_features=feats, frame_features=frames, probe_series=series,
        done=done, noise_strip=strip, x_values=xs, y_values=ys,
        probe_steps=probe_steps, snapshot_steps=snaps, measure_steps=measures,
        raw_index=raw_idx,
        cell_names=np.array(pm.CELL_NAMES), frame_names=np.array(pm.FRAME_NAMES),
        probe_names=np.array(pm.PROBE_NAMES),
        meta=np.array(json.dumps({
            "preset": str(path), "preset_name": path.stem,
            "brain_layout": harness.sim.brain_layout.signature(),
            "x_param": args.x_param, "y_param": args.y_param,
            "x_range": list(args.x_range), "y_range": list(args.y_range),
            "grid": int(args.grid), "steps": args.steps,
            "probe_every": args.probe_every, "measure_frames": args.measure_frames,
            "raw_stride": (0 if args.no_raw else args.raw_stride),
            "seed": args.seed, "world_size": args.world_size,
            "particle_density": args.density,
            "canvas": [harness.width, harness.height],
            "entity_count": harness.sim.entity_count,
            "coverage_threshold": args.coverage_threshold,
            "pr_bracket": [args.pr_lo, args.pr_hi],
            "particle_subsample": len(harness.sub),
            "config": harness.raw_config,
        })))


def sweep(args) -> int:
    path = find_preset(args.preset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    harness = Harness(path, args.world_size, args.density)
    state = harness.state
    for axis in (args.x_param, args.y_param):
        if not hasattr(state, axis):
            harness.release()
            raise SystemExit(f"{axis!r} is not a SimState field")

    r = int(args.grid)
    xs = np.linspace(args.x_range[0], args.x_range[1], r)
    ys = np.linspace(args.y_range[0], args.y_range[1], r)
    snaps = snapshot_steps(args.steps, args.snapshots)
    probe_steps = np.arange(args.probe_every, args.steps + 1, args.probe_every)
    measures = probe_steps[-max(1, args.measure_frames):]
    n_snap = len(snaps)
    n_probe = len(probe_steps)

    raw_idx = (np.zeros(0, dtype=int) if args.no_raw
               else np.arange(0, r, max(1, args.raw_stride)))
    raw_pos = {int(i): j for j, i in enumerate(raw_idx)}
    rr = len(raw_idx)

    print(f"preset      {path.name}  ({harness.sim.brain_layout.signature()})")
    print(f"axes        {args.x_param} {args.x_range} x "
          f"{args.y_param} {args.y_range}")
    print(f"grid        {r}x{r} = {r * r} cells, {args.steps} steps each")
    print(f"world       size {args.world_size}, canvas "
          f"{harness.width}x{harness.height}, {harness.sim.entity_count} particles")
    print(f"measure     {list(measures)} (frame features averaged over these)")
    if args.no_raw:
        print("raw         off")
    else:
        print(f"raw         at {list(snaps)}, every {args.raw_stride} "
              f"-> {rr}x{rr} = {rr * rr} cells")

    feats = np.full((r, r, len(pm.CELL_NAMES)), np.nan, dtype=np.float32)
    frames = np.full((r, r, len(measures), len(pm.FRAME_NAMES)), np.nan,
                     dtype=np.float32)
    series = np.full((r, r, n_probe, len(pm.PROBE_NAMES)), np.nan, dtype=np.float32)
    done = np.zeros((r, r), dtype=bool)

    canvas_mm = np.lib.format.open_memmap(
        out / "canvas.npy", mode="w+", dtype=np.float16,
        shape=(rr, rr, n_snap, harness.height, harness.width, 2))
    particle_mm = np.lib.format.open_memmap(
        out / "particles.npy", mode="w+", dtype=np.float16,
        shape=(rr, rr, n_snap, len(harness.sub), 4))

    strip = np.zeros((0, len(pm.CELL_NAMES)), dtype=np.float32)

    def checkpoint():
        """A long sweep must be readable before it finishes.

        Cells are filled in row-major order, so a partial file is a COMPLETE
        diagram of however many rows are done - useful rather than merely
        recoverable.
        """
        _write_sidecar(out, args, harness, path, feats, frames, series, done,
                       strip, xs, ys, probe_steps, snaps, measures, raw_idx)

    order = (progressive_order(r) if args.order == "progressive"
             else [(iy, ix) for iy in range(r) for ix in range(r)])
    started = time.time()
    total = r * r
    every = max(1, min(512, total // 128))
    for n, (iy, ix) in enumerate(order, start=1):
        want_raw = ix in raw_pos and iy in raw_pos
        s, probed, f, canvas, parts = harness.run_cell(
            {args.x_param: xs[ix], args.y_param: ys[iy]},
            args.seed, args.steps, args.probe_every, snaps, measures,
            args.coverage_threshold, want_raw)

        series[iy, ix, :len(s)] = s[:n_probe]
        frames[iy, ix, :len(f)] = f
        feats[iy, ix] = pm.cell_row(f, probed, s, args.pr_lo, args.pr_hi,
                                    args.steps)
        done[iy, ix] = True
        if want_raw:
            canvas_mm[raw_pos[iy], raw_pos[ix]] = canvas
            particle_mm[raw_pos[iy], raw_pos[ix]] = parts

        if n % every == 0 or n == total:
            per = (time.time() - started) / n
            left = per * (total - n)
            print(f"\r  {n}/{total}  {per:.2f}s/cell  "
                  f"eta {left / 3600:.2f} h   ", end="", flush=True)
            checkpoint()
    print()

    if args.noise_strip:
        # EVERYTHING identical, both seeds included: both are held fixed across
        # the grid, so the intrinsic splat race is the grid's only noise source
        # and reproducing exactly that is the only way to size it.
        print(f"noise floor: {args.noise_strip} repeats at the preset's own values")
        base = {args.x_param: getattr(state, args.x_param),
                args.y_param: getattr(state, args.y_param)}
        rows = []
        for _ in range(args.noise_strip):
            s, probed, f, _, _ = harness.run_cell(
                base, args.seed, args.steps, args.probe_every, snaps, measures,
                args.coverage_threshold, False)
            rows.append(pm.cell_row(f, probed, s, args.pr_lo, args.pr_hi,
                                    args.steps))
        strip = np.asarray(rows, dtype=np.float32)

    canvas_mm.flush()
    particle_mm.flush()
    harness.release()
    checkpoint()

    elapsed = (time.time() - started) / 60.0
    print(f"\nwrote {out}  ({elapsed:.1f} min)\n")

    from tools.phase_view import summarise
    summarise(feats, strip, list(pm.CELL_NAMES), (args.pr_lo, args.pr_hi))
    return 0


def pilot(args) -> int:
    """A coarse sweep at several world sizes, before any grid is sized.

    Answers the three things that cannot be answered on paper: seconds per run,
    where the features stop moving with world size, and whether the step budget
    is enough.
    """
    path = find_preset(args.preset)
    sizes = [float(s) for s in args.pilot_sizes.split(",")]
    r = int(args.pilot_grid)
    interesting = ["participation_ratio", "structure", "spec_peak_wavelen",
                   "change_rate", "polar_order", "alive_steps"]

    print(f"pilot: {path.name}, {r}x{r} cells at each of {sizes}\n")
    for ws in sizes:
        harness = Harness(path, ws, args.density)
        snaps = snapshot_steps(args.steps, args.snapshots)
        probes = np.arange(args.probe_every, args.steps + 1, args.probe_every)
        measures = probes[-max(1, args.measure_frames):]
        xs = np.linspace(args.x_range[0], args.x_range[1], r)
        ys = np.linspace(args.y_range[0], args.y_range[1], r)

        rows, started = [], time.time()
        for yv in ys:
            for xv in xs:
                s, probed, f, _, _ = harness.run_cell(
                    {args.x_param: xv, args.y_param: yv}, args.seed,
                    args.steps, args.probe_every, snaps, measures,
                    args.coverage_threshold, False)
                rows.append(pm.cell_row(f, probed, s, args.pr_lo, args.pr_hi,
                                        args.steps))
        per = (time.time() - started) / len(rows)
        arr = np.asarray(rows)

        print(f"world_size {ws:<7} canvas {harness.width:>4}  "
              f"{harness.sim.entity_count:>7} particles  {per:.2f} s/cell")
        for name in interesting:
            col = arr[:, pm.CELL_NAMES.index(name)]
            n_bad = int((~np.isfinite(col)).sum())
            note = f"   ({n_bad} undefined)" if n_bad else ""
            print(f"    {name:<20} median {np.nanmedian(col):9.4f}  "
                  f"range {np.nanmin(col):9.4f} .. {np.nanmax(col):9.4f}{note}")
        est = per * (args.grid ** 2) / 3600.0
        print(f"    a {args.grid}x{args.grid} grid would take {est:.1f} h\n")
        harness.release()
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="fish soup")
    ap.add_argument("--x-param", default="AXIAL_FORCE")
    ap.add_argument("--y-param", default="LATERAL_FORCE")
    ap.add_argument("--x-range", type=float, nargs=2, default=(-1.0, 1.0))
    ap.add_argument("--y-range", type=float, nargs=2, default=(-1.0, 1.0))
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--snapshots", type=int, default=5)
    ap.add_argument("--order", choices=("progressive", "raster"),
                    default="progressive",
                    help="progressive fills a complete coarse diagram first, "
                         "so an interrupted sweep is still full-extent")
    ap.add_argument("--measure-frames", type=int, default=8,
                    help="frame features are averaged over this many late "
                         "probes; a single frame samples the fluctuation, not "
                         "the state")
    ap.add_argument("--raw-stride", type=int, default=4,
                    help="store full fields on every Nth cell in each axis")
    ap.add_argument("--no-raw", action="store_true",
                    help="features only - the large grids cannot afford fields")
    ap.add_argument("--world-size", type=float, default=0.10,
                    help="smaller is a smaller WINDOW on the same phase, not "
                         "different physics - but it buys much less time than "
                         "it looks like it should; see the module docstring")
    ap.add_argument("--density", type=float, default=1.0,
                    help="NOT a speed knob - it changes trail intensity, which "
                         "feeds back into speed and moves the diagram")
    ap.add_argument("--seed", type=float, default=12345.0)
    ap.add_argument("--coverage-threshold", type=float,
                    default=DEFAULT_COVERAGE_THRESHOLD)
    ap.add_argument("--pr-lo", type=float, default=DEFAULT_PR_BRACKET[0])
    ap.add_argument("--pr-hi", type=float, default=DEFAULT_PR_BRACKET[1])
    ap.add_argument("--noise-strip", type=int, default=32,
                    help="repeats at identical parameters, to size the speckle")
    ap.add_argument("--out", default="phase_out")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-sizes", default="0.20,0.10,0.05,0.025")
    ap.add_argument("--pilot-grid", type=int, default=5)
    args = ap.parse_args(argv)

    return pilot(args) if args.pilot else sweep(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
