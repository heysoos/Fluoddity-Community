"""Does vision-guided search actually train, and which brain trains best?

    python -m tools.brain_search_bench --smoke
    python -m tools.brain_search_bench --out runs/bench.json \
        --configs fourier:5,fourier:19,gabor:11,lenia:15,mlp:16 \
        --algorithms "CMA-ES,Random Search" --generations 20

Runs the REAL Auto-tournament loop headlessly: the same AutoTournamentService
state machine, the same PromptDriver, the same CaptureView the app scores. Only
the GLFW window is hidden and the per-frame budget is gone. Nothing here
reimplements a rollout, a capture or a fitness - a benchmark that drifts from
the app measures the benchmark.

WHY THE RANDOM SEARCH CONTROL IS NOT OPTIONAL. Modalities have different priors,
so their absolute fitness is not comparable: a brain whose random draws happen to
make busier pictures starts higher and can then go nowhere. What a search is for
is the DIFFERENCE between an optimizer and drawing the same number of samples at
random, and that difference is the only number here that answers "does it train".
Report both; rank on the difference.

WHY EVERY RUN SHARES ONE PRESET. Brain-only search shapes local behaviour on top
of the gross morphology the physics preset fixes, so the preset is at least as
strong a variable as the brain. It is held constant and named in the output.

Note that the preset also decides the BRAIN'S INPUT SCALE, which varies ~900x
across the library (tools/brain_input_scale.py). Gabor compares its input to a
centre in 4-D, so a centre spread that does not match the input range degenerates
its envelope into a plain oscillation - which would read here as "Gabor trains
badly" when what happened is that it was handed a mis-set slider. Pass
--gabor-input-scale with the preset's measured p90, or accept the default and
read the caveat in the output header.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui.physics_params -> ui/__init__ -> ui.core -> services.config_saver. It
# resolves only when `ui` is imported first. Prime it, as tests/conftest.py does.
import ui  # noqa: F401,E402

from state.view_modes import CAMERA as VIEW_CAM_BRUSH  # one home: state/view_modes.py
WINDOW_PX = 512         # only has to exist; the capture re-renders at grid*224

DEFAULT_PROMPTS = (
    "flowing water",
    "tree branches",
    "a spiral galaxy",
)


def count_key(modality) -> str:
    """The setting that changes a modality's parameter COUNT.

    Derived from settings_schema() rather than a hand-written map. The
    hand-written map is exactly what let two 'declared but never read' settings
    through before; see tests/test_brain_scales.py for the same reasoning.
    """
    for s in modality.settings_schema():
        if s.kind == "int":
            return s.key
    raise ValueError(f"{modality.name} declares no int setting")


def parse_configs(text: str, extra: dict) -> list:
    """'fourier:19,gabor:11' -> [BrainLayout, ...]."""
    from services.brains import REGISTRY

    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, n = item.partition(":")
        m = REGISTRY[name.strip()]
        settings = dict(extra.get(name.strip(), {}))
        if n:
            settings[count_key(m)] = int(n)
        out.append(m.layout_from_settings(settings))
    return out


# ---- the rollout ---------------------------------------------------------

class Bench:
    """One GL context, one CLIP session, reused across every config."""

    def __init__(self, preset: str, world_size, density, window_px=WINDOW_PX):
        import glfw
        import moderngl

        from camera import Camera
        from services.capture_blit import CaptureBlit
        from services.capture_view import CaptureView
        from services.vision_scorer import VisionScorer
        from services.config_saver import ConfigSaver
        from sim import Sim
        from state import SimState
        from state.ui_state import UIState
        from utilities.paths import get_app_physics_configs_dir

        # A hidden window, not a standalone context: Camera calls
        # glfw.get_framebuffer_size(self.window), and borrowing the real Camera
        # is what keeps the capture from drifting away from the app's.
        if not glfw.init():
            raise RuntimeError("glfw init failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(window_px, window_px, "bench", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create a hidden window")
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()

        self.ui_state = UIState()
        prefs = self.ui_state.preferences
        if world_size is not None:
            prefs.world_size = world_size
        if density is not None:
            prefs.particle_density = density
        self.prefs = prefs

        self.sim = Sim(self.ctx, world_size=prefs.world_size,
                       canvas_aspect_ratio="1:1",
                       particle_density=prefs.particle_density)
        self.camera = Camera(self.ctx, self.sim, self.window)
        self.camera.cam_brush_mode = True
        self.camera.BRIGHTNESS = prefs.brightness
        self.camera.trail_overlay_strength = prefs.trail_overlay_strength

        self.capture_view = CaptureView(self.ctx, self.sim, self.camera)
        self.blit = CaptureBlit(self.ctx)
        self.tile_capture = None
        self.scorer = VisionScorer()

        self._saver = ConfigSaver()
        self._presets_root = get_app_physics_configs_dir()
        self.state = SimState()
        self.load_preset(preset)

    def load_preset(self, name: str) -> None:
        """The physics substrate every run in a sweep shares.

        Swappable so one GL context and one CLIP session can scout several
        substrates; a sweep that compares modalities must still hold it fixed.
        """
        from state import SimState

        path = None
        for sub in ("Core", "Advanced"):
            p = self._presets_root / sub / f"{name}.json"
            if p.is_file():
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"no preset named {name!r}")
        cfg = self._saver.load_from_file(path)
        self.state = SimState()
        self.rule = self._saver.apply_config(cfg, self.state)
        self.preset = name
        # UIState.sim is what CaptureView reads for watercolor_mode.
        self.ui_state.sim = self.state
        self.camera.watercolor_mode = self.state.watercolor_mode
        self.camera.ink_weight = self.state.ink_weight

    # -- the app's own capture path, with nothing re-derived --------------

    def _assemble_kwargs(self) -> dict:
        """Only the entries _capture_kwargs does NOT neutralise.

        Everything camera-dependent is overwritten inside CaptureView, so
        supplying it here would be theatre. assemble_frame defaults the rest.
        """
        return dict(
            view_mode=VIEW_CAM_BRUSH,
            brightness=self.camera.BRIGHTNESS,
            exposure=self.prefs.exposure,
            ink_weight=self.state.ink_weight,
            watercolor_mode=self.state.watercolor_mode,
            canvas_resolution=self.sim.get_canvas_dimensions(),
            tonemap_softness=self.prefs.tonemap_softness,
            trail_overlay_strength=self.camera.trail_overlay_strength,
        )

    def capture(self, grid):
        from services.tile_capture import TILE_PX, TileCapture

        tex = self.capture_view.render(self.ui_state, self._assemble_kwargs(),
                                       grid * TILE_PX)
        if tex is None:
            return None
        if self.tile_capture is None:
            self.tile_capture = TileCapture(self.ctx, grid=grid)
        self.tile_capture.resize(grid)
        return self.tile_capture.capture(
            lambda fbo: self.capture_view.draw_grid(
                fbo, tex, grid, self.blit, self.ui_state, TILE_PX))

    # -- one run ----------------------------------------------------------

    def run(self, layout, prompt, algorithm, generations, grid, seed,
            steps_per_gen, snapshots, steps_per_frame=10, log=print):
        from services.auto_tournament_service import Action, AutoTournamentService
        from services.genome_spec import spec_for
        from services.tournament_service import TournamentService
        from ui.brain_window import saturation_fraction

        self.sim.realloc_brain_buffers(layout)
        self.sim.apply_rule(None)

        tour = TournamentService(grid=grid, rng=np.random.default_rng(seed),
                                 layout=layout)
        svc = AutoTournamentService(tour, self.scorer, spec=spec_for(layout),
                                    base_seed=seed)
        svc.configure(algorithm=algorithm, steps_per_gen=steps_per_gen,
                      snapshots_per_gen=snapshots,
                      sim_steps_per_frame=steps_per_frame)
        svc.set_layout(layout)
        svc.start(prompt)

        # apply_tournament only sets flags; apply_state is what pushes them to
        # the uniforms, so it has to run after, and every frame as in the app.
        self.sim.apply_tournament(True, grid=grid, mutation=0.0,
                                  plain_colour=True, physics=False)

        curve, t0 = [], time.time()
        guard = generations * (steps_per_gen // max(1, steps_per_frame) + 8) + 64
        while svc.generation < generations and guard > 0:
            guard -= 1
            self.sim.apply_state(self.state)
            action = svc.update()
            if action is Action.WRITE_RULES:
                self.sim.write_tournament_rules(tour.pack_rule_bytes())
                tour.clear_dirty()
                # One seed per generation, shared by every tile: the population
                # is compared on equal footing, and it moves between
                # generations so a genome cannot win by suiting one layout.
                self.sim.reset_seed = float(svc.gen_seed)
                self.sim.reset()
            elif action is Action.CAPTURE:
                crops = self.capture(grid)
                if crops is not None:
                    svc.submit_frames(crops)
            elif action is Action.SCORE:
                fit = svc.score_and_tell()
                # std, because a rank-based optimizer climbs the SPREAD within a
                # generation, not the absolute level. A substrate on which every
                # tile scores the same gives it nothing to select on however
                # high or low that score is.
                curve.append((float(fit.max()), float(fit.mean()),
                              float(fit.std())))
                if log:
                    log(f"      gen {svc.generation:3d}  best {fit.max():.4f}"
                        f"  mean {fit.mean():.4f}  std {fit.std():.4f}"
                        f"  sigma {svc.sigma:.3f}")
            elif action is Action.STEP:
                for _ in range(max(1, steps_per_frame)):
                    self.sim.update(self.ctx)

        best_z, best_f = (svc.optimizer.best() if svc.optimizer
                          else (None, float("nan")))
        return {
            "modality": layout.modality,
            "signature": layout.signature(),
            "dim": int(layout.length),
            "prompt": prompt,
            "algorithm": algorithm,
            "grid": grid,
            "popsize": grid * grid,
            "seed": seed,
            "generations": len(curve),
            "best": [c[0] for c in curve],
            "mean": [c[1] for c in curve],
            "std": [c[2] for c in curve],
            "best_fitness": float(best_f) if np.isfinite(best_f) else None,
            # A saturated coordinate is one the search can no longer move, and
            # the reported sigma does not reveal it - so this is the honest
            # "is the optimizer still able to steer" reading.
            "saturation": (saturation_fraction(best_z)
                           if best_z is not None and np.isfinite(best_f) else None),
            "seconds": time.time() - t0,
        }

    def close(self):
        import glfw

        glfw.terminate()


# ---- reporting -----------------------------------------------------------

def cell_key(row: dict) -> tuple:
    """What identifies one run within a sweep, for --resume.

    Every run is independent - a fresh optimizer at a fixed seed, and a
    generation seed derived from it - so nothing carries between cells and a
    resumed sweep is EXACT rather than approximate. The signature rather than
    the dimension, because Fourier at 19 centres and Gabor at 11 filters are
    both 152 floats.
    """
    return (row.get("preset", ""), row["signature"], row["algorithm"],
            row["prompt"], row["seed"])


def load_previous(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text()).get("runs", [])
    except (ValueError, OSError):
        return []


def summarise(rows: list[dict]) -> dict:
    """Per (preset, modality, dim, algorithm): final best and gain over gen 0.

    Gain matters independently of the absolute score because the priors differ:
    a modality can start high and learn nothing, or start low and climb.
    """
    out: dict = {}
    for r in rows:
        if not r["best"]:
            continue
        key = (r.get("preset", ""), r["modality"], r["dim"], r["algorithm"])
        acc = out.setdefault(key, {"final": [], "gain": [], "spread": [],
                                   "sat": [], "sec": []})
        # Best-so-far at the end, not the last generation's best: the search
        # keeps its elite, and one unlucky final population is not a regression.
        acc["final"].append(max(r["best"]))
        acc["gain"].append(max(r["best"]) - r["best"][0])
        acc["spread"].append(float(np.mean(r["std"])) if r.get("std") else 0.0)
        if r["saturation"] is not None:
            acc["sat"].append(r["saturation"])
        acc["sec"].append(r["seconds"])
    return out


def report(rows: list[dict]) -> None:
    s = summarise(rows)
    if not s:
        print("no completed runs")
        return
    algos = sorted({k[3] for k in s})
    print(f"\n{'preset':12s} {'modality':10s} {'dim':>5s} {'algorithm':14s} "
          f"{'final':>8s} {'gain':>8s} {'spread':>8s} {'sat':>6s} {'n':>3s} "
          f"{'sec':>7s}")
    for key in sorted(s):
        a = s[key]
        sat = f"{np.mean(a['sat']):6.1%}" if a["sat"] else "     -"
        print(f"{key[0]:12s} {key[1]:10s} {key[2]:5d} {key[3]:14s} "
              f"{np.mean(a['final']):8.4f} {np.mean(a['gain']):+8.4f} "
              f"{np.mean(a['spread']):8.4f} {sat} "
              f"{len(a['final']):3d} {np.mean(a['sec']):7.1f}")

    if "Random Search" in algos and len(algos) > 1:
        print("\n--- lift over Random Search (the load-bearing number) ---")
        print(f"{'preset':12s} {'modality':10s} {'dim':>5s} {'algorithm':14s} "
              f"{'lift':>9s}")
        for key in sorted(s):
            if key[3] == "Random Search":
                continue
            ctrl = s.get((key[0], key[1], key[2], "Random Search"))
            if ctrl is None:
                continue
            lift = np.mean(s[key]["final"]) - np.mean(ctrl["final"])
            print(f"{key[0]:12s} {key[1]:10s} {key[2]:5d} {key[3]:14s} "
                  f"{lift:+9.4f}")
        print("\nA lift at or below zero means the optimizer found nothing that")
        print("drawing the same number of random samples would not have found.")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="fourier:10,gabor:12,lenia:12,mlp:16",
                    help="comma-separated modality:count")
    ap.add_argument("--algorithms", default="CMA-ES,Random Search")
    ap.add_argument("--prompts", default="", help="default: three fixed prompts")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps-per-gen", type=int, default=300)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--preset", default="_Default",
                    help="comma-separated to scout substrates; a modality "
                         "comparison must use exactly one")
    ap.add_argument("--world-size", type=float, default=None)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--gabor-input-scale", type=float, default=None,
                    help="the preset's measured |input| p90; see the module docstring")
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true",
                    help="keep the runs already in --out and do only the rest")
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny run, to time a generation before committing")
    args = ap.parse_args(argv)

    if args.smoke:
        args.configs = args.configs.split(",")[0]
        args.algorithms = "CMA-ES"
        args.generations = 2

    extra: dict = {}
    if args.gabor_input_scale is not None:
        extra["gabor"] = {"input_scale": args.gabor_input_scale}

    layouts = parse_configs(args.configs, extra)
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    prompts = ([p.strip() for p in args.prompts.split(",") if p.strip()]
               if args.prompts else list(DEFAULT_PROMPTS))
    if args.smoke:
        prompts = prompts[:1]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    presets = [p.strip() for p in args.preset.split(",") if p.strip()]
    rows: list[dict] = load_previous(args.out) if (args.resume and args.out) else []
    done = {cell_key(r) for r in rows}

    # Every cell listed, so the plan is visible before an hour of GPU time is
    # spent on it - and so a resume says how much it is actually skipping.
    plan = [(p, lay, a, pr, s) for p in presets for lay in layouts
            for a in algorithms for pr in prompts for s in seeds]
    todo = [c for c in plan
            if (c[0], c[1].signature(), c[2], c[3], c[4]) not in done]

    bench = Bench(presets[0], args.world_size, args.density)
    print(f"preset {'/'.join(presets)}   {bench.sim.entity_count} particles   "
          f"canvas {bench.sim.can.size}   grid {args.grid} "
          f"({args.grid ** 2} tiles)")
    print(f"{len(todo)} runs to do of {len(plan)} "
          f"({len(rows)} already in {args.out or '-'}) x {args.generations} "
          f"generations x {args.steps_per_gen} steps, "
          f"{args.snapshots} snapshots\n")

    t0, current = time.time(), None
    try:
        for preset, lay, algo, prompt, seed in todo:
            if preset != current:
                bench.load_preset(preset)
                current = preset
            print(f"  {preset:12s} {lay.signature():18s} dim {lay.length:4d}  "
                  f"{algo:14s} seed {seed}  {prompt!r}")
            row = bench.run(lay, prompt, algo, args.generations, args.grid,
                            seed, args.steps_per_gen, args.snapshots)
            row["preset"] = preset
            rows.append(row)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps({"runs": rows}, indent=1))
    finally:
        bench.close()

    print(f"\n{len(rows)} runs in {time.time() - t0:.0f}s")
    report(rows)
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
