# Merging tournament-mode and brain-modalities

> **Status.** Stages 1–3 are done, in merge commit `c6e76ef` on `integration`,
> plus the post-merge fixes that followed it. The suite is green and
> `tools.shader_compile_check` reports COMPILE OK. `integration` is now the
> working branch, checked out in the repository root; the merge worktree is
> gone. Two things are outstanding: Stage 4 — making the tile's edge agree with
> the world's edge — has not been started, and Stage 3's by-hand checks from
> `docs/testing_checklist.md` have not been run.

Merge base `1cf194f` ("a tournament tile is a small world, not a walled box").
Since then: 29 commits on `tournament-mode` (tip `1f157ed`), 45 on
`worktree-brain-modalities` (tip `85c6719`). Both suites are green at their
tips.

The two branches barely overlap by accident and heavily by intent. 22 files
were touched by both; a `git merge-tree` dry run produces **6 textual
conflicts**, all of them small. The work is not in the conflicts. It is in the
files that merge *clean* and are wrong afterwards.

## Direction

Neither branch receives the merge. An `integration` branch is cut from the
merge base and both are merged into it, so `tournament-mode` stays runnable
throughout and the brain worktree stays isolated.

```bash
git worktree add .claude/worktrees/integration -b integration 1cf194f
```

Merge `tournament-mode` first, then `worktree-brain-modalities`. That order
puts the larger, newer branch second, where its conflict side is the one being
adapted — and every reconciliation below is written as "keep tournament's
structure, thread brain's layout through it", which reads naturally in that
direction.

## Stage 1 — the six textual conflicts

Each is a keep-both or a mechanical combination. None needs a design decision.

### `ui/menu_bar.py`
Two Extras checkboxes inserted at the same line. Keep both, tournament's
"Archive Browser" first.

### `main.py` line ~142
`self.command_handler.release_archive = self._release_archive` and
`self.command_handler.apply_brain_layout = self._apply_brain_layout`. Keep both.

### `command_handler.py` line ~39
`self.release_archive = None` and `self.apply_brain_layout = None`. Keep both —
along with tournament's `archive_store`, `_run_physics_written` and
`_run_config_cache`, added by `93d9524` in the same block.

### `services/archive.py` `Archive.__init__`
Both sides widened the signature. Combined:

```python
def __init__(self, store=None, capacity: int = 20000, k: int = 10,
             liveness_min: float = 0.002, dim: int = 512,
             min_separation: float = DEFAULT_MIN_SEPARATION, layout=None):
```

Keep brain's `default_layout` import and the `self.layout` fallback chain, and
tournament's `self.min_separation`. The two comment blocks about `seed_n` say
the same thing twice — keep one.

### `services/imgep_driver.py` imports
Union of both:

```python
from services.genome_spec import encode, layout_of, spec_for
from services.goal_source import LATENT_DIMS, Goal, latent_goal, novelty_goal
```

`BRAIN_SPEC` goes. Everything else in this file auto-merges: tournament
rewrote it wholesale (487 lines) and brain touched 18, all of which land.

### `command_handler.py` — the save paths (3 hunks)
Tournament routed every save through the one save dialog; brain added
`layout=`. Both, in each case:

```python
config = self.config_saver.create_config(
    ui_state.sim, svc.population[tile], layout=svc.layout)
filepath = self.user_configs_dir / f"{stem}.json"      # from save_targets
```

```python
path = self.user_configs_dir / f"{filename}.json"
export_genome(path, z, sim_state, meta, layout=layout)
```

```python
path = self.user_configs_dir / f"{filename}.json"
export_genome(path, brain_z, sim_state, meta, layout=svc.spec.layout)
```

Tournament's `notice` assignments must survive all three — a save that only
prints to the console reads as a no-op.

Brain's three new methods (`_rule_fits`, `_brain_layout`,
`_restore_brain_settings`) land unchanged above `_save_tournament_selection`.

### `main.py` line ~382 — the large insert-vs-insert
Tournament added `_open_archive`, `_open_archive_browser`, `_release_archive`,
`_save_archive_settings`, `_load_archive_settings`, and the crash-safe
`run()`/`_write_crash_log()`/`_cleanup_safely()`. Brain added
`_render_brain_preview`, `_refresh_driver_specs`, `_apply_brain_layout`.

Keep all of them. Two joins matter:

- Tournament's `_ensure_archive_service` (which delegates to `_open_archive`)
  is the one that survives, not brain's.
- Tournament's `run()` is the one that survives — brain's is the pre-crash-log
  version.

### `CLAUDE.md`
One insert-vs-insert at the tail. Keep both bodies; brain's caveats want their
own section beside the GPU/tiling one. Reconcile the headings so there is one
list, not two.

## Stage 2 — the clean merges that are wrong

These files auto-merge and compile. They are the actual work.

### 2a. `encode()` with no layout silently means Fourier

`genome_spec.encode(genome, layout=None)` falls back to `default_layout()`.
Three tournament-mode call sites pass nothing:

- `services/imgep_driver.py` `_parent_z` — `encode(self.archive.brains[i])`
- `command_handler.py` `_save_archive_entry` — brain already passes `layout`
  here; confirm it survives the conflict resolution
- `command_handler.py` `_seed_from_archive` — `encode(self.archive.brains[i])`

Under a Gabor, Lenia or MLP archive these re-encode through Fourier's squash
and hand the optimizer a genome nobody chose. Pass `self.archive.layout` (the
archive is the object that knows what its stored floats mean) at each.

Two of the three are latent on `worktree-brain-modalities` today, so this is a
fix-forward rather than a merge artifact — but the merge is where it becomes
reachable, because tournament-mode is what made the archive browser and the
seed button easy to hit.

### 2b. Two store members are orphaned by the signature directory

Brain moved the store to `<archive>/<layout-signature>/` and wrote
`migrate_to_signature_dir()` against:

```python
_LEGACY_MEMBERS = ("index.jsonl", "vectors.npz", "goals.json", "thumbs")
```

Tournament has since added **two** more members, both after that list was
written, and both resolve inside the signature directory once brain's change
lands:

- `settings.json` — `ArchiveStore.settings_path`, the per-archive Explore
  settings
- `runs/` — `ArchiveStore.run_config_path`, added by `93d9524`, holding the
  whole `PhysicsConfig` each run was carried out under

Neither is moved by the migration, so on an existing archive the Explore
settings silently reset to defaults and every entry loses the physics it was
made under — which is precisely the defect `93d9524` exists to fix, reappearing
by a different route.

Add both to `_LEGACY_MEMBERS` and extend `tests/test_archive_signature.py` to
assert they move with the rest. `runs` is a directory, like `thumbs`, so the
existing `os.replace` loop handles it unchanged.

### 2b-bis. The run config does not record its brain

`_record_run_physics` calls `ConfigSaver().create_config(ui_state.sim, None)`,
which under brain's signature writes `brain_layout=""` and no
`brain_settings`. Pass `layout=self._brain_layout()`.

This is honesty rather than correctness: the store is per layout, so
`<archive>/<signature>/runs/<id>.json` can only ever have come from the
signature in its own path. But the file is read back through
`PhysicsConfig.from_json`, and a config claiming no brain is one that
`_restore_brain_settings` will decline to act on — so the run's decode scales
would be lost even though nothing about the layout is ambiguous.

### 2c. A layout switch rebuilds the archive without releasing it

`_apply_brain_layout` ends with a bare `self._build_archive_set(path)`.
Tournament introduced `_release_archive` precisely because `ArchiveStore` holds
`index.jsonl` open for append and Windows will not let go of the directory —
and because the thumbnail cache is keyed by a filename derived from the entry
id, which restarts at 0 in every archive.

Route the layout switch through the same sequence a name switch uses:
`_save_archive_settings` → `_release_archive` → `_build_archive_set` →
`_load_archive_settings`.

Consequence worth stating out loud in `CLAUDE.md`: Explore settings are now
per **layout**, not per named archive, because they live beside `goals.json`
inside the signature directory. That is defensible — the settings that suit a
Fourier archive need not suit a Gabor one — but it is a behaviour change and
should be written down rather than discovered.

### 2d. `set_layout` does not cancel an in-flight scoring pass

Tournament moved CLIP off the frame loop: `AutoTournamentService._precomputed`
submits `driver.precompute(list(self._buffer))` to a one-thread pool.
`set_layout` calls `_resolve_spec()` and `_sync_driver()` but never touches
`_pre_future`, so a scoring pass launched under the old brain can land after
the switch and be told about the new one.

Make `set_layout` do what `abort_generation` does to the pending future.

### 2e. The duplicate `#define SQRT_WORLD_SIZE` — already fine

Both branches delete the same line (`1f157ed` on tournament, incidentally on
brain). Identical deletions merge without complaint. Noted only so nobody
re-adds it while resolving the shader by hand.

### 2f. `.gitignore`

The uncommitted working-tree line `.claude/settings.local.json` duplicates one
brain already committed in `98e7c88` (which also carries `.claude/worktrees/`).
Drop the local edit before cutting the integration branch.

## Stage 3 — verification

In order, cheapest first. Nothing below is optional; the shader in particular
cannot be verified by the suite.

1. **Import and assemble.** `python -m tools.shader_compile_check` — brain's
   tool, which compiles the assembled compute shader with all four brain
   prepends. This is the only automated check that would catch the merged
   `entity_update.glsl` failing to compile, since the test suite has no GPU and
   can only assert on source text.

2. **Full suite.** `.venv/Scripts/python.exe -m pytest -q`. Baseline is exit 0
   on both branches, so anything red is the merge. Expect the pressure at:
   - `tests/test_shader_source.py` and `tests/test_brain_shader_source.py` —
     both parse the shader, and `V_MAX` joined `MultiLoadConfig`
   - `tests/test_tile_isolation_gl.py` — runs real GL at 647/8, 647/3 and 641/7
     and now has to survive the brain prepends
   - `tests/test_archive*.py` — `_brain` changed shape from `(N, 10, 8)` to
     `(N, layout.length)` under tournament's separation and novelty-column work
   - `tests/test_archive_io.py` and `tests/test_archive_preview.py` — both
     sides touched the store, and the preview now applies a whole
     `PhysicsConfig` rather than a dict of physics fields
   - `tests/test_v_max.py` — runs the sim, and the step-cap code now sits
     downstream of `eval_brain`

3. **The struct offset check.** `MULTI_LOAD_CONFIG_SIZE` is `11*7*4 + 6*4 +
   3*4`, and both raw std430 writers pack by offset. Confirm
   `tests/test_v_max.py` still guards the `"V Max"` label lookup — a physics
   parameter whose label is not `title()` of its field name silently gets the
   default slider range.

4. **By hand**, from `docs/testing_checklist.md`, plus specifically:
   - switch brain modality with the archive browser open, and confirm the
     archive follows to the sibling signature directory and back
   - save a tournament tile under a non-Fourier brain, load it from
     File > Load > Custom, confirm it plays back
   - run Explore for a few generations under Gabor and confirm the entries
     admitted decode to what the tiles showed
   - `V Max` at its minimum under a strafing preset, in and out of tournament
     mode

## Stage 4 — make the tile's edge agree with the world's edge

Run this only once Stage 3 is green. Doing it during the merge makes a red test
unattributable: it could be the reconciliation or it could be the physics.

Both branches fixed the same bug — a probe leaving its world fell through to a
`repeat` sampler and came back with the opposite edge of the canvas. `1cf194f`
fixed it for tournament tiles, `20f0d58` for the single world. They then chose
different replacements for the non-wrap case, and the tile got the two that
`20f0d58` measured and rejected:

| | single sim, bounce/reset | tournament tile, bounce/reset |
|---|---|---|
| sensor past the edge | reads zero (**void**) | reads the nearest texel (**clamp**) |
| trail past the edge | leaves (**absorbing**) | comes back (**zero-flux mirror**) |

The measurements, from `20f0d58`, as border/interior luminance over
`physics_configs/Core`:

| treatment | LavaLamp | Streamers | % at wall |
|---|---|---|---|
| wrap (the original leak) | 10.6x | 2.9x | 4.70% |
| clamp | 16.1x | — | — |
| mirror | 11.2x | 8.1x | 10.50% |
| void | 9.5x | 3.4x | 1.89% |

and on the trail side, a zero-flux mirror took LavaLamp from 10.6x to 14.2x,
where absorbing took it to 7.5x and Streamers from 2.9x to 0.4x. The leak that
both fixes replaced was acting as a *sink* draining the bright edge; sealing
the boundary keeps the pile-up and removes the drain.

**Under wrap nothing below changes.** A torus has no outside, both paths
already agree, and the fold stays exactly as it is.

### 4a. The sensor — `confine_sample` in `entity_update.glsl`

It currently clamps unconditionally. It needs to keep folding under wrap and
report off-world under bounce and reset, so the caller can substitute zero the
way `sense_off_world` already does for the canvas:

```glsl
// -> the sample point, and whether it is off this tile's world.
//
// Under wrap the tile is a torus and nothing is ever off it. Under bounce or
// reset the tile edge IS the world's edge, and beyond it there is nothing to
// smell - the same answer sense_off_world() gives for the canvas.
vec2 confine_sample(vec2 p, vec2 lo, vec2 hi, int boundary_mode, out bool off){
    off = false;
    if(boundary_mode == 2){
        p = lo + mod(p - lo, hi - lo);
    } else if(any(lessThan(p, lo)) || any(greaterThan(p, hi))){
        off = true;
        return p;
    }
    float ca = canvas_resolution.x / canvas_resolution.y;
    vec2 half_texel = vec2(sqrt(ca), 1.0 / sqrt(ca)) / canvas_resolution;
    return clamp(p, lo + half_texel, hi - half_texel);
}
```

The half-texel inset stays, and stays load-bearing: it is what stops a sample
that *is* inside the tile from bilinearly blending the neighbour's texel.

At the call site, the tap becomes conditional:

```glsl
bool loff = false, roff = false;
if(TOURNAMENT_MODE == 1){
    vec2 tlo, thi; particle_world_box(index, tlo, thi);
    int smode = get_particle_boundary_conditions();
    lsample = confine_sample(lsample, tlo, thi, smode, loff);
    rsample = confine_sample(rsample, tlo, thi, smode, roff);
}
vec4 ltap = loff ? vec4(0) : get_can(lsample);
vec4 rtap = roff ? vec4(0) : get_can(rsample);
```

### 4b. The diffusion — `tile_tap` in `canvas.frag`

Its non-wrap tail is `return centre;` — zero net flux across the seam. It
becomes `return vec4(0.0);`, matching what `getCan` now does at the canvas
edge. The `centre` parameter falls out of use and should go, along with its
argument at the four call sites in `getBlur`.

### 4c. What this does NOT change

Tile isolation is untouched: nothing leaks between tiles, in either direction,
under any mode. The texel-exact seam arithmetic (`tile_lo_texel`, integer, the
banned `%`) is untouched. Wrap is untouched.

### 4d. Tests that assert the old behaviour and must move with it

Three in `tests/test_tile_isolation_gl.py`, and none of them is wrong today —
they guard the treatment being replaced:

- `test_a_tile_conserves_its_own_trail[BOUNCE]` — conservation is exactly what
  absorbing gives up. It should split: still exact under wrap, and under bounce
  it becomes the tile's version of brain's
  `test_a_wall_drains_the_trail_rather_than_sealing_it`.
- `test_a_sensor_never_leaves_its_tile[BOUNCE|RESET]` — the postcondition
  becomes "inside the tile by half a texel, *or* flagged off-world".
- the bounce case asserting two sensors past a wall collapse to one point —
  under void they still agree, both reading zero, so the assertion holds but
  its reasoning changes. It is worth rewriting to say what it now means: there
  is no steering differential to be had past a wall in either treatment, and
  the question is only whether the pair reads the wall's own bright trail or
  reads nothing.

`test_bounce_holds_the_trail_off_the_seam` gets strictly stronger and needs no
change.

Brain's `tests/test_canvas_boundary_gl.py` is the model for all of this — it
already covers the single world's version of every one of these cases.

### 4e. Confirm it on the presets, not just in the unit tests

`tools/edge3.py` measures border/interior and %-at-wall over eight Core presets
but only for the whole canvas. Add a tournament-mode variant that measures
per-tile, and check the tile numbers land where the canvas numbers did. If they
do not, that is a real finding and this stage should stop rather than ship on
the assumption that the single-world measurement transfers.

### 4f. Consequence to write down

This changes what a candidate was scored under, for non-wrap presets in
tournament mode. Existing archive entries were admitted under clamp-and-mirror
and will not reproduce identically. Note it in `CLAUDE.md` beside the tile
caveat; it is not a reason to avoid the change, but it is a reason not to
compare novelty across the boundary.
