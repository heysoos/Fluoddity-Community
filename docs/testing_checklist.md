# Fluoddity Manual Testing Checklist

Use this after large refactors or significant new features. Items roughly ordered by breakage risk.

## 1. Config Save/Load System
- [ ] **a.** File → Save: enter name, verify JSON appears in Custom folder
- [ ] **b.** File → Save existing name: overwrite confirmation dialog works
- [ ] **c.** File → Load: click config, verify rule + sliders + appearance applied
- [ ] **d.** Hover preview: hover over config names, verify live preview (particles change)
- [ ] **e.** Preview restore: move mouse away from menu, verify original state restored
- [ ] **f.** Watercolor lock: right-click in Load submenu toggles watercolor for all previews
- [ ] **g.** Category headers: Core/Custom/Advanced collapse/expand, state persists across opens
- [ ] **h.** N button: shows notes tooltip (blue when notes exist)
- [ ] **i.** X button: opens delete confirmation, file removed on confirm
- [ ] **j.** Clipboard: Ctrl+C copies config, Ctrl+V pastes and applies
- [ ] **k.** Saves preserve: jitter values, custom slider ranges, parameter sweep assignments, notes

## 2. Particle Selection & Rule History
- [ ] **a.** Left click selects particle, applies its mutated rule (1-frame deferred readback)
- [ ] **b.** Right click undoes last selection (pops rule stack)
- [ ] **c.** History window shows colored jersey numbers, newest first
- [ ] **d.** Hover history entry: live preview of that rule
- [ ] **e.** Click history entry: moves rule to top of stack
- [ ] **f.** X button in history: deletes that rule entry
- [ ] **g.** Z key: full reset (zero rule + new seed + push to history)
- [ ] **h.** M key: randomize mutations (same rule, new seed, push to history)
- [ ] **i.** R key: simple reset (particles only, rule unchanged)

## 3. Physics Sliders
- [ ] **a.** All 12 sliders respond and affect simulation in real-time
- [ ] **b.** Right-click context menu opens on each slider
- [ ] **c.** Jitter slider works (orange tint, range shown in label)
- [ ] **d.** Jitter hidden for Hazard Rate and Mutation Scale
- [ ] **e.** Min/Max fields adjust range; "Reset Range" restores defaults
- [ ] **f.** "Reset Value" button works (shows loaded config name if applicable)
- [ ] **g.** Ctrl+click on slider allows direct number entry
- [ ] **h.** Hazard Rate uses power scaling (fine control at low values)
- [ ] **i.** Hard limits enforced: Drag/Sensor Angle (±1.0), Trail Persistence/Diffusion (0-1.0)
- [ ] **j.** Physics tooltips: enable in preferences, hover slider shows animated diagram

## 4. Parameter Sweeps
- [ ] **a.** Enable checkbox toggles X/Y/C buttons on sliders
- [ ] **b.** X button: left-click = normal (bright red), right-click = inverse (dark red)
- [ ] **c.** Y button: same pattern, green
- [ ] **d.** C button (cohort): same pattern, yellow
- [ ] **e.** Left click on canvas: updates slider values from position (XY) or particle cohort (C)
- [ ] **f.** Right click on canvas: enters preview mode (sweeps disabled, window tints blue)
- [ ] **g.** Any click while preview pending: re-enables sweeps
- [ ] **h.** Sweep reticle visible on canvas, hidden during recording/screenshot
- [ ] **i.** Range adjust buttons (^ v): widen/narrow range around current value

## 5. Video Recording
- [ ] **a.** Record key toggles recording on/off
- [ ] **b.** While recording: speedmult/motion blur locked to recording settings
- [ ] **c.** Stop recording: user settings restored
- [ ] **d.** Delayed start: set Video End Frame > 0, recording starts at calculated frame
- [ ] **e.** Pending state: shows countdown, can cancel with record key
- [ ] **f.** Video saved to Documents/Fluoddity/ with timestamp
- [ ] **g.** Recording window shows status (RECORDING / WAITING / idle)

## 6. Screenshots
- [ ] **a.** Shift+P takes screenshot
- [ ] **b.** Settings temporarily overridden (max quality motion blur)
- [ ] **c.** Settings restored after save (including pause state)
- [ ] **d.** File saved to Documents/Fluoddity/screenshots/ with timestamp
- [ ] **e.** Supersample factor applied

## 7. Multi-Load Mode
- [ ] **a.** Extras → Multi Load Mode enables
- [ ] **b.** File → Load adds configs (max 64), menu stays open
- [ ] **c.** Physics window switches to multi-load layout
- [ ] **d.** Mouse mode forced to Draw Trail
- [ ] **e.** Parameter sweeps force-disabled
- [ ] **f.** Simultaneous configs / Progression Pace / Current Progress sliders work
- [ ] **g.** Remove buttons remove individual configs
- [ ] **h.** Per-config toggles (Initial Conditions, Cohorts, Hazard Rate) grey out respective controls

## 8. Appearance & View
- [ ] **a.** Color by Cohort toggle (hides Hue Sensitivity when on)
- [ ] **b.** Watercolor Mode toggle (V key), shows Ink Weight when on
- [ ] **c.** Emboss Mode combo (Off/Canvas/Brush), shows Intensity + Smoothness when on
- [ ] **d.** Brightness slider affects output
- [ ] **e.** Exposure slider works
- [ ] **f.** View option dropdown cycles views
- [ ] **g.** Tiling mode (view option 3): camera wraps, exiting wraps position back to center

## 9. Preferences
- [ ] **a.** World size change triggers full rebuild (expensive, console output)
- [ ] **b.** Physics frequency slider (locked label during recording)
- [ ] **c.** Motion blur toggle + blur quality slider
- [ ] **d.** Mouse mode dropdown (locked text in multi-load)
- [ ] **e.** Draw size / Draw power visible only in Draw Trail mode
- [ ] **f.** Debug arrows toggle + sensitivity slider
- [ ] **g.** Preferences saved on exit, restored on next launch

## 10. Menu Auto-Close
- [ ] **a.** Main menu bar: menus close when mouse moves far away
- [ ] **b.** Physics settings menu bar: same behavior
- [ ] **c.** Slider context menus: same behavior
- [ ] **d.** Save dialog open prevents auto-close

## 11. Camera & Input
- [ ] **a.** WASD movement
- [ ] **b.** QE zoom in/out
- [ ] **c.** Scroll wheel zoom (centered on mouse pointer)
- [ ] **d.** V key: reload shaders (hot reload)
- [ ] **e.** Keybindings from keyboard_controls.json respected

## 12. Help Windows
- [ ] **a.** Help → Controls: lists all shortcuts
- [ ] **b.** Help → Tutorial: all collapsible sections open/close
- [ ] **c.** Help → Parameter Sweeps: info window opens
- [ ] **d.** Help → Performance: opens
- [ ] **e.** Help → Video Recording: shows recording status + all controls

## 13. Explore (IMGEP) mode
- [ ] Enabling Explore forces a 1:1 canvas; disabling restores the previous ratio
- [ ] Start with an empty archive: regime reads `bootstrap`, archive grows by up to N^2 per generation
- [ ] At `seed_n` the regime becomes `expansion`; admission drops as separation starts rejecting
- [ ] `Documents/Fluoddity/archives/<name>/index.jsonl` gains one line per admission; `thumbs/` fills
- [ ] With `Expansion Between` = 3 and a goal in the list, an expedition fires and names the goal
- [ ] With the goal list empty, every expedition reads `(latent)` or `(novelty)`
- [ ] `Expansion Between` = 0 never expeditions
- [ ] Right-click a tile: an expedition starts immediately with goal `(chase)`
- [ ] Left-click a tile: it enters the archive as pinned, bypassing the gates
- [ ] Reset Search keeps the archive; the size readout does not drop
- [ ] Quit and relaunch: archive size and goal list are preserved
- [ ] Kill the app mid-run (no clean quit): the archive reloads, reporting the dropped trailing entries
- [ ] Archive browser: sorting, pinned-only filter, hover stats, "Save as config..." writes a loadable config
- [ ] Archive browser opens large enough to show the thumbnails AND the Save/Seed/Delete row
- [ ] Map: the cloud grows outward; the goal marker sits outside it during a latent expedition
- [ ] "Refit projection" does not mirror the layout
- [ ] Switching Explore -> Auto -> Manual and back leaves each mode working
- [ ] The Auto tab still has Grid, Steps per Gen, Sim Steps per Frame, Snapshots per Gen, Start/Reset and Initial Sigma
- [ ] Toggling physics search mid-run ends any expedition without crashing
- [ ] Changing the grid mid-expedition does not crash

## 14. Archive management

- [ ] First launch after updating: the old `Documents/Fluoddity/archive/`
      folder is gone and `Documents/Fluoddity/archives/default/` holds its
      contents. Explore mode's archive count matches what it was before.
- [ ] Explore tab: the Archive row names the active archive and its entry
      count matches the Archive Browser's. (While a search is running the
      browser can lead by up to 200 - the row counts what is on disk, and
      `vectors.npz` is only rewritten every 200 admissions.)
- [ ] New -> type `a/b` -> the preview says it will be saved as `ab`; Create
      makes `archives/ab/` and the row switches to it.
- [ ] New -> type the name of an existing archive -> Create is disabled and the
      reason is shown.
- [ ] Switch archives while a search is running: it stops, the entry count
      changes, the gallery shows the new archive's thumbnails (not the old
      one's), and pressing Start explores into the new archive.
- [ ] Switch back: the first archive's entries and goal list are exactly as
      they were.
- [ ] Empty -> confirm: the archive shows 0 entries and a
      `<name>.cleared-<time>` folder appears beside it with the old contents.
- [ ] Delete: the button is disabled when only one archive exists; otherwise it
      requires typing the name, and afterwards the row switches to another
      archive.
- [ ] Delete an archive folder in Explorer while the app is running, then pick
      it in the dropdown: a warning appears and nothing else changes.
- [ ] Quit while a non-default archive is active; relaunch: the same archive is
      loaded.

## 15. Expeditions

The offline evidence for this change is all proxies on stored descriptors: it
shows the landscape has a gradient built from achievable creatures, not that
CMA-ES walks it in 50 generations at dim 80. These checks are the part that
only the running app can answer.

- [ ] Run Explore until the archive passes `seed_n` and an expedition starts.
      The Explore status shows `Regime: expedition` and a goal label.
- [ ] **Fitness moves off its starting value rather than decaying.** This is the
      whole fix: previously the expedition seeded on the optimum of its own
      objective, so the score could only fall. A flat or falling curve across a
      whole expedition means it is not working.
- [ ] A latent expedition (`Latent Goal Share` = 1.0) produces tiles that differ
      from its seed rather than converging back onto it.
- [ ] A text expedition (`Latent Goal Share` = 0.0, with goals in the list)
      moves toward the phrase rather than toward noise.
- [ ] **Cancel** appears next to the goal only during an expedition, ends it,
      and the regime returns to `expansion`.
- [ ] After cancelling, the next expedition does not start immediately - it is a
      full `Expansion Between` interval away.
- [ ] Chase a tile mid-expedition: the goal switches to that tile rather than
      the expedition simply ending.
- [ ] Switch to the Auto (CLIP) tab and confirm its prompt still works. Explore
      must never call `scorer.set_prompt()`, which owns that tab's cache.

## 16. Saving from tournament modes
- [ ] Manual: select tiles, "Save Selected..." opens a dialog prefilled `tile<N>` (one tile) or `tournament` (several)
- [ ] Manual: with nothing selected the button is greyed and says "(select tiles first)"
- [ ] Manual: saving two tiles writes `<name>_tile<a>.json` and `<name>_tile<b>.json`, both listed in the confirmation
- [ ] Auto: "Save best genome..." opens a dialog prefilled `best_gen<NNNN>`
- [ ] Auto: right-click a tile opens a dialog prefilled `tile<N>_gen<NNNN>`
- [ ] Explore: "Save as config..." on a selected entry, and on the map, both open the dialog
- [ ] Saving an existing name asks to overwrite; Cancel leaves the file untouched
- [ ] Every save reports the filename on screen, and the file appears under File > Load > Custom
- [ ] A name containing `/` or `\` still saves, into the configs folder, under a sanitised name
- [ ] "Save checkpoint" says it is not a config and names the run folder

## 17. Per-archive Explore settings
- [ ] Change alpha / min separation / grid, switch archives, switch back: the values return
- [ ] Two archives hold different settings independently
- [ ] Quit and relaunch: the active archive reopens with its own settings, not the defaults
- [ ] A brand new archive inherits the settings currently on screen rather than snapping to defaults
- [ ] Restoring an archive saved at a different grid rebuilds the tournament grid to match
- [ ] Deleting `settings.json` by hand loads defaults without crashing

## 18. Archive map view modes
- [ ] Defaults are unchanged: Colour=source, Show=All entries, Draw=points
- [ ] Colour > novelty / liveness: points take a ramp, and a legend with a low->high scale appears
- [ ] Show > Recent generations: a Generations slider appears; the remaining points expand to fill the canvas
- [ ] Show > Most novel: a Top % slider appears
- [ ] Show > One goal / One regime: the combo lists only values the archive actually holds
- [ ] Show > Kept for matching on an archive with no summits/records/pins: reads "No entries match this filter", not a broken map
- [ ] Draw > density: a log-scaled heatmap; zooming in raises the cell resolution
- [ ] Draw > density with Colour=novelty/liveness: cells take the RAMP, not a count colour, and the legend still shows the ramp
- [ ] Draw > density with Colour=source: cells take the regime colour of whatever is in them
- [ ] In every density mode, a crowded cell is more opaque than a lone one, and a lone cell is still visible
- [ ] Draw > points+density: dots drawn over the heatmap
- [ ] "showing N of M" is correct under every filter
- [ ] With a filter on, hovering a dot shows THAT entry, and clicking selects it (not the entry at the same row index)
- [ ] Switching archives restores that archive's map view settings

## 19. Panel width and the map wheel
- [ ] Drag the Tournament window as narrow as it will go: no slider label is cut off, and every button is reachable
- [ ] Same for the Archive browser, on both the Gallery and Map tabs
- [ ] The gallery wraps to fewer thumbnails per row as the window narrows, instead of clipping the right-hand column
- [ ] Scrolling with the pointer over the map zooms the map and does NOT scroll the panel behind it
- [ ] Scrolling with the pointer anywhere else in the tab scrolls the panel and does NOT zoom the map
- [ ] Zoom still centres on the point under the cursor, and drag-to-pan still works

## 20. Live preview from the archive browser
- [ ] Extras > Archive Browser opens the gallery straight from a fresh launch, with no Tournament window and no visible model-loading pause
- [ ] Unticking it closes the window; reticking reopens it without reloading the archive
- [ ] Opening it, then opening Explore, does not reload the archive a second time (watch the console for a second '[archive] loaded' line)
- [ ] An archive last closed with the browser shut still opens when picked from the menu
- [ ] Open the Explore tab, open the Archive Browser, then CLOSE the Tournament window: the canvas goes back to one simulation, not a grid
- [ ] "Live preview" is disabled while the Tournament window is open, and says why
- [ ] With it on, hovering a gallery thumbnail runs that creature in the live sim
- [ ] Moving the pointer off the gallery puts your own creature back, unchanged
- [ ] Hovering across many entries in a row and then leaving still restores the original (nothing accumulates)
- [ ] Clicking an entry keeps it after the pointer leaves, and says which one it loaded
- [ ] The same works by hovering and clicking dots on the Map tab
- [ ] An entry saved from a physics-search run brings its physics sliders, and leaving puts your sliders back
- [ ] Toggling Live preview off while hovering restores immediately
- [ ] Deleting the entry you are hovering does not crash
- [ ] The toggle's state is remembered per archive across a relaunch
