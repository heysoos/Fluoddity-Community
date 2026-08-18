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
- [ ] **h.** In a camera view (Camera / Tiled / Particles+Trails), zoom out so
      black bars show around the canvas, then record: the file has no bars.
- [ ] **i.** Zoom IN past the canvas edges and record: the framing is exactly
      what is on screen, unchanged.
- [ ] **j.** In the raw Canvas view the recording is unchanged from before.
- [ ] **k.** Resize the window mid-take in a camera view: it stays ONE file.
- [ ] **l.** Pan while zoomed out during a take: the world stays framed rather
      than sliding, and no bar appears at the edge it was panned away from.
- [ ] **m.** The framerate barely moves when recording starts, and there are NO
      periodic freezes. A lurch of a second or more means something is writing
      to ffmpeg on the frame loop again.
- [ ] **n.** The video plays and is the right way up. A vertical flip means the
      two capture paths stopped agreeing about which way is up.
- [ ] **o.** Set Video End Frame and let a take stop on its own: the file has
      exactly the frames asked for, not one more. The readback runs a frame
      behind, so the last one is collected at the end.
- [ ] **p.** Record at a window size with an odd pixel dimension: the video is
      one pixel smaller on that axis and encodes fine, rather than failing.
- [ ] **q.** Record for several minutes and watch memory: it settles rather
      than climbing. The frame queue is capped, so a slow encoder must slow the
      sim instead of growing.
- [ ] **r.** Shift+P during a recording still writes an upright screenshot at
      the full window size, not the even-rounded video size.

### Video Recording with audio
Sync is the thing only a person can confirm, so use material with a hard beat.
- [ ] **a.** With audio capture running, tick Record Audio and record: the mp4
      plays with sound, and a visible hit lands on the beat you can hear.
- [ ] **b.** The same at the END of a two-minute take - drift accumulates, so a
      short clip proves nothing.
- [ ] **c.** With Record Audio ticked, the Preferences physics Rate slider is
      NOT locked during recording; with it unticked, it is.
- [ ] **d.** Record with audio capture stopped: the take completes silently and
      is not an error.
- [ ] **e.** Stop the capture device mid-take (unplug, or stop it in the Audio
      tab): the recording still saves, and the window shows a drift warning.
- [ ] **f.** No `.video.mp4` or `.pcm` file is left behind after a take.
- [ ] **g.** Audio Delay at 0 records exactly as before; the slider only
      appears with Record Audio ticked, and right-click resets it.
- [ ] **h.** Record the same passage at 0 and at 0.15: the second one has the
      hit landing visibly later against the same picture. If both look
      identical the delay is being accepted and ignored — the failure mode is
      silent, so compare two takes rather than judging one.

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
- [ ] **c.** Brightness slider affects output
- [ ] **d.** Exposure slider works
- [ ] **e.** View option dropdown cycles views
- [ ] **f.** Tiling mode (view option 3): camera wraps, exiting wraps position back to center

## 9. Preferences
- [ ] **a.** World size change triggers full rebuild (expensive, console output)
- [ ] **b.** Physics frequency slider (locked label during recording)
- [ ] **c.** Motion blur toggle + blur quality slider
- [ ] **d.** Mouse mode dropdown (locked text in multi-load)
- [ ] **e.** Draw size / Draw power visible only in Draw Trail mode
- [ ] **f.** Debug arrows toggle + sensitivity slider
- [ ] **g.** Preferences saved on exit, restored on next launch
- [ ] **h.** Window layout: open Physics, Screen Recording, Config Clipboard
      and Audio Reactive, move and resize them, quit and relaunch — each is
      back where it was. The windows left closed stay closed.
- [ ] **i.** `Documents/Fluoddity/imgui.ini` exists and is the file that
      changes; running from a different folder gets the same layout.
- [ ] **j.** With the Archive Browser open at quit, the next launch reopens it
      on the same archive (it costs a moment while the archive loads).
- [ ] **k.** Tournament Mode is OFF at launch however it was left.

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
- [ ] Switch to the Auto (Prompt) tab and confirm its prompt still works. Explore
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
- [ ] Run Explore briefly, then hover the entries it made: they recreate their thumbnails, whatever your sliders were set to beforehand
- [ ] Move every physics slider somewhere else and hover them again: still recreated, and leaving puts your own sliders back
- [ ] `<archive>/runs/<run_id>.json` appears when a run starts, and a pause-and-resume does not add a second file or change the first
- [ ] Entries from an archive made before this (no `runs/` folder) still preview, and leave the sliders alone
- [ ] Toggling Live preview off while hovering restores immediately
- [ ] Deleting the entry you are hovering does not crash
- [ ] The toggle's state is remembered per archive across a relaunch

## 21. The picker and the map's Home icon
- [ ] The Archive Browser has its own archive combo; switching from it changes what the gallery and map show
- [ ] Switching from the browser does NOT close the browser, even for an archive last closed with it shut
- [ ] New / Empty / Delete / Refresh all work from the browser, and their dialogs appear
- [ ] With Live preview on, switching archives mid-hover puts your own creature back rather than leaving the old one running
- [ ] Both the Explore tab and the browser can be open at once with no ImGui "conflicting ID" dialog
- [ ] The map's recentre icon sits inside the canvas, top-right, and lights up on hover
- [ ] Clicking it resets zoom and centre; the hover card for a dot underneath does not fight it
- [ ] Scrolling over the icon still zooms the map

## 22. V Max (the per-step speed limit)
- [ ] Physics > Forces shows a "V Max" slider sitting at the far right of its track on a fresh launch, reading "Off"
- [ ] Nudging it one notch down from the top turns the readout into a number, and dragging back to the top says "Off" again
- [ ] Load several Core and Advanced presets: each looks exactly as it did before this parameter existed
- [ ] Dragging V Max left slows particles down; the pattern shrinks in scale rather than freezing all at once
- [ ] The change is visible well before the far left — around the middle of the track on a typical preset
- [ ] On a fast preset (Zipper, Karst, Shrimp) the top third of the track already bites
- [ ] Dragging it fully left stops the particles dead, and dragging back right revives them
- [ ] It stops them dead on a preset with high Strafe Power too (Zipper, Adrift, Shrimp) — strafe moves particles without touching velocity
- [ ] Holding it low for a while and then releasing it does not make particles lurch
- [ ] With V Max fully left, painting a strong strafe field in Advanced Drawing does not move particles either; with V Max at the top the brush pushes exactly as it always did
- [ ] Take a preset that blows up (high Sensor Gain, Drag near 1) and lower V Max: it settles instead of tearing
- [ ] Save a preset, relaunch, load it: V Max comes back at the value you saved
- [ ] Copy to clipboard and paste back: same
- [ ] Right-click the slider > "Reset value to ..." puts back the loaded preset's value
- [ ] Alt-click locks it, and a preset load leaves a locked V Max alone
- [ ] Change World Size: the same V Max produces the same apparent speed
- [ ] Multi-load two presets with different V Max values: each config's particles obey their own limit
- [ ] Run a tournament: every tile obeys the global V Max, and lowering it mid-run affects all tiles together
- [ ] Run Explore with physics search on: V Max stays where you left it and is not varied between tiles

## Audio Reactive

Needs sound playing on the machine, or a microphone. `PyAudioWPatch` must be
installed; without it the panel says so and everything else runs unchanged.

- [ ] Extras > Audio Reactive opens the panel; it is closed on a fresh launch.
- [ ] The device list contains both `input:` and `loopback:` entries.
- [ ] Picking a loopback device and pressing Start with the machine SILENT
      reads `waiting`, not `active` - a loopback endpoint sends nothing until
      something plays, so this is the normal state, not a fault.
- [ ] Starting music then flips it to `active` and the spectrum moves.
- [ ] Stop halts the traces; the status reads `idle`.
- [ ] The spectrum is BARS, coloured red/orange/green/blue low to high, and it
      does not rescale itself frame to frame - hold a steady note and the bars
      hold still rather than heaving about.
- [ ] The five band traces move while audio plays, and each band prints its
      current value. On a steady note a trace is FLAT, not a fuzzy hash: the
      bands are smoothed, and a trace never draws more samples than it has
      pixels.
- [ ] Nothing playing but room noise leaves every band at ZERO, not part way
      up. This is what the measures were changed for; if a band sits high with
      nothing playing, raise its Floor under Bands.
- [ ] A quiet passage of a track reads far lower than a loud one, and the bands
      return to zero between hits rather than resting half-lit.
- [ ] Hi-hats and snare hits SNAP: the `hi` and `presence` traces jump on the
      transient and ease back down, rather than swelling into it. A steady note
      still draws a flat trace - both halves matter, and a change to the
      smoothing that fixes one usually breaks the other.
- [ ] Binding bass to Sensor Gain visibly changes the simulation on a beat.
- [ ] Unticking `Modulate` at the top freezes the simulation's response while
      the spectrum and every trace KEEP moving; the sliders lose their hatching.
      Ticking it back resumes without touching any mapping.
- [ ] A bound row's own checkbox silences every band on that parameter at once
      and dims its name. The band dots stay clickable, and a band whose `On`
      box was already unticked in the drawer is still unticked after the row is
      switched back on.
- [ ] Muting a Brain row under one modality leaves the same-named row under
      another alone (mlp and lenia both have Weight Scale).
- [ ] Both switches survive a quit and relaunch.
- [ ] A bound row shows a sparkline of what audio is doing to that parameter;
      an unbound row shows none and is dimmed.
- [ ] Clicking a bound row's name opens its drawer; clicking another row's name
      closes the first - only one is ever open.
- [ ] Clicking a band dot toggles the mapping and does NOT open or close a
      drawer.
- [ ] The drawer has one tab per bound band, and a Total tab once two or more
      bands are bound.
- [ ] A band tab offers Depth, Gain, mode, an On toggle and a Shaper; picking
      each shaper shows only the controls that shaper uses (smooth: attack and
      release; gate: threshold and hold; lfo: rates and wave).
- [ ] Right-clicking Depth, Gain, Strength or any shaper slider offers a reset
      naming that control's default, and picking it puts the value back.
- [ ] A band tab's trace shows the SHAPER'S output bright over the raw band
      faint: pick `lfo` and the bright line oscillates while the faint one
      follows the music. Picking `none` makes the two identical.
- [ ] The Total tab shows live value, base, delta, a range bar, and the
      contributing bands overlaid with the parameter's own trace.
- [ ] The Sensor Gain slider shows hatching, a pale base tick, and a handle that
      rides with the music. Its printed value is the live one.
- [ ] Every mark stays INSIDE the slider and none of it reaches the label. In
      `add` mode the hatching starts under the grab and runs up to where a
      full-scale signal would take it; the pale tick sits exactly under the
      grab at any value, on a bipolar slider (Lateral Force) as well as a
      0-based one.
- [ ] Give a mapping an `lfo` or `smooth` shaper: the hatching still shows the
      whole swing, not half of it.
- [ ] Dragging that slider moves the base tick; the hatching follows it.
- [ ] File > Save writes the slider value, not the momentary modulated one.
- [ ] Right-click on a bound slider > Audio... opens the panel.
- [ ] Setting an X sweep on a bound parameter badges the row `swept`, and the
      parameter stops responding to audio.
- [ ] Switching brain modality changes the Brain rows; switching back restores
      the mappings that were there.
- [ ] EVERY modality offers at least one Brain row, MLP included (Weight Scale
      and Bias Scale). Mapping a band to Weight Scale visibly changes how hard
      the brain drives the particles.
- [ ] Enabling Auto (Prompt) stops audio modulating anything; disabling restores it.
- [ ] Quitting and relaunching restores the mappings, with capture stopped.
- [ ] With the panel closed, the frame rate matches a run with the feature never
      enabled.
- [ ] The Sonification window (Extras) still opens and works alongside this one;
      the two panels are separate and neither replaces the other.

### How each band is measured

- [ ] The Bands section shows a measure per band, a floor and a ceiling, with
      `volume` naming its own measure and offering only the window.
- [ ] Switching a band to `mean_db` makes it sit high with quiet material, and
      back to `power` drops it. That contrast is the whole change.
- [ ] Switching a measure moves the floor and ceiling to that measure's own
      defaults; right-clicking either puts it back.
- [ ] Raising a band's Floor above the music makes it read zero, immediately,
      with the track still playing - no Stop and Start.
- [ ] Setting Release to 0 makes the traces snap back the instant the sound
      stops; the rise is identical at every setting. The spectrum bars stay
      smooth either way.
- [ ] With Release at 0 and a `phase` shaper bound, the wave stops dead in the
      gaps rather than drifting on.
- [ ] The measures, the windows and Release all survive a quit, and a rig saved
      before they existed still loads.
- [ ] Setting a band to `flux` makes it fire on ATTACKS and drop straight back:
      a sustained bass note holds `power` open for its whole length and leaves
      `flux` at zero after the first moment. `hi`/flux tracks the hi-hats.
- [ ] `centroid` moves with brightness and NOT with level: turn the system
      volume down and it stays put while every other trace drops. A filter
      sweep or the hats entering moves it.
- [ ] Stopping the music leaves `centroid` where it was rather than diving; it
      is at zero on a fresh launch, before anything has played.
- [ ] Bind `centroid` to a `phase` shaper, let it run, then PAUSE the music on
      a loopback device: the wave must stop where it is. It holds a non-zero
      value, so an integrator would otherwise keep travelling with nothing
      playing.
- [ ] On a MICROPHONE with the room quiet, `centroid` is frozen - not drifting.
      Its trace must be a flat line. If it still moves, the room is above the
      gate: raise `volume`'s Floor until volume itself reads near zero.
- [ ] The shaper's Attack and Release reach 30 s and the low end is still
      controllable - the track is logarithmic, so 0.05 s is a real position on
      it and not the first pixel.
- [ ] Loading the `Dancing` preset binds four rows (centroid, volume, bass,
      hi), sets bass and hi to `flux`, and Release to 0. The device you had
      selected is unchanged.

### Rig presets, and the rig surviving a second copy

- [ ] Build a rig, press Save at the top of the panel, name it: the dialog says
      it saves to `Documents/Fluoddity/audio_rigs`, NOT to physics_configs, and
      the name appears in the Preset combo straight away.
- [ ] The saved rig does not appear anywhere under File > Load.
- [ ] Saving under a name that already exists asks before overwriting.
- [ ] Clear the rig, pick the preset, press Load: every mapping, strength and
      mute comes back.
- [ ] Quit and relaunch with no preset touched: the rig you left is the rig you
      get, with capture stopped.
- [ ] **The wipe.** Open two copies of Fluoddity. Build a rig in the first;
      touch nothing in the second. Close the SECOND, then the first. Relaunch:
      the rig is still there. (Before this rule the untouched copy overwrote
      it, which is why the rig seemed to save only sometimes.)
- [ ] Load a preset in one copy and close it last: that preset is what comes
      back next launch.

## Archive browser: gallery sizing, the wheel, and the map atlas

Launch with the venv interpreter - `umap-learn` is installed there and nowhere
else, so any other interpreter shows the Layout combo as PCA-only:

```
.venv/Scripts/python.exe main.py
```

### A. The wheel (fix; check first, it is the quickest)
- [ ] **a.** Open Extras > Archive Browser, Map tab. Make the window SHORT
      enough to scroll, and scroll it to the MIDDLE - at the top a wheel-up
      moves nothing and everything looks fine either way.
- [ ] **b.** Wheel over the map canvas: the map zooms and the panel behind it
      does NOT move. Both directions.
- [ ] **c.** Wheel over the panel outside the canvas: it scrolls normally. The
      failure this guards is over-claiming the wheel and making the browser
      unscrollable.
- [ ] **d.** Start a wheel gesture over the canvas and run the pointer off it
      mid-gesture: the panel should not lurch.

### B. Gallery size and list mode
- [ ] **e.** Gallery tab: the size slider sits between a list glyph and a grid
      glyph. Clicking either glyph snaps to that end.
- [ ] **f.** Drag it up and down: tiles scale; at 48px and below it becomes a
      table. Default (96) looks exactly as it always did.
- [ ] **g.** Small sizes on a WIDE, TALL window: scrolling stays smooth. Stutter
      means the thumbnail cache is thrashing.
- [ ] **h.** Table: right-click a header for show/hide. Hide a column, reopen
      the app, it stays hidden (ImGui keeps this in imgui.ini).
- [ ] **i.** Click column headers to sort; switch back to grid - the order is
      the same and the Sort combo agrees. The arrow beside it flips direction.
- [ ] **j.** Hover a row with Live preview on: it previews. Click: it keeps.
      Same as the grid.
- [ ] **k.** Reopen an archive: its size and sort come back.

### C. The map layout
- [ ] **l.** Layout combo offers PCA and UMAP. On PCA everything is as before.
- [ ] **m.** Switch to UMAP on a large archive: the readout says "fitting...",
      the UI STAYS RESPONSIVE, and the map swaps when it lands. A freeze means
      the fit is on the frame loop.
- [ ] **n.** The result should visibly cluster where PCA was one blob.
- [ ] **o.** Close and reopen the archive: the UMAP map returns instantly, in
      the same orientation. A relayout or a mirrored map is the bug.
- [ ] **p.** Readout shows "fitted at N of M" as entries are admitted; it
      refits itself once M is a quarter past N.
- [ ] **q.** Relayout: refits under UMAP, and does something visible under PCA
      too.
- [ ] **r.** Switch archives: the map is the new archive's, not the old one's.

### D. The thumbnail atlas
- [ ] **s.** Thumbnails checkbox: pictures ONLY - no dots anywhere, and the
      Colour and Draw combos grey out.
- [ ] **t.** Size slider 16..64 changes the cell size.
- [ ] **u.** Zoom in: cells subdivide and more pictures appear. Pan and zoom
      slowly - the pictures must SLIDE, never reshuffle which entry is where.
- [ ] **v.** Leave the map open and STOP TOUCHING IT: within a second or two
      the pictures must stop appearing. A sweep that keeps rolling across the
      atlas means the working set no longer fits the cache.
- [ ] **w.** Move the pointer across a settled atlas: nothing redraws. The
      hover card's own picture must not cost a cell.
- [ ] **x.** Drag the size slider slowly: the pictures resize CONTINUOUSLY,
      not in two or three jumps.
- [ ] **x2.** Switch Layout between PCA and UMAP: the old pictures hold and
      are replaced all at once. No vertical sweep of thumbnails.
- [ ] **x3.** Zoom right in: every cell on screen has a picture, and new
      entries keep appearing rather than the same few.
- [ ] **x4.** Switch to a SMALLER archive with the map open: it redraws
      rather than crashing.
- [ ] **x5.** Gallery list mode: the thumbnail column's width tracks the size
      slider, and offers no resize handle of its own.
- [ ] **y.** A big archive is no slower to draw than a small one at the same
      zoom. The count follows the viewport, not the entry count.

Numbers behind the UMAP choice: `python -m tools.measure_map_layout`.
