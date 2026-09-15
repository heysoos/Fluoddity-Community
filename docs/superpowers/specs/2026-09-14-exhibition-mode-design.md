# Exhibition mode

A set list of presets on the keyboard, MIDI knobs on the physics sliders, and
a lock that hands the keyboard to visitors. It extends the Perform panel: the
audience watches the projector, the laptop keeps its UI.

## Decisions

- **A slot may also name a saved audio rig**, loaded through the Audio
  panel's own Load path after the preset; "" leaves the current rig alone.
- **Cycle** loads the next filled slot every `cycle_seconds` (a preference)
  while `PerformState.cycle_enabled`; any slot load restarts the clock from
  that slot. `services/slot_cycle.py`, driven from `_handle_slot_load`.
- **A key press is an ordinary cold load.** Physics with locks respected, the
  rule pushed, the field stack applied, undo tagged `Load <stem>`. No reset;
  the canvas keeps evolving.
- **Knobs are relative encoders and write the real state.** A detent adds a
  fraction of the slider's live range to `ui_state.sim`, so the Physics
  slider moves and Save writes it. A preset load recentres by definition.
  Pushing an encoder returns its slider to the value the current preset
  loaded with. The audio rig's `global_strength` and `rate_scale` are knob
  targets too (`AUDIO_KNOB_TARGETS`), recentred when a rig loads; Rate Scale
  steps in log space because its slider is logarithmic.
- **Audio bends on top.** MIDI runs before the audio runtime, whose output is
  what the sim runs under. The LED ring shows THAT value; the colour switches
  while it differs from the slider by more than a dead band.
- **The lock takes the keyboard before ImGui.** Set-list keys and the unlock
  chord (Ctrl+Shift+F11) are the only keys that do anything; held keys are
  dropped too. The mouse is untouched.
- **Unlocked, a set-list key still loads** so a show can be rehearsed, unless
  the app already binds that key, which wins.

## Components

| Piece | Home |
|---|---|
| `Slot`, `KnobBinding`, `SetList` | `state/set_list.py` |
| set list files, `Documents/Fluoddity/setlists/<name>.json` | `services/set_list_io.py` |
| device, port, Twister protocol constants | `services/midi_control.py` |
| `MidiRuntime.update` / `feedback` | `services/midi_runtime.py` |
| key handling and the lock | `ui/core.py` `key_callback` |
| slot dispatch, `recentre_knobs` | `command_handler.py` |
| the panel | `ui/exhibition_panel.py`, drawn by `ui/perform_window.py` |

State: `PerformState.exhibition_locked`, `set_list`, `slot_load`,
`recentre_knobs`, `midi_status` (none persisted);
`PreferencesState.set_list_name`, `midi_device` (both `NOT_UNDOABLE`).

## Midi Fighter Twister

Channels are 0-indexed as mido reports them. Encoders in and ring position
out on channel 0; encoder push in (value 127) and RGB colour out on channel 1;
RGB brightness on channel 2 (17 dimmest, 47 brightest). Encoders must be set
to ENC 3FH/41H in the Midi Fighter Utility: 65 per clockwise detent, 63 per
anti-clockwise. Bank 2 is CC 16-31, so a binding by CC number covers all four
banks. The device's MIDI IN cannot be remapped, so these are fixed.

## Not built

Reset on switch, absolute-encoder takeover, per-preset knob
assignments, MIDI on the audio rig's signal list.
