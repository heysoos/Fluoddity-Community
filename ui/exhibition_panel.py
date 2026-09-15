"""The exhibition half of the Perform panel: set list, knobs, device, lock."""
import time

from imgui_bundle import imgui

from services import midi_control
from services import set_list_io
from state.set_list import KnobBinding, SetList, Slot
from utilities.keybinding_management import KeybindingManager
from . import hints, layout, notices

SLOT_KEYS = tuple(k for k in KeybindingManager.KEY_NAME_TO_GLFW
                  if k not in ("ESC",))
DEVICE_RESCAN_SECONDS = 1.0
NONE_DEVICE = "(none)"


def _param_labels(sim_state):
    from services.midi_runtime import knob_targets
    return [(t.key, t.label) for t in knob_targets(sim_state)]


class ExhibitionPanelMixin:
    """Mixin for the exhibition controls. Combined into UI via inheritance."""

    _set_list_name_edit = None
    _exhibition_notice = ""
    _rig_cache = ()
    _rig_cache_at = -1e9
    _device_cache = ()
    _device_cache_at = -1e9

    def _render_exhibition(self, perform, prefs):
        imgui.separator()
        imgui.text("Exhibition")
        layout.text_disabled_wrapped(
            "Keys load presets from a set list and MIDI knobs nudge sliders. "
            "Lock takes the keyboard for the visitors; Ctrl+Shift+"
            f"{self.keybindings.get_key_display_name('toggle_perform_mode')} "
            "gives it back.")

        changed, want = imgui.checkbox("Lock keyboard##exhibition",
                                       perform.exhibition_locked)
        if changed:
            perform.exhibition_locked = want
        hints.tip("Only set-list keys and the unlock chord do anything while on.")

        changed, want = imgui.checkbox("Cycle presets##exhibition",
                                       perform.cycle_enabled)
        if changed:
            perform.cycle_enabled = want
        hints.tip("Load the next set-list slot on a timer; a key press restarts it.")
        imgui.same_line()
        imgui.set_next_item_width(imgui.calc_text_size("M" * 7).x)
        changed, secs = imgui.drag_float("s##cycle", prefs.cycle_seconds,
                                         1.0, 5.0, 3600.0, "%.0f")
        if changed:
            prefs.cycle_seconds = max(5.0, min(3600.0, secs))
        hints.tip("Seconds each slot stays up while cycling.")

        self._render_set_list_file(perform, prefs)
        self._render_slots(perform)
        self._render_device(perform, prefs)
        self._render_knobs(perform)

    # --- the file ---------------------------------------------------------

    def _render_set_list_file(self, perform, prefs):
        sl = perform.set_list
        if self._set_list_name_edit is None:
            self._set_list_name_edit = sl.name
        layout.push_settings_width()
        changed, text = imgui.input_text("Set list##exhibition",
                                         self._set_list_name_edit)
        if changed:
            self._set_list_name_edit = text
        imgui.pop_item_width()
        hints.tip("The name this set list is saved under.")

        right = layout.row_right_edge()
        if imgui.button("Save##setlist"):
            sl.name = self._set_list_name_edit
            try:
                set_list_io.save_set_list(sl)
                prefs.set_list_name = sl.name
                self._exhibition_notice = f"Saved set list '{sl.name}'."
            except (OSError, ValueError) as e:
                perform.notice = f"Could not save set list: {e}"
        layout.wrap_row(right, layout.button_width("New"))
        if imgui.button("New##setlist"):
            perform.set_list = SetList.default("")
            self._set_list_name_edit = ""
            prefs.set_list_name = ""

        names = set_list_io.list_set_lists()
        if names:
            layout.push_settings_width()
            current = names.index(sl.name) if sl.name in names else -1
            changed, picked = imgui.combo("Open##setlist", current, names)
            imgui.pop_item_width()
            if changed and 0 <= picked < len(names):
                loaded = set_list_io.load_set_list(names[picked])
                if loaded is not None:
                    perform.set_list = loaded
                    prefs.set_list_name = loaded.name
                    self._set_list_name_edit = loaded.name
                else:
                    perform.notice = f"Could not read set list '{names[picked]}'."
        if self._exhibition_notice:
            imgui.text_colored(imgui.ImVec4(*notices.OK), self._exhibition_notice)

    # --- slots ------------------------------------------------------------

    def _preset_choices(self):
        """Every loadable preset as (category, stem), with a display label."""
        by_cat = getattr(self, "config_files_by_category", {}) or {}
        out = []
        for cat, files in by_cat.items():
            for f in files:
                out.append((cat, f, f"{cat} / {f}"))
        return out

    def _rig_names(self):
        """Named rigs on disk, re-read once a second rather than per frame."""
        now = time.monotonic()
        if now - self._rig_cache_at > DEVICE_RESCAN_SECONDS:
            from services.audio_rig_io import list_rigs
            self._rig_cache = tuple(list_rigs())
            self._rig_cache_at = now
        return self._rig_cache

    def _render_slots(self, perform):
        imgui.spacing()
        imgui.text("Keys")
        layout.text_disabled_wrapped("Key, the preset it loads, and the audio "
                                     "rig loaded with it.")
        choices = self._preset_choices()
        labels = ["(empty)"] + [c[2] for c in choices]
        rigs = list(self._rig_names())
        key_w = imgui.calc_text_size("MMMMMM").x + 24
        remove = -1
        for i, slot in enumerate(perform.set_list.slots):
            imgui.push_id(f"slot{i}")
            imgui.set_next_item_width(key_w)
            cur = SLOT_KEYS.index(slot.key) if slot.key in SLOT_KEYS else -1
            changed, picked = imgui.combo("##key", cur, list(SLOT_KEYS))
            if changed:
                slot.key = SLOT_KEYS[picked]
            imgui.same_line()
            cur = next((j + 1 for j, c in enumerate(choices)
                        if c[0] == slot.category and c[1] == slot.filename), 0)
            if slot.filename and cur == 0:
                # Named but not in the library: offered back, never replaced.
                shown = labels + [f"{slot.category} / {slot.filename} (missing)"]
                cur = len(shown) - 1
            else:
                shown = labels
            room = (imgui.get_content_region_avail().x
                    - layout.button_width("X") - 16) / 2
            imgui.set_next_item_width(max(100.0, room))
            changed, picked = imgui.combo("##preset", cur, shown)
            if changed:
                if picked == 0:
                    slot.category = slot.filename = ""
                elif picked <= len(choices):
                    slot.category, slot.filename = choices[picked - 1][:2]
            imgui.same_line()
            rig_shown = ["(keep rig)"] + rigs
            if slot.rig and slot.rig not in rigs:
                rig_shown.append(f"{slot.rig} (missing)")
                rig_cur = len(rig_shown) - 1
            else:
                rig_cur = rig_shown.index(slot.rig) if slot.rig in rigs else 0
            imgui.set_next_item_width(max(100.0, room))
            changed, picked = imgui.combo("##rig", rig_cur, rig_shown)
            if changed and picked <= len(rigs):
                slot.rig = "" if picked == 0 else rigs[picked - 1]
            imgui.same_line()
            if imgui.small_button("X"):
                remove = i
            imgui.pop_id()
        if remove >= 0:
            del perform.set_list.slots[remove]
        if imgui.button("Add key##exhibition"):
            used = {s.key for s in perform.set_list.slots}
            free = next((k for k in SLOT_KEYS if k not in used), SLOT_KEYS[0])
            perform.set_list.slots.append(Slot(key=free))

    # --- device -----------------------------------------------------------

    def _devices(self):
        now = time.monotonic()
        if now - self._device_cache_at > DEVICE_RESCAN_SECONDS:
            self._device_cache = tuple(midi_control.list_devices())
            self._device_cache_at = now
        return self._device_cache

    def _render_device(self, perform, prefs):
        imgui.spacing()
        imgui.text("MIDI")
        if not midi_control.available():
            layout.text_disabled_wrapped(
                "Install mido and python-rtmidi to use a controller.")
            return
        devices = list(self._devices())
        if prefs.midi_device and prefs.midi_device not in devices:
            devices.append(prefs.midi_device)
        options = [NONE_DEVICE] + devices
        cur = options.index(prefs.midi_device) if prefs.midi_device in options else 0
        layout.push_settings_width()
        changed, picked = imgui.combo("Device##midi", cur, options)
        imgui.pop_item_width()
        if changed:
            prefs.midi_device = "" if picked == 0 else options[picked]
        hints.tip("The controller, by name. Encoders must send relative "
                  "(ENC 3FH/41H) messages and their switches CC Hold.")
        if perform.midi_status:
            imgui.text_disabled(perform.midi_status)

    # --- knobs ------------------------------------------------------------

    def _render_knobs(self, perform):
        imgui.spacing()
        imgui.text("Knobs")
        params = _param_labels(self.state.sim)
        names = [p[0] for p in params]
        labels = ["(none)"] + [p[1] for p in params]
        narrow = imgui.calc_text_size("MMMM").x + 20
        remove = -1
        for i, knob in enumerate(perform.set_list.knobs):
            imgui.push_id(f"knob{i}")
            imgui.set_next_item_width(narrow)
            changed, cc = imgui.input_int("##cc", knob.cc, 0)
            if changed:
                knob.cc = max(0, min(127, cc))
            hints.tip("Encoder CC number.")
            imgui.same_line()
            cur = names.index(knob.param) + 1 if knob.param in names else 0
            imgui.set_next_item_width(imgui.calc_text_size("M" * 16).x)
            changed, picked = imgui.combo("##param", cur, labels)
            if changed:
                knob.param = "" if picked == 0 else names[picked - 1]
            imgui.same_line()
            imgui.set_next_item_width(narrow + 20)
            changed, step = imgui.slider_float(
                "##step", knob.step, 0.001, 0.1, "%.3f",
                imgui.SliderFlags_.logarithmic)
            if changed:
                knob.step = step
            hints.tip("Fraction of the slider range per detent.")
            imgui.same_line()
            imgui.set_next_item_width(narrow)
            changed, c = imgui.slider_int("##rest", knob.rest_colour, 0, 127)
            if changed:
                knob.rest_colour = c
            hints.tip("Ring colour at rest.")
            imgui.same_line()
            imgui.set_next_item_width(narrow)
            changed, c = imgui.slider_int("##active", knob.active_colour, 0, 127)
            if changed:
                knob.active_colour = c
            hints.tip("Ring colour while audio is bending the value.")
            imgui.same_line()
            if imgui.small_button("X"):
                remove = i
            imgui.pop_id()
        if remove >= 0:
            del perform.set_list.knobs[remove]
        if imgui.button("Add knob##exhibition"):
            used = {k.cc for k in perform.set_list.knobs}
            free = next((c for c in range(midi_control.ENCODER_COUNT)
                         if c not in used), 0)
            perform.set_list.knobs.append(KnobBinding(cc=free))
