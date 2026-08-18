"""The Audio Reactive panel: source, spectrum, band traces, mapping matrix.

Named audio_reactive_window, not audio_window, because the sonification feature
owns that name for the MIDI-out side. Two mixins with one method name would
shadow each other in the UI's MRO without raising.

The matrix follows the boids panel it was ported from: one row per target with
a sparkline of what audio is doing to it, five band dots, and ONE drawer open
at a time - a tab per bound band plus a total. Traces go through
imgui.plot_lines over a float32 array, one crossing into C++ per trace rather
than one per segment.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services import audio_capture
from services.audio_analysis import BAND_NAMES, SIGNAL_NAMES
from services.audio_mapping import (MODES, Mapping, brain_targets,
                                    deaf_targets, physics_targets)
from services.audio_shapers import SHAPER_KINDS, ShaperParams
from ui import hints, layout, notices

SIGNAL_COLORS: dict[str, tuple] = {
    "bass": (0.88, 0.31, 0.38, 1.0),
    "mid": (0.88, 0.56, 0.13, 1.0),
    "presence": (0.50, 0.82, 0.38, 1.0),
    "hi": (0.25, 0.63, 0.88, 1.0),
    "volume": (0.71, 0.55, 0.94, 1.0),
    "centroid": (0.94, 0.85, 0.45, 1.0),
}

SIGNAL_ABBR: dict[str, str] = {
    "bass": "B", "mid": "M", "presence": "P", "hi": "H", "volume": "V",
    "centroid": "C",
}

WAVE_KINDS: tuple[str, ...] = ("sine", "triangle", "ramp")

TRACE_LEN = 180          # capped near the trace's width in pixels

# Which shaper fields each kind actually reads. A kind that ignores a field
# must not offer it - an attack slider on a gate is a control that does nothing.
SHAPER_FIELDS: dict[str, tuple[str, ...]] = {
    "none": (),
    "smooth": ("attack", "release"),
    "gate": ("threshold", "hold"),
    "envelope": ("threshold", "release"),
    "phase": ("rate", "wave"),
    "sample_hold": ("threshold",),
}

# Attack and release reach half a minute, which is a SONG SECTION rather than
# a smoothing time: `smooth` over tens of seconds is what turns any band into
# the slow envelope that follows a track's arrangement. Both are logarithmic,
# or everything under a second - which is every percussive setting - would
# share the first pixel of the track.
_FIELD_RANGE: dict[str, tuple[float, float, str]] = {
    "attack": (0.0, 30.0, "%.3f s"),
    "release": (0.0, 30.0, "%.3f s"),
    "threshold": (0.0, 1.0, "%.2f"),
    "hold": (0.0, 1.0, "%.3f s"),
    "rate": (0.0, 20.0, "%.2f Hz"),
}

# Which shaper fields need a logarithmic track.
_FIELD_LOG: frozenset = frozenset({"attack", "release"})

# What a signal whose measure is not a choice is measuring instead.
_FIXED_MEASURE_BLURB: dict[str, str] = {
    "volume": "the block's own loudness",
    "centroid": "brightness: where the energy sits",
}

# The track a band's window slider covers, by unit.
_WINDOW_TRACK: dict[str, tuple[float, float]] = {
    "dB": (-120.0, 0.0),
    "Hz": (20.0, 16000.0),
}

_FIELD_TIP: dict[str, str] = {
    "rate": "How fast the wave travels while the band is at full scale; a "
            "quieter band moves it proportionally slower, and silence stops it.",
}


def visible_samples(values: np.ndarray, width: float) -> np.ndarray:
    """The newest samples that fit the widget, one per pixel.

    plot_lines spreads whatever it is handed across the whole graph, so giving
    a 56-pixel sparkline a 180-sample history draws three samples per pixel and
    reads as noise however smooth the signal underneath is.
    """
    n = int(max(2.0, min(float(values.size), width)))
    return values[-n:]


class TraceRing:
    """A rolling window kept as float32 so it can go straight to plot_lines."""

    def __init__(self, length: int = TRACE_LEN) -> None:
        self._buf = np.zeros(int(length), dtype=np.float32)

    def push(self, value: float) -> None:
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = np.float32(value)

    @property
    def values(self) -> np.ndarray:
        return self._buf


class AudioReactiveWindowMixin:
    """Combined into UI via multiple inheritance."""

    # --- ring buffers ---------------------------------------------------

    def _audio_rings(self) -> dict:
        """One ring per band, for the source section and the drawers."""
        if not hasattr(self, "_audio_trace_rings"):
            self._audio_trace_rings = {n: TraceRing() for n in SIGNAL_NAMES}
        return self._audio_trace_rings

    def _audio_target_rings(self) -> dict:
        """One ring per bound target, holding its live value as a fraction of
        its own track, so a row's sparkline reads the same at any range."""
        if not hasattr(self, "_audio_target_trace_rings"):
            self._audio_target_trace_rings = {}
        return self._audio_target_trace_rings

    def _audio_shaped_rings(self) -> dict:
        """One ring per mapping, holding what its shaper puts out."""
        if not hasattr(self, "_audio_shaped_trace_rings"):
            self._audio_shaped_trace_rings = {}
        return self._audio_shaped_trace_rings

    def _audio_devices(self, refresh: bool = False) -> list:
        if refresh or not hasattr(self, "_audio_device_cache"):
            self._audio_device_cache = audio_capture.list_devices()
        return self._audio_device_cache

    # --- window ---------------------------------------------------------

    def render_audio_reactive_window(self):
        ast = self.state.audio
        if not ast.show_window:
            return

        imgui.set_next_window_size(imgui.ImVec2(440, 660),
                                   imgui.Cond_.first_use_ever)
        layout.constrain_panel()
        _expanded, opened = imgui.begin("Audio Reactive", True)
        if not opened:
            ast.show_window = False
            imgui.end()
            return

        self._audio_pump_rings(ast)

        # push_settings_width pairs with imgui.pop_item_width, not a layout call.
        layout.push_settings_width()
        self._render_audio_presets(ast)
        notices.render_banner(ast, "warning", notices.BAD, scope="audio")
        notices.render_banner(ast, "notice", notices.OK, scope="audio")
        imgui.separator()
        self._render_audio_source(ast)
        imgui.separator()
        self._render_audio_signals(ast)
        imgui.separator()
        self._render_audio_matrix(ast)
        imgui.pop_item_width()
        imgui.end()

    def _audio_pump_rings(self, ast):
        """Advance every ring once per frame, before anything draws.

        Done here rather than inside a draw call so a collapsed section or a
        closed drawer does not freeze the history it will show when reopened.
        """
        snap = getattr(ast, "snapshot", None)
        rings = self._audio_rings()
        if snap is not None:
            for name in SIGNAL_NAMES:
                rings[name].push(snap.signals.get(name, 0.0))

        overlays = getattr(self, "audio_overlays", {}) or {}
        trings = self._audio_target_rings()
        for key, ov in overlays.items():
            span = ov["hi"] - ov["lo"]
            frac = (ov["live"] - ov["lo"]) / span if span > 0 else 0.0
            trings.setdefault(key, TraceRing()).push(min(1.0, max(0.0, frac)))
        for gone in [k for k in trings if k not in overlays]:
            del trings[gone]

        shaped = getattr(ast, "shaped", None) or {}
        srings = self._audio_shaped_rings()
        for key, value in shaped.items():
            srings.setdefault(key, TraceRing()).push(
                min(1.0, max(0.0, float(value))))
        for gone in [k for k in srings if k not in shaped]:
            del srings[gone]

    # --- presets --------------------------------------------------------

    def _audio_preset_names(self, refresh: bool = False) -> list:
        """The rig names on disk, re-read only when a save or load moves them.

        A directory listing every frame is a syscall per frame for a folder
        that changes about twice a session.
        """
        if refresh or not hasattr(self, "_audio_preset_cache"):
            from services.audio_rig_io import list_rigs
            self._audio_preset_cache = list_rigs()
        return self._audio_preset_cache

    def _render_audio_presets(self, ast):
        """Pick a named rig, load it over the live one, or save the live one."""
        from services import save_targets

        # Every save and load sets a notice, so a change to it is exactly when
        # the folder can have gained a file.
        if getattr(self, "_audio_preset_notice", None) != ast.notice:
            self._audio_preset_notice = ast.notice
            self._audio_preset_names(refresh=True)

        names = self._audio_preset_names()
        current = names.index(ast.preset_name) if ast.preset_name in names else -1
        changed, idx = imgui.combo("Preset", current, names)
        if changed and 0 <= idx < len(names):
            ast.preset_name = names[idx]
        hints.tip("A saved rig: every mapping, strength and mute.")

        if imgui.button("Load##audio_preset") and ast.preset_name:
            ast.request_load_preset = True
        imgui.same_line()
        if imgui.button("Save##audio_preset"):
            self.open_save_popup(save_targets.AUDIO_RIG,
                                 name=ast.preset_name or "rig")
        imgui.same_line()
        if imgui.button("Rescan##audio_preset"):
            self._audio_preset_names(refresh=True)

    # --- source ---------------------------------------------------------

    def _render_audio_source(self, ast):
        if not audio_capture.is_available():
            imgui.text_wrapped(
                "Audio input needs PyAudioWPatch, which is not installed.")
            return

        devices = self._audio_devices()
        names = [f"{'loopback' if d['loopback'] else 'input'}: {d['name']}"
                 for d in devices]
        current = next((i for i, d in enumerate(devices)
                        if d["name"] == ast.device_name), 0)
        if names:
            changed, idx = imgui.combo("Device", current, names)
            if changed:
                ast.device_name = devices[idx]["name"]
        hints.tip("Which input the bands are read from.")

        if ast.enabled:
            if imgui.button("Stop##audio_source"):
                ast.request_stop = True
        else:
            if imgui.button("Start##audio_source"):
                ast.request_start = True
        imgui.same_line()
        if imgui.button("Rescan##audio_source"):
            self._audio_devices(refresh=True)
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(0.5, 0.8, 0.4, 1.0) if ast.status == "active"
            else imgui.ImVec4(0.8, 0.35, 0.35, 1.0) if ast.status == "error"
            else imgui.ImVec4(0.85, 0.65, 0.25, 1.0) if ast.status == "waiting"
            else imgui.ImVec4(0.5, 0.5, 0.55, 1.0),
            ast.status)

        if ast.status == "error" and ast.last_error:
            imgui.text_wrapped(ast.last_error)
        if ast.status == "waiting":
            imgui.text_disabled("Play something - a silent output sends no audio.")

        changed, value = imgui.checkbox("Modulate", ast.modulate)
        if changed:
            ast.modulate = value
        hints.tip(
            "Master switch for every mapping. Capture keeps running, so the "
            "traces still move while it is off.")

        imgui.same_line()
        changed, value = imgui.checkbox("Auto Gain", ast.auto_gain)
        if changed:
            ast.auto_gain = value
        hints.tip("Normalises each band against its recent peak, "
                              "which also lifts a quiet room's hiss.")

        changed, value = self._audio_slider("Strength", ast.global_strength,
                                            0.0, 2.0, "%.2f", 1.0)
        if changed:
            ast.global_strength = value
        hints.tip("Scales every mapping at once.")

        changed, value = self._audio_slider("Rate Scale", ast.rate_scale,
                                            0.05, 8.0, "x%.2f", 1.0, log=True)
        if changed:
            ast.rate_scale = value
        hints.tip(
            "Multiplies every phase shaper's rate, so a rig can be slid onto "
            "another tempo.")

        changed, value = self._audio_slider("Release", ast.release_seconds,
                                            0.0, 0.5, "%.3f s", 0.075)
        if changed:
            ast.release_seconds = value
        hints.tip(
            "How long a band takes to fall. Zero hands every shaper the raw "
            "measurement; the rise is never smoothed.")

        self._render_audio_bands(ast)

    # --- how each band is measured --------------------------------------

    def _render_audio_bands(self, ast):
        """A measure and a dB window per band.

        Both are calibration, deliberately fixed rather than adaptive: the same
        sound must always produce the same number, which is what a running
        peak or floor cannot promise.
        """
        from services import audio_analysis as aa

        if not imgui.collapsing_header("Bands##audio_bands"):
            return
        effective = aa.effective_bands(ast.bands)
        for name in SIGNAL_NAMES:
            setting = effective[name]
            imgui.text_colored(imgui.ImVec4(*SIGNAL_COLORS[name]), f"{name:9}")
            imgui.same_line()
            if name in aa.FIXED_MEASURES:
                imgui.text_disabled(_FIXED_MEASURE_BLURB[name])
            else:
                imgui.set_next_item_width(110)
                current = list(aa.MEASURES).index(setting["measure"])
                changed, idx = imgui.combo(f"##measure_{name}", current,
                                           list(aa.MEASURES))
                if changed:
                    lo, hi = aa.MEASURE_WINDOWS[aa.MEASURES[idx]]
                    self._write_band(ast, name, measure=aa.MEASURES[idx],
                                     floor=lo, ceiling=hi)
                hints.tip(
                    "power sums the band's energy and takes decibels once, so "
                    "it rests at the floor between hits; mean_db averages "
                    "every bin's level, including the ones carrying nothing.")

            lo_default, hi_default = aa.MEASURE_WINDOWS[setting["measure"]]
            unit = aa.MEASURE_UNITS.get(setting["measure"], "dB")
            track_lo, track_hi = _WINDOW_TRACK[unit]
            imgui.set_next_item_width(90)
            changed, value = self._audio_slider(
                f"##floor_{name}", setting["floor"], track_lo, track_hi,
                f"from %.0f {unit}", lo_default, log=unit == "Hz")
            if changed:
                self._write_band(ast, name, floor=value)
            hints.tip("Everything below this reads zero.")
            imgui.same_line()
            imgui.set_next_item_width(90)
            changed, value = self._audio_slider(
                f"##ceiling_{name}", setting["ceiling"], track_lo, track_hi,
                f"to %.0f {unit}", hi_default, log=unit == "Hz")
            if changed:
                self._write_band(ast, name, ceiling=value)
            hints.tip("At this level and above the band reads one.")

    @staticmethod
    def _write_band(ast, name, **changes):
        """Store one band's settings, filling in what it is using now.

        A band is absent from the rig until it is touched, so the stored entry
        has to be complete rather than a lone key.
        """
        from services.audio_analysis import effective_bands

        setting = dict(effective_bands(ast.bands)[name])
        setting.update(changes)
        ast.bands = dict(ast.bands, **{name: setting})

    # --- spectrum and band traces ---------------------------------------

    def _render_audio_signals(self, ast):
        snap = getattr(ast, "snapshot", None)
        rings = self._audio_rings()
        if snap is not None:
            self._render_audio_spectrum(snap)
        else:
            imgui.text_disabled("No signal yet.")

        width = max(40.0,
                    imgui.get_content_region_avail().x / len(SIGNAL_NAMES) - 4)
        for i, name in enumerate(SIGNAL_NAMES):
            if i:
                imgui.same_line()
            imgui.begin_group()
            self._audio_trace(f"##audio_trace_{name}", rings[name].values,
                              SIGNAL_COLORS[name], width, 26)
            value = snap.signals.get(name, 0.0) if snap is not None else 0.0
            imgui.text_colored(imgui.ImVec4(*SIGNAL_COLORS[name]),
                               f"{name[:4]} {value:.2f}")
            imgui.end_group()

    def _render_audio_spectrum(self, snap):
        """Mel bars on a FIXED scale, each coloured by the band it falls in.

        A fixed scale is what makes the bars readable: left to itself
        plot_histogram rescales to the frame's own extremes every frame, so the
        whole graph heaves about even when nothing in the music changed.
        """
        mel = np.asarray(snap.mel, dtype=np.float32)
        bands = np.asarray(getattr(snap, "mel_bands", np.zeros(mel.size)))
        size = imgui.ImVec2(imgui.get_content_region_avail().x, 46)
        origin = imgui.get_cursor_screen_pos()
        drawn = 0
        for index, name in enumerate(BAND_NAMES):
            masked = np.where(bands == index, mel, 0.0).astype(np.float32)
            if not masked.any():
                continue
            if drawn:
                # Rewind onto the first graph and clear this one's backing, so
                # the bars share one canvas rather than sitting side by side.
                imgui.set_cursor_screen_pos(origin)
                imgui.push_style_color(imgui.Col_.frame_bg,
                                       imgui.ImVec4(0, 0, 0, 0))
            imgui.push_style_color(imgui.Col_.plot_histogram,
                                   imgui.ImVec4(*SIGNAL_COLORS[name]))
            imgui.plot_histogram(f"##audio_spectrum_{name}", masked,
                                 scale_min=0.0, scale_max=1.0, graph_size=size)
            imgui.pop_style_color()
            if drawn:
                imgui.pop_style_color()
            drawn += 1
        if not drawn:
            imgui.plot_histogram("##audio_spectrum", mel, scale_min=0.0,
                                 scale_max=1.0, graph_size=size)

    def _audio_slider(self, label, value, lo, hi, fmt, default, log=False):
        """A slider whose right-click menu puts it back to its default.

        Every number in this panel came from a dataclass field, so there is
        always exactly one value to go back to. `log` is for the tracks that
        span decades - seconds from a frame to half a minute, hertz from a
        rumble to a cymbal - where a linear grab has no resolution at the
        bottom, which is where the useful settings are.
        """
        flags = imgui.SliderFlags_.logarithmic if log else 0
        changed, v = imgui.slider_float(label, float(value), lo, hi, fmt,
                                        flags)
        if imgui.begin_popup_context_item(f"{label}_reset"):
            # p_selected has no default in this binding, and a popup body only
            # runs while the popup is OPEN - so a wrong call here reaches the
            # user rather than a test. See the drawer's forced-popup test.
            if imgui.selectable(f"Reset to {default:g}##do", False)[0]:
                v, changed = float(default), True
                imgui.close_current_popup()
            imgui.end_popup()
        return changed, v

    def _audio_trace(self, ident, values, colour, width, height,
                     lo=0.0, hi=1.0):
        """One plot_lines call with the band's colour pushed around it."""
        pixels = width if width > 0 else imgui.get_content_region_avail().x
        imgui.push_style_color(imgui.Col_.plot_lines, imgui.ImVec4(*colour))
        imgui.plot_lines(ident, visible_samples(values, pixels),
                         scale_min=lo, scale_max=hi,
                         graph_size=imgui.ImVec2(width, height))
        imgui.pop_style_color()

    # --- the matrix -----------------------------------------------------

    def _mapping_list(self, ast, group: str) -> list:
        if group == "physics":
            return ast.mappings
        return ast.brain_mappings.setdefault(self.state.brain.modality, [])

    def _render_audio_matrix(self, ast):
        deaf = deaf_targets(self.state.sim)

        if imgui.collapsing_header("Physics##audio_group",
                                   imgui.TreeNodeFlags_.default_open):
            targets = physics_targets(self.state.sim)
            mappings = self._mapping_list(ast, "physics")
            for target in targets:
                self._render_audio_row(ast, "physics", mappings, target,
                                       target.key in deaf)

        if not imgui.collapsing_header("Brain##audio_group"):
            return

        from services import brains
        modality_name = self.state.brain.modality
        modality = brains.get(modality_name)
        b_layout = modality.layout_from_settings(self.state.brain.settings)
        targets = brain_targets(modality, b_layout)
        if not targets:
            # Not a fault: a modality whose settings are all structural has no
            # decode scale to move. Name it, or this reads as the panel failing
            # after a preset load.
            imgui.text_disabled(
                f"The {modality_name} brain has no continuous scales, so there "
                f"is nothing here to modulate.")
            hints.tip(
                "Only settings that rescale an existing brain can be "
                "modulated; ones that change how many numbers it has cannot.")
            return
        mappings = self._mapping_list(ast, "brain")
        for target in targets:
            self._render_audio_row(ast, "brain", mappings, target, False)

    def _render_audio_row(self, ast, group, mappings, target, is_deaf):
        bound = [m for m in mappings if m.target == target.key]
        overlay = (getattr(self, "audio_overlays", {}) or {}).get(target.key)
        imgui.push_id(f"{group}:{target.key}")

        # The row's own switch: every band on this parameter at once, without
        # touching what each band's On box says. A brain row's key carries its
        # modality, since two brains may declare the same scale name.
        mute_key = (target.key if group == "physics"
                    else f"{self.state.brain.modality}:{target.key}")
        muted = bool(ast.muted.get(mute_key))
        if bound:
            changed, on = imgui.checkbox("##row_on", not muted)
            if changed:
                ast.muted[mute_key] = not on
            hints.tip(
                "Silences every band mapped to this parameter at once.")
        else:
            imgui.dummy(imgui.ImVec2(imgui.get_frame_height(),
                                     imgui.get_frame_height()))
        imgui.same_line()

        # Clicking the name opens this row's drawer and closes any other.
        open_here = ast.open_target == target.key
        arrow = "v " if open_here else "> " if bound else "  "
        if bound:
            if muted:
                imgui.push_style_color(imgui.Col_.text,
                                       imgui.get_style().color_(
                                           imgui.Col_.text_disabled))
            if imgui.selectable(f"{arrow}{target.label}", open_here, 0,
                                imgui.ImVec2(140, 0))[0]:
                ast.open_target = "" if open_here else target.key
                ast.open_band = ""
            if muted:
                imgui.pop_style_color()
        else:
            imgui.text_disabled(f"{arrow}{target.label}")
            imgui.same_line(0, 0)
            imgui.dummy(imgui.ImVec2(max(0.0, 140 - imgui.calc_text_size(
                f"{arrow}{target.label}").x), 1))

        # The row's own sparkline: what audio is doing to this parameter.
        imgui.same_line()
        if bound and overlay is not None:
            ring = self._audio_target_rings().get(target.key)
            self._audio_trace(f"##rowtrace", ring.values if ring else
                              np.zeros(2, dtype=np.float32),
                              SIGNAL_COLORS[bound[0].signal], 56, 16)
        else:
            imgui.dummy(imgui.ImVec2(56, 16))

        if is_deaf:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(0.85, 0.65, 0.25, 1.0), "swept")
            hints.tip(
                "A swept parameter ignores its slider value, so audio cannot "
                "move it. Zero the sweeps to use this row.")

        for signal in SIGNAL_NAMES:
            imgui.same_line()
            existing = next((m for m in bound if m.signal == signal), None)
            colour = imgui.ImVec4(*SIGNAL_COLORS[signal])
            if existing is None:
                colour = imgui.ImVec4(colour.x, colour.y, colour.z, 0.30)
            imgui.push_style_color(imgui.Col_.button, colour)
            if imgui.button(f"{SIGNAL_ABBR[signal]}##{signal}"):
                # A dot toggles its mapping and never opens or closes a drawer.
                if existing is None:
                    mappings.append(Mapping(signal=signal, target=target.key))
                else:
                    mappings.remove(existing)
            imgui.pop_style_color()

        if open_here and bound:
            self._render_audio_drawer(ast, mappings, target, bound, overlay)
        imgui.pop_id()

    # --- the drawer -----------------------------------------------------

    def _render_audio_drawer(self, ast, mappings, target, bound, overlay):
        imgui.indent()
        if imgui.begin_tab_bar("##audio_drawer"):
            for m in sorted(bound, key=lambda x: SIGNAL_NAMES.index(x.signal)):
                if imgui.begin_tab_item(f"{SIGNAL_ABBR[m.signal]}##tab_{m.signal}")[0]:
                    ast.open_band = m.signal
                    self._render_audio_band_tab(m)
                    imgui.end_tab_item()
            if len(bound) > 1 and imgui.begin_tab_item("Total##tab_total")[0]:
                ast.open_band = ""
                self._render_audio_total_tab(target, bound, overlay)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.unindent()

    def _render_audio_band_tab(self, m):
        imgui.text_colored(imgui.ImVec4(*SIGNAL_COLORS[m.signal]), m.signal)
        imgui.same_line()
        if imgui.button(f"{m.mode}##mode"):
            m.mode = MODES[(MODES.index(m.mode) + 1) % len(MODES)]
        hints.tip("Cycles add, subtract and multiply.")
        imgui.same_line()
        changed, value = imgui.checkbox("On##enabled", m.enabled)
        if changed:
            m.enabled = value

        blank = Mapping(signal=m.signal, target=m.target)
        changed, value = self._audio_slider("Depth", m.depth, 0.0, 1.0, "%.2f",
                                            blank.depth)
        if changed:
            m.depth = value
        hints.tip("How far this band can move the parameter.")
        changed, value = self._audio_slider("Gain", m.gain, 0.0, 4.0, "%.2f",
                                            blank.gain)
        if changed:
            m.gain = value
        hints.tip("Amplifies the band before it is used.")

        kind_idx = (SHAPER_KINDS.index(m.shaper.kind)
                    if m.shaper.kind in SHAPER_KINDS else 0)
        changed, idx = imgui.combo("Shaper", kind_idx, list(SHAPER_KINDS))
        if changed:
            m.shaper.kind = SHAPER_KINDS[idx]
        hints.tip("Reshapes the band before it drives anything.")

        for fname in SHAPER_FIELDS.get(m.shaper.kind, ()):
            if fname == "wave":
                w = WAVE_KINDS.index(m.shaper.wave) if m.shaper.wave in WAVE_KINDS else 0
                changed, widx = imgui.combo("Wave", w, list(WAVE_KINDS))
                if changed:
                    m.shaper.wave = WAVE_KINDS[widx]
                continue
            lo, hi, fmt = _FIELD_RANGE[fname]
            changed, value = self._audio_slider(
                fname.replace("_", " ").title(),
                float(getattr(m.shaper, fname)), lo, hi, fmt,
                float(getattr(ShaperParams(), fname)),
                log=fname in _FIELD_LOG)
            if changed:
                setattr(m.shaper, fname, value)
            if fname in _FIELD_TIP:
                hints.tip(_FIELD_TIP[fname])

        # The raw band faint, and what the shaper makes of it bright over the
        # top - the difference between the two IS the shaper's effect, which is
        # the only reason this drawer is open.
        raw = self._audio_rings().get(m.signal)
        out = self._audio_shaped_rings().get(m.uid)
        colour = SIGNAL_COLORS[m.signal]
        width = imgui.get_content_region_avail().x
        origin = imgui.get_cursor_screen_pos()
        if raw is not None:
            self._audio_trace(f"##band_{m.signal}", raw.values,
                              (*colour[:3], 0.35), width, 42)
        if out is not None:
            if raw is not None:
                imgui.set_cursor_screen_pos(origin)
                imgui.push_style_color(imgui.Col_.frame_bg,
                                       imgui.ImVec4(0, 0, 0, 0))
            self._audio_trace("##shaped", out.values, colour, width, 42)
            if raw is not None:
                imgui.pop_style_color()
        elif m.shaper.kind != "none":
            imgui.text_disabled("Start audio to see the shaper's output.")

    def _render_audio_total_tab(self, target, bound, overlay):
        if overlay is None:
            imgui.text_disabled("Start audio to see this parameter move.")
            return

        base, live = overlay["base"], overlay["live"]
        delta = live - base
        imgui.text(f"{live:.4f}")
        imgui.same_line()
        imgui.text_disabled(f"base {base:.4f}")
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(0.5, 0.8, 0.4, 1.0) if delta >= 0
            else imgui.ImVec4(0.8, 0.45, 0.35, 1.0),
            f"{delta:+.4f}")

        # Range bar: where the live value sits between the slider's own rails.
        lo, hi = overlay["lo"], overlay["hi"]
        span = hi - lo
        frac = (live - lo) / span if span > 0 else 0.0
        imgui.progress_bar(min(1.0, max(0.0, frac)), imgui.ImVec2(-1, 6), "")
        imgui.text_disabled(f"{lo:.3g}")
        imgui.same_line()
        imgui.text_disabled(f"{hi:.3g}")

        # Stacked traces: each contributing band faint, the parameter bright.
        # plot_lines draws one series per canvas, so the later ones are drawn
        # over the first by rewinding the cursor and clearing their frame - one
        # numpy buffer still crosses into C++ once per series, which is the
        # property that matters, and it needs no second global context.
        rings = self._audio_rings()
        series = [(m.signal, rings[m.signal].values,
                   (*SIGNAL_COLORS[m.signal][:3], 0.5))
                  for m in bound if m.signal in rings]
        tring = self._audio_target_rings().get(target.key)
        if tring is not None:
            series.append((target.key, tring.values, (0.95, 0.95, 1.0, 1.0)))
        if not series:
            return

        origin = imgui.get_cursor_screen_pos()
        size = imgui.ImVec2(imgui.get_content_region_avail().x, 80)
        for i, (name, values, colour) in enumerate(series):
            if i:
                imgui.set_cursor_screen_pos(origin)
                imgui.push_style_color(imgui.Col_.frame_bg,
                                       imgui.ImVec4(0, 0, 0, 0))
            self._audio_trace(f"##total_{name}", values, colour,
                              size.x, size.y)
            if i:
                imgui.pop_style_color()

        for name, _v, colour in series:
            imgui.text_colored(imgui.ImVec4(*colour), name[:4])
            imgui.same_line()
        imgui.new_line()
