"""The Audio Reactive panel: source, spectrum, band traces, mapping matrix.

Named audio_reactive_window, not audio_window, because the sonification feature
owns that name for the MIDI-out side. Two mixins with one method name would
shadow each other in the UI's MRO without raising.

Traces go through imgui.plot_lines over a float32 array - one crossing into C++
per trace rather than one per segment.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services import audio_capture
from services.audio_analysis import SIGNAL_NAMES
from services.audio_mapping import (MODES, Mapping, brain_targets,
                                    deaf_targets, physics_targets)
from ui import layout

SIGNAL_COLORS: dict[str, tuple] = {
    "bass": (0.88, 0.31, 0.38, 1.0),
    "mid": (0.88, 0.56, 0.13, 1.0),
    "presence": (0.50, 0.82, 0.38, 1.0),
    "hi": (0.25, 0.63, 0.88, 1.0),
    "volume": (0.71, 0.55, 0.94, 1.0),
}

SIGNAL_ABBR: dict[str, str] = {
    "bass": "B", "mid": "M", "presence": "P", "hi": "H", "volume": "V",
}

TRACE_LEN = 180          # capped near the trace's width in pixels


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

    def _audio_rings(self) -> dict:
        if not hasattr(self, "_audio_trace_rings"):
            self._audio_trace_rings = {n: TraceRing() for n in SIGNAL_NAMES}
        return self._audio_trace_rings

    def _audio_devices(self, refresh: bool = False) -> list:
        if refresh or not hasattr(self, "_audio_device_cache"):
            self._audio_device_cache = audio_capture.list_devices()
        return self._audio_device_cache

    def render_audio_reactive_window(self):
        ast = self.state.audio
        if not ast.show_window:
            return

        imgui.set_next_window_size(imgui.ImVec2(420, 640),
                                   imgui.Cond_.first_use_ever)
        layout.constrain_panel()
        _expanded, opened = imgui.begin("Audio Reactive", True)
        if not opened:
            ast.show_window = False
            imgui.end()
            return

        # push_settings_width pairs with imgui.pop_item_width, not a layout call.
        layout.push_settings_width()
        self._render_audio_source(ast)
        imgui.separator()
        self._render_audio_signals(ast)
        imgui.separator()
        self._render_audio_matrix(ast)
        imgui.pop_item_width()
        imgui.end()

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
        self._delayed_tooltip("Which input the bands are read from.")

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

        changed, value = imgui.checkbox("Auto Gain", ast.auto_gain)
        if changed:
            ast.auto_gain = value
        self._delayed_tooltip("Normalises each band against its recent peak.")

        changed, value = imgui.slider_float("Strength", ast.global_strength,
                                            0.0, 2.0, "%.2f")
        if changed:
            ast.global_strength = value
        self._delayed_tooltip("Scales every mapping at once.")

    # --- spectrum and band traces ---------------------------------------

    def _render_audio_signals(self, ast):
        runtime = getattr(self, "audio_runtime", None)
        snap = runtime.capture.snapshot() if runtime is not None else None
        rings = self._audio_rings()
        if snap is not None:
            for name in SIGNAL_NAMES:
                rings[name].push(snap.signals.get(name, 0.0))
            imgui.plot_lines("##audio_spectrum",
                             np.asarray(snap.mel, dtype=np.float32),
                             graph_size=imgui.ImVec2(0, 46))

        width = max(40.0,
                    imgui.get_content_region_avail().x / len(SIGNAL_NAMES) - 4)
        for i, name in enumerate(SIGNAL_NAMES):
            if i:
                imgui.same_line()
            imgui.begin_group()
            imgui.push_style_color(imgui.Col_.plot_lines,
                                   imgui.ImVec4(*SIGNAL_COLORS[name]))
            imgui.plot_lines(f"##audio_trace_{name}", rings[name].values,
                             scale_min=0.0, scale_max=1.0,
                             graph_size=imgui.ImVec2(width, 26))
            imgui.pop_style_color()
            imgui.text_colored(imgui.ImVec4(*SIGNAL_COLORS[name]), name[:4])
            imgui.end_group()

    # --- the matrix -----------------------------------------------------

    def _mapping_list(self, ast, group: str) -> list:
        if group == "physics":
            return ast.mappings
        return ast.brain_mappings.setdefault(self.state.brain.modality, [])

    def _render_audio_matrix(self, ast):
        deaf = deaf_targets(self.state.sim)
        groups = [("physics", "Physics", physics_targets(self.state.sim))]

        from services import brains
        modality = brains.get(self.state.brain.modality)
        b_layout = getattr(getattr(self, "sim", None), "brain_layout", None)
        if b_layout is None:
            b_layout = modality.layout_from_settings(self.state.brain.settings)
        groups.append(("brain", "Brain", brain_targets(modality, b_layout)))

        for group, title, targets in groups:
            if not imgui.collapsing_header(f"{title}##audio_group"):
                continue
            if not targets:
                imgui.text_disabled("This brain has no scales to modulate.")
                continue
            mappings = self._mapping_list(ast, group)
            for target in targets:
                self._render_audio_row(ast, group, mappings, target,
                                       target.key in deaf)

    def _render_audio_row(self, ast, group, mappings, target, is_deaf):
        bound = [m for m in mappings if m.target == target.key]
        imgui.push_id(f"{group}:{target.key}")

        if bound:
            imgui.text(target.label)
        else:
            imgui.text_disabled(target.label)
        if is_deaf:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(0.85, 0.65, 0.25, 1.0), "swept")
            self._delayed_tooltip(
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
                if existing is None:
                    mappings.append(Mapping(signal=signal, target=target.key))
                else:
                    mappings.remove(existing)
            imgui.pop_style_color()

        for m in bound:
            self._render_audio_mapping(ast, mappings, m)
        imgui.pop_id()

    def _render_audio_mapping(self, ast, mappings, m):
        imgui.push_id(m.signal)
        imgui.indent()
        imgui.text_colored(imgui.ImVec4(*SIGNAL_COLORS[m.signal]), m.signal)

        imgui.same_line()
        if imgui.button(m.mode):
            m.mode = MODES[(MODES.index(m.mode) + 1) % len(MODES)]
        self._delayed_tooltip("Cycles add, subtract and multiply.")

        changed, value = imgui.slider_float("Depth", m.depth, 0.0, 1.0, "%.2f")
        if changed:
            m.depth = value
        changed, value = imgui.slider_float("Gain", m.gain, 0.0, 4.0, "%.2f")
        if changed:
            m.gain = value

        imgui.unindent()
        imgui.pop_id()
