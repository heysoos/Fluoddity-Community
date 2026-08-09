"""Brain modality selection and per-modality settings.

Passive, like every other UI mixin: it renders widgets and exposes state. The
orchestrator reads `request_layout_change` and performs the switch, because
applying a layout tears down and rebuilds the archive.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services.brains import REGISTRY, get

# tanh(2.65) ~ 0.99, so |z| beyond this decodes to within 1% of its rail: the
# parameter has stopped responding to the optimizer.
SATURATION_Z = 2.65

# Settings that change the PARAMETER COUNT, and so the genome's meaning. These
# reset the optimizer and switch archive; everything else only moves a squash
# and is free to change mid-run.
_COUNT_KEYS = ("centers", "filters", "bumps", "hidden", "activation")


def saturation_fraction(z) -> float:
    """Fraction of search coordinates within 1% of their decoded limit.

    Worth showing because nothing else on screen reveals it: measured over five
    runs, evolved genomes drift from ~0% to 10% hard-saturated while the
    reported sigma barely moves. A saturated coordinate is one the search can no
    longer move, so a rising number means the run is quietly losing dimensions.
    """
    if z is None:
        return 0.0
    a = np.abs(np.asarray(z, dtype=np.float32).reshape(-1))
    return float(np.mean(a >= SATURATION_Z)) if a.size else 0.0


def layout_change_needed(old: dict, new: dict) -> bool:
    """True when a setting that changes the parameter count was edited."""
    return any(k in new and old.get(k) != new[k] for k in _COUNT_KEYS)


def layout_for(modality_name: str, settings: dict):
    """The layout for a modality and its settings, never raising.

    An unknown modality or a stale setting comes from a config written by a
    different build, and the app has to keep running - get() already falls back
    to Fourier, and a modality ignores settings it does not have.
    """
    modality = get(modality_name)
    try:
        return modality.layout_from_settings(dict(settings or {}))
    except (TypeError, ValueError, KeyError):
        return modality.layout_from_settings({})


class BrainWindowMixin:
    """Renders the Brain window. Combined into UI via multiple inheritance."""

    def render_brain_window(self) -> None:
        state = self.state.brain
        if not state.enabled:
            return

        expanded, opened = imgui.begin("Brain - EXPERIMENTAL", True)
        if not opened:
            state.enabled = False
            imgui.end()
            return
        if not expanded:
            imgui.end()
            return

        names = sorted(REGISTRY, key=lambda n: REGISTRY[n].modality_id)
        cur = state.modality if state.modality in names else names[0]
        changed, idx = imgui.combo("Modality", names.index(cur), names)
        if changed and names[idx] != state.modality:
            # A different modality is always a different genome: drop the old
            # settings rather than carrying "centers" into a brain with none.
            self._pending_brain = (names[idx], {})
            imgui.open_popup("Change brain layout?")

        modality = get(state.modality)
        pending = dict(state.settings)
        for s in modality.settings_schema():
            value = pending.get(s.key, s.default)
            if s.kind == "int":
                ch, v = imgui.slider_int(s.label, int(value), int(s.lo), int(s.hi))
            elif s.kind == "choice":
                ch, v = imgui.combo(s.label, int(value), list(s.choices))
            else:
                ch, v = imgui.slider_float(s.label, float(value), s.lo, s.hi)
            if ch:
                pending[s.key] = v

        if pending != state.settings:
            if layout_change_needed(state.settings, pending):
                self._pending_brain = (state.modality, pending)
                imgui.open_popup("Change brain layout?")
            else:
                state.settings = pending      # a squash change costs nothing

        self._render_brain_layout_popup()

        layout = layout_for(state.modality, state.settings)
        imgui.separator()
        imgui.text(f"search dim: {layout.length}")
        imgui.text(f"archive: {layout.signature()} "
                   f"({state.archive_entries} entries)")

        frac = saturation_fraction(getattr(self, "brain_best_z", None))
        imgui.text(f"saturated params: {frac:.0%}")
        imgui.progress_bar(frac, (-1.0, 0.0))
        if frac >= 0.10:
            imgui.text_colored((1.0, 0.7, 0.2, 1.0),
                               "the search is losing dimensions")

        self._render_brain_inspector(state, layout)
        imgui.end()

    def _render_brain_inspector(self, state, layout) -> None:
        """The response field of each unit, and of the whole brain.

        The texture is rendered by the orchestrator from the SAME GLSL the
        particles run - this only places it.
        """
        from services.brain_preview import AXES, CHANNELS

        if not imgui.collapsing_header("Inspector")[0]:
            return

        ch, v = imgui.combo("Slice", state.preview_axes, [a[0] for a in AXES])
        if ch:
            state.preview_axes = v
        ch, v = imgui.combo("Output", state.preview_channel, list(CHANNELS))
        if ch:
            state.preview_channel = v
        ch, v = imgui.slider_float("Input Range", state.preview_range, 0.25, 8.0)
        if ch:
            state.preview_range = v
        ch, v = imgui.slider_float("Contrast", state.preview_gain, 0.05, 8.0)
        if ch:
            state.preview_gain = v

        preview = getattr(self, "brain_preview", None)
        tex = getattr(self, "brain_preview_tex", None)
        if preview is None or tex is None:
            imgui.text_disabled("preview unavailable")
            return

        n = preview.unit_count(layout)
        imgui.text(f"whole brain, then {n} units "
                   f"(blue negative, orange positive)")
        avail = max(imgui.get_content_region_avail().x, 64.0)
        per_row = max(1, min(preview.grid, int(avail // 74)))
        size = min(72.0, (avail - 8.0 * per_row) / per_row)
        for slot in range(n + 1):
            if slot % per_row:
                imgui.same_line()
            (u0, v0), (u1, v1) = preview.uv_for(slot)
            imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(size, size),
                        imgui.ImVec2(u0, v0), imgui.ImVec2(u1, v1))
            if imgui.is_item_hovered():
                imgui.set_tooltip("whole brain" if slot == 0
                                  else f"unit {slot - 1}")

    def _render_brain_layout_popup(self) -> None:
        """Changing the parameter count is not undoable in place - it resets the
        optimizer and moves to a different archive - so it is confirmed."""
        if not imgui.begin_popup_modal("Change brain layout?")[0]:
            return
        name, settings = getattr(self, "_pending_brain", (None, None))
        if name is not None:
            layout = layout_for(name, settings)
            imgui.text("This changes the genome layout.")
            imgui.text("The optimizer resets and searching moves to")
            imgui.text(f"a separate archive: {layout.signature()}")
            imgui.separator()
            imgui.text(f"search dim {layout.length}")
        if imgui.button("Apply"):
            if name is not None:
                self.state.brain.modality = name
                self.state.brain.settings = dict(settings)
                self.state.brain.request_layout_change = True
            self._pending_brain = (None, None)
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel"):
            self._pending_brain = (None, None)
            imgui.close_current_popup()
        imgui.end_popup()
