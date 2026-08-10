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
            state.modality = names[idx]
            state.settings = {}
            state.request_layout_change = True
            self._brain_draft.clear()

        modality = get(state.modality)
        settings = dict(state.settings)
        for s in modality.settings_schema():
            # An int slider is the only widget here that can fire an expensive
            # change on every frame of a drag: it changes the parameter COUNT,
            # which rebuilds the archive and resets the search. So its value is
            # held in a draft while the mouse is down and committed once, on
            # release. Everything else only moves a decode scale - the light
            # path - and commits live, so the picture tracks the slider.
            key = (state.modality, s.key)
            shown = self._brain_draft.get(key, settings.get(s.key, s.default))
            if s.kind == "int":
                ch, v = imgui.slider_int(s.label, int(shown), int(s.lo), int(s.hi))
                if ch:
                    self._brain_draft[key] = v
                # Gated on "not being held" rather than on a one-shot
                # deactivated-after-edit event, so a draft can never get
                # stranded: whatever happens, the frame after the widget stops
                # being active commits it. A slider that silently never
                # commits is a worse failure than the modal this replaced.
                if key in self._brain_draft and not imgui.is_item_active():
                    settings[s.key] = self._brain_draft.pop(key)
                # Only while it actually differs from what is committed, so the
                # consequence is stated at the moment it is being chosen and
                # never nags. Not a prompt: nothing here waits on an answer.
                committed = settings.get(s.key, s.default)
                if layout_change_needed({s.key: committed}, {s.key: shown}):
                    imgui.text_disabled("   on release: resets the search, "
                                        "switches archive")
                continue
            if s.kind == "choice":
                ch, v = imgui.combo(s.label, int(shown), list(s.choices))
            else:
                ch, v = imgui.slider_float(s.label, float(shown), s.lo, s.hi)
            if ch:
                settings[s.key] = v

        if settings != state.settings:
            state.settings = settings

        layout = layout_for(state.modality, state.settings)
        imgui.separator()
        imgui.text(f"search dim: {layout.length}")
        imgui.text(f"archive: {layout.signature()} "
                   f"({state.archive_entries} entries)")

        frac = saturation_fraction(getattr(self, "brain_best_z", None))
        imgui.text(f"saturated params: {frac:.0%}")
        imgui.progress_bar(frac, imgui.ImVec2(-1.0, 0.0))
        if frac >= 0.10:
            imgui.text_colored(imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
                               "the search is losing dimensions")

        self._render_brain_inspector(state, layout)
        imgui.end()

    def _render_brain_inspector(self, state, layout) -> None:
        """The response field of each unit, and of the whole brain.

        The texture is rendered by the orchestrator from the SAME GLSL the
        particles run - this only places it.
        """
        from services.brain_preview import AXES, CHANNELS

        # One-arg collapsing_header returns a bare bool; only the p_visible
        # overload returns a tuple. Subscripting it crashed the app on open.
        if not imgui.collapsing_header("Inspector"):
            return

        ch, v = imgui.combo("Slice", state.preview_axes, [a[0] for a in AXES])
        if ch:
            state.preview_axes = v
        if AXES[min(state.preview_axes, len(AXES) - 1)][1] is None:
            # Only meaningful for the random projection; the four axis-aligned
            # slices are fixed planes and have nothing to reseed.
            if imgui.button("Reseed plane"):
                state.preview_seed += 1
            imgui.same_line()
            imgui.text_disabled(f"plane #{state.preview_seed}")
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

    @property
    def _brain_draft(self) -> dict:
        """In-progress int-slider values, keyed by (modality, setting).

        There used to be an Apply/Cancel modal here instead, because a count
        change rebuilds the archive and resets the search and a drag fires one
        per frame. Confirming every edit was the wrong answer to that - it
        interrupted scale sliders too, which are free. Deferring the commit to
        release fixes the actual problem and asks nothing of the user.
        """
        d = getattr(self, "_brain_draft_values", None)
        if d is None:
            d = {}
            self._brain_draft_values = d
        return d
