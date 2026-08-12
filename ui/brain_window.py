"""Brain modality selection and per-modality settings.

Passive, like every other UI mixin: it renders widgets and exposes state. The
orchestrator reads `request_layout_change` and performs the switch, because
applying a layout tears down and rebuilds the archive.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services.brains import MAX_BRAIN_FLOATS, REGISTRY, get, settings_of
from ui import layout as layout_helpers

# tanh(2.65) ~ 0.99, so |z| beyond this decodes to within 1% of its rail: the
# parameter has stopped responding to the optimizer.
SATURATION_Z = 2.65


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


def layout_or_none(modality_name: str, settings: dict):
    """The layout these settings name, or None when they do not build one.

    Over budget is the reachable case: BrainLayout refuses anything wider than
    MAX_BRAIN_FLOATS, and the layer editor derives its limits from that refusal
    rather than repeating the packing formula a third time.
    """
    try:
        return get(modality_name).layout_from_settings(dict(settings or {}))
    except (TypeError, ValueError, KeyError):
        return None


def layout_for(modality_name: str, settings: dict):
    """The layout for a modality and its settings, never raising.

    An unknown modality or a stale setting comes from a config written by a
    different build, and the app has to keep running - get() already falls back
    to Fourier, and a modality ignores settings it does not have.
    """
    return (layout_or_none(modality_name, settings)
            or get(modality_name).layout_from_settings({}))


def layout_change_needed(modality_name: str, old: dict, new: dict) -> bool:
    """True when an edit changes the parameter count, and so the genome's
    meaning - it resets the optimizer and switches archive.

    Asked of the LAYOUTS rather than of a list of setting keys. BrainLayout
    leaves the decode scales out of its equality, so a scale move reads as no
    change without anything here knowing which keys are scales, and a modality
    that renames or replaces a setting cannot leave this stale - which is
    exactly what the hand-written list it replaces did.
    """
    return layout_for(modality_name, old) != layout_for(modality_name, new)


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

        self._render_source_selector(state)

        modality = get(state.modality)
        settings = dict(state.settings)
        for s in modality.settings_schema():
            if s.kind == "layers":
                self._render_layer_rows(state, settings, s)
                continue
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
                if layout_change_needed(state.modality,
                                        {s.key: committed}, {s.key: shown}):
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

        if state.best_z is None:
            imgui.text_disabled("saturated params: no search yet")
        else:
            frac = saturation_fraction(state.best_z)
            imgui.text(f"saturated params: {frac:.0%}")
            imgui.progress_bar(frac, imgui.ImVec2(-1.0, 0.0))
            if frac >= 0.10:
                imgui.text_colored(imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
                                   "the search is losing dimensions")

        self._render_brain_inspector(state, layout)
        imgui.end()

    # ---- which brain ------------------------------------------------------

    @staticmethod
    def source_kind(state) -> str:
        """"rule", "cohort" or "tile" - which brain the window is pointing at.

        Decided by the sim, not by the user, because only one of the three
        exists at a time: a loaded rule is what EVERY particle reads, no rule
        loaded means one generated brain per cohort, and a tournament means one
        per tile. The tile case wins, because a rule-less startup leaves the
        per-cohort flag set underneath it.
        """
        if state.preview_tile0:
            return "tile"
        return "cohort" if state.preview_per_cohort else "rule"

    def _render_source_selector(self, state) -> None:
        kind = self.source_kind(state)
        n = max(1, int(state.source_count))
        if kind == "rule":
            state.source_index = 0
            imgui.text("Source    Loaded rule")
            return

        word = "Cohort" if kind == "cohort" else "Tile"
        i = min(max(int(state.source_index), 0), n - 1)
        ch, v = imgui.combo("Source", i, [f"{word} {k}" for k in range(n)])
        if ch:
            state.source_index = int(v)
        else:
            state.source_index = i

        if kind == "tile":
            # The grid owns these slots and rewrites them every generation, so
            # there is nothing an edit could survive.
            imgui.text_disabled("   the tournament owns these; read only")
            return
        imgui.same_line()
        if imgui.button("Adopt as loaded rule"):
            state.adopt_requested = True
        imgui.text_disabled("   edits show at once; File > Save writes the "
                            "loaded rule, so adopt to keep one")

    def _source_locked(self, state) -> str:
        """Why layer operations are refused right now, or "" if they are not."""
        if self.source_kind(state) == "tile":
            return "the tournament owns this brain"
        if state.borrow_active:
            # Slot 0 holds someone else's brain for as long as the pointer sits
            # on a menu item, and the commit path would keep the edit.
            return "a preview is borrowing this brain"
        return ""

    # ---- the layer stack ---------------------------------------------------

    def _render_layer_rows(self, state, settings, s) -> None:
        """One row per hidden layer: width, activation, remove.

        Two kinds of edit live on the same row and only one of them is free.
        A layer's COUNT, WIDTH or ACTIVATION is a layout edit - it resets the
        search and switches archive - so the row carries that warning. Its
        weights are a genome edit and carry none.
        """
        # Read back off the LAYOUT, so a config that names the pre-stack
        # `hidden`/`activation` arrives here already as one row.
        layers = [[int(w), int(a)] for w, a
                  in settings_of(layout_for(state.modality, settings)
                                 ).get(s.key, [])]
        committed = [list(p) for p in layers]    # what is on the GPU right now
        acts = list(s.choices)
        drafts: dict[int, int] = {}              # widths a slider is still held on
        row_hovered: dict[int, bool] = {}
        removed = None

        imgui.text("Layers")
        style = imgui.get_style()
        combo_w = imgui.calc_text_size(max(acts, key=len) if acts
                                       else "tanh").x + 40.0
        room = combo_w + layout_helpers.button_width("x") + style.item_spacing.x * 2.0
        for i, (w, a) in enumerate(layers):
            imgui.text(f"L{i + 1}")
            imgui.same_line()
            # A width drag fires once per frame and a count change rebuilds the
            # archive, so it is held in a draft and committed on release - the
            # same deferral the other count sliders use.
            key = (state.modality, s.key, i)
            held = self._brain_draft.get(key, w)
            imgui.push_item_width(-room)
            ch, v = imgui.slider_int(f"##w{i}", int(held),
                                     int(s.lo), self._max_width(
                                         state, settings, s, layers, i))
            imgui.pop_item_width()
            # Captured HERE, because the draft commit below asks the same
            # question of the last item and the popup would answer for it.
            row_hovered[i] = imgui.is_item_hovered()
            if ch:
                self._brain_draft[key] = v
            drafts[i] = int(self._brain_draft.get(key, w))
            if key in self._brain_draft and not imgui.is_item_active():
                layers[i][0] = self._brain_draft.pop(key)
            imgui.same_line()
            imgui.push_item_width(combo_w)
            # One click, one layout change - the same immediate commit the
            # activation combo has always had.
            ch, v = imgui.combo(f"##a{i}", int(a), acts)
            imgui.pop_item_width()
            if ch:
                layers[i][1] = int(v)
            imgui.same_line()
            # The last layer cannot go: a stack of none is not a brain.
            imgui.begin_disabled(len(layers) <= 1)
            if imgui.small_button(f"x##rm{i}"):
                removed = i
            imgui.end_disabled()
            if imgui.is_item_hovered():
                imgui.set_tooltip("remove this layer")
            self._render_layer_menu(state, i, layers[i], row_hovered[i])

        if removed is not None:
            layers.pop(removed)
            self._brain_draft.clear()   # the drafts below it now name other rows

        added = self._add_layer_width(state, settings, s, layers)
        imgui.begin_disabled(added is None)
        if imgui.button("+ Add layer") and added is not None:
            layers.append([added, 0])
        imgui.end_disabled()
        if added is None and imgui.is_item_hovered():
            imgui.set_tooltip("no room for another layer")
        # Stated because it is not guessable from the sliders: a second layer
        # narrows every layer, this one included.
        deep = self._max_width(state, settings, s, [[1, 0], [1, 0]], 0)
        if len(layers) == 1 and layers[0][0] > deep:
            imgui.text_disabled(f"   a second layer narrows every layer "
                                f"to {deep}")

        if layers != committed:
            # The BUILT stack, not the asked-for one: adding a second layer
            # narrows every layer to what the deep path can carry, and settings
            # that disagreed with the layout would put the old width back the
            # next time a row was removed.
            settings[s.key] = self._built(state, settings, s, layers) or layers
            # hidden/activation named a one-layer stack before this existed.
            # Dropped once the stack is edited, so the file cannot carry two
            # answers - layout_from_settings still ACCEPTS them, forever.
            settings.pop("hidden", None)
            settings.pop("activation", None)

        shown = [list(p) for p in layers]
        for i, w in drafts.items():
            if i < len(shown):
                shown[i][0] = w
        self._render_budget(state, settings, s, layers, shown)

    def _render_layer_menu(self, state, i, layer, hovered) -> None:
        """Right-click a layer: reroll, rescale or redistribute its WEIGHTS.

        A genome edit, not a layout edit - it is free, it is undoable with Z,
        and it touches nothing structural. That is why this menu carries none of
        the warning the row's own controls do.
        """
        pid = f"layer_ctx_{i}"
        if hovered and imgui.is_mouse_clicked(1):
            imgui.open_popup(pid)
        if not imgui.begin_popup(pid):
            if self._layer_popup == i:
                # Closed: one history entry for the whole drag, not one a frame.
                self._layer_popup = None
                state.layer_op = (i, "scale_end", None)
            return
        if self._layer_popup != i:
            self._layer_popup = i
            self._layer_scale = 1.0
            state.layer_op = (i, "scale_begin", None)

        # The physics panel's auto-close, so a menu left behind by the pointer
        # goes away on its own.
        pos, size = imgui.get_window_pos(), imgui.get_window_size()
        mouse = imgui.get_mouse_pos()
        dx = max(pos.x - mouse.x, 0.0, mouse.x - (pos.x + size.x))
        dy = max(pos.y - mouse.y, 0.0, mouse.y - (pos.y + size.y))
        if ((dx * dx + dy * dy) ** 0.5
                > self.state.preferences.menu_close_threshold):
            imgui.close_current_popup()

        imgui.text(f"Layer {i + 1} - {int(layer[0])} units")
        locked = self._source_locked(state)
        if locked:
            imgui.text_disabled(locked)
        imgui.begin_disabled(bool(locked))

        dists = list(getattr(get(state.modality), "distributions", ()))
        cur = int(state.layer_dist.get(i, 0)) if dists else 0
        imgui.push_item_width(imgui.calc_text_size("heavy-tail").x + 48.0)
        if dists:
            ch, v = imgui.combo("Distribution", min(cur, len(dists) - 1), dists)
            if ch:
                state.layer_dist[i] = int(v)
                cur = int(v)
        # Relative to a snapshot taken when the popup opened. Applied per frame
        # it would compound over a drag and the layer would explode.
        ch, v = imgui.slider_float("Scale", float(self._layer_scale), 0.0, 4.0)
        imgui.pop_item_width()
        if ch:
            self._layer_scale = float(v)
            state.layer_op = (i, "scale", float(v))
        imgui.separator()
        for label, op in (("Reroll weights", "reroll_weights"),
                          ("Reroll biases", "reroll_biases")):
            if imgui.button(label):
                state.layer_op = (i, op, cur)
        imgui.separator()
        if imgui.button("Reset layer"):
            state.layer_op = (i, "reset", None)
        if imgui.is_item_hovered():
            imgui.set_tooltip("back to how this layer was when the menu opened")
        imgui.end_disabled()
        imgui.end_popup()

    @property
    def _layer_popup(self):
        return getattr(self, "_layer_popup_index", None)

    @_layer_popup.setter
    def _layer_popup(self, value):
        self._layer_popup_index = value

    @property
    def _layer_scale(self) -> float:
        return float(getattr(self, "_layer_scale_value", 1.0))

    @_layer_scale.setter
    def _layer_scale(self, value) -> None:
        self._layer_scale_value = float(value)

    def _render_budget(self, state, settings, s, layers, shown) -> None:
        length = layout_for(state.modality, settings).length
        imgui.text(f"{length} / {MAX_BRAIN_FLOATS} floats")
        imgui.progress_bar(min(length / float(MAX_BRAIN_FLOATS), 1.0),
                           imgui.ImVec2(-1.0, 0.0))
        # Only while a held slider actually differs from what is committed, so
        # the consequence is stated as it is being chosen and never nags.
        if layout_change_needed(state.modality, {**settings, s.key: layers},
                                {**settings, s.key: shown}):
            imgui.text_disabled("   on release: resets the search, "
                                "switches archive")

    @staticmethod
    def _widest(fits, lo: int, hi: int):
        """The largest value in [lo, hi] that `fits`, or None if none does.

        Bisected on whether the layout BUILDS, because an over-budget layout has
        to be unconstructable rather than merely refused after the fact:
        layout_for() falls back to the defaults when it cannot build one, so an
        unclamped control would make the window snap back. Asking the layout
        also keeps the packing formula out of the UI - it has two homes already.
        """
        if fits(hi):
            return hi
        if not fits(lo):
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    @staticmethod
    def _built(state, settings, s, layers):
        """The stack these layers actually BUILD, or None if they build nothing.

        Not the same list back: a modality clamps a width or a depth it will not
        carry, and the clamped value is what the GPU runs. Asked through the
        modality's own settings_of, so the UI never learns how a stack is
        encoded.
        """
        got = layout_or_none(state.modality, {**settings, s.key: layers})
        return None if got is None else settings_of(got).get(s.key)

    def _max_width(self, state, settings, s, layers, i) -> int:
        """The widest layer i can be with its neighbours where they are.

        Clamping counts as not fitting. A slider whose range runs past what the
        modality will build looks stuck, which reads as a broken control rather
        than as a limit.
        """
        def fits(w):
            trial = [list(p) for p in layers]
            trial[i][0] = int(w)
            built = self._built(state, settings, s, trial)
            return built is not None and int(built[i][0]) == int(w)

        return self._widest(fits, int(s.lo), int(s.hi)) or int(s.lo)

    def _add_layer_width(self, state, settings, s, layers):
        """The width a new layer would get, or None when one will not fit.

        None covers every limit without naming any of them: over budget, and at
        the depth cap, where the extra layer is clamped away.
        """
        def fits(w):
            grown = self._built(state, settings, s, layers + [[int(w), 0]])
            return (grown is not None and len(grown) == len(layers) + 1
                    and int(grown[-1][0]) == int(w))

        return self._widest(fits, int(s.lo), int(s.default))

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
        if state.preview_tile0:
            # The grid owns slot 0, so this is ONE tile of however many are on
            # screen - not the grid, and not a cohort.
            imgui.text_disabled("tile 0 of the tournament grid")
        elif state.preview_per_cohort:
            # A statement, not a warning. Every modality does this when no rule
            # is loaded, and it is what makes a fresh canvas interesting.
            imgui.text_disabled("cohort 0 of an unsaved random brain per cohort")
        imgui.text(f"whole brain, then {n} units "
                   f"(blue negative, orange positive)")

        # A CHILD with a permanent scrollbar, and the tile layout derived from
        # the unit count rather than from the width available.
        #
        # Reading get_content_region_avail().x to choose the column count is a
        # feedback loop: the tiles decide the content height, the height decides
        # whether a scrollbar appears, and the scrollbar takes ~20px off the
        # width - which changes the column count, the height, and round again.
        # That is the grid visibly resizing every frame. Reserving the scrollbar
        # unconditionally makes the width constant, and a child of fixed height
        # keeps the parent's own scrollbar out of the same loop.
        SIZE, PAD = 72.0, 8.0
        per_row = max(1, min(preview.grid, 8))
        height = np.ceil((n + 1) / per_row) * (SIZE + PAD) + PAD
        avail_y = imgui.get_content_region_avail().y
        imgui.begin_child(
            "brain_units",
            imgui.ImVec2(0.0, float(min(height, max(avail_y, 120.0)))),
            imgui.ChildFlags_.none,
            imgui.WindowFlags_.always_vertical_scrollbar,
        )
        for slot in range(n + 1):
            if slot % per_row:
                imgui.same_line()
            (u0, v0), (u1, v1) = preview.uv_for(slot)
            imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(SIZE, SIZE),
                        imgui.ImVec2(u0, v0), imgui.ImVec2(u1, v1))
            if imgui.is_item_hovered():
                imgui.set_tooltip("whole brain" if slot == 0
                                  else f"unit {slot - 1}")
        imgui.end_child()

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
