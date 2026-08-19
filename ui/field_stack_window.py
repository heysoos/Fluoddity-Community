"""Inject Texture: the ordered list of layers feeding the sim's fields."""
from imgui_bundle import imgui

from services import field_sources, file_picker
from state.field_stack import BLENDS, DESTINATIONS, MAPPINGS, FieldLayer
from ui import hints, layout
from ui.notices import BAD

WINDOW_TITLE = "Inject Texture"

SCALES = (("Full", 1.0), ("Half", 0.5), ("Quarter", 0.25))
NO_FILE = "(none)"

# A GL texture has v=0 at the BOTTOM; imgui draws uv0 at the TOP-LEFT. Every
# preview here is a texture the bus rendered, so it needs the flip or it is a
# mirror image of what the particles are reading - which reads as the sim
# being upside down rather than the picture.
UV0 = imgui.ImVec2(0.0, 1.0)
UV1 = imgui.ImVec2(1.0, 0.0)

# The names on screen, and one sentence each. A key never reaches the user.
SOURCE_TIPS = {
    "noise":    "Animated fractal noise generated on the GPU.",
    "gradient": "A smooth ramp or radial falloff.",
    "image":    "A picture from disk, held still.",
    "shader":   "Your own fragment shader from a .frag file.",
    "brush":    "What you paint with the mouse, in Drawing Controls.",
    "feedback": "The simulation's own trail canvas, fed back in.",
}

MAPPING_LABELS = {
    "rg_direct": "Direct RG",
    "polar":     "Hue and value",
    "gradient":  "Gradient (uphill)",
    "curl":      "Curl (swirl)",
    "luminance": "Brightness",
}
MAPPING_TIPS = {
    "rg_direct": "Reads the red and green channels as the vector itself.",
    "polar":     "Reads hue as the direction and value as the strength.",
    "gradient":  "Points up or down the brightness slope, so particles gather "
                 "on peaks or flee them.",
    "curl":      "Points along the brightness contours, so particles circle "
                 "the bright regions instead of piling into them.",
    "luminance": "Uses brightness alone, with no direction.",
}

DESTINATION_LABELS = {"force": "Force", "strafe": "Strafe"}
DESTINATION_TIPS = {
    "force":  "Accelerates particles, so the effect builds up and carries.",
    "strafe": "Slides particles sideways each step, with no momentum.",
}

BLEND_LABELS = {"replace": "Replace", "add": "Add",
                "multiply": "Multiply", "max": "Max"}
BLEND_TIPS = {
    "replace":  "Discards whatever the layers beneath contributed.",
    "add":      "Sums with the layers beneath.",
    "multiply": "Scales the layers beneath, so it masks rather than adds.",
    "max":      "Keeps whichever of the two is stronger.",
}

# The sign control only means something where the mapping takes a derivative.
SIGNED_MAPPINGS = ("gradient", "curl")
SIGN_LABELS = ("Toward bright", "Away from bright")

VIEW_LABELS = ("Source texture", "After mapping", "Whole field")
VIEWS = ("source", "mapped", "destination")

STRENGTH_DEFAULT = 1.0
BLUR_DEFAULT = 0.0


def shader_options(current: str, names: list[str]) -> tuple[list[str], int]:
    """The shader combo's entries and which one is selected.

    A name that no longer resolves is OFFERED BACK, never replaced: silently
    adopting another file hides the missing one, and the layer's own error is
    the only place the loss is reported. The empty choice is listed for the
    same reason - a combo naming a file the layer was never pointed at is the
    same lie.
    """
    if current in names:
        return list(names), names.index(current)
    return [current or NO_FILE] + list(names), 0


class FieldStackWindowMixin:
    """Mixin for the injection layer list. Combined into UI."""

    def render_field_stack_window(self, collect=None):
        expanded, opened = imgui.begin(WINDOW_TITLE, True)
        if not opened:
            self.state.preferences.show_field_stack = False
            # Row previews cost nothing once nobody is looking at them.
            if getattr(self, "field_bus", None) is not None:
                self.field_bus.set_thumbnails_enabled(False)
            imgui.end()
            return
        if expanded:
            self._draw_field_stack(collect)
        imgui.end()

    # -- internals ------------------------------------------------------

    def _tag(self, text, collect):
        if collect is not None:
            collect.append(text)
        return text

    def _mark_dirty(self):
        if getattr(self, "field_bus", None) is not None:
            self.field_bus.mark_dirty()

    def _bus(self):
        return getattr(self, "field_bus", None)

    def _draw_field_stack(self, collect):
        stack = self.state.field_stack
        bus = self._bus()
        if bus is not None:
            bus.set_thumbnails_enabled(True)
        layout.push_settings_width()

        if imgui.button(self._tag("+ Add layer##fieldstack", collect)):
            stack.layers.append(FieldLayer())
            self._mark_dirty()
        hints.tip("Add a texture layer on top of the stack.")

        imgui.same_line()
        passes = bus.pass_count if bus is not None else 0
        res = bus.resolution if bus is not None else (0, 0)
        imgui.text_disabled(self._tag(
            f"{passes} passes | {res[0]}x{res[1]}", collect))

        if not stack.layers:
            imgui.text_disabled(self._tag(
                "No layers. Add one to push the particles around.", collect))

        imgui.separator()

        remove_index = None
        for index, layer in enumerate(stack.layers):
            if self._draw_layer_row(layer, collect):
                remove_index = index
        if remove_index is not None:
            stack.layers.pop(remove_index)
            self._mark_dirty()

        self._draw_inspect(stack, collect)
        self._draw_advanced(collect)

        imgui.pop_item_width()

    def _draw_advanced(self, collect) -> None:
        """The performance knob, out of the way of everything a user wants."""
        if not imgui.collapsing_header(self._tag("Advanced##fieldstack",
                                                 collect)):
            return
        prefs = self.state.preferences
        names = [n for n, _ in SCALES]
        current = min(range(len(SCALES)),
                      key=lambda i: abs(SCALES[i][1] - prefs.field_bus_scale))
        changed, current = imgui.combo(
            self._tag("Resolution##fieldbus", collect), current, names)
        if changed:
            prefs.field_bus_scale = SCALES[current][1]
            self._mark_dirty()
        hints.tip("How finely the layers are composited. Lower is faster and "
                  "softer.")

    # -- one layer ------------------------------------------------------

    def _summary(self, layer) -> str:
        source = field_sources.get(layer.source).label
        return (f"{source} -> {MAPPING_LABELS.get(layer.mapping, layer.mapping)}"
                f" -> {DESTINATION_LABELS.get(layer.destination, layer.destination)}")

    def _draw_layer_row(self, layer, collect) -> bool:
        uid = layer.uid
        changed_any = False

        changed, layer.enabled = imgui.checkbox(f"##en{uid}", layer.enabled)
        changed_any |= changed
        hints.tip("Turn this layer off without losing its settings.")
        imgui.same_line()

        self._draw_thumbnail(layer, collect)

        # Open by default: adding a layer and getting a collapsed line with
        # nothing to adjust is the same silent no-op as a source with no file.
        open_now = imgui.collapsing_header(
            self._tag(f"{self._summary(layer)}##hdr{uid}", collect),
            imgui.TreeNodeFlags_.default_open)
        if layer.error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BAD))
            imgui.text_wrapped(self._tag(layer.error, collect))
            imgui.pop_style_color()
        if not open_now:
            if changed_any:
                self._mark_dirty()
            return False

        imgui.push_id(uid)
        changed_any |= self._draw_layer_body(layer, collect)
        remove = imgui.button(self._tag(f"Remove##{uid}", collect))
        hints.tip("Delete this layer.")
        imgui.same_line()
        if imgui.button(self._tag(f"Inspect##{uid}", collect)):
            self._inspect_uid = uid
        hints.tip("Show this layer large, at each stage of the chain.")
        imgui.pop_id()
        imgui.separator()

        if changed_any:
            self._mark_dirty()
        return remove

    def _draw_layer_body(self, layer, collect) -> bool:
        uid = layer.uid
        changed_any = False

        keys = [d.key for d in field_sources.descriptors()]
        labels = [field_sources.get(k).label for k in keys]
        pos = keys.index(layer.source) if layer.source in keys else 0
        changed, pos = imgui.combo(
            self._tag(f"Source##src{uid}", collect), pos, labels)
        hints.tip(SOURCE_TIPS.get(layer.source, "Where this layer's picture "
                                                "comes from."))
        if changed:
            layer.source = keys[pos]
            layer.params = {}
            changed_any = True

        changed, layer.mapping = self._enum(
            "Mapping", layer.mapping, MAPPINGS, MAPPING_LABELS,
            f"map{uid}", collect,
            MAPPING_TIPS.get(layer.mapping, "How the picture becomes a vector."))
        changed_any |= changed

        changed, layer.destination = self._enum(
            "Destination", layer.destination, DESTINATIONS, DESTINATION_LABELS,
            f"dst{uid}", collect,
            DESTINATION_TIPS.get(layer.destination, "What this layer drives."))
        changed_any |= changed

        changed, layer.blend = self._enum(
            "Blend", layer.blend, BLENDS, BLEND_LABELS, f"bl{uid}", collect,
            BLEND_TIPS.get(layer.blend, "How this layer combines with the "
                                        "ones beneath it."))
        changed_any |= changed

        changed, layer.strength = self._slider(
            "Strength", f"{uid}s", layer.strength, 0.0, 4.0, STRENGTH_DEFAULT,
            collect, "How hard this layer pushes.")
        changed_any |= changed

        changed, layer.blur = self._slider(
            "Blur", f"{uid}b", layer.blur, 0.0, 6.0, BLUR_DEFAULT, collect,
            "Softens the picture before the mapping reads it.")
        changed_any |= changed

        # Only the derivative mappings have a direction to reverse; on the
        # others this control was live and did nothing.
        if layer.mapping in SIGNED_MAPPINGS:
            pos = 0 if layer.sign >= 0.0 else 1
            changed, pos = imgui.combo(
                self._tag(f"Direction##sign{uid}", collect), pos,
                list(SIGN_LABELS))
            hints.tip("Whether particles are drawn to the bright regions or "
                      "driven out of them.")
            if changed:
                layer.sign = 1.0 if pos == 0 else -1.0
                changed_any = True

        changed_any |= self._draw_file_picker(layer, collect)
        changed_any |= self._draw_layer_params(layer, collect)
        return changed_any

    # -- widgets --------------------------------------------------------

    def _slider(self, label, key, value, lo, hi, default, collect, tip):
        """A float slider that resets to `default` on right-click."""
        changed, value = imgui.slider_float(
            self._tag(f"{label}##{key}", collect), value, lo, hi)
        hints.tip(tip)
        if self._reset_menu(key, collect):
            return True, default
        return changed, value

    def _reset_menu(self, key, collect) -> bool:
        """Right-click menu on the last widget. -> True if Reset was clicked."""
        hit = False
        if imgui.begin_popup_context_item(f"##rst{key}"):
            if imgui.selectable(self._tag(f"Reset to default##rst{key}",
                                          collect), False)[0]:
                hit = True
            imgui.end_popup()
        return hit

    def _enum(self, label, current, options, labels, tag, collect, tip):
        pos = options.index(current) if current in options else 0
        changed, pos = imgui.combo(
            self._tag(f"{label}##{tag}", collect), pos,
            [labels.get(o, o) for o in options])
        hints.tip(tip)
        return changed, options[pos]

    def _draw_file_picker(self, layer, collect) -> bool:
        """`shader` and `image` are the two sources that name a file.

        Both read `params["_file"]`, so without this the source can be selected
        and never pointed at anything - the silent no-op the stack exists to
        remove.
        """
        if layer.source == "shader":
            names = field_sources.available_shader_files()
            current = layer.params.get("_file", "")
            if not names and not current:
                imgui.text_disabled(self._tag(
                    f"no .frag files found##nofrag{layer.uid}", collect))
            options, pos = shader_options(current, names)
            changed, pos = imgui.combo(
                self._tag(f"Shader##file{layer.uid}", collect), pos, options)
            hints.tip("Fragment shader this layer renders.")
            changed_any = False
            if changed:
                layer.params["_file"] = ("" if options[pos] == NO_FILE
                                         else options[pos])
                changed_any = True
            if file_picker.available():
                if imgui.button(self._tag(f"Browse...##frag{layer.uid}",
                                          collect)):
                    self._pending_pick = (layer.uid, file_picker.open_shader(
                        file_picker.folder_of(current)))
                hints.tip("Choose a .frag from anywhere on disk.")
            changed_any |= self._collect_pick(layer)
            return changed_any

        if layer.source == "image":
            return self._draw_image_picker(layer, collect)
        return False

    def _draw_image_picker(self, layer, collect) -> bool:
        """Browse... plus the typed path, which stays the way through.

        The dialog runs in another process, so it is asked for on one frame and
        collected on a later one - waiting on it here would freeze the sim
        behind the window.
        """
        changed_any = False
        current = layer.params.get("_file", "")

        changed, value = imgui.input_text(
            self._tag(f"Image##file{layer.uid}", collect), current)
        hints.tip("Picture this layer reads.")
        if changed:
            layer.params["_file"] = value
            changed_any = True

        if file_picker.available():
            if imgui.button(self._tag(f"Browse...##img{layer.uid}", collect)):
                self._pending_pick = (layer.uid, file_picker.open_image(
                    file_picker.folder_of(current)))
            hints.tip("Choose a picture from disk.")
        changed_any |= self._collect_pick(layer)
        return changed_any

    def _collect_pick(self, layer) -> bool:
        """Take the chosen path once the dialog closes. -> True if it changed."""
        pending = getattr(self, "_pending_pick", None)
        if pending is None or pending[0] != layer.uid:
            return False
        chosen = pending[1].result()
        if chosen is None:
            return False
        self._pending_pick = None
        if not chosen:
            return False
        layer.params["_file"] = chosen
        return True

    def _draw_thumbnail(self, layer, collect) -> None:
        """The row's preview: hover peeks larger, click opens Inspect."""
        bus = self._bus()
        thumb = bus.thumbnail_for(layer) if bus is not None else None
        if thumb is None:
            imgui.dummy(imgui.ImVec2(24, 24))
            imgui.same_line()
            return
        imgui.image(imgui.ImTextureRef(thumb.glo), imgui.ImVec2(24, 24),
                    UV0, UV1)
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.image(imgui.ImTextureRef(thumb.glo), imgui.ImVec2(256, 256),
                        UV0, UV1)
            imgui.end_tooltip()
        if imgui.is_item_clicked():
            self._inspect_uid = layer.uid
        imgui.same_line()

    def _draw_inspect(self, stack, collect) -> None:
        bus = self._bus()
        uid = getattr(self, "_inspect_uid", None)
        if bus is None or uid is None:
            return
        layer = next((l for l in stack.layers if l.uid == uid), None)
        if layer is None:
            self._inspect_uid = None
            return

        imgui.separator()
        view = getattr(self, "_inspect_view", 0)
        _, view = imgui.combo(self._tag("Inspect##fieldinspect", collect),
                              view, list(VIEW_LABELS))
        self._inspect_view = view
        hints.tip("Which stage of this layer to show: the picture, the vectors "
                  "it maps to, or the whole composited field.")
        tex = bus.inspect(layer, VIEWS[view])
        if tex is not None:
            imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(384, 384),
                        UV0, UV1)
        else:
            imgui.text_disabled(self._tag(
                "nothing to show##fieldinspect", collect))
        if imgui.button(self._tag("Close##fieldinspect", collect)):
            self._inspect_uid = None

    def _draw_layer_params(self, layer, collect) -> bool:
        changed_any = False
        for param in field_sources.params_for(layer):
            key = param.name
            default = (param.default[0] if param.components == 1
                       else list(param.default))
            value = layer.params.get(key, default)
            tag = self._tag(f"{param.label}##{layer.uid}{key}", collect)
            if param.kind == "bool":
                changed, value = imgui.checkbox(tag, bool(value))
            elif param.kind == "int":
                changed, value = imgui.slider_int(
                    tag, int(value), int(param.lo), int(param.hi))
            elif param.components == 1:
                changed, value = imgui.slider_float(
                    tag, float(value), param.lo, param.hi)
            elif param.kind == "color" and param.components == 3:
                changed, value = imgui.color_edit3(tag, list(value))
            else:
                changed, value = imgui.slider_float2(
                    tag, list(value)[:2], param.lo, param.hi)
            if self._reset_menu(f"{layer.uid}{key}", collect):
                changed, value = True, default
            if changed:
                layer.params[key] = value
                changed_any = True
        return changed_any
