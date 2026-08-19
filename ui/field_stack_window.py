"""Field Stack window: the ordered list of injection layers."""
from imgui_bundle import imgui

from services import field_sources
from state.field_stack import BLENDS, DESTINATIONS, MAPPINGS, FieldLayer
from ui import hints, layout
from ui.notices import BAD

SCALES = (("1/1", 1.0), ("1/2", 0.5), ("1/4", 0.25))
NO_FILE = "(none)"


class FieldStackWindowMixin:
    """Mixin for the field injection layer list. Combined into UI."""

    def render_field_stack_window(self, collect=None):
        expanded, opened = imgui.begin("Field Stack", True)
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

    def _draw_field_stack(self, collect):
        prefs = self.state.preferences
        stack = self.state.field_stack
        if getattr(self, "field_bus", None) is not None:
            self.field_bus.set_thumbnails_enabled(True)
        layout.push_settings_width()

        names = [n for n, _ in SCALES]
        current = min(range(len(SCALES)),
                      key=lambda i: abs(SCALES[i][1] - prefs.field_bus_scale))
        changed, current = imgui.combo(
            self._tag("Bus Resolution##fieldbus", collect), current, names)
        if changed:
            prefs.field_bus_scale = SCALES[current][1]
            self._mark_dirty()
        hints.tip(
            "Resolution the injection layers are composited at, as a fraction "
            "of the canvas.")

        imgui.separator()

        remove_index = None
        for index, layer in enumerate(stack.layers):
            if self._draw_layer_row(layer, collect):
                remove_index = index
        if remove_index is not None:
            stack.layers.pop(remove_index)
            self._mark_dirty()

        if imgui.button(self._tag("+ Add layer##fieldstack", collect)):
            stack.layers.append(FieldLayer())
            self._mark_dirty()

        imgui.same_line()
        bus = getattr(self, "field_bus", None)
        passes = bus.pass_count if bus is not None else 0
        res = bus.resolution if bus is not None else (0, 0)
        imgui.text_disabled(self._tag(
            f"{passes} passes | {res[0]}x{res[1]}", collect))

        self._draw_inspect(stack, collect)

        imgui.pop_item_width()

    def _draw_layer_row(self, layer, collect) -> bool:
        uid = layer.uid
        changed_any = False

        self._draw_thumbnail(layer, collect)

        changed, layer.enabled = imgui.checkbox(f"##en{uid}", layer.enabled)
        changed_any |= changed
        imgui.same_line()

        keys = [d.key for d in field_sources.descriptors()]
        pos = keys.index(layer.source) if layer.source in keys else 0
        changed, pos = imgui.combo(
            self._tag(f"{layer.source}##src{uid}", collect), pos, keys)
        if changed:
            layer.source = keys[pos]
            layer.params = {}
            changed_any = True

        changed, layer.mapping = self._enum(layer.mapping, MAPPINGS,
                                            f"map{uid}", collect)
        changed_any |= changed
        changed, layer.destination = self._enum(layer.destination, DESTINATIONS,
                                                f"dst{uid}", collect)
        changed_any |= changed
        changed, layer.blend = self._enum(layer.blend, BLENDS,
                                          f"bl{uid}", collect)
        changed_any |= changed
        hints.tip(
            "How this layer combines with the ones beneath it.")

        changed, layer.strength = imgui.slider_float(
            self._tag(f"Strength##{uid}", collect), layer.strength, 0.0, 4.0)
        changed_any |= changed

        changed, layer.blur = imgui.slider_float(
            self._tag(f"Blur##{uid}", collect), layer.blur, 0.0, 6.0)
        changed_any |= changed
        hints.tip(
            "Softens the source before the mapping reads it.")

        attract = layer.sign >= 0.0
        changed, attract = imgui.checkbox(
            self._tag(f"Attract##{uid}", collect), attract)
        if changed:
            layer.sign = 1.0 if attract else -1.0
            changed_any = True
        hints.tip(
            "Whether the gradient pulls toward the bright regions or away.")

        changed_any |= self._draw_file_picker(layer, collect)

        if layer.error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BAD))
            imgui.text_wrapped(self._tag(layer.error, collect))
            imgui.pop_style_color()

        changed_any |= self._draw_layer_params(layer, collect)

        remove = imgui.button(self._tag(f"Remove##{uid}", collect))
        imgui.separator()

        if changed_any:
            self._mark_dirty()
        return remove

    def _draw_file_picker(self, layer, collect) -> bool:
        """`shader` and `image` are the two sources that name a file.

        Both read `params["_file"]`, so without this the source can be selected
        and never pointed at anything - the silent no-op the stack exists to
        remove.
        """
        if layer.source == "shader":
            names = field_sources.available_shader_files()
            current = layer.params.get("_file", "")
            if not names:
                imgui.text_disabled(self._tag(
                    f"no .frag files found##nofrag{layer.uid}", collect))
                return False
            # A name that no longer resolves is OFFERED BACK, never replaced:
            # silently adopting another file hides the missing one, and the
            # layer's own error is the only place the loss is reported. The
            # empty choice is listed too, or the combo names a file that the
            # layer has not in fact been pointed at.
            options = names if current in names else [current or NO_FILE] + names
            pos = options.index(current) if current in options else 0
            # The label names the selection, as every other combo in the row
            # does, so what is chosen is legible without opening the list.
            changed, pos = imgui.combo(
                self._tag(f"Shader: {current or NO_FILE}##file{layer.uid}",
                          collect), pos, options)
            if changed:
                layer.params["_file"] = "" if options[pos] == NO_FILE else options[pos]
                return True
            return False

        if layer.source == "image":
            changed, value = imgui.input_text(
                self._tag(f"Image##file{layer.uid}", collect),
                layer.params.get("_file", ""))
            hints.tip("Image file, absolute or relative to the Fluoddity folder.")
            if changed:
                layer.params["_file"] = value
                return True
        return False

    def _draw_thumbnail(self, layer, collect) -> None:
        """The row's 48px preview: hover peeks, click pins the Inspect panel."""
        bus = getattr(self, "field_bus", None)
        thumb = bus.thumbnail_for(layer) if bus is not None else None
        if thumb is None:
            return
        imgui.image(imgui.ImTextureRef(thumb.glo), imgui.ImVec2(48, 48))
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.image(imgui.ImTextureRef(thumb.glo), imgui.ImVec2(256, 256))
            imgui.end_tooltip()
        if imgui.is_item_clicked():
            self._inspect_uid = layer.uid
        imgui.same_line()

    def _draw_inspect(self, stack, collect) -> None:
        bus = getattr(self, "field_bus", None)
        uid = getattr(self, "_inspect_uid", None)
        if bus is None or uid is None:
            return
        layer = next((l for l in stack.layers if l.uid == uid), None)
        if layer is None:
            self._inspect_uid = None
            return

        imgui.separator()
        views = ["source", "mapped", "destination"]
        view = getattr(self, "_inspect_view", 0)
        _, view = imgui.combo(self._tag("Inspect##fieldinspect", collect),
                              view, views)
        self._inspect_view = view
        # "mapped" shows the raw source until Stage 2, where optical flow makes
        # a vector-field view worth drawing with the arrow renderer.
        tex = bus.inspect(layer, views[view])
        if tex is not None:
            imgui.image(imgui.ImTextureRef(tex.glo), imgui.ImVec2(384, 384))
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
            if changed:
                layer.params[key] = value
                changed_any = True
        return changed_any

    def _enum(self, current, options, tag, collect):
        pos = options.index(current) if current in options else 0
        changed, pos = imgui.combo(
            self._tag(f"{current}##{tag}", collect), pos, list(options))
        return changed, options[pos]
